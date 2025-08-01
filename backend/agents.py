import logging
from enum import IntEnum, StrEnum
from functools import lru_cache
from types import TracebackType
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import WebSocketException

from backend.models import Contour, State
from backend.services import Redis

logger = logging.getLogger(__name__)


class Actions(StrEnum):
    """Enum for WebSocket actions."""

    INIT = "INIT"
    """Initialization of the WebSocket agent."""
    CONNECTING = "CONNECTING"
    """Connecting to the WebSocket."""
    CONNECTED = "CONNECTED"
    """Connected to the WebSocket."""
    DISCONNECTING = "DISCONNECTING"
    """Disconnecting from the WebSocket."""
    DISCONNECTED = "DISCONNECTED"
    """Disconnected from the WebSocket."""
    SENDING = "SENDING"
    """Sending data to the client."""
    WAITING = "WAITING"
    """Waiting for a response from the client."""
    ACTING = "ACTING"
    """Acting on the response from the client."""
    DONE = "DONE"
    """Agent has completed its task."""
    UNEXPECTED_ERROR = "UNEXPECTED ERROR"
    """Unexpected error during the Agent communication."""


class WebSocketAgent:
    """Base class for WebSocket agents."""

    LOG_TEMPLATE = "[{name}] [{identifier}] [{action}]: {message}"
    LOG_TEMPLATE_NO_MSG = LOG_TEMPLATE.rstrip(": {message}")

    def __init__(self, image_id: str, websocket: WebSocket):
        """Initialize the WebSocket agent."""
        self.identifier = uuid4()
        self.image_id = image_id
        self.websocket = websocket
        self.name = self.__class__.__name__
        self.log_action(Actions.INIT, message=self.image_id)

    def log_action(self, action: Actions, message: Any = ""):
        """Log the action performed by the agent in a structured format."""
        template = self.LOG_TEMPLATE if message else self.LOG_TEMPLATE_NO_MSG
        log = template.format(
            name=self.name,
            identifier=self.identifier,
            action=action.value,
            message=message,
        )
        match action:
            case Actions.UNEXPECTED_ERROR:
                logger.error(log)
            case _:
                logger.info(log)

    async def __aenter__(self):
        """Enter the context manager, connecting to the WebSocket."""
        self.log_action(Actions.CONNECTING)
        await self.websocket.accept()
        self.log_action(Actions.CONNECTED)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        """Exit the context manager, disconnecting from the WebSocket."""
        if exc_value is None:
            code = status.WS_1000_NORMAL_CLOSURE
        elif isinstance(exc_value, (WebSocketException, WebSocketDisconnect)):
            code = exc_value.code
        else:
            self.log_action(Actions.UNEXPECTED_ERROR, exc_value)
            logger.debug(exc_value, exc_info=True)
            code = status.WS_1011_INTERNAL_ERROR

        self.log_action(Actions.DISCONNECTING, message=f"{code} - {exc_value}")
        # NOTE: In case of WebSocketDisconnect, starlette will handle closing the connection.
        if exc_type is not WebSocketDisconnect:
            await self.websocket.close(code=code)
        self.log_action(Actions.DISCONNECTED)


class ValidationSignal(IntEnum):
    """Signals for validation decisions sent by the client."""

    UNDO = -1
    """Undo the last signal."""
    INVALID = 0
    """Mark the contour as invalid."""
    VALID = 1
    """Mark the contour as valid."""
    SKIP = 2
    """Skip the contour (Move it to the end of the queue)."""

    @classmethod
    @lru_cache(maxsize=1)
    def to_list(cls) -> list[int]:
        """Return a list of all signal values."""
        return [signal.value for signal in cls]

    @classmethod
    def is_signal(cls, value: int) -> bool:
        """Check if the value is a valid signal."""
        return value in cls.to_list()


class ValidationAgent(WebSocketAgent):
    """Agent for handling validation WebSocket connections."""

    def __init__(
        self, image_id: str, websocket: WebSocket, state: State, contours: list[Contour]
    ):
        """Initialize the ValidationAgent."""
        super().__init__(image_id=image_id, websocket=websocket)
        self.state = state
        self.contours = contours
        self._current_id = state.next()

    @property
    def done(self) -> bool:
        """Check if the validation is done."""
        return self.state.done

    @property
    def current_id(self) -> int:
        """Get the current id of the contour being validated."""
        if self._current_id is None:
            self.log_action(
                Actions.UNEXPECTED_ERROR,
                "Websocket should have been closed after sending the last contour.",
            )
            raise WebSocketDisconnect(
                code=status.WS_1000_NORMAL_CLOSURE, reason="Validation completed."
            )
        return self._current_id

    def get_data_to_send(self):
        """Get the next contour coordinates to send to the client."""
        return {
            "coords": self.contours[self.current_id].coords,
            "id": self.current_id,
            "total": len(self.state),
            "session": self.identifier,
        }

    async def send_contour(self):
        """Send the next contour coordinates to the client to validate."""
        self.log_action(Actions.SENDING, message=self.current_id)
        try:
            data = self.get_data_to_send()
            await self.websocket.send_json(data=data, mode="text")
        except (WebSocketException, WebSocketDisconnect) as e:
            raise e
        except Exception as e:
            self.log_action(
                Actions.UNEXPECTED_ERROR, f"Sending coordinates has failed: {e}"
            )
            logger.debug(e, exc_info=True)
            raise WebSocketException(
                code=status.WS_1011_INTERNAL_ERROR,
                detail="Unexpected error occurred while sending coordinates.",
            ) from None
        # NOTE: Update the current id after successfully sending the data
        self._current_id = self.state.next()

    async def receive_decision(self):
        """Receive the client's signal on the contour sent."""
        self.log_action(Actions.WAITING, message=self.current_id)
        try:
            text = await self.websocket.receive_text()
        except (WebSocketException, WebSocketDisconnect) as e:
            raise e
        except Exception as e:
            self.log_action(
                Actions.UNEXPECTED_ERROR, f"Receiving signal has failed: {e}"
            )
            logger.debug(e, exc_info=True)
            raise WebSocketException(
                code=status.WS_1011_INTERNAL_ERROR,
                detail="Unexpected error occurred while receiving signal.",
            ) from None

        # NOTE: Expecting a single digit signal
        if len(text) != 1 or not (
            text.isdigit() and ValidationSignal.is_signal(int(text))
        ):
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Received invalid signal from client. Expected one of "
                f"{ValidationSignal.to_list()} but got: {text}",
            )

        signal = ValidationSignal(int(text))
        self.log_action(Actions.ACTING, message=signal.name)
        await self.act_on_decision(signal)

    async def act_on_decision(self, signal: ValidationSignal):
        """Act on the signal received from the client."""
        match signal:
            case ValidationSignal.UNDO:
                self.state.undo(self.current_id)
            case ValidationSignal.INVALID:
                self.state.mark_invalid(self.current_id)
            case ValidationSignal.VALID:
                self.state.mark_valid(self.current_id)
            case ValidationSignal.SKIP:
                self.state.skip(self.current_id)
            case _:
                raise RuntimeError(f"Unexpected signal received: {signal}.")

        # NOTE: Persist the state after each recieved ValidationSignal
        if not await Redis.set_state_by_id(self.image_id, self.state):
            self.log_action(
                Actions.UNEXPECTED_ERROR,
                f"Failed to persist state for image ID: {self.image_id}",
            )


class InferenceAgent(WebSocketAgent):
    """Agent for handling inference WebSocket connections."""
