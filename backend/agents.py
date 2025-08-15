import logging
from enum import IntEnum, StrEnum
from functools import lru_cache
from types import TracebackType
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import WebSocketException
from myosam.inference.models.base import Myotubes, NucleiClusters, Nucleis

from backend.models import Contour, Point, State
from backend.services import Redis

logger = logging.getLogger(__name__)


class Action(StrEnum):
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

    def __init__(self, websocket: WebSocket):
        """Initialize the WebSocket agent."""
        self.identifier = uuid4()
        self.websocket = websocket
        self.name = self.__class__.__name__
        self.log_action(Action.INIT)

    def log_action(self, action: Action, message: Any = ""):
        """Log the action performed by the agent in a structured format."""
        log = self.LOG_TEMPLATE.format(
            name=self.name,
            identifier=self.identifier,
            action=action.value,
            message=message,
        )
        match action:
            case Action.UNEXPECTED_ERROR:
                logger.error(log)
            case _:
                logger.info(log)

    async def __aenter__(self):
        """Enter the context manager, connecting to the WebSocket."""
        self.log_action(Action.CONNECTING)
        await self.websocket.accept()
        self.log_action(Action.CONNECTED)
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
            self.log_action(Action.UNEXPECTED_ERROR, exc_value)
            code = status.WS_1011_INTERNAL_ERROR

        self.log_action(Action.DISCONNECTING, message=f"{code} - {exc_value}")
        # NOTE: In case of WebSocketDisconnect | WebSocketException, starlette handles
        # closing the connection.
        if not isinstance(exc_value, (WebSocketDisconnect, WebSocketException)):
            await self.websocket.close(code=code)
        self.log_action(Action.DISCONNECTED)


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
        self, websocket: WebSocket, image_id: str, state: State, contours: list[Contour]
    ):
        """Initialize the ValidationAgent."""
        super().__init__(websocket=websocket)
        self.image_id = image_id
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
                Action.UNEXPECTED_ERROR,
                "Websocket should have been closed after sending the last contour.",
            )
            msg = "Validation completed."
            raise WebSocketDisconnect(code=status.WS_1000_NORMAL_CLOSURE, reason=msg)
        return self._current_id

    def get_data_to_send(self):
        """Get the next contour coordinates to send to the client."""
        return {
            "id": self.current_id,
            "coords": self.contours[self.current_id].coords,
            "total": len(self.state),
            "session": str(self.identifier),
        }

    async def send_contour(self):
        """Send the next contour coordinates to the client to validate."""
        self.log_action(Action.SENDING, message=self.current_id)
        try:
            data = self.get_data_to_send()
            await self.websocket.send_json(data=data, mode="text")
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected while sending contour.")
            raise
        except Exception as e:
            self.log_action(
                Action.UNEXPECTED_ERROR, message=f"Sending coordinates has failed: {e}"
            )
            raise e

    async def receive_decision(self):
        """Receive the client's signal on the contour sent."""
        self.log_action(Action.WAITING, message=self.current_id)
        try:
            digit = await self.websocket.receive_json()
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected while waiting for decision.")
            raise
        except Exception as e:
            self.log_action(
                Action.UNEXPECTED_ERROR, message=f"Receiving signal has failed: {e}"
            )
            raise e

        # NOTE: Expecting a single digit signal
        if not isinstance(digit, int) or not ValidationSignal.is_signal(digit):
            msg = (
                "Received invalid signal from client. Expected one of "
                f"{ValidationSignal.to_list()} but got: {digit}"
            )
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason=msg)

        signal = ValidationSignal(digit)
        self.log_action(Action.ACTING, message=signal.name)
        await self.act_on_decision(signal)
        # NOTE: Update the current id after successfully validating the decision
        self._current_id = self.state.next()

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
                msg = f"Unexpected signal received: {signal}."
                raise RuntimeError(msg)

        # NOTE: Persist the state after each received ValidationSignal
        if not await Redis.set_state_by_id(self.image_id, self.state):
            self.log_action(
                Action.UNEXPECTED_ERROR,
                f"Failed to persist state for image ID: {self.image_id}",
            )


class InferenceAgent(WebSocketAgent):
    """Agent for handling inference WebSocket connections."""

    def __init__(
        self,
        websocket: WebSocket,
        myotubes: Myotubes,
        nucleis: Nucleis,
        clusters: NucleiClusters,
    ):
        """Initialize the InferenceAgent."""
        super().__init__(websocket=websocket)
        self.myotubes = myotubes
        self.nucleis = nucleis
        if not any((len(myotubes), len(nucleis))):
            msg = "No Myotubes or Nucleis found, Agent cannot do inference."
            raise WebSocketDisconnect(code=status.WS_1000_NORMAL_CLOSURE, reason=msg)
        self.clusters = clusters

    async def receive_point(self):
        """Receive a point from the client and return the inference data for it."""
        try:
            self.log_action(Action.WAITING)
            data = await self.websocket.receive_json()
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected while waiting for point.")
            raise
        except Exception as e:
            self.log_action(
                Action.UNEXPECTED_ERROR, message=f"Receiving Point has failed: {e}"
            )
            raise e
        try:
            point = Point(**data)
        except TypeError as e:
            msg = f"Invalid data for Point: {data}. Expected: {{'x': int, 'y': int}}"
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION, reason=msg
            ) from e
        return point

    def get_data_to_send(self, point: Point):
        """Get the data to send for a specific point."""
        if not (myotube := self.myotubes.get_instance_by_point(point)):
            return {"myotube": None, "clusters": None}

        clusters = self.clusters.get_clusters_by_myotube_id(myotube.identifier)
        myotube = myotube.model_dump(exclude=["roi_coords", "nuclei_ids"])
        data = {
            "myotube": myotube,
            "clusters": [cluster.model_dump() for cluster in clusters],
        }

        # NOTE: Rounding floating values, ugly but lets come back later
        for k, v in myotube.items():
            if isinstance(v, float):
                v = round(v, 2)
            elif isinstance(v, (tuple, list)):
                v = [round(x, 2) if isinstance(x, float) else x for x in v]
            data["myotube"][k] = v

        return data

    async def send_data(self, point: Point):
        """Send inference data for a specific point."""
        data = self.get_data_to_send(point)
        self.log_action(Action.SENDING, message=point)
        try:
            await self.websocket.send_json(data=data, mode="text")
        except WebSocketDisconnect:
            raise
        except Exception as e:
            self.log_action(
                Action.UNEXPECTED_ERROR, message=f"Sending data has failed: {e}"
            )
            raise e
        self.log_action(Action.ACTING, message=f"Sent data for point: {point}")
