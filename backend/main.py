import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from read_roi._read_roi import UnrecognizedRoiType
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from backend.logger import setup_logging
from backend.models import Config, HealthCheck, RootInfo
from backend.routers.contours import router as contours_router
from backend.routers.inference import router as inference_router
from backend.routers.validation import router as validation_router
from backend.services import MyoSam, Redis
from backend.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan of application."""
    setup_logging(get_settings().log_level)
    MyoSam.setup()

    yield

    MyoSam.cleanup()
    await Redis.close()
    await asyncio.sleep(0)  # Graceful shutdown


app = FastAPI(lifespan=lifespan, title="MyoVision API", version="0.1.0")
app.include_router(contours_router, prefix="/contours", tags=["contours"])
app.include_router(inference_router, prefix="/inference", tags=["inference"])
app.include_router(validation_router, prefix="/validation", tags=["validation"])
app.mount(f"/{MyoSam.cache_dir.strip('/')}", StaticFiles(directory=MyoSam.cache_dir))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", status_code=status.HTTP_200_OK, response_model=RootInfo)
def get_root() -> RootInfo:
    """Root endpoint."""
    return RootInfo()


@app.post("/health", status_code=status.HTTP_200_OK, response_model=HealthCheck)
@app.get("/health", status_code=status.HTTP_200_OK, response_model=HealthCheck)
def get_health() -> HealthCheck:
    """Check the health of the application."""
    return HealthCheck(status="OK")


@app.get("/check-redis", status_code=status.HTTP_200_OK, response_model=HealthCheck)
async def redis_status() -> HealthCheck:
    """Check status of the redis conncetion."""
    await Redis.ping()
    return HealthCheck(status="OK")


@app.get("/get-config-schema/", response_model=dict[str, Any])
def get_config_chema() -> dict[str, Any]:
    """Get the configuration schema of the pipeline."""
    return Config.model_json_schema()["$defs"]


def _redis_connection_error_handler(request: Request, exc: Exception):
    """Handle Redis connection errors."""
    logger.error(
        "Redis Error occurred when processing request for %s: %s",
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Service is temporarily unavailable. Please try again later."
        },
    )


async def _request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Handle request validation errors."""
    logger.error(
        "Validation error occurred when processing request for %s: %s",
        request.url.path,
        exc,
    )
    return await request_validation_exception_handler(request, exc)


def _unrecognized_roi_type_handler(request: Request, exc: UnrecognizedRoiType):
    """Handle unrecognized ROI type errors."""
    logger.error("Unrecognized ROI type for request %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Unrecognized ROI type in the uploaded file."},
    )


app.add_exception_handler(RedisConnectionError, _redis_connection_error_handler)
app.add_exception_handler(RedisTimeoutError, _redis_connection_error_handler)
app.add_exception_handler(RequestValidationError, _request_validation_exception_handler)
app.add_exception_handler(UnrecognizedRoiType, _unrecognized_roi_type_handler)
