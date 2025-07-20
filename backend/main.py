import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from redis.exceptions import ConnectionError, TimeoutError

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
    assert setup_logging(get_settings().log_level)
    MyoSam.setup()    

    yield

    MyoSam.cleanup()
    await Redis.close()
    await asyncio.sleep(0)  # Graceful shutdown


app = FastAPI(lifespan=lifespan, title="MyoVision API", version="0.1.0")
app.include_router(contours_router, prefix="/contours", tags=["contours"])
app.include_router(inference_router, prefix="/inference", tags=["inference"])
app.include_router(validation_router, prefix="/validation", tags=["validation"])
app.mount(MyoSam.cache_dir, StaticFiles(directory=MyoSam.cache_dir), name="cache")


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


@app.get("/get-config-schema/", response_model=dict[str, str])
def get_config_chema() -> dict[str, str]:
    """Get the configuration schema of the pipeline."""
    return Config.model_json_schema()["$defs"]


async def _redis_connection_error_handler(request: Request, exc: Exception):
    """Handle Redis connection errors."""
    logger.info(f"Redis Error occured when processing request for {request.url.path}")
    logger.error(exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Service is temporarily unavailable. Please try again later."
        },
    )


app.add_exception_handler(ConnectionError, _redis_connection_error_handler)
app.add_exception_handler(TimeoutError, _redis_connection_error_handler)
