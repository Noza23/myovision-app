import logging
from typing import Annotated

from fastapi import Depends, HTTPException, UploadFile, status
from myosam.inference.models.base import MyoObjects
from myosam.inference.pipeline import Pipeline as _Pipeline

from backend.models import Contour, ImageContours, State
from backend.services import MyoSamManager, Redis

logger = logging.getLogger(__name__)

Pipeline = Annotated[_Pipeline, Depends(MyoSamManager.get_pipeline)]


async def get_objects_by_id(image_id: str) -> MyoObjects:
    """Get MyoObjects from Redis by image ID."""
    logger.info(f"Fetching objects for image ID: {image_id}")
    if (objects_json := await Redis.get_by_id(image_id)) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Objects not found for the provided image ID.",
        )
    return MyoObjects.model_validate_json(objects_json)


ImageObjectsByID = Annotated[MyoObjects, Depends(get_objects_by_id)]


async def get_contours_by_id(objects: ImageObjectsByID) -> ImageContours:
    """Get contours from MyoObjects for the response model."""
    logger.info(f"Fetching contours for objects: {objects}")
    return ImageContours(contours=[Contour(coords=x.roi_coords) for x in objects])


ImageContoursByID = Annotated[ImageContours, Depends(get_contours_by_id)]

async def get_objects_or_none_by_id(image_id: str) -> MyoObjects | None:
    """Get MyoObjects from Redis by image ID or return None if not found."""
    logger.info(f"Fetching objects or [None] for image ID: {image_id}")
    if (objects_json := await Redis.get_by_id(image_id)) is None:
        return None
    logger.info(f"Objects found for image ID: {image_id}")
    return MyoObjects.model_validate_json(objects_json)


ImageObjectsOrNoneByID = Annotated[MyoObjects | None, Depends(get_objects_or_none_by_id)]


async def get_validation_state_by_id(image_id: str) -> State:
    """Get validation state from Redis by image ID."""
    logger.info(f"Fetching validation state for image ID: {image_id}")
    if (state_json := await Redis.get_state_by_id(image_id)) is None:
        return State()  # Return an empty state if not found
    return State.model_validate_json(state_json)


ValidationStateByID = Annotated[State, Depends(get_validation_state_by_id)]

async def recieve_file(file: UploadFile, extensions: tuple[str]) -> bytes:
    """Receive a file and validate its extension."""
    logger.info(f"Receving file: {file.filename}")
    if not file.filename.lower().endswith(extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed extensions: {', '.join(extensions)}",
        )
    return await file.read()

ZIP_FILE_EXTENSIONS = (".zip",)
ROIZip = Annotated[
    bytes, Depends(lambda file: recieve_file(file, extensions=ZIP_FILE_EXTENSIONS))
]


IMAGE_FILE_EXTENSIONS = (".png", ".jpeg", ".tif", ".tiff")
ImageFile = Annotated[
    bytes, Depends(lambda file: recieve_file(file, extensions=IMAGE_FILE_EXTENSIONS))
]
