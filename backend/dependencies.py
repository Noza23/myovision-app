import logging
from typing import Annotated

from fastapi import Depends, HTTPException, UploadFile, status
from myosam.inference.models.base import MyoObjects, Myotubes
from myosam.inference.pipeline import Pipeline as _Pipeline

from backend.models import Contour, ImageContours, State
from backend.services import MyoSamManager, Redis

logger = logging.getLogger(__name__)

Pipeline = Annotated[_Pipeline, Depends(MyoSamManager.get_pipeline)]


async def get_objects_by_id(image_id: str) -> MyoObjects:
    """Get MyoObjects from Redis by image ID."""
    if (objects := await Redis.get_myoobjects_by_id(image_id)) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Objects not found for the provided image ID.",
        )
    return objects


ObjectsByID = Annotated[MyoObjects, Depends(get_objects_by_id)]


async def get_objects_or_none_by_id(image_id: str) -> MyoObjects | None:
    """Get MyoObjects from Redis by image ID or return None if not found."""
    logger.info(f"[GET] Objects or None for image ID: {image_id}")
    return await Redis.get_myoobjects_by_id(image_id)


ObjectsOrNoneByID = Annotated[MyoObjects | None, Depends(get_objects_or_none_by_id)]


async def get_myotubes_by_id(image_id: str) -> Myotubes:
    """Get Myotubes from Redis by image ID."""
    if (myotubes := await Redis.get_myoobjects_by_id(image_id)) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Myotubes not found for the provided image ID.",
        )
    return myotubes


MyotubesByID = Annotated[Myotubes, Depends(get_myotubes_by_id)]


async def get_myotubes_or_none_by_id(image_id: str) -> Myotubes | None:
    """Get Myotubes from Redis by image ID or return None if not found."""
    logger.info(f"[GET] Myotubes or None for image ID: {image_id}")
    return await Redis.get_myoobjects_by_id(image_id)


MyotubesOrNoneByID = Annotated[Myotubes | None, Depends(get_myotubes_or_none_by_id)]


async def get_contours_by_id(image_id: str, objects: ObjectsByID) -> ImageContours:
    """Get contours from MyoObjects for the response model."""
    logger.info(f"[GET] ImageContours for image ID: {image_id}")
    return ImageContours(contours=[Contour(coords=x.roi_coords) for x in objects])


ContoursByID = Annotated[ImageContours, Depends(get_contours_by_id)]


async def get_validation_state_by_id(image_id: str) -> State:
    """Get validation state from Redis by image ID."""
    return await Redis.get_state_by_id(image_id) or State()


StateByID = Annotated[State, Depends(get_validation_state_by_id)]


async def recieve_file(file: UploadFile, extensions: tuple[str]) -> bytes:
    """Receive a file and validate its extension."""
    logger.info(f"Received File: {file.filename}")
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
