import logging
from typing import Annotated

from fastapi import Depends, File, HTTPException, status
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


async def get_validation_state_by_id(image_id: str) -> State:
    """Get validation state from Redis by image ID."""
    logger.info(f"Fetching validation state for image ID: {image_id}")
    if (state_json := await Redis.get_state_by_id(image_id)) is None:
        return State.empty()
    return State.model_validate_json(state_json)


StateByID = Annotated[State, Depends(get_validation_state_by_id)]


ROIZip = Annotated[bytes, File(description="ImageJ generated .zip file with ROIs")]
