import logging

from anyio import NamedTemporaryFile
from fastapi import APIRouter, HTTPException, status
from read_roi import read_roi_zip

from backend.dependencies import (
    ContoursByID,
    ObjectsByID,
    ROIZip,
    StateByID,
)
from backend.models import Contour, ImageContours
from backend.services import Redis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{image_id}")
async def get_contours(contours: ContoursByID) -> ImageContours:
    """Get contours for a specific image by image ID."""
    return contours


@router.patch("/{image_id}/upload")
async def upload_contours(
    image_id: str, objects: ObjectsByID, state: StateByID, file: ROIZip
):
    """Upload contours from a ROI zip file exported from ImageJ."""
    if not (coords := await load_coords_from_zip(file)):
        logger.warning(f"Empty contours uploaded for image ID {image_id}.")
        return ImageContours(contours=[])
    objects.add_instances_from_coords(coords)
    # NOTE: Update the state: we assume all uploaded contours are valid.
    state.add_valids(len(coords))
    await Redis.set_objects_and_state_by_id(image_id, objects=objects, state=state)
    return ImageContours(contours=[Contour(coords=x) for x in coords])


async def load_coords_from_zip(file: bytes) -> list[list[list[int]]]:
    """Load coordinatres from a byte stream of a zip file from ImageJ."""
    async with NamedTemporaryFile(mode="wb+", delete=True) as f:
        await f.write(file)
        await f.seek(0)
        try:
            rois = read_roi_zip(zip_path=f.name)
        except Exception as e:
            logger.error(f"Failed to read ROIs from file: {e}")
            logger.debug(e, exc_info=True)
            raise e
    return [[[x, y] for x, y in zip(roi["x"], roi["y"])] for _, roi in rois.items()]
