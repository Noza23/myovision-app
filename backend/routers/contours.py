import logging

from anyio import NamedTemporaryFile
from fastapi import APIRouter, HTTPException, status
from read_roi import read_roi_zip

from backend.dependencies import ImageContoursByID, ImageObjectsByID, ROIZip, StateByID
from backend.models import Contour, ImageContours
from backend.services import Redis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{image_id}")
async def get_contours(contours: ImageContoursByID) -> ImageContours:
    """Get contours for a specific image by image ID."""
    return contours


@router.patch("/{image_id}/upload")
async def upload_contours(
    image_id: str, objects: ImageObjectsByID, state: StateByID, file: ROIZip
):
    """Upload contours from a zip file."""
    if not (coords := await load_coords_from_zip(file)):
        logger.warning(f"Empty contours uploaded for image ID {image_id}.")
        return ImageContours(contours=[])
    objects.add_instances_from_coords(coords)
    # NOTE: Append contours to the State. we assume all uploaded contours are valid.
    state.shift_all(len(coords))
    await Redis.set_by_id(image_id, objects.model_dump_json())
    await Redis.set_state_by_id(image_id, state.model_dump_json())
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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed ROI file."
            )
    return [[[x, y] for x, y in zip(roi["x"], roi["y"])] for _, roi in rois.items()]
