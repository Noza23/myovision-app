import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, HTTPException, UploadFile, WebSocket, status
from myosam.inference.models.base import MyoObjects, Myotubes, NucleiClusters, Nucleis
from myosam.inference.pipeline import Pipeline as _Pipeline

from backend.agents import InferenceAgent as _InferenceAgent
from backend.agents import ValidationAgent as _ValidationAgent
from backend.models import Contour, ImageContours, State
from backend.services import MyoSamManager, Redis

logger = logging.getLogger(__name__)

Pipeline = Annotated[_Pipeline, Depends(MyoSamManager.get_pipeline)]


async def get_objects_by_id(image_id: str) -> MyoObjects:
    """Get MyoObjects from Redis by image ID."""
    if (objects := await Redis.get_myoobjects_by_id(image_id)) is None:
        detail = "Objects not found for the provided image ID."
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    return objects


ObjectsByID = Annotated[MyoObjects, Depends(get_objects_by_id)]


async def get_objects_or_none_by_id(image_id: str | None) -> MyoObjects | None:
    """Get MyoObjects from Redis by image ID or return None if not found."""
    return image_id and await Redis.get_myoobjects_by_id(image_id)


ObjectsOrNoneByID = Annotated[MyoObjects | None, Depends(get_objects_or_none_by_id)]


async def get_myotubes_by_id(image_id: str) -> Myotubes:
    """Get Myotubes from Redis by image ID."""
    if (myotubes := await Redis.get_myotubes_by_id(image_id)) is None:
        detail = "Myotubes not found for the provided image ID."
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    return myotubes


MyotubesByID = Annotated[Myotubes, Depends(get_myotubes_by_id)]


async def get_myotubes_or_none_by_id(image_id: str | None) -> Myotubes | None:
    """Get Myotubes from Redis by image ID or return None if not found."""
    return image_id and await Redis.get_myotubes_by_id(image_id)


MyotubesOrNoneByID = Annotated[Myotubes | None, Depends(get_myotubes_or_none_by_id)]


async def get_nucleis_by_id(image_id: str) -> Nucleis | None:
    """Get Nucleis from Redis by image ID or return None if not found."""
    if (nucleis := await Redis.get_nucleis_by_id(image_id)) is None:
        msg = "Nuclei not found for the provided image ID."
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
    return nucleis


NucleisByID = Annotated[Nucleis, Depends(get_nucleis_by_id)]


async def get_nucleis_or_none_by_id(image_id: str | None) -> Nucleis | None:
    """Get Nuclei from Redis by image ID or return None if not found."""
    return image_id and await Redis.get_nucleis_by_id(image_id)


NucleisOrNoneByID = Annotated[Nucleis | None, Depends(get_nucleis_or_none_by_id)]


async def get_myotubes_and_nucleis_by_id(
    myotubes_id: str, nucleis_id: str
) -> tuple[Myotubes, Nucleis, NucleiClusters]:
    """Get Myotubes and Nuclei from Redis by image ID or return None if not found."""
    objects = await Redis.get_myotubes_and_nucleis_by_id(myotubes_id, nucleis_id)
    myotubes, nucleis = objects[0] or Myotubes(), objects[1] or Nucleis()
    clusters = NucleiClusters.compute_clusters(nucleis)
    return myotubes, nucleis, clusters


MyotubesAndNucleisByID = Annotated[
    tuple[Myotubes, Nucleis, NucleiClusters], Depends(get_myotubes_and_nucleis_by_id)
]


async def get_contours_by_id(objects: ObjectsByID) -> ImageContours:
    """Get contours from MyoObjects for the response model."""
    return ImageContours(contours=[Contour(coords=x.roi_coords) for x in objects])


ContoursByID = Annotated[ImageContours, Depends(get_contours_by_id)]


async def get_validation_state_by_id(image_id: str) -> State:
    """Get validation state from Redis by image ID."""
    return await Redis.get_state_by_id(image_id) or State()


StateByID = Annotated[State, Depends(get_validation_state_by_id)]


async def recieve_file(file: UploadFile, extensions: tuple[str]) -> bytes:
    """Receive a file and validate its extension."""
    logger.info("Received file %s", file.filename)
    if not file.filename.lower().endswith(extensions):
        msg = f"Invalid file extension. Allowed extensions: {', '.join(extensions)}"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return await file.read()


ZIP_FILE_EXTENSIONS = (".zip",)
ROIZip = Annotated[
    bytes, Depends(lambda file: recieve_file(file, extensions=ZIP_FILE_EXTENSIONS))
]


IMAGE_FILE_EXTENSIONS = (".png", ".jpeg", ".tif", ".tiff")
ImageFile = Annotated[
    bytes, Depends(lambda file: recieve_file(file, extensions=IMAGE_FILE_EXTENSIONS))
]


async def get_validation_agent(
    image_id: str, websocket: WebSocket, state: StateByID, contours: ContoursByID
) -> AsyncGenerator[_ValidationAgent, None]:
    """Get a ValidationAgent instance for the WebSocket connection."""
    async with _ValidationAgent(
        websocket=websocket,
        image_id=image_id,
        state=state,
        contours=contours.contours,
    ) as agent:
        yield agent


ValidationAgent = Annotated[_ValidationAgent, Depends(get_validation_agent)]


async def get_inference_agent(
    objects: MyotubesAndNucleisByID, websocket: WebSocket
) -> AsyncGenerator[_InferenceAgent, None]:
    """Get an InferenceAgent instance for the WebSocket connection."""
    async with _InferenceAgent(
        websocket=websocket,
        myotubes=objects[0],
        nucleis=objects[1],
        clusters=objects[2],
    ) as agent:
        yield agent


InferenceAgent = Annotated[_InferenceAgent, Depends(get_inference_agent)]
