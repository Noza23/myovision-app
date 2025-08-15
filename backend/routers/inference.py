import hashlib
import logging

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from backend.dependencies import (
    InferenceAgent,
    MyotubeFile,
    MyotubesAndNucleisByID,
    NucleiFile,
    Pipeline,
    get_myotubes_or_none_by_id,
    get_nucleis_or_none_by_id,
)
from backend.models import InferenceDataResponse, InferenceResponse
from backend.services import MyoSam, Redis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=InferenceResponse)
async def predict(myotube: MyotubeFile, nuclei: NucleiFile, pipeline: Pipeline):
    """Run the pipeline and return the inference Response."""
    myotube_id = hashlib.blake2b(myotube, digest_size=8).digest().hex()
    nuclei_id = hashlib.blake2b(nuclei, digest_size=8).digest().hex()

    pipeline.set_myotube_image(myotube, name=myotube_id)
    pipeline.set_nuclei_image(nuclei, name=nuclei_id)

    myotubes = await get_myotubes_or_none_by_id(myotube_id)
    nucleis = await get_nucleis_or_none_by_id(nuclei_id)
    try:
        result = (
            await run_in_threadpool(pipeline.execute, myotubes, nucleis)
        ).information_metrics
    except Exception as e:
        logger.error("Unexpected error occurred during pipeline execution: %s", e)
        logger.debug(e, exc_info=True)
        msg = f"Pipeline failed for images {myotube_id} and {nuclei_id}"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg
        ) from e

    await Redis.set_objects_by_id(myotube_id, result.myotubes)
    await Redis.set_objects_by_id(nuclei_id, result.nucleis)

    general_info = result.model_dump(exclude={"myotubes", "nucleis", "nuclei_clusters"})
    image_path = MyoSam.generate_fp()
    image = pipeline.draw_contours_on_myotube_image(
        myotubes=result.myotubes, nucleis=result.nucleis
    )
    pipeline.save_image(path=image_path, img=image)
    return InferenceResponse(
        image_path=image_path,
        image_hash=myotube_id,
        image_secondary_hash=nuclei_id,
        general_info=general_info,
    )


@router.websocket("/ws/{myotubes_id}/{nucleis_id}")
async def inference(agent: InferenceAgent):
    """Analyse the inference results using the InferenceAgent."""
    while True:
        # NOTE: Receive point from the client
        point = await agent.receive_point()
        # NOTE: Send the inference data for the point back to the client
        await agent.send_data(point)


@router.get("/{myotubes_id}/{nucleis_id}")
async def get_inference_data(objects: MyotubesAndNucleisByID):
    """Get inference data given myotube and nuclei ids."""
    return InferenceDataResponse(
        myotubes=objects[0], nucleis=objects[1], nuclei_clusters=objects[2]
    )
