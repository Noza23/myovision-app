import logging

from fastapi import APIRouter, HTTPException, status
from myosam.inference.predictors.utils import invert_image
from starlette.concurrency import run_in_threadpool

from backend.agents import Action
from backend.dependencies import (
    ImageFile,
    Pipeline,
    ValidationAgent,
    get_myotubes_or_none_by_id,
    get_validation_state_by_id,
)
from backend.models import Config, ValidationResponse
from backend.services import MyoSam, Redis

logger = logging.getLogger(__name__)

router = APIRouter()

CONTOUR_COLOR = (0, 255, 0)  # Green color for contours
CONTOUR_THICKNESS = 3  # Thickness of the contour lines


@router.post("/", response_model=ValidationResponse)
async def predict(config: Config, image: ImageFile, pipeline: Pipeline):
    """Run the pipeline and return the validation response."""
    image_id = str(hash(image))
    pipeline.set_myotube_image(image=image, name=image_id)
    validated_image = pipeline.myotube_image_np.copy()
    if config.invert:  # NOTE: Invert the image if the config requires it
        validated_image = invert_image(validated_image)

    myotubes = await get_myotubes_or_none_by_id(image_id)
    state = await get_validation_state_by_id(image_id)

    if myotubes:  # NOTE: There is already a validation state for uploaded image
        myotubes.adjust_measure_unit(measure_unit=config.mu)
        validated_image = pipeline.draw_contours(
            validated_image,
            [myotubes[i].roi_coords_np for i in state.valid],
            color=CONTOUR_COLOR,
            thickness=CONTOUR_THICKNESS,
        )
    else:  # NOTE: There is no validation state, so we need to run the pipeline
        try:
            myotubes = (
                await run_in_threadpool(pipeline.execute)
            ).information_metrics.myotubes
            # NOTE: All new myotubes are recorded in the state with status "no decision"
            state.add_no_decisions(len(myotubes))
        except Exception as e:
            logger.error("Unexpected error during pipeline execution: %s", e)
            logger.debug(e, exc_info=True)
            msg = f"Pipeline failed for image {image_id}"
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=msg,
            ) from e

    await Redis.set_objects_and_state_by_id(image_id, objects=myotubes, state=state)
    image_path = MyoSam.generate_fp()
    pipeline.save_image(path=image_path, img=validated_image)
    return ValidationResponse(image_id, image_path)


@router.websocket("/ws/{image_id}")
async def validate(agent: ValidationAgent):
    """Validate contours interactively using ValidationAgent."""
    while not agent.done:
        # NOTE: Send the next contour coordinates to the client for validation
        await agent.send_contour()
        # NOTE: wait for the client to send a decision for the contour sent
        await agent.receive_decision()
    agent.log_action(Action.DONE)
