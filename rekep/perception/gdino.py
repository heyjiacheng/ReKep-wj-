from dds_cloudapi_sdk import Config, Client, DetectionTask, TextPrompt, DetectionModel, DetectionTarget
import os

API_TOKEN = os.getenv("DDS_CLOUDAPI_TEST_TOKEN")
MODEL = "GDino1_5_Pro"
DETECTION_TARGETS = ["Mask", "BBox"]

class GroundingDINO:
    def __init__(self):
        config = Config(API_TOKEN)
        self.client = Client(config)

    def detect_objects(self, image_path, prompts):
        image_url = self.client.upload_file(image_path)
        task = DetectionTask(
            image_url=image_url,
            prompts=[TextPrompt(text=prompt) for prompt in prompts],
            targets=[getattr(DetectionTarget, target) for target in DETECTION_TARGETS],
            model=getattr(DetectionModel, MODEL),
        )
        self.client.run_task(task)
        return task.result

    def rle2rgba(self, rle_mask):
        # Create a dummy task with minimal required arguments
        dummy_task = DetectionTask(
            image_url="dummy",
            prompts=[TextPrompt(text="dummy")],
            targets=[DetectionTarget.Mask],
            model=getattr(DetectionModel, MODEL)
        )
        return dummy_task.rle2rgba(rle_mask)