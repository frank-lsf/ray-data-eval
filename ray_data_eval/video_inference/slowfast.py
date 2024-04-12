import json
import time

import numpy as np
import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

DATA_DIR = "/mnt/data/ray-data-eval/kinetics"
DEVICE = "cuda"


def _get_kinetics_classnames() -> dict[str, str]:
    with open(f"{DATA_DIR}/kinetics_classnames.json", "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    return kinetics_id_to_classname


def repeat(x):
    return torch.cat([x] * 12, dim=0)


class Classifier:
    def __init__(self):
        start_time = time.time()
        self.model = (
            torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
            .eval()
            .to(DEVICE)
        )
        self.post_act = torch.nn.Softmax(dim=1)
        self.kinetics_id_to_classname = _get_kinetics_classnames()
        print(f"Time to initialize model: {time.time() - start_time}")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        start_time = time.time()
        inputs = batch["frames"]
        for i, _ in enumerate(inputs):
            inputs[i] = repeat(inputs[i])
        preds = self.model(inputs)
        print(preds.shape)
        preds = self.post_act(preds)
        print(preds.shape)
        topk_indices = preds.topk(k=5, dim=1).indices
        batch_pred_class_names = []
        for pred_classes in topk_indices:
            pred_class_names = [self.kinetics_id_to_classname[int(i)] for i in pred_classes]
            pred_class_names = ", ".join(pred_class_names)
            batch_pred_class_names.append(pred_class_names)
        print(f"Time to process batch: {time.time() - start_time}")
        return {"results": np.array(batch_pred_class_names)}


SIDE_SIZE = 256
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
CROP_SIZE = 256
NUM_FRAMES = 32
SAMPLING_RATE = 2
FPS = 30
SLOWFAST_ALPHA = 4

# The duration of the input clip is also specific to the model.
CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FPS


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // SLOWFAST_ALPHA).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(NUM_FRAMES),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(MEAN, STD),
            ShortSideScale(size=SIDE_SIZE),
            CenterCropVideo(CROP_SIZE),
            PackPathway(),
        ]
    ),
)

start_time = time.time()

video_path = f"{DATA_DIR}/archery.mp4"
# video_path = "/mnt/data/ray-data-eval/kinetics/Kinetics700-2020-test/-LO2DhhIdp0_000111_000121.mp4"

# Select the duration of the clip to load by specifying the start and end duration
# The start_sec should correspond to where the action occurs in the video
START_SEC = 0
END_SEC = START_SEC + CLIP_DURATION

# Initialize an EncodedVideo helper class and load the video
video = EncodedVideo.from_path(video_path)

# Load the desired clip
video_data = video.get_clip(start_sec=START_SEC, end_sec=CLIP_DURATION)

# Apply a transform to normalize the video input
video_data = transform(video_data)

# Move the inputs to the desired device
inputs = video_data["video"]
inputs = [i.to(DEVICE)[None, ...] for i in inputs]

print(f"Time to load and preprocess video: {time.time() - start_time}")

classifier = Classifier()
results = classifier({"frames": inputs})
print(results)
