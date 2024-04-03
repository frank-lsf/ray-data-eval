import functools
import io
import json
import os
import time

import humanize
from pytorchvideo.data.encoded_video import EncodedVideoPyAV
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

DEVICE = "cuda"
SIDE_SIZE = 256
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
CROP_SIZE = 256
NUM_FRAMES = 32
SAMPLING_RATE = 2
FPS = 30
CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FPS
START_SEC = 0
END_SEC = START_SEC + CLIP_DURATION

DATA_PATH = "/mnt/data/ray-data-eval/kinetics"
INPUT_PATH = "/mnt/data/ray-data-eval/kinetics/Kinetics700-2020-test"

ModelInputType = tuple[torch.Tensor, torch.Tensor]


def tensor_size(t: torch.Tensor) -> str:
    return humanize.naturalsize(t.element_size() * t.nelement())


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time() - start:.2f} seconds")
        return result

    return wrapper


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    SLOWFAST_ALPHA = 4

    def forward(self, frames: torch.Tensor) -> ModelInputType:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.SLOWFAST_ALPHA).long(),
        )
        return slow_pathway, fast_pathway


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


@timeit
def load_file(file_path: str) -> io.BytesIO:
    with open(file_path, "rb") as f:
        return io.BytesIO(f.read())


@timeit
def preprocess_video(video_bytes: io.BytesIO) -> torch.Tensor:
    print("video bytes", humanize.naturalsize(video_bytes.getbuffer().nbytes))
    video = EncodedVideoPyAV(video_bytes)
    video_data = video.get_clip(start_sec=START_SEC, end_sec=CLIP_DURATION)
    print("decoded video size", tensor_size(video_data["video"]))
    video_data = transform(video_data)
    inputs = video_data["video"]
    return inputs


def _get_kinetics_classnames() -> dict[str, str]:
    with open(f"{DATA_PATH}/kinetics_classnames.json", "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    return kinetics_id_to_classname


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

    @timeit
    def __call__(self, batch: ModelInputType) -> list:
        for tensor in batch:
            print(tensor_size(tensor))
        batch = [tensor.to(DEVICE) for tensor in batch]
        preds = self.model(batch)
        preds = self.post_act(preds)
        topk_indices = preds.topk(k=5, dim=1).indices
        batch_pred_class_names = []
        for pred_classes in topk_indices:
            pred_class_names = [self.kinetics_id_to_classname[int(i)] for i in pred_classes]
            pred_class_names = ", ".join(pred_class_names)
            print(pred_class_names)
            batch_pred_class_names.append(pred_class_names)
        return batch_pred_class_names


class SimpleIterator:
    def __init__(self):
        self.files = os.listdir(INPUT_PATH)
        self.file_index = 0

    def __iter__(self):
        return self

    def __next__(self) -> ModelInputType:
        if self.file_index >= len(self.files):
            raise StopIteration
        file = self.files[self.file_index]
        file_path = os.path.join(INPUT_PATH, file)
        video_bytes = load_file(file_path)
        model_input = preprocess_video(video_bytes)
        self.file_index += 1
        return model_input


class BatchIterator:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self._iter = SimpleIterator()

    def __iter__(self):
        return self

    @timeit
    def __next__(self) -> ModelInputType:
        tensor_lists = [], []
        for _ in range(self.batch_size):
            slow, fast = next(self._iter)
            tensor_lists[0].append(slow)
            tensor_lists[1].append(fast)
        return [torch.stack(tensor_list) for tensor_list in tensor_lists]


def main():
    n_batches = 0
    batch_size = 8
    classifier = Classifier()
    batch_iterator = BatchIterator(batch_size=batch_size)
    for batch in batch_iterator:
        classifier(batch)
        n_batches += 1
        if n_batches >= 2:
            break


if __name__ == "__main__":
    main()
