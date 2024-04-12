import numpy as np

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)
import torch

from decord import VideoReader

DEVICE = "cuda"
MODEL_ID = "MCG-NJU/videomae-huge-finetuned-kinetics"
DATA_DIR = "/mnt/data/ray-data-eval/kinetics"
video_path = f"{DATA_DIR}/archery.mp4"

print(DEVICE)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
model = model.eval().to(DEVICE)
print(f"Number of parameters: {model.num_parameters()}")

# video clip consists of 300 frames (10 seconds at 30 FPS)
vr = VideoReader(video_path, num_threads=1)


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    str_idx = end_idx - converted_len
    index = np.linspace(str_idx, end_idx, num=clip_len)
    index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
    return index


vr.seek(0)
index = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(vr))
video = vr.get_batch(index).asnumpy()  # [nframes, height, width, channels]
video = list(video)  # as a list of frames

inputs = processor(video, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    print(inputs["pixel_values"].shape)
    outputs = model(**inputs)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
