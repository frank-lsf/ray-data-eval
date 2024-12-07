import time
import pickle
from PIL import Image

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration
from pyflink.datastream.functions import (
    FlatMapFunction,
    RuntimeContext,
    MapFunction,
    ProcessFunction,
)

import torch
import numpy as np
from diffusers import AutoPipelineForImage2Image
from ray_data_eval.image_generation.raydata.ray_data_pipeline_helpers import (
    ChromeTracer,
    # CsvLogger,
)


from ray_data_eval.image_generation.common import (
    get_image_prompt,
    transform_image,
    encode_and_upload,
    IMAGE_PROMPTS_DF,
    S3_BUCKET_NAME,
)
from ray_data_eval.image_generation.flink.common import postprocess_logs
from ray_data_eval.image_generation.spark.spark_fused import S3Handler

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("flink_logs.log", mode="w")
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

LOGGER = logger

EXECUTION_MODE = "process"
MB = 1024 * 1024

NUM_TASKS = 160 * 5
BLOCK_SIZE = int(1 * MB)
TIME_UNIT = 0.5

NUM_CPUS = 8

PRODUCER_PARALLELISM = 8
INFERENCE_PARALLELISM = 1
CONSUMER_PARALLELISM = 8

NUM_ROWS_PER_PRODUCER = 1
NUM_ROWS_PER_CONSUMER = 1


CSV_FILENAME = "log.csv"
GPU_TIMELINE_FILENAME = "gpu_timeline.json"
RESOLUTION = 512
BATCH_SIZE = 10


def record(log, logger=LOGGER):
    """
    @ronyw: When running with thread mode, flush the output to `flink_logs.log`
    `python run.py > flink_logs.log`
    """
    if EXECUTION_MODE == "thread":
        print(log)
    else:
        logger.info(log)


class Model:
    def __init__(self):
        self.tracer = ChromeTracer(GPU_TIMELINE_FILENAME)
        self.model = AutoPipelineForImage2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")  # StableDiffusionImg2ImgPipeline

        self.start_time = time.perf_counter()
        self.last_end_time = self.start_time
        self.total_num_rows = 0

    def __call__(self, images, prompts):
        inference_start_time = time.perf_counter()

        with self.tracer.profile("task:gpu_execution"):
            output_batch = self.model(
                prompt=prompts,
                image=images,
                height=RESOLUTION,
                width=RESOLUTION,
                num_inference_steps=2,
                output_type="np",
            )

        inference_end_time = time.perf_counter()
        num_rows = len(images)
        self.total_num_rows += num_rows
        self.last_end_time = inference_end_time
        self.tracer.save()
        return output_batch


class Producer(FlatMapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def flat_map(self, value):
        producer_start = time.perf_counter()
        image: Image = S3Handler(S3_BUCKET_NAME).download_image(value)
        image = transform_image(image, busy=True)
        producer_end = time.perf_counter()
        log = {
            "cat": "producer:" + str(self.task_index),
            "name": "producer:" + str(self.task_index),
            "pid": "",
            "tid": "",
            "ts": f"{producer_start * 1e6:.0f}",
            "dur": f"{producer_end * 1e6 - producer_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        record(log)
        for _ in range(NUM_ROWS_PER_PRODUCER):
            yield np.array([np.array(image)] * 100), value


class Inference(ProcessFunction):
    def open(self, runtime_context: RuntimeContext):
        self.images_batch = []
        self.prompts_batch = []
        self.paths = []
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()
        self.model = Model()

    def process_element(self, value, ctx: "ProcessFunction.Context"):
        self.images_batch.append(value[0])
        self.paths.append(value[1])
        self.prompts_batch.append(get_image_prompt(value[1]))

        if len(self.images_batch) >= BATCH_SIZE:
            inference_start = time.perf_counter()
            output = self.model(self.images_batch, self.prompts_batch)
            output = output.images
            inference_end = time.perf_counter()
            log = {
                "cat": "inference:" + str(self.task_index),
                "name": "inference:" + str(self.task_index),
                "pid": "",
                "tid": "",
                "ts": f"{inference_start * 1e6:.0f}",
                "dur": f"{inference_end * 1e6 - inference_start * 1e6:.0f}",
                "ph": "X",
                "args": {},
            }
            record(log)
            paths = self.paths.copy()
            self.images_batch.clear()
            self.prompts_batch.clear()
            self.paths.clear()
            output = pickle.dumps(output)
            yield output, paths


class Consumer(MapFunction):
    def open(self, runtime_context):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def map(self, value):
        output = pickle.loads(value[0])
        path = value[1]
        batch = {"path": path, "image": output}

        consumer_start = time.perf_counter()
        encode_and_upload(batch, busy=True)
        consumer_end = time.perf_counter()
        log = {
            "cat": "consumer:" + str(self.task_index),
            "name": "consumer:" + str(self.task_index),
            "pid": "",
            "tid": "",
            "ts": f"{consumer_start * 1e6:.0f}",
            "dur": f"{consumer_end * 1e6 - consumer_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        record(log)
        return BATCH_SIZE


def run_flink(env):
    start = time.perf_counter()

    ds = env.from_collection(IMAGE_PROMPTS_DF.index[:NUM_TASKS], type_info=Types.STRING())

    ds = ds.flat_map(
        Producer(), output_type=Types.TUPLE([Types.PICKLED_BYTE_ARRAY(), Types.STRING()])
    ).set_parallelism(PRODUCER_PARALLELISM)

    ds = ds.process(
        Inference(),
        output_type=Types.TUPLE([Types.PICKLED_BYTE_ARRAY(), Types.BASIC_ARRAY(Types.STRING())]),
    ).set_parallelism(INFERENCE_PARALLELISM)

    ds = ds.map(Consumer(), output_type=Types.INT()).set_parallelism(CONSUMER_PARALLELISM)

    result = []
    for length in ds.execute_and_collect():
        result.append(length)
        print(f"Processed block of size: {sum(result)}/{NUM_TASKS}\n")

    total_length = sum(result)
    end = time.perf_counter()
    print(f"Total data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")


def run_experiment():
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)
    config.set_integer("taskmanager.numberOfTaskSlots", NUM_CPUS)

    env = StreamExecutionEnvironment.get_execution_environment(config)
    run_flink(env)
    postprocess_logs()


def main():
    run_experiment()


if __name__ == "__main__":
    main()
