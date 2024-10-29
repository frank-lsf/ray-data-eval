import io

import boto3
import pandas as pd
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms.functional as F

BUCKET = "ray-data-eval-us-west-2"
IMAGENET_LOCAL_DIR = "/mnt/data/ray-data-eval/ILSVRC/Data/CLS-LOC/10k"


def get_image_file_paths(limit: int = -1) -> list[str]:
    ret = []
    with open("../manifests/imagenet-manifest.txt", "r") as fin:
        for line in fin:
            try:
                _, _, _, path = line.strip().split(maxsplit=3)
                ret.append(path)
                if len(ret) % 100_000 == 0:
                    print(len(ret))
                if limit > 0 and len(ret) >= limit:
                    break
            except ValueError as e:
                print(line.strip().split(maxsplit=3))
                raise e
    return ret


spark = (
    SparkSession.builder.appName("Image Batch Inference")
    .config("spark.executor.resource.gpu.amount", "1")  # Total GPUs per executor
    .config("spark.task.resource.gpu.amount", "1")  # GPUs per task
    .config("spark.executor.memory", "2g")
    .config("spark.executor.instances", "13")
    .getOrCreate()
)
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "100")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
sc = spark.sparkContext

print("Nums gpus per task: ", spark.conf.get("spark.task.resource.gpu.amount"))
print("Executor memory: ", spark.conf.get("spark.executor.memory"))


# model_state = resnet50(weights=ResNet50_Weights.DEFAULT).state_dict()
# bc_model_state = sc.broadcast(model_state)


def transform_image(image: Image) -> torch.Tensor:
    image = image.resize((232, 232), resample=Image.BILINEAR)
    image = image.convert("RGB")
    image = F.pil_to_tensor(image)
    image = F.center_crop(image, 224)
    image = F.convert_image_dtype(image, torch.float)
    image = F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return image


def load_image(file_path: str) -> torch.Tensor:
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET, Key=file_path)
    image_bytes = response["Body"].read()

    image = Image.open(io.BytesIO(image_bytes))
    image = transform_image(image)
    return image


@pandas_udf(ArrayType(FloatType()))
def load_and_transform(path: pd.Series) -> pd.Series:
    print(f"number of paths: {len(path)}")
    return path.apply(load_image)


# limit = 1 * 1000
# file_paths = get_image_file_paths(limit)
# print(len(file_paths))

# df = spark.createDataFrame(file_paths, "string").toDF("path")
# df = df.select(load_and_transform(col("path")))
# df.write.format("parquet").mode("overwrite").save("output.parquet")
# print(df)

df = spark.read.format("image").option("dropInvalid", True).load(IMAGENET_LOCAL_DIR)
print(df.select("image.origin", "image.width", "image.height").show(truncate=False))
