import json
import os
import pathlib
import subprocess

from ray_data_eval.infra.config import JobConfig
from ray_data_eval.infra import shell_utils

SCRIPT_DIR = pathlib.Path(os.path.dirname(__file__))
TERRAFORM_DIR = SCRIPT_DIR / "terraform"


def get_tf_output(cluster_name: str, key: str | list[str]) -> list[str] | list[list[str]]:
    tf_dir = get_tf_dir(cluster_name, must_exist=True)
    p = shell_utils.run("terraform output -json", cwd=tf_dir, stdout=subprocess.PIPE)
    data = json.loads(p.stdout.decode("ascii"))
    if isinstance(key, list):
        return [data[k]["value"] for k in key]
    return data[key]["value"]


def get_terraform_vars(**kwargs: dict) -> str:
    def serialize(v):
        if isinstance(v, str):
            return f'\\"{v}\\"'
        if isinstance(v, list):
            return "[" + ",".join([serialize(x) for x in v]) + "]"
        if isinstance(v, dict):
            return "{" + ",".join([f'\\"{k}\\"={serialize(v)}' for k, v in v.items()]) + "}"
        return v

    return "".join([f' -var="{k}={serialize(v)}"' for k, v in kwargs.items()])


def get_tf_dir(_cluster_name: str, must_exist: bool = True) -> pathlib.Path:
    return TERRAFORM_DIR / "aws"


def terraform_provision(cluster_name: str, cfg: JobConfig) -> None:
    tf_dir = get_tf_dir(cluster_name)
    shell_utils.run("terraform init", cwd=tf_dir)
    cmd = "terraform apply -auto-approve" + get_terraform_vars(
        cluster_name=cluster_name,
        instances=cfg.cluster.terraform_instances_map,
    )
    shell_utils.run(cmd, cwd=tf_dir)


def terraform_destroy(cluster_name: str, cfg: JobConfig) -> None:
    tf_dir = get_tf_dir(cluster_name, must_exist=True)
    cmd = "terraform destroy -auto-approve" + get_terraform_vars(cluster_name=cluster_name)
    shell_utils.run(cmd, cwd=tf_dir)
