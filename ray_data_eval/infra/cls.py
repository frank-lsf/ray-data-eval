"""Cluster management toolkit."""

import json
import os
import pathlib

import boto3
import click

from ray_data_eval.infra import ansible_utils
from ray_data_eval.infra import config
from ray_data_eval.infra import shell_utils
from ray_data_eval.infra import terraform_utils
from ray_data_eval.infra import yarn_utils

cfg = config.get()

RAY_METRICS_EXPORT_PORT = 8090
RAY_OBJECT_MANAGER_PORT = 8076
SSH_KEY = "/home/ubuntu/.aws/login-us-west-2.pem"


def get_cluster_name() -> str:
    user = os.getenv("USER", os.getenv("USERNAME"))
    assert user, "$USER or $USERNAME is not set"
    return f"{user}-{cfg.cluster.name}"


def get_instances(filters: dict) -> list[dict]:
    ec2 = boto3.client("ec2")
    paginator = ec2.get_paginator("describe_instances")
    ret = []
    for page in paginator.paginate(Filters=[{"Name": k, "Values": v} for k, v in filters.items()]):
        ret.extend(page["Reservations"])
    return [item["Instances"][0] for item in ret]


def check_cluster_existence(cluster_name: str, raise_if_exists: bool = False) -> bool:
    instances = get_instances(
        {
            "tag:ClusterName": [cluster_name],
            # Excluding "shutting-down" (0x20) and "terminated" (0x30).
            # https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instances.html
            "instance-state-code": [str(code) for code in [0x00, 0x10, 0x40, 0x50]],
        }
    )
    cnt = len(instances)
    ret = cnt > 0
    if raise_if_exists and ret:
        shell_utils.error(f"{cluster_name} must not exist (found {cnt} instances)")
    return ret


def get_current_ip() -> str:
    return shell_utils.run_output("ec2metadata --local-ipv4")


def json_dump_no_space(data) -> str:
    return json.dumps(data, separators=(",", ":"))


def get_ray_start_cmd() -> tuple[str, dict, dict]:
    system_config = {
        "local_fs_capacity_threshold": 1.0,
        "memory_usage_threshold": 1.0,
    }
    system_config_str = json_dump_no_space(system_config)
    resources = json_dump_no_space({"head": 1})
    cmd = "ray start --head"
    cmd += f" --metrics-export-port={RAY_METRICS_EXPORT_PORT}"
    cmd += f" --object-manager-port={RAY_OBJECT_MANAGER_PORT}"
    cmd += f" --system-config='{system_config_str}'"
    cmd += f" --resources='{resources}'"
    env = {}
    env = {k: str(v) for k, v in env.items()}
    return cmd, env, system_config


def restart_ray(inventory_path: pathlib.Path) -> None:
    shell_utils.run("ray stop -f")
    ray_cmd, ray_env, _ = get_ray_start_cmd()
    shell_utils.run(ray_cmd, env=dict(os.environ, **ray_env))
    head_ip = get_current_ip()
    ev = {
        "head_ip": head_ip,
        "ray_object_manager_port": RAY_OBJECT_MANAGER_PORT,
        "ray_merics_export_port": RAY_METRICS_EXPORT_PORT,
    }
    ansible_utils.run_ansible_playbook(inventory_path, "ray", ev=ev)
    shell_utils.sleep(3, "waiting for Ray nodes to connect")
    shell_utils.run("ray status")


def common_setup(cluster_name: str, _cluster_exists: bool) -> pathlib.Path:
    ips = terraform_utils.get_tf_output(cluster_name, "instance_ips")
    inventory_path = ansible_utils.get_or_create_ansible_inventory(cluster_name, ips=ips)
    if not os.environ.get("HADOOP_HOME"):
        click.secho("$HADOOP_HOME not set, skipping Hadoop setup", color="yellow")
    else:
        yarn_utils.setup_yarn(ips)
    # TODO: use boto3 to wait for describe_instance_status to be "ok" for all
    # shell_utils.sleep(60, "worker nodes starting up")
    ansible_utils.run_ansible_playbook(inventory_path, "setup", ev={}, retries=10)
    return inventory_path


def print_after_setup(cluster_name: str) -> None:
    success_msg = f"Cluster {cluster_name} is up and running."
    click.secho("\n" + "-" * len(success_msg), fg="green")
    click.secho(success_msg, fg="green")
    click.echo(f"  Terraform config directory: {terraform_utils.get_tf_dir(cluster_name)}")


# ------------------------------------------------------------
#     CLI Interface
# ------------------------------------------------------------


@click.group()
def cli():
    pass


@cli.command()
def up(ray: bool = True, yarn: bool = True):
    cluster_name = get_cluster_name()
    cluster_exists = check_cluster_existence(cluster_name)
    config_exists = os.path.exists(terraform_utils.get_tf_dir(cluster_name))
    if cluster_exists and not config_exists:
        shell_utils.error(f"{cluster_name} exists on the cloud but nothing is found locally")
    terraform_utils.terraform_provision(cluster_name, cfg)
    inventory_path = common_setup(cluster_name, cluster_exists)
    if ray:
        restart_ray(inventory_path)
    if yarn:
        yarn_utils.restart_yarn(inventory_path)
    print_after_setup(cluster_name)


@cli.command()
def down():
    cluster_name = get_cluster_name()
    terraform_utils.terraform_destroy(cluster_name, cfg)
    check_cluster_existence(cluster_name, raise_if_exists=True)


@cli.command()
@click.argument("worker_id_or_ip", type=str, default="0")
def ssh(worker_id_or_ip: str):
    cluster_name = get_cluster_name()
    try:
        idx = int(worker_id_or_ip)
        ips = terraform_utils.get_tf_output(cluster_name, "instance_ips")
        click.echo(f"worker_ips = {ips}")
        ip = ips[idx]
    except ValueError:
        ip = worker_id_or_ip
    shell_utils.run(
        f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {ip}"
    )


if __name__ == "__main__":
    cli()
