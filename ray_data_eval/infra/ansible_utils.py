import json
import os
import pathlib
import subprocess
import yaml

import click

from ray_data_eval.infra import shell_utils

SCRIPT_DIR = pathlib.Path(os.path.dirname(__file__))
ANSIBLE_DIR = SCRIPT_DIR / "ansible"
SSH_KEY = "/home/ubuntu/.aws/login-us-west-2.pem"
PARALLELISM = (os.cpu_count() or 1) * 4


def run_ansible_playbook(
    inventory_path: pathlib.Path,
    playbook: str,
    *,
    ev: dict = {},
    retries: int = 1,
    time_between_retries: float = 10,
) -> subprocess.CompletedProcess:
    if not playbook.endswith(".yml"):
        playbook += ".yml"
    playbook_path = ANSIBLE_DIR / playbook
    cmd = "ansible-playbook"
    cmd += " --key-file " + SSH_KEY
    cmd += f" -f {PARALLELISM} -i {inventory_path} {playbook_path}"
    if ev:
        cmd += f" --extra-vars '{json.dumps(ev)}'"
    return shell_utils.run(
        cmd,
        cwd=ANSIBLE_DIR,
        retries=retries,
        time_between_retries=time_between_retries,
    )


def get_or_create_ansible_inventory(
    cluster_name: str, ips: list[str] | None = None
) -> pathlib.Path:
    path = ANSIBLE_DIR / f"_{cluster_name}.yml"
    if not ips:
        if os.path.exists(path):
            return path
        raise ValueError("No hosts provided to Ansible")
    with open(path, "w") as fout:
        fout.write(get_ansible_inventory_content(ips))
    click.secho(f"Created {path}", fg="green")
    return path


def get_ansible_inventory_content(node_ips: list[str]) -> str:
    def get_item(ip):
        host = "node_" + ip.replace(".", "_")
        return host, {"ansible_host": ip}

    hosts = [get_item(ip) for ip in node_ips]
    ret = {
        "all": {
            "hosts": dict(hosts),
        },
    }
    return yaml.dump(ret)
