import click
import os
import pathlib
import string

from ray_data_eval.infra import ansible_utils
from ray_data_eval.infra import shell_utils

SSH_KEY = "/home/ubuntu/.aws/login-us-west-2.pem"

SCRIPT_DIR = pathlib.Path(os.path.dirname(__file__))
HADOOP_TEMPLATE_DIR = SCRIPT_DIR / "config" / "hadoop"


def update_hosts_file(ips: list[str]) -> None:
    PATH = "/etc/hosts"
    MARKER = "### HADOOP_YARN_HOSTS ###\n"
    with open(PATH) as fin:
        content = fin.read()

    marker_idx = content.find(MARKER)
    content = content if marker_idx < 0 else content[:marker_idx]
    content += "\n" + MARKER

    for i, ip in enumerate(ips, start=1):
        content += f"{ip} dn{i}\n"

    shell_utils.run(f"sudo cp {PATH} {PATH}.backup")
    shell_utils.run(f"sudo echo <<EOF\n{content}\nEOF > {PATH}")
    click.secho(f"Updated {PATH}", fg="green")


def update_workers_file(ips: list[str]) -> None:
    PATH = os.path.join(os.getenv("HADOOP_HOME"), "etc/hadoop/workers")
    shell_utils.run(f"cp {PATH} {PATH}.backup")
    with open(PATH, "w") as fout:
        fout.write("\n".join(ips))
    click.secho(f"Updated {PATH}", fg="green")


def update_hadoop_xml(filename: str, head_ip: str) -> None:
    with open(HADOOP_TEMPLATE_DIR / (filename + ".template")) as fin:
        template = string.Template(fin.read())
    content = template.substitute(
        DEFAULT_FS=f"hdfs://{head_ip}:9000",
        HEAD_IP=head_ip,
    )
    output_path = os.path.join(os.getenv("HADOOP_HOME"), "etc/hadoop", filename)
    with open(output_path, "w") as fout:
        fout.write(content)
    click.secho(f"Updated {output_path}", fg="green")


def update_hadoop_config() -> None:
    head_ip = shell_utils.run_output("ec2metadata --local-ipv4")
    for filename in ["core-site.xml", "hdfs-site.xml", "yarn-site.xml"]:
        update_hadoop_xml(filename, head_ip)


def setup_yarn(ips: list[str]) -> None:
    update_hosts_file(ips)
    update_workers_file(ips)
    update_hadoop_config


def restart_yarn(inventory_path: pathlib.Path) -> None:
    env = dict(
        os.environ,
        HADOOP_SSH_OPTS=f"-i {SSH_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR",
        HADOOP_OPTIONAL_TOOLS="hadoop-aws",
    )
    HADOOP_HOME = os.getenv("HADOOP_HOME")
    SPARK_HOME = os.getenv("SPARK_HOME")
    SPARK_EVENTS_PATH = os.getenv("SPARK_EVENTS_PATH")
    shell_utils.run(f"{SPARK_HOME}/sbin/stop-history-server.sh")
    shell_utils.run(f"{HADOOP_HOME}/sbin/stop-yarn.sh", env=env)
    shell_utils.run(f"{HADOOP_HOME}/sbin/stop-dfs.sh", env=env)
    shell_utils.run(f"{HADOOP_HOME}/bin/hdfs namenode -format -force")
    ansible_utils.run_ansible_playbook(inventory_path, "yarn")
    shell_utils.run(f"{HADOOP_HOME}/sbin/start-dfs.sh", env=env)
    shell_utils.run(f"{HADOOP_HOME}/sbin/start-yarn.sh", env=env)
    shell_utils.run(
        f"{SPARK_HOME}/sbin/start-history-server.sh",
        env=dict(
            env,
            SPARK_HISTORY_OPTS=f"-Dspark.history.fs.logDirectory={SPARK_EVENTS_PATH}",
        ),
    )
