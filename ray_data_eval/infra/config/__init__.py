from enum import Enum
from dataclasses import InitVar, dataclass, field
import os


CONFIG_NAME_ENV_VAR = "CONFIG"

InstanceType = str


@dataclass
class Cluster:
    name: str
    instance_types: dict[InstanceType, int]

    @property
    def instances(self) -> list[InstanceType]:
        return [inst for inst, cnt in self.instance_types.items() for _ in range(cnt)]


@dataclass
class JobConfig:
    name: str
    cluster: Cluster


CONFIGS = [
    JobConfig(
        name="1+1",
        cluster=Cluster(
            name="1+1",
            instance_types={
                "g5.xlarge": 1,
                "m7i.xlarge": 1,
            },
        ),
    )
]


__config_dict__ = {cfg.name: cfg for cfg in CONFIGS}


def get(config_name: str | None = None) -> JobConfig:
    if config_name is None:
        config_name = os.getenv(CONFIG_NAME_ENV_VAR)
    assert config_name, f"No configuration specified, please set ${CONFIG_NAME_ENV_VAR}"
    assert config_name in __config_dict__, f"Unknown configuration: {config_name}"
    return __config_dict__[config_name]
