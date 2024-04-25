from dataclasses import dataclass
import os


CONFIG_NAME_ENV_VAR = "CONFIG"

InstanceType = str


@dataclass
class Cluster:
    name: str
    instance_types: dict[InstanceType, int]

    @property
    def terraform_instances_map(self) -> dict[str, str]:
        i = 1
        ret = {}
        for inst, cnt in self.instance_types.items():
            for _ in range(cnt):
                ret[f"node_{i:02d}"] = inst
                i += 1
        return ret


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
                "g5.xlarge": 0,
                "m7i.2xlarge": 1,
            },
        ),
    ),
    JobConfig(
        name="2+2",
        cluster=Cluster(
            name="1+1",
            instance_types={
                "g5.xlarge": 1,
                "m7i.2xlarge": 2,
            },
        ),
    ),
]


__config_dict__ = {cfg.name: cfg for cfg in CONFIGS}


def get(config_name: str | None = None) -> JobConfig:
    if config_name is None:
        config_name = os.getenv(CONFIG_NAME_ENV_VAR)
    assert config_name, f"No configuration specified, please set ${CONFIG_NAME_ENV_VAR}"
    assert config_name in __config_dict__, f"Unknown configuration: {config_name}"
    return __config_dict__[config_name]
