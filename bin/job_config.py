from hydra.conf import JobConf 
from dataclasses import dataclass, field
@dataclass
class JobConfig:
    # Original problematic line
    # override_dirname: str = JobConf.JobConfig.OverrideDirname  # Incorrect

    # Updated line
    override_dirname: str = field(default_factory=lambda: JobConf.JobConfig.OverrideDirname())