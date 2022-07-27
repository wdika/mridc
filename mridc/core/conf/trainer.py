# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/config/pytorch_lightning.py

from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore

__all__ = ["TrainerConfig"]

cs = ConfigStore.instance()


@dataclass
class TrainerConfig:
    """TrainerConfig is a dataclass that holds all the hyperparameters for the training process."""

    logger: Any = True
    callbacks: Optional[Any] = None
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    num_nodes: int = 1
    gpus: Optional[Any] = None
    auto_select_gpus: bool = False
    tpu_cores: Optional[Any] = None
    enable_progress_bar: bool = True
    overfit_batches: Any = 0.0
    track_grad_norm: Any = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: Any = 1
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = -1
    min_steps: Optional[int] = None
    limit_train_batches: Any = 1.0
    limit_val_batches: Any = 1.0
    limit_test_batches: Any = 1.0
    val_check_interval: Any = 1.0
    log_every_n_steps: int = 50
    accelerator: Optional[str] = None
    sync_batchnorm: bool = False
    precision: Any = 32
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Optional[str] = None
    profiler: Optional[Any] = None
    benchmark: bool = False
    deterministic: bool = False
    auto_lr_find: Any = False
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: Any = False
    amp_backend: str = "native"
    amp_level: Optional[str] = None
    plugins: Optional[Any] = None  # Optional[Union[str, list]]
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = "max_size_cycle"
    limit_predict_batches: float = 1.0
    gradient_clip_algorithm: str = "norm"
    max_time: Optional[Any] = None  # can be one of Union[str, timedelta, Dict[str, int], None]
    reload_dataloaders_every_n_epochs: int = 0
    ipus: Optional[int] = None
    devices: Any = None
    strategy: Any = None
    enable_checkpointing: bool = False
    enable_model_summary: bool = True


# Register the trainer config.
cs.store(group="trainer", name="trainer", node=TrainerConfig)
