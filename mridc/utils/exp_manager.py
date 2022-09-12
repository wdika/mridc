# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/exp_manager.py
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shutil import copy, move
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loggers import LoggerCollection as _LoggerCollection, TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import mridc.utils
from mridc.constants import MRIDC_ENV_VARNAME_TESTING, MRIDC_ENV_VARNAME_VERSION
from mridc.utils import logging, timers
from mridc.utils.app_state import AppState
from mridc.utils.env_var_parsing import get_envbool
from mridc.utils.exceptions import MRIDCBaseException
from mridc.utils.get_rank import is_global_rank_zero
from mridc.utils.lightning_logger_patch import add_filehandlers_to_pl_logger


class NotFoundError(MRIDCBaseException):
    """Raised when a file or folder is not found"""


class LoggerMisconfigurationError(MRIDCBaseException):
    """Raised when a mismatch between trainer.logger and exp_manager occurs"""

    def __init__(self, message):
        message = (
            message + "You can disable lightning's trainer from creating a logger by passing logger=False to its "
            "constructor. "
        )
        super().__init__(message)


class CheckpointMisconfigurationError(MRIDCBaseException):
    """Raised when a mismatch between trainer.callbacks and exp_manager occurs"""


@dataclass
class CallbackParams:
    """Parameters for a callback"""

    filepath: Optional[str] = None  # Deprecated
    dirpath: Optional[str] = None  # If None, exp_manager will attempt to handle the filepath
    filename: Optional[str] = None  # If None, exp_manager will attempt to handle the filepath
    monitor: Optional[str] = "val_loss"
    verbose: Optional[bool] = True
    save_last: Optional[bool] = True
    save_top_k: Optional[int] = 3
    save_weights_only: Optional[bool] = False
    mode: Optional[str] = "min"
    every_n_epochs: Optional[int] = 1
    prefix: Optional[str] = None  # If None, exp_manager will attempt to handle the filepath
    postfix: str = ".mridc"
    save_best_model: bool = False
    always_save_mridc: bool = False
    save_mridc_on_train_end: Optional[bool] = True  # Automatically save .mridc file during on_train_end hook
    model_parallel_size: Optional[int] = None  # tensor parallel size * pipeline parallel size


@dataclass
class StepTimingParams:
    """Parameters for the step timing callback."""

    reduction: Optional[str] = "mean"
    # if True torch.cuda.synchronize() is called on start/stop
    sync_cuda: Optional[bool] = False
    # if positive, defines the size of a sliding window for computing mean
    buffer_size: Optional[int] = 1


@dataclass
class ExpManagerConfig:
    """Configuration for the experiment manager."""

    # Log dir creation parameters
    explicit_log_dir: Optional[str] = None
    exp_dir: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    use_datetime_version: Optional[bool] = True
    resume_if_exists: Optional[bool] = False
    resume_past_end: Optional[bool] = False
    resume_ignore_no_checkpoint: Optional[bool] = False
    # Logging parameters
    create_tensorboard_logger: Optional[bool] = True
    summary_writer_kwargs: Optional[Dict[Any, Any]] = None
    create_wandb_logger: Optional[bool] = False
    wandb_logger_kwargs: Optional[Dict[Any, Any]] = None
    # Checkpointing parameters
    create_checkpoint_callback: Optional[bool] = True
    checkpoint_callback_params: Optional[CallbackParams] = CallbackParams()
    # Additional exp_manager arguments
    files_to_copy: Optional[List[str]] = None
    # logs timing of train/val/test steps
    log_step_timing: Optional[bool] = True
    step_timing_kwargs: Optional[StepTimingParams] = StepTimingParams()
    # Configures creation of log files for different ranks
    log_local_rank_0_only: Optional[bool] = False
    log_global_rank_0_only: Optional[bool] = False
    model_parallel_size: Optional[int] = None


class TimingCallback(Callback):
    """Logs execution time of train/val/test steps"""

    def __init__(self, timer_kwargs=None):
        """Initialize TimingCallback"""
        if timer_kwargs is None:
            timer_kwargs = {}
        self.timer = timers.NamedTimer(**timer_kwargs)

    def _on_batch_start(self, name):
        """Called at the beginning of each batch"""
        # reset only if we do not return mean of a sliding window
        if self.timer.buffer_size <= 0:
            self.timer.reset(name)

        self.timer.start(name)

    def _on_batch_end(self, name, pl_module):
        """Called at the end of each batch"""
        self.timer.stop(name)
        pl_module.log(name, self.timer[name], on_step=True, on_epoch=False)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
        """Called at the beginning of each training batch"""
        self._on_batch_start("train_step_timing")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        """Logs the time taken by the training batch"""
        self._on_batch_end("train_step_timing", pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Logs the time taken by the validation batch"""
        self._on_batch_start("validation_step_timing")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Logs the time taken by the validation step"""
        self._on_batch_end("validation_step_timing", pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Logs execution time of test steps"""
        self._on_batch_start("test_step_timing")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Logs execution time of test steps"""
        self._on_batch_end("test_step_timing", pl_module)

    def on_before_backward(self, trainer, pl_module, loss):
        """Logs the time taken for backward pass"""
        self._on_batch_start("train_backward_timing")

    def on_after_backward(self, trainer, pl_module):
        """Note: this is called after the optimizer step"""
        self._on_batch_end("train_backward_timing", pl_module)


def exp_manager(trainer: Trainer, cfg: Optional[Union[DictConfig, Dict]] = None) -> Optional[Path]:
    """
    exp_manager is a helper function used to manage folders for experiments. It follows the pytorch lightning \
    paradigm of exp_dir/model_or_experiment_name/version. If the lightning trainer has a logger, exp_manager will \
    get exp_dir, name, and version from the logger. Otherwise, it will use the exp_dir and name arguments to create \
    the logging directory. exp_manager also allows for explicit folder creation via explicit_log_dir.

    The version can be a datetime string or an integer. Datetime version can be disabled if you use_datetime_version \
    is set to False. It optionally creates TensorBoardLogger, WandBLogger, ModelCheckpoint objects from pytorch \
    lightning. It copies sys.argv, and git information if available to the logging directory. It creates a log file \
    for each process to log their output into.

    exp_manager additionally has a resume feature (resume_if_exists) which can be used to continuing training from \
    the constructed log_dir. When you need to continue the training repeatedly (like on a cluster which you need \
    multiple consecutive jobs), you need to avoid creating the version folders. Therefore, from v1.0.0, when \
    resume_if_exists is set to True, creating the version folders is ignored.

    Parameters
    ----------
    trainer: The lightning trainer object.
    cfg: Can have the following keys:
        - explicit_log_dir: Can be used to override exp_dir/name/version folder creation. Defaults to None, which \
        will use exp_dir, name, and version to construct the logging directory.
        - exp_dir: The base directory to create the logging directory. Defaults to None, which logs to \
         ./mridc_experiments.
        - name: The name of the experiment. Defaults to None which turns into "default" via name = name or "default".
        - version: The version of the experiment. Defaults to None which uses either a datetime string or lightning's \
         TensorboardLogger system of using version_{int}.
        - use_datetime_version: Whether to use a datetime string for version. Defaults to True.
        - resume_if_exists: Whether this experiment is resuming from a previous run. If True, it sets \
        trainer._checkpoint_connector.resume_from_checkpoint_fit_path so that the trainer should auto-resume. \
        exp_manager will move files under log_dir to log_dir/run_{int}. Defaults to False. From v1.0.0, when \
        resume_if_exists is True, we would not create version folders to make it easier to find the log folder for \
        next runs.
        - resume_past_end: exp_manager errors out if resume_if_exists is True and a checkpoint matching \*end.ckpt \
        indicating a previous training run fully completed. This behaviour can be disabled, in which case the \
        \*end.ckpt will be loaded by setting resume_past_end to True. Defaults to False.
        - resume_ignore_no_checkpoint: exp_manager errors out if resume_if_exists is True and no checkpoint could be \
         found. This behaviour can be disabled, in which case exp_manager will print a message and continue without \
         restoring, by setting resume_ignore_no_checkpoint to True. Defaults to False.
        - create_tensorboard_logger: Whether to create a tensorboard logger and attach it to the pytorch lightning \
        trainer. Defaults to True.
        - summary_writer_kwargs: A dictionary of kwargs that can be passed to lightning's TensorboardLogger class. \
        Note that log_dir is passed by exp_manager and cannot exist in this dict. Defaults to None.
        - create_wandb_logger: Whether to create a Weights and Biases logger and attach it to the pytorch lightning \
        trainer. Defaults to False.
        - wandb_logger_kwargs: A dictionary of kwargs that can be passed to lightning's WandBLogger class. Note that \
         name and project are required parameters if create_wandb_logger is True. Defaults to None.
        - create_checkpoint_callback: Whether to create a ModelCheckpoint callback and attach it to the pytorch \
        lightning trainer. The ModelCheckpoint saves the top 3 models with the best "val_loss", the most recent \
        checkpoint under \*last.ckpt, and the final checkpoint after training completes under \*end.ckpt. \
        Defaults to True.
        - files_to_copy: A list of files to copy to the experiment logging directory. Defaults to None which copies \
        no files.
        - log_local_rank_0_only: Whether to only create log files for local rank 0. Defaults to False. Set this to \
        True if you are using DDP with many GPUs and do not want many log files in your exp dir.
        - log_global_rank_0_only: Whether to only create log files for global rank 0. Defaults to False. Set this to \
        True if you are using DDP with many GPUs and do not want many log files in your exp dir.

    Returns
    -------
    The final logging directory where logging files are saved. Usually the concatenation of exp_dir, name, and version.
    """
    # Add rank information to logger
    # Note: trainer.global_rank and trainer.is_global_zero are not set until trainer.fit, so have to hack around it
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = trainer.node_rank * trainer.num_devices + local_rank
    logging.rank = global_rank

    if cfg is None:
        logging.error("exp_manager did not receive a cfg argument. It will be disabled.")
        return None

    if trainer.fast_dev_run:
        logging.info("Trainer was called with fast_dev_run. exp_manager will return without any functionality.")
        return None

    # Ensure passed cfg is compliant with ExpManagerConfig
    schema = OmegaConf.structured(ExpManagerConfig)
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg = OmegaConf.merge(schema, cfg)

    error_checks(trainer, cfg)  # Ensures that trainer options are compliant with MRIDC and exp_manager arguments

    log_dir, exp_dir, name, version = get_log_dir(
        trainer=trainer,
        exp_dir=cfg.exp_dir,
        name=cfg.name,
        version=cfg.version,
        explicit_log_dir=cfg.explicit_log_dir,
        use_datetime_version=cfg.use_datetime_version,
        resume_if_exists=cfg.resume_if_exists,
    )

    if cfg.resume_if_exists:
        check_resume(trainer, str(log_dir), cfg.resume_past_end, cfg.resume_ignore_no_checkpoint)

    checkpoint_name = name
    # If name returned from get_log_dir is "", use cfg.name for checkpointing
    if checkpoint_name is None or checkpoint_name == "":
        checkpoint_name = cfg.name or "default"
    cfg.name = name  # Used for configure_loggers so that the log_dir is properly set even if name is ""
    cfg.version = version

    # update app_state with log_dir, exp_dir, etc
    app_state = AppState()
    app_state.log_dir = log_dir
    app_state.exp_dir = exp_dir
    app_state.name = name
    app_state.version = version
    app_state.checkpoint_name = checkpoint_name
    app_state.create_checkpoint_callback = cfg.create_checkpoint_callback
    app_state.checkpoint_callback_params = cfg.checkpoint_callback_params

    # Create the logging directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
    logging.info(f"Experiments will be logged at {log_dir}")
    trainer._default_root_dir = log_dir

    if cfg.log_local_rank_0_only is True and cfg.log_global_rank_0_only is True:
        raise ValueError(
            "Cannot set both log_local_rank_0_only and log_global_rank_0_only to True."
            "Please set either one or neither."
        )

    # This is set if the env var MRIDC_TESTING is set to True.
    mridc_testing = get_envbool(MRIDC_ENV_VARNAME_TESTING, False)

    log_file = log_dir / f"mridc_log_globalrank-{global_rank}_localrank-{local_rank}.txt"

    logging.add_file_handler(log_file)
    # For some reason, LearningRateLogger requires trainer to have a logger. Safer to create logger on all ranks
    # not just global rank 0.
    if cfg.create_tensorboard_logger or cfg.create_wandb_logger:
        configure_loggers(
            trainer,
            [Path(exp_dir)],
            cfg.name,
            cfg.version,
            cfg.create_tensorboard_logger,
            cfg.summary_writer_kwargs,
            cfg.create_wandb_logger,
            cfg.wandb_logger_kwargs,
        )

    # add loggers timing callbacks
    if cfg.log_step_timing:
        timing_callback = TimingCallback(timer_kwargs=cfg.step_timing_kwargs or {})
        trainer.callbacks.insert(0, timing_callback)

    if cfg.create_checkpoint_callback:
        configure_checkpointing(
            trainer, log_dir, checkpoint_name, cfg.resume_if_exists, cfg.checkpoint_callback_params
        )

    if is_global_rank_zero():
        # Move files_to_copy to folder and add git information if present
        if cfg.files_to_copy:
            for _file in cfg.files_to_copy:
                copy(Path(_file), log_dir)

        # Create files for cmd args and git info
        with open(log_dir / "cmd-args.log", "w", encoding="utf-8") as _file:
            _file.write(" ".join(sys.argv))

        # Try to get git hash
        git_repo, git_hash = get_git_hash()
        if git_repo:
            with open(log_dir / "git-info.log", "w", encoding="utf-8") as _file:
                _file.write(f"commit hash: {git_hash}")
                _file.write(get_git_diff())

        # Add err_file logging to global_rank zero
        logging.add_err_file_handler(log_dir / "mridc_error_log.txt")

        # Add lightning file logging to global_rank zero
        add_filehandlers_to_pl_logger(log_dir / "lightning_logs.txt", log_dir / "mridc_error_log.txt")

    return log_dir


def error_checks(trainer: Trainer, cfg: Optional[Union[DictConfig, Dict]] = None):
    """
    Checks that the passed trainer is compliant with MRIDC and exp_manager's passed configuration. Checks that:
        - Throws error when hydra has changed the working directory. This causes issues with lightning's DDP
        - Throws error when trainer has loggers defined but create_tensorboard_logger or create_WandB_logger is True
        - Prints error messages when 1) run on multi-node and not Slurm, and 2) run on multi-gpu without DDP
    """
    if HydraConfig.initialized() and get_original_cwd() != os.getcwd():
        raise ValueError(
            "Hydra changed the working directory. This interferes with ExpManger's functionality. Please pass "
            "hydra.run.dir=. to your python script."
        )

    if trainer.logger is not None and (cfg.create_tensorboard_logger or cfg.create_wandb_logger):  # type: ignore
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger, and either "
            "create_tensorboard_logger or create_wandb_logger was set to True. These can only be used if trainer does "
            "not already have a logger."
        )

    if trainer.num_nodes > 1 and not check_slurm(trainer):  # type: ignore
        logging.error(
            "You are running multi-node training without SLURM handling the processes."
            " Please note that this is not tested in MRIDC and could result in errors."
        )

    if trainer.num_devices > 1 and not isinstance(trainer.strategy, DDPStrategy):  # type: ignore
        logging.error(
            "You are running multi-gpu without ddp.Please note that this is not tested in MRIDC and could result in "
            "errors."
        )


def check_resume(
    trainer: Trainer,
    log_dir: str,
    resume_past_end: bool = False,
    resume_ignore_no_checkpoint: bool = False,
):
    """
    Checks that resume=True was used correctly with the arguments pass to exp_manager. Sets
    trainer._checkpoint_connector.resume_from_checkpoint_fit_path as necessary.

    Parameters
    ----------
    trainer: The trainer that is being used.
    log_dir: The directory where the logs are being saved.
    resume_past_end: Whether to resume from the end of the experiment.
    resume_ignore_no_checkpoint: Whether to ignore if there is no checkpoint to resume from.

    Returns
    -------
    NotFoundError: If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
    ValueError: If resume is True, and there were more than 1 checkpoint could found.
    """
    if not log_dir:
        raise ValueError(f"Resuming requires the log_dir {log_dir} to be passed to exp_manager")

    checkpoint_dir = Path(Path(log_dir) / "checkpoints")

    checkpoint = None
    end_checkpoints = list(checkpoint_dir.rglob("*end.ckpt"))
    last_checkpoints = list(checkpoint_dir.rglob("*last.ckpt"))
    if not checkpoint_dir.exists():
        if not resume_ignore_no_checkpoint:
            raise NotFoundError(f"There was no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Cannot resume.")
        logging.warning(f"There was no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Training from scratch.")
        return
    if end_checkpoints:
        if not resume_past_end:
            raise ValueError(
                f"Found {end_checkpoints[0]} indicating that the last training run has already completed."
            )
        if len(end_checkpoints) > 1:
            if "mp_rank" in str(end_checkpoints[0]):
                checkpoint = end_checkpoints[0]
            else:
                raise ValueError(f"Multiple checkpoints {end_checkpoints} that matches *end.ckpt.")
        logging.info(f"Resuming from {end_checkpoints[0]}")
    elif not last_checkpoints:
        if not resume_ignore_no_checkpoint:
            raise NotFoundError(f"There were no checkpoints found in {checkpoint_dir}. Cannot resume.")
        logging.warning(f"There were no checkpoints found in {checkpoint_dir}. Training from scratch.")
        return
    elif len(last_checkpoints) > 1:
        if "mp_rank" not in str(last_checkpoints[0]) and "tp_rank" not in str(last_checkpoints[0]):
            raise ValueError(f"Multiple checkpoints {last_checkpoints} that matches *last.ckpt.")
        checkpoint = last_checkpoints[0]
        checkpoint = mridc.utils.model_utils.uninject_model_parallel_rank(checkpoint)  # type: ignore
    else:
        logging.info(f"Resuming from {last_checkpoints[0]}")
        checkpoint = last_checkpoints[0]

    trainer._checkpoint_connector.resume_from_checkpoint_fit_path = str(checkpoint)

    if is_global_rank_zero():
        if files_to_move := [child for child in Path(log_dir).iterdir() if child.is_file()]:
            # Move old files to a new folder
            other_run_dirs = Path(log_dir).glob("run_*")
            run_count = sum(bool(fold.is_dir()) for fold in other_run_dirs)
            new_run_dir = Path(Path(log_dir) / f"run_{run_count}")
            new_run_dir.mkdir()
            for _file in files_to_move:
                move(str(_file), str(new_run_dir))


def check_explicit_log_dir(
    trainer: Trainer, explicit_log_dir: List[Union[Path, str]], exp_dir: str, name: str, version: str
) -> Tuple[Path, str, str, str]:
    """
    Checks that the passed arguments are compatible with explicit_log_dir.

    Parameters
    ----------
    trainer: The trainer to check.
    explicit_log_dir: The explicit log dir to check.
    exp_dir: The experiment directory to check.
    name: The experiment name to check.
    version: The experiment version to check.

    Returns
    -------
    The log_dir, exp_dir, name, and version that should be used.

    Raises
    ------
    LoggerMisconfigurationError
    """
    if trainer.logger is not None:
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger and explicit_log_dir: "
            f"{explicit_log_dir} was pass to exp_manager. Please remove the logger from the lightning trainer."
        )
    # Checking only (explicit_log_dir) vs (exp_dir and version).
    # The `name` will be used as the actual name of checkpoint/archive.
    if exp_dir or version:
        logging.error(
            f"exp_manager received explicit_log_dir: {explicit_log_dir} and at least one of exp_dir: {exp_dir}, "
            f"or version: {version}. Please note that exp_dir, name, and version will be ignored."
        )
    if is_global_rank_zero() and Path(str(explicit_log_dir)).exists():
        logging.warning(f"Exp_manager is logging to {explicit_log_dir}, but it already exists.")
    return Path(str(explicit_log_dir)), str(explicit_log_dir), "", ""


def get_log_dir(
    trainer: Trainer,
    exp_dir: str = None,
    name: str = None,
    version: str = None,
    explicit_log_dir: str = None,
    use_datetime_version: bool = True,
    resume_if_exists: bool = False,
) -> Tuple[Path, str, str, str]:
    """
    Obtains the log_dir used for exp_manager.

    Parameters
    ----------
    trainer: The trainer to check.
    exp_dir: The experiment directory to check.
    name: The experiment name to check.
    version: The experiment version to check.
    explicit_log_dir: The explicit log dir to check.
    use_datetime_version: Whether to use datetime versioning.
    resume_if_exists: Whether to resume if the log_dir already exists.

    Raises
    -------
    LoggerMisconfigurationError: If trainer is incompatible with arguments
    NotFoundError: If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
    ValueError: If resume is True, and there were more than 1 checkpoint could found.
    """
    if explicit_log_dir:  # If explicit log_dir was passed, short circuit
        return check_explicit_log_dir(trainer, [Path(explicit_log_dir)], exp_dir, name, version)  # type: ignore

    # Default exp_dir to ./mridc_experiments if None was passed
    _exp_dir = exp_dir
    if exp_dir is None:
        _exp_dir = str(Path.cwd() / "mridc_experiments")

    # If the user has already defined a logger for the trainer, use the logger defaults for logging directory
    if trainer.logger is not None:
        if trainer.logger.save_dir:
            if exp_dir:
                raise LoggerMisconfigurationError(
                    "The pytorch lightning trainer that was passed to exp_manager contained a logger, the logger's "
                    f"save_dir was not None, and exp_dir ({exp_dir}) was not None. If trainer.logger.save_dir "
                    "exists, exp_manager will use trainer.logger.save_dir as the logging directory and exp_dir "
                    "must be None."
                )
            _exp_dir = trainer.logger.save_dir
        if name:
            raise LoggerMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a logger, and name: "
                f"{name} was also passed to exp_manager. If the trainer contains a "
                "logger, exp_manager will use trainer.logger.name, and name passed to exp_manager must be None."
            )
        name = trainer.logger.name
        version = f"version_{trainer.logger.version}"
    # Use user-defined exp_dir, project_name, exp_name, and versioning options
    else:
        name = name or "default"
        version = version or os.environ.get(MRIDC_ENV_VARNAME_VERSION)

        if not version:
            if resume_if_exists:
                logging.warning(
                    "No version folders would be created under the log folder as 'resume_if_exists' is enabled."
                )
                version = None
            elif is_global_rank_zero():
                if use_datetime_version:
                    version = time.strftime("%Y-%m-%d_%H-%M-%S")
                else:
                    tensorboard_logger = TensorBoardLogger(save_dir=_exp_dir, name=name, version=version)
                    version = f"version_{tensorboard_logger.version}"
                os.environ[MRIDC_ENV_VARNAME_VERSION] = "" if version is None else version

    log_dir = Path(str(_exp_dir)) / Path(str(name)) / Path("" if version is None else str(version))
    return log_dir, str(_exp_dir), str(name), str(version)


def get_git_hash():
    """
    Helper function that tries to get the commit hash if running inside a git folder.

    Returns
    -------
    Bool: Whether the git subprocess ran without error.
    String: git subprocess output or error message
    """
    try:
        return True, subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as err:
        return False, f'{err.output.decode("utf-8")}\n'


def get_git_diff():
    """
    Helper function that tries to get the git diff if running inside a git folder.

    Returns
    -------
    Bool: Whether the git subprocess ran without error.
    String: git subprocess output or error message
    """
    try:
        return subprocess.check_output(["git", "diff"], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as err:
        return f'{err.output.decode("utf-8")}\n'


class LoggerList(_LoggerCollection):
    """A thin wrapper on Lightning's LoggerCollection such that name and version are better aligned with exp_manager"""

    def __init__(self, _logger_iterable, mridc_name=None, mridc_version=""):
        super().__init__(_logger_iterable)
        self._mridc_name = mridc_name
        self._mridc_version = mridc_version

    @property
    def name(self) -> str:
        """The name of the experiment."""
        return self._mridc_name

    @property
    def version(self) -> str:
        """The version of the experiment. If the logger was created with a version, this will be the version."""
        return self._mridc_version


def configure_loggers(
    trainer: Trainer,
    exp_dir: List[Union[Path, str]],
    name: str,
    version: str,
    create_tensorboard_logger: bool,
    summary_writer_kwargs: dict,
    create_wandb_logger: bool,
    wandb_kwargs: dict,
):
    """
    Creates TensorboardLogger and/or WandBLogger and attach them to trainer. Raises ValueError if summary_writer_kwargs
    or wandb_kwargs are miss configured.

    Parameters
    ----------
    trainer: The trainer to attach the loggers to.
    exp_dir: The experiment directory.
    name: The name of the experiment.
    version: The version of the experiment.
    create_tensorboard_logger: Whether to create a TensorboardLogger.
    summary_writer_kwargs: The kwargs to pass to the TensorboardLogger.
    create_wandb_logger: Whether to create a Weights & Biases logger.
    wandb_kwargs: The kwargs to pass to the Weights & Biases logger.

    Returns
    -------
    LoggerList: A list of loggers.
    """
    # Potentially create tensorboard logger and/or WandBLogger
    logger_list = []
    if create_tensorboard_logger:
        if summary_writer_kwargs is None:
            summary_writer_kwargs = {}
        elif "log_dir" in summary_writer_kwargs:
            raise ValueError(
                "You cannot pass `log_dir` as part of `summary_writer_kwargs`. `log_dir` is handled by lightning's "
                "TensorBoardLogger logger."
            )
        tensorboard_logger = TensorBoardLogger(
            save_dir=exp_dir[0], name=name, version=version, **summary_writer_kwargs
        )
        logger_list.append(tensorboard_logger)
        logging.info("TensorboardLogger has been set up")

    if create_wandb_logger:
        if wandb_kwargs is None:
            wandb_kwargs = {}
        if "name" not in wandb_kwargs and "project" not in wandb_kwargs:
            raise ValueError("name and project are required for wandb_logger")
        # if wandb_kwargs.get("save_dir", None) is None:
        #     wandb_kwargs["save_dir"] = str(exp_dir[0])
        #     os.makedirs(wandb_kwargs["save_dir"], exist_ok=True)
        wandb_logger = WandbLogger(save_dir=str(exp_dir[0]), version=version, **wandb_kwargs)

        logger_list.append(wandb_logger)
        logging.info("WandBLogger has been set up")

    logger_list = (
        LoggerList(logger_list, mridc_name=name, mridc_version=version) if len(logger_list) > 1 else logger_list[0]
    )
    trainer._logger_connector.configure_logger(logger_list)


class MRIDCModelCheckpoint(ModelCheckpoint):
    """Light wrapper around Lightning's ModelCheckpoint to force a saved checkpoint on train_end"""

    def __init__(
        self,
        always_save_mridc=False,
        save_mridc_on_train_end=True,
        save_best_model=False,
        postfix=".mridc",
        n_resume=False,
        model_parallel_size=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        always_save_mridc: Whether to save the model even if it is not the best model. Default: False.
        save_mridc_on_train_end: Whether to save the model at the end of training. Default: True.
        save_best_model: Whether to save the model if it is the best model. Default: False.
        postfix: The postfix to add to the model name. Default: ".mridc".
        n_resume: Whether to resume training from a checkpoint. Default: False.
        model_parallel_size: The size of the model parallel group. Default: None.
        kwargs: The kwargs to pass to ModelCheckpoint.
        """
        # Parse and store "extended" parameters: save_best model and postfix.
        self.always_save_mridc = always_save_mridc
        self.save_mridc_on_train_end = save_mridc_on_train_end
        self.save_best_model = save_best_model
        self.previous_model_path = None
        self.last_model_path: Union[Any, str] = None
        if self.save_best_model and not self.save_mridc_on_train_end:
            logging.warning(
                (
                    "Found save_best_model is True and save_mridc_on_train_end is False. "
                    "Set save_mridc_on_train_end to True to automatically save the best model."
                )
            )
        self.postfix = postfix
        self.previous_best_path = ""
        self.model_parallel_size = model_parallel_size

        # `prefix` is deprecated
        self.prefix = kwargs.pop("prefix") if "prefix" in kwargs else ""
        # Call the parent class constructor with the remaining kwargs.
        super().__init__(**kwargs)

        if self.save_top_k != -1 and n_resume:
            logging.debug("Checking previous runs")
            self.mridc_topk_check_previous_run()

    def mridc_topk_check_previous_run(self):
        """Check if there are previous runs with the same topk value."""
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""

        checkpoints = list(Path(self.dirpath).rglob("*.ckpt"))
        for checkpoint in checkpoints:
            if "mp_rank" in str(checkpoint) or "tp_rank" in str(checkpoint):
                checkpoint = mridc.utils.model_utils.uninject_model_parallel_rank(checkpoint)
            checkpoint = str(checkpoint)
            if checkpoint.endswith("-last.ckpt"):
                continue
            index = checkpoint.find(self.monitor) + len(self.monitor) + 1  # Find monitor in str + 1 for '='
            if index != -1:
                if match := re.search("[A-z]", checkpoint[index:]):
                    value = checkpoint[index : index + match.start() - 1]  # -1 due to separator hypen
                    self.best_k_models[checkpoint] = float(value)
        if not self.best_k_models:
            return  # No saved checkpoints yet

        _reverse = self.mode != "min"

        best_k_models = sorted(self.best_k_models, key=self.best_k_models.get, reverse=_reverse)

        # This section should be ok as rank zero will delete all excess checkpoints, since all other ranks are
        # instantiated after rank zero. models_to_delete should be 0 for all other ranks.
        if self.model_parallel_size is not None:
            models_to_delete = len(best_k_models) - self.model_parallel_size * self.save_top_k
        else:
            models_to_delete = len(best_k_models) - self.save_top_k
        logging.debug(f"Number of models to delete: {models_to_delete}")
        for _ in range(models_to_delete):
            model = best_k_models.pop(-1)
            self.best_k_models.pop(model)
            self._del_model_without_trainer(model)
            logging.debug(f"Removed checkpoint: {model}")

        self.kth_best_model_path = best_k_models[-1]
        self.best_model_path = best_k_models[0]
        self.best_model_score = self.best_k_models[self.best_model_path]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Override the default on_save_checkpoint to save the best model if needed.

        Parameters
        ----------
        trainer: The trainer object.
        pl_module: The PyTorch-Lightning module.
        checkpoint: The checkpoint object.
        """
        output = super().on_save_checkpoint(trainer, pl_module, checkpoint)
        if not self.always_save_mridc:
            return output
        # Load the best model and then re-save it
        app_state = AppState()

        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            raise ValueError("always_save_mridc is not implemented for model parallel models.")

        # since we are creating tarfile artifacts we need to update .mridc path
        app_state.model_restore_path = os.path.abspath(
            os.path.expanduser(os.path.join(self.dirpath, self.prefix + self.postfix))
        )

        if self.save_best_model:
            if not os.path.exists(self.best_model_path):
                return output

            if self.best_model_path == self.previous_best_path:
                return output

            self.previous_model_path = self.best_model_path
            old_state_dict = deepcopy(pl_module.state_dict())
            checkpoint = torch.load(self.best_model_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            # get a new instance of the model
            pl_module.load_state_dict(checkpoint, strict=True)
            pl_module.save_to(save_path=app_state.model_restore_path)
            pl_module.load_state_dict(old_state_dict, strict=True)
        else:
            pl_module.save_to(save_path=app_state.model_restore_path)
        return output

    def on_train_end(self, trainer, pl_module):
        """
        This is called at the end of training.

        Parameters
        ----------
        trainer: The trainer object.
        pl_module: The PyTorch-Lightning module.
        """
        if trainer.fast_dev_run:
            return None

        # check if we need to save a last checkpoint manually as validation isn't always run based on the interval
        if self.save_last and trainer.val_check_interval != 0:
            should_save_last_checkpoint = False
            if isinstance(trainer.val_check_interval, float) and trainer.val_check_interval % trainer.global_step != 0:
                should_save_last_checkpoint = True
            if isinstance(trainer.val_check_interval, int) and trainer.global_step % trainer.val_check_interval != 0:
                should_save_last_checkpoint = True
            if should_save_last_checkpoint:
                monitor_candidates = self._monitor_candidates(trainer)
                super()._save_last_checkpoint(trainer, monitor_candidates)

        # Call parent on_train_end() to save the -last checkpoint
        super().on_train_end(trainer, pl_module)

        # Load the best model and then re-save it
        if self.save_best_model:
            # wait for all processes to finish
            trainer.strategy.barrier("SaveBestCheckpointConnector.resume_end")
            if self.best_model_path == "":
                logging.warning(
                    f"{self} was told to save the best checkpoint at the end of training, but no saved checkpoints "
                    "were found. Saving latest model instead."
                )
            else:
                self.best_model_path = trainer.strategy.broadcast(self.best_model_path)
                trainer._checkpoint_connector.restore(self.best_model_path)

        if self.save_mridc_on_train_end:
            pl_module.save_to(save_path=os.path.join(self.dirpath, self.prefix + self.postfix))

    def _del_model_without_trainer(self, filepath: str) -> None:
        """
        Delete a model without a trainer.

        Parameters
        ----------
        filepath: The path to the model to delete.
        """
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            # filepath needs to be updated to include mp_rank
            filepath = mridc.utils.model_utils.inject_model_parallel_rank(filepath)  # type: ignore

        # each model parallel rank needs to remove its model
        if is_global_rank_zero() or (app_state.model_parallel_size is not None and app_state.data_parallel_rank == 0):
            try:
                self._fs.rm(filepath)
                logging.info(f"Removed checkpoint: {filepath}")
            except FileNotFoundError:
                logging.info(f"Tried to remove checkpoint: {filepath} but failed.")


def configure_checkpointing(trainer: Trainer, log_dir: Path, name: str, resume: bool, params: "DictConfig"):
    """Adds ModelCheckpoint to trainer. Raises CheckpointMisconfigurationError if trainer already has a ModelCheckpoint
    callback or if trainer.weights_save_path was passed to Trainer.
    """
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            raise CheckpointMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a ModelCheckpoint "
                "and create_checkpoint_callback was set to True. Please either set create_checkpoint_callback "
                "to False, or remove ModelCheckpoint from the lightning trainer"
            )
    if Path(trainer.weights_save_path) != Path.cwd():
        raise CheckpointMisconfigurationError(
            "The pytorch lightning was passed weights_save_path. This variable is ignored by exp_manager"
        )

    # Create the callback and attach it to trainer
    if "filepath" in params:
        if params.filepath is not None:
            logging.warning("filepath is deprecated. Please switch to dirpath and filename instead")
            if params.dirpath is None:
                params.dirpath = Path(params.filepath).parent
            if params.filename is None:
                params.filename = Path(params.filepath).name
        with open_dict(params):
            del params["filepath"]
    if params.dirpath is None:
        params.dirpath = Path(log_dir / "checkpoints")
    if params.filename is None:
        params.filename = f"{name}--{{{params.monitor}:.4f}}-{{epoch}}"
    if params.prefix is None:
        params.prefix = name
    MRIDCModelCheckpoint.CHECKPOINT_NAME_LAST = f"{params.filename}-last"

    logging.debug(params.dirpath)
    logging.debug(params.filename)
    logging.debug(params.prefix)

    if "val" in params.monitor:
        if (
            trainer.max_epochs is not None
            and trainer.max_epochs != -1
            and trainer.max_epochs < trainer.check_val_every_n_epoch
        ):
            logging.error(
                "The checkpoint callback was told to monitor a validation value but trainer.max_epochs("
                f"{trainer.max_epochs}) was less than trainer.check_val_every_n_epoch("
                f"{trainer.check_val_every_n_epoch}). It is very likely this run will fail with "
                f"ModelCheckpoint(monitor='{params.monitor}') not found in the returned metrics. Please ensure that "
                "validation is run within trainer.max_epochs."
            )
        elif trainer.max_steps is not None:
            logging.warning(
                "The checkpoint callback was told to monitor a validation value and trainer's max_steps was set to "
                f"{trainer.max_steps}. Please ensure that max_steps will run for at least "
                f"{trainer.check_val_every_n_epoch} epochs to ensure that checkpointing will not error out."
            )

    checkpoint_callback = MRIDCModelCheckpoint(n_resume=resume, **params)
    checkpoint_callback.last_model_path = trainer._checkpoint_connector.resume_from_checkpoint_fit_path or ""
    if "mp_rank" in checkpoint_callback.last_model_path or "tp_rank" in checkpoint_callback.last_model_path:
        checkpoint_callback.last_model_path = mridc.utils.model_utils.uninject_model_parallel_rank(  # type: ignore
            checkpoint_callback.last_model_path
        )
    trainer.callbacks.append(checkpoint_callback)


def check_slurm(trainer):
    """
    Checks if the trainer is running on a slurm cluster. If so, it will check if the trainer is running on the master
    node. If it is not, it will exit.

    Parameters
    ----------
    trainer: The trainer to check.

    Returns
    -------
    True if the trainer is running on the master node, False otherwise.
    """
    try:
        return trainer.accelerator_connector.is_slurm_managing_tasks
    except AttributeError:
        return False


class StatelessTimer(Timer):
    """Extension of PTL timers to be per run."""

    def state_dict(self) -> Dict[str, Any]:  # type: ignore
        """Saves the state of the timer."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state of the timer."""
