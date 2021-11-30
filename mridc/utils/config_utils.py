# encoding: utf-8
import sys

__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/config_utils.py

import copy
import inspect
from dataclasses import is_dataclass
from typing import Dict, List, Optional, Set

from omegaconf import DictConfig, OmegaConf, open_dict

from mridc.core.conf.modelPT import MRIDCConfig
from mridc.utils import logging

_HAS_HYDRA = True


def update_model_config(model_cls: MRIDCConfig, update_cfg: "DictConfig", drop_missing_subconfigs: bool = True):
    """
    Helper class that updates the default values of a ModelPT config class with the values in a DictConfig that \
    mirrors the structure of the config class. Assumes the `update_cfg` is a DictConfig (either generated manually, \
    via hydra or instantiated via yaml/model.cfg). This update_cfg is then used to override the default values \
    preset inside the ModelPT config class. If `drop_missing_subconfigs` is set, the certain sub-configs of the \
    ModelPT config class will be removed, if they are not found in the mirrored `update_cfg`. The following \
    sub-configs are subject to potential removal:
        -   `train_ds`
        -   `validation_ds`
        -   `test_ds`
        -   `optim` + nested sched

    Parameters
    ----------
    model_cls: A subclass of MRIDC, that details in entirety all the parameters that constitute the MRIDC Model.
    update_cfg: A DictConfig that mirrors the structure of the MRIDCConfig data class. Used to update the default \
    values of the config class.
    drop_missing_subconfigs: Bool which determines whether to drop certain sub-configs from the MRIDCConfig class, \
    if the corresponding sub-config is missing from `update_cfg`.

    Returns
    -------
    A DictConfig with updated values that can be used to instantiate the MRIDC Model along with supporting \
    infrastructure.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        sys.exit(1)
    if not (is_dataclass(model_cls) or isinstance(model_cls, DictConfig)):
        raise ValueError("`model_cfg` must be a dataclass or a structured OmegaConf object")

    if not isinstance(update_cfg, DictConfig):
        update_cfg = OmegaConf.create(update_cfg)

    if is_dataclass(model_cls):
        model_cls = OmegaConf.structured(model_cls)

    # Update optional configs
    model_cls = _update_subconfig(
        model_cls, update_cfg, subconfig_key="train_ds", drop_missing_subconfigs=drop_missing_subconfigs
    )
    model_cls = _update_subconfig(
        model_cls, update_cfg, subconfig_key="validation_ds", drop_missing_subconfigs=drop_missing_subconfigs
    )
    model_cls = _update_subconfig(
        model_cls, update_cfg, subconfig_key="test_ds", drop_missing_subconfigs=drop_missing_subconfigs
    )
    model_cls = _update_subconfig(
        model_cls, update_cfg, subconfig_key="optim", drop_missing_subconfigs=drop_missing_subconfigs
    )

    # Add optim and sched additional keys to model cls
    model_cls = _add_subconfig_keys(model_cls, update_cfg, subconfig_key="optim")

    # Perform full merge of model config class and update config
    # Remove ModelPT artifact `target`
    if "target" in update_cfg.model and "target" not in model_cls.model:  # type: ignore
        with open_dict(update_cfg.model):
            update_cfg.model.pop("target")

    # Remove ModelPT artifact `mridc_version`
    if "mridc_version" in update_cfg.model and "mridc_version" not in model_cls.model:  # type: ignore
        with open_dict(update_cfg.model):
            update_cfg.model.pop("mridc_version")

    return OmegaConf.merge(model_cls, update_cfg)


def _update_subconfig(
    model_cfg: "DictConfig", update_cfg: "DictConfig", subconfig_key: str, drop_missing_subconfigs: bool
):
    """
    Updates the MRIDCConfig DictConfig such that:
        1)  If the sub-config key exists in the `update_cfg`, but does not exist in ModelPT config:
            - Add the sub-config from update_cfg to ModelPT config
        2) If the sub-config key does not exist in `update_cfg`, but exists in ModelPT config:
            - Remove the sub-config from the ModelPT config; iff the `drop_missing_subconfigs` flag is set.

    Parameters
    ----------
    model_cfg: A DictConfig instantiated from the MRIDCConfig subclass.
    update_cfg: A DictConfig that mirrors the structure of `model_cfg`, used to update its default values.
    subconfig_key: A str key used to check and update the sub-config.
    drop_missing_subconfigs: A bool flag, whether to allow deletion of the MRIDCConfig sub-config, if its mirror
    sub-config does not exist in the `update_cfg`.

    Returns
    -------
    The updated DictConfig for the MRIDCConfig
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        sys.exit(1)
    with open_dict(model_cfg.model):
        # If update config has the key, but model cfg doesnt have the key
        # Add the update cfg subconfig to the model cfg
        if subconfig_key in update_cfg.model and subconfig_key not in model_cfg.model:
            model_cfg.model[subconfig_key] = update_cfg.model[subconfig_key]

        # If update config does not the key, but model cfg has the key
        # Remove the model cfg subconfig in order to match layout of update cfg
        if subconfig_key not in update_cfg.model and subconfig_key in model_cfg.model and drop_missing_subconfigs:
            model_cfg.model.pop(subconfig_key)

    return model_cfg


def _add_subconfig_keys(model_cfg: "DictConfig", update_cfg: "DictConfig", subconfig_key: str):
    """
    For certain sub-configs, the default values specified by the MRIDCConfig class is insufficient.
    In order to support every potential value in the merge between the `update_cfg`, it would require explicit
    definition of all possible cases.
    An example of such a case is Optimizers, and their equivalent Schedulers. All optimizers share a few basic details
    - such as name and lr, but almost all require additional parameters - such as weight decay.
    It is impractical to create a config for every single optimizer + every single scheduler combination.
    In such a case, we perform a dual merge. The Optim and Sched Dataclass contain the bare minimum essential
    components. The extra values are provided via update_cfg.
    In order to enable the merge, we first need to update the update sub-config to incorporate the keys, with dummy
    temporary values (merge update config with model config). This is done on a copy of the update sub-config, as the
    actual override values might be overridden by the MRIDCConfig defaults.
    Then we perform a merge of this temporary sub-config with the actual override config in a later step (merge
    model_cfg with original update_cfg, done outside this function).

    Parameters
    ----------
    model_cfg: A DictConfig instantiated from the MRIDCConfig subclass.
    update_cfg: A DictConfig that mirrors the structure of `model_cfg`, used to update its default values.
    subconfig_key: A str key used to check and update the sub-config.

    Returns
    -------
    A ModelPT DictConfig with additional keys added to the sub-config.
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        sys.exit(1)
    with open_dict(model_cfg.model):
        # Create copy of original model sub config
        if subconfig_key in update_cfg.model:
            if subconfig_key not in model_cfg.model:
                # create the key as a placeholder
                model_cfg.model[subconfig_key] = None

            subconfig = copy.deepcopy(model_cfg.model[subconfig_key])
            update_subconfig = copy.deepcopy(update_cfg.model[subconfig_key])

            # Add the keys and update temporary values, will be updated during full merge
            subconfig = OmegaConf.merge(update_subconfig, subconfig)
            # Update sub config
            model_cfg.model[subconfig_key] = subconfig

    return model_cfg


def assert_dataclass_signature_match(
    cls: "class_type",  # type: ignore
    datacls: "dataclass",  # type: ignore
    ignore_args: Optional[List[str]] = None,
    remap_args: Optional[Dict[str, str]] = None,
):
    """
    Analyses the signature of a provided class and its respective data class,
    asserting that the dataclass signature matches the class __init__ signature.
    Note:
        This is not a value based check. This function only checks if all argument
        names exist on both class and dataclass and logs mismatches.

    Parameters
    ----------
    cls: Any class type - but not an instance of a class. Pass type(x) where x is an instance
        if class type is not easily available.
    datacls: A corresponding dataclass for the above class.
    ignore_args: (Optional) A list of string argument names which are forcibly ignored,
        even if mismatched in the signature. Useful when a dataclass is a superset of the
        arguments of a class.
    remap_args: (Optional) A dictionary, mapping an argument name that exists (in either the
        class or its dataclass), to another name. Useful when argument names are mismatched between
        a class and its dataclass due to indirect instantiation via a helper method.

    Returns
    -------
    A tuple containing information about the analysis:
        1) A bool value which is True if the signatures matched exactly / after ignoring values.
            False otherwise.
        2) A set of arguments names that exist in the class, but *do not* exist in the dataclass.
            If exact signature match occurs, this will be None instead.
        3) A set of argument names that exist in the data class, but *do not* exist in the class itself.
            If exact signature match occurs, this will be None instead.
    """
    class_sig = inspect.signature(cls.__init__)

    class_params = dict(**class_sig.parameters)
    class_params.pop("self")

    dataclass_sig = inspect.signature(datacls)

    dataclass_params = dict(**dataclass_sig.parameters)
    dataclass_params.pop("_target_", None)

    class_params = set(class_params.keys())  # type: ignore
    dataclass_params = set(dataclass_params.keys())  # type: ignore

    if remap_args is not None:
        for original_arg, new_arg in remap_args.items():
            if original_arg in class_params:
                class_params.remove(original_arg)  # type: ignore
                class_params.add(new_arg)  # type: ignore
                logging.info(f"Remapped {original_arg} -> {new_arg} in {cls.__name__}")

            if original_arg in dataclass_params:
                dataclass_params.remove(original_arg)  # type: ignore
                dataclass_params.add(new_arg)  # type: ignore
                logging.info(f"Remapped {original_arg} -> {new_arg} in {datacls.__name__}")

    if ignore_args is not None:
        ignore_args = set(ignore_args)  # type: ignore

        class_params = class_params - ignore_args  # type: ignore
        dataclass_params = dataclass_params - ignore_args  # type: ignore
        logging.info(f"Removing ignored arguments - {ignore_args}")

    intersection: Set[type] = set.intersection(class_params, dataclass_params)  # type: ignore
    subset_cls = class_params - intersection  # type: ignore
    subset_datacls = dataclass_params - intersection  # type: ignore

    if (len(class_params) != len(dataclass_params)) or len(subset_cls) > 0 or len(subset_datacls) > 0:
        logging.error(f"Class {cls.__name__} arguments do not match " f"Dataclass {datacls.__name__}!")

        if len(subset_cls) > 0:
            logging.error(f"Class {cls.__name__} has additional arguments :\n" f"{subset_cls}")

        if len(subset_datacls):
            logging.error(f"Dataclass {datacls.__name__} has additional arguments :\n{subset_datacls}")

        return False, subset_cls, subset_datacls
    return True, None, None
