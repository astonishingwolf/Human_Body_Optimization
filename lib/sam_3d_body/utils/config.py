# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError
from yacs.config import CfgNode as CN


# OmegaConf support for variable interpolation (e.g. ${paths.MODEL}/weights.ckpt)
# Skips interpolations it can't resolve, which lets us load Hydra .yamls without a Hydra runtime
def resolve_omegaconf_to_dict(conf):
    """
    Recursively convert an OmegaConf object to a dictionary, resolving interpolations
    where possible and leaving unsupported ones as-is.
    """
    if isinstance(conf, DictConfig):
        result = {}
        for k, v in conf.items():
            try:
                result[k] = resolve_omegaconf_to_dict(v)
            except InterpolationResolutionError:
                # Convert unresolved OmegaConf objects to containers without resolving
                result[k] = OmegaConf.to_container(v, resolve=False)
        return result
    elif isinstance(conf, ListConfig):
        result = []
        for item in conf:
            try:
                result.append(resolve_omegaconf_to_dict(item))
            except InterpolationResolutionError:
                # Convert unresolved OmegaConf objects to containers without resolving
                result.append(OmegaConf.to_container(item, resolve=False))
        return result
    else:
        # Base case: conf is a primitive value or an interpolation
        if OmegaConf.is_config(conf):
            try:
                return OmegaConf.to_container(conf, resolve=True)
            except InterpolationResolutionError:
                # Convert unresolved OmegaConf objects to containers without resolving
                return OmegaConf.to_container(conf, resolve=False)
        else:
            # conf is a primitive value
            return conf


def to_lower(x: Dict) -> Dict:
    """
    Convert all dictionary keys to lowercase
    Args:
      x (dict): Input dictionary
    Returns:
      dict: Output dictionary with all keys converted to lowercase
    """
    return {k.lower(): v for k, v in x.items()}


def get_config(config_file: str) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    cfg = CN(new_allowed=True)

    # Resolve config with OmegaConf
    conf = OmegaConf.load(config_file)
    conf_dict = resolve_omegaconf_to_dict(conf)
    conf_cfg = CN(conf_dict)

    # Merge resolved config with the default or new one
    cfg.merge_from_other_cfg(conf_cfg)

    cfg.freeze()
    return cfg


def get_config_from_sections(model_section: dict, training_section=None) -> CN:
    """
    Build a SAM3D config from nested training config sections.
    Expects required keys to be present in the provided sections.
    """
    cfg = CN(new_allowed=True)

    model_dict = model_section
    if OmegaConf.is_config(model_section):
        model_dict = OmegaConf.to_container(model_section, resolve=True)

    training_dict = training_section
    if OmegaConf.is_config(training_section):
        training_dict = OmegaConf.to_container(training_section, resolve=True)

    conf_dict = {}
    conf_dict["MODEL"] = model_dict["sam3dbody"]["MODEL"]
    conf_dict["TRAIN"] = model_dict["sam3dbody"]["TRAIN"]
    conf_dict["CHECKPOINT_PATH"] = conf_dict["MODEL"]["CHECKPOINT_PATH"]
    conf_dict["avatar"] = model_dict["avatar"]
    conf_dict["triplane"] = model_dict.get("triplane", {}) # triplane is optional
    

    use_fp16 = training_dict.get("use_fp16")
    if use_fp16 is None:
        mixed_precision = training_dict.get("mixed_precision", "no")
        use_fp16 = mixed_precision != "no"

    cfg.merge_from_other_cfg(CN(conf_dict))
    cfg.freeze()
    return cfg
