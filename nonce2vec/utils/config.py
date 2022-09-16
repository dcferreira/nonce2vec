"""Config utilities.

Methods used to manipulate YAML-based configuration files.
"""

import yaml
from loguru import logger

from nonce2vec.utils.immutables import ImmutableConfig

__all__ = "load"


def load(config_file):
    """Load an ImmutableConfig from a YAML configuration file."""
    logger.info("Loading config from file {}".format(config_file))
    with open(config_file, "r", encoding="utf-8") as config_stream:
        config = yaml.safe_load(config_stream)
        return ImmutableConfig(config)
