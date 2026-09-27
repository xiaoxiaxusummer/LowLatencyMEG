from .MEGEnv_multiconstraints import MEGEnv_multiconstraints

"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from gymnasium.envs.registration import (
    # load_plugin_envs,
    register,
)

# MEG env

register(
    id="MEG-multiconstraints",
    entry_point="src.env.MEG_envs.MEGEnv_multiconstraints:MEGEnv_multiconstraints",
    kwargs={'dataset': True},
    max_episode_steps = 64, # Maximum length of an episode
)

# Hook to load plugins from entry points
# load_plugin_envs()
