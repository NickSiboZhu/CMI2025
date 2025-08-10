#!/usr/bin/env python3
"""Generic Registry implementation for CMI Competition."""

from typing import Any, Dict, Optional, Type


class Registry:
    """A simple registry mapping strings to classes."""
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        return self._name

    def get(self, key: str) -> Optional[Type]:
        return self._module_dict.get(key, None)

    def _register_module(self, module_class: Type, module_name: Optional[str] = None):
        if module_name is None:
            module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self._name}")
        self._module_dict[module_name] = module_class

    def register_module(self, name: Optional[str] = None):
        def _register(cls):
            self._register_module(module_class=cls, module_name=name)
            return cls
        return _register

    def __repr__(self):
        return f"Registry(name={self._name}, items={list(self._module_dict)})"


def build_from_cfg(cfg: Dict[str, Any], registry: Registry, default_args: Optional[Dict[str, Any]] = None):
    """Instantiate an object from a config dict (strict, config-driven)."""
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError("`cfg` must contain the key `type`")

    args = cfg.copy()
    obj_type = args.pop("type")
    obj_cls = registry.get(obj_type)
    if obj_cls is None:
        raise KeyError(f"{obj_type} is not registered in the `{registry.name}` registry")

    # Remove fallback default_args merging to ensure everything comes from cfg
    return obj_cls(**args) 