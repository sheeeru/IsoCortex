"""
IsoCortex Desktop App — Plugin System
======================================
A lightweight plugin system that allows users to extend IsoCortex
with custom Python plugins.

Plugin types:
  - extractor: Custom text extractors for new file formats
  - pre_search: Modify queries before search
  - post_search: Filter/rank results after search
  - pre_ingest: Modify text before chunking/embedding
  - command: Custom commands accessible from the chat

Plugins are loaded from ~/.isocortex/plugins/
Each plugin is a .py file with a `register()` function.
"""

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("IsoCortex.plugins")

PLUGIN_DIR = Path.home() / ".isocortex" / "plugins"


@dataclass
class PluginInfo:
    """Metadata about a loaded plugin."""
    name: str
    version: str
    description: str
    author: str
    file_path: str
    hooks: dict[str, Callable] = field(default_factory=dict)
    enabled: bool = True


class PluginManager:
    """
    Manages plugin discovery, loading, and execution.

    Usage:
        pm = PluginManager()
        pm.discover_plugins()
        pm.load_all()

        # Run a hook
        results = pm.run_hook("pre_search", query="my search")
    """

    def __init__(self):
        self._plugins: dict[str, PluginInfo] = {}
        self._hook_registry: dict[str, list[tuple[str, Callable]]] = {}

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        return self._plugins

    @property
    def loaded_count(self) -> int:
        return len(self._plugins)

    def discover_plugins(self) -> list[Path]:
        """Scan the plugin directory for .py files.

        Returns list of plugin file paths found.
        """
        PLUGIN_DIR.mkdir(parents=True, exist_ok=True)

        plugin_files = []
        for f in PLUGIN_DIR.glob("*.py"):
            if f.name.startswith("_"):
                continue
            plugin_files.append(f)

        if plugin_files:
            logger.info("Discovered %d plugins", len(plugin_files))

        return plugin_files

    def load_plugin(self, file_path: Path) -> Optional[PluginInfo]:
        """Load a single plugin from a .py file.

        The plugin file must define a `register()` function that returns
        a dict with:
          - name: str
          - version: str (optional, default "1.0")
          - description: str (optional)
          - author: str (optional)
          - hooks: dict[str, callable]  — hook_name -> function
        """
        try:
            spec = importlib.util.spec_from_file_location(
                f"isocortex_plugin_{file_path.stem}",
                str(file_path),
            )
            if spec is None or spec.loader is None:
                logger.error("Failed to create module spec for %s", file_path.name)
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"isocortex_plugin_{file_path.stem}"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "register"):
                logger.error("Plugin %s has no register() function", file_path.name)
                return None

            info_dict = module.register()
            if not isinstance(info_dict, dict):
                logger.error("Plugin %s register() must return a dict", file_path.name)
                return None

            name = info_dict.get("name", file_path.stem)
            version = info_dict.get("version", "1.0")
            description = info_dict.get("description", "")
            author = info_dict.get("author", "")
            hooks = info_dict.get("hooks", {})

            if not isinstance(hooks, dict):
                hooks = {}

            plugin_info = PluginInfo(
                name=name,
                version=version,
                description=description,
                author=author,
                file_path=str(file_path),
                hooks=hooks,
            )

            self._plugins[name] = plugin_info

            # Register hooks
            for hook_name, hook_func in hooks.items():
                if callable(hook_func):
                    if hook_name not in self._hook_registry:
                        self._hook_registry[hook_name] = []
                    self._hook_registry[hook_name].append((name, hook_func))

            logger.info(
                "Loaded plugin '%s' v%s with %d hooks: %s",
                name, version, len(hooks), ", ".join(hooks.keys()),
            )

            return plugin_info

        except Exception as exc:
            logger.error("Failed to load plugin %s: %s", file_path.name, exc)
            return None

    def load_all(self) -> int:
        """Discover and load all plugins. Returns count loaded."""
        plugin_files = self.discover_plugins()
        loaded = 0
        for pf in plugin_files:
            if self.load_plugin(pf):
                loaded += 1
        return loaded

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin by name. Returns True if found."""
        if name not in self._plugins:
            return False

        plugin = self._plugins.pop(name)

        # Remove from hook registry
        for hook_name in plugin.hooks:
            if hook_name in self._hook_registry:
                self._hook_registry[hook_name] = [
                    (n, f) for n, f in self._hook_registry[hook_name] if n != name
                ]

        # Remove from sys.modules
        module_name = f"isocortex_plugin_{Path(plugin.file_path).stem}"
        sys.modules.pop(module_name, None)

        logger.info("Unloaded plugin '%s'", name)
        return True

    def run_hook(self, hook_name: str, **kwargs) -> Any:
        """Execute all registered functions for a hook.

        Each hook function receives the keyword arguments and can:
        - Return a modified value (for pre_* hooks)
        - Return None to pass through unchanged
        - Raise an exception to stop processing

        The return value from the LAST hook is returned.
        """
        hooks = self._hook_registry.get(hook_name, [])
        if not hooks:
            return kwargs.get("default", None)

        result = kwargs.get("default", None)

        for plugin_name, hook_func in hooks:
            plugin = self._plugins.get(plugin_name)
            if not plugin or not plugin.enabled:
                continue

            try:
                internal_keys = {"default"}
                hook_kwargs = {k: v for k, v in kwargs.items() if k not in internal_keys}
                hook_result = hook_func(**hook_kwargs)
                if hook_result is not None:
                    result = hook_result
            except Exception as exc:
                logger.warning(
                    "Plugin '%s' hook '%s' failed: %s",
                    plugin_name, hook_name, exc,
                )

        return result

    def get_plugin_list(self) -> list[dict]:
        """Return info about all loaded plugins."""
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "author": p.author,
                "enabled": p.enabled,
                "hooks": list(p.hooks.keys()),
                "file": Path(p.file_path).name,
            }
            for p in self._plugins.values()
        ]

    def toggle_plugin(self, name: str, enabled: bool) -> bool:
        """Enable or disable a plugin."""
        if name not in self._plugins:
            return False
        self._plugins[name].enabled = enabled
        return True


def create_example_plugin():
    """Create an example plugin file for users to learn from."""
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)

    example_path = PLUGIN_DIR / "_example_plugin.py"
    if example_path.exists():
        return

    example_code = '''\
"""
IsoCortex Example Plugin
========================
This is an example plugin that demonstrates all available hooks.
Copy this file and modify it to create your own plugin.
"""

def register():
    return {
        "name": "Example Plugin",
        "version": "1.0",
        "description": "Demonstrates all available plugin hooks",
        "author": "IsoCortex User",
        "hooks": {
            "pre_search": pre_search_hook,
            "post_search": post_search_hook,
            "pre_ingest": pre_ingest_hook,
        },
    }


def pre_search_hook(query, **kwargs):
    """Modify the search query before it\'s executed.

    Args:
        query: The user\'s search query string.

    Returns:
        Modified query string, or None to leave unchanged.
    """
    # Example: add context for common abbreviations
    expansions = {
        "api": "API application programming interface",
        "db": "database DB",
        "ml": "machine learning ML",
    }

    query_lower = query.lower()
    for abbr, expansion in expansions.items():
        if abbr in query_lower:
            return f"{query} ({expansion})"

    return None  # No modification


def post_search_hook(results, query, **kwargs):
    """Filter or re-rank search results after retrieval.

    Args:
        results: List of SearchResult objects.
        query: The original search query.

    Returns:
        Modified list of results, or None to leave unchanged.
    """
    # Example: boost results from .pdf files
    # (This would need SearchResult import, simplified here)
    return None  # No modification


def pre_ingest_hook(text, file_path, **kwargs):
    """Modify extracted text before chunking and embedding.

    Args:
        text: The extracted text content.
        file_path: Path to the source file.

    Returns:
        Modified text, or None to leave unchanged.
    """
    # Example: clean up common OCR artifacts
    # text = text.replace("fi", "fi")  # ligature fix
    return None  # No modification
'''

    example_path.write_text(example_code, encoding="utf-8")
    logger.info("Created example plugin at %s", example_path)