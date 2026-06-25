"""
IsoCortex Desktop App — Update Checker
======================================
Checks GitHub releases API for newer versions.
Uses urllib.request (stdlib) — no external dependencies.
"""

import json
import logging
import threading
import urllib.request
import urllib.error
from typing import Callable, Optional

from desktop_app.version import __version__, GITHUB_REPO

logger = logging.getLogger("IsoCortex.updater")


def check_for_updates(
    current_version: str | None = None,
    callback: Callable[[Optional[str]], None] | None = None,
) -> None:
    """Check GitHub releases API for a newer version.

    Calls callback(latest_version) if an update is available,
    or callback(None) if up-to-date or on error.
    Runs in a background daemon thread.
    """
    if current_version is None:
        current_version = __version__

    def _check():
        try:
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "IsoCortex-Desktop",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            latest = data.get("tag_name", "").lstrip("v")
            if latest and latest != current_version:
                logger.info("Update available: %s -> %s", current_version, latest)
                callback(latest)  # type: ignore[misc]
                return

            logger.debug("App is up-to-date (%s)", current_version)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                logger.debug("No releases found (repo may be private)")
            else:
                logger.debug("Update check HTTP error: %s", exc)
        except Exception as exc:
            logger.debug("Update check failed: %s", exc)

        try:
            callback(None)  # type: ignore[misc]
        except Exception:
            pass

    t = threading.Thread(target=_check, daemon=True)
    t.start()