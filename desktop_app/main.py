"""
IsoCortex Desktop App — Entry Point
=====================================
Launches the native desktop GUI application.
No web server, no browser, no terminal commands required.
"""

import sys
import os
import logging

# ── Ensure the project root is on the Python path ─────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Configure logging before anything else ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IsoCortex")

# ── Launch ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import customtkinter as ctk
    except ImportError:
        logger.error(
            "customtkinter is not installed. "
            "Run: pip install customtkinter"
        )
        sys.exit(1)

    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")

    # Import our app
    from desktop_app.app import IsoCortexApp

    logger.info("Starting IsoCortex Desktop App...")
    app = IsoCortexApp()
    app.mainloop()
