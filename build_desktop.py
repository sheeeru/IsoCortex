"""
IsoCortex Desktop App — PyInstaller Build Script
================================================
Builds the desktop app into a standalone executable.

Usage:
    python build_desktop.py            # Build for current platform
    python build_desktop.py --onefile   # Build as single .exe file
    python build_desktop.py --onedir     # Build as directory
    python build_desktop.py --clean     # Clean build artifacts
"""

import subprocess
import sys
import os
from pathlib import Path

# Project paths
DESKTOP_DIR = Path(__file__).parent
PROJECT_ROOT = DESKTOP_DIR.parent
APP_ENTRY = DESKTOP_DIR / "main.py"
SPEC_FILE = DESKTOP_DIR / "isocortex.spec"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def clean():
    """Remove previous build artifacts."""
    for d in [BUILD_DIR, DIST_DIR]:
        if d.exists():
            print(f"Removing {d}...")
            import shutil
            shutil.rmtree(d)
    if SPEC_FILE.exists():
        SPEC_FILE.unlink()
    print("Clean complete.")


def build(onefile: bool = False):
    """Build the desktop app with PyInstaller."""
    # Ensure dependencies
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-r", str(DESKTOP_DIR / "requirements.txt"),
        "--target", str(BUILD_DIR),
        "--no-warn-script-location",
    ])

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "IsoCortex",
        "--windowed",
        "--noconfirm",
        "--clean",
    ]

    if onefile:
        cmd.append("--onefile")
        cmd.extend(["--icon", str(PROJECT_ROOT / "website" / "public" / "favicon.ico")])

    # Hidden imports to ensure all modules are bundled
    cmd.extend([
        "--hidden-import=sentence_transformers",
        "--hidden-import=numpy",
        "--hidden-import=pymupdf",
        "--hidden-import=docx",
        "--hidden-import=pptx",
        "--hidden-import=openpyxl",
        "--hidden-import=bcrypt",
        "--collect-all", "sentence_transformers",
        "--collect-all", "customtkinter",
        "--paths", str(DESKTOP_DIR),
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        str(APP_ENTRY),
    ])

    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    print()
    print("=" * 50)
    print("  Build Complete!")
    print("=" * 50)
    print(f"  Output: {DIST_DIR}")
    print(f"  Entry:  {APP_ENTRY}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build IsoCortex Desktop App")
    parser.add_argument("--onefile", action="store_true", help="Build as single executable")
    parser.add_argument("--clean", action="store_true", help="Clean previous builds")
    args = parser.parse_args()

    if args.clean:
        clean()
    else:
        build(onefile=args.onefile)

# mac app
#pyinstaller -y --name "IsoCortex" --noconsole --onedir \
#  --add-data "desktop_app/assets:desktop_app/assets" \
#  --hidden-import=desktop_app \
#  --hidden-import=desktop_app.app \
#  --hidden-import=desktop_app.theme \
#  --hidden-import=desktop_app.engine \
#  --hidden-import=desktop_app.workers \
#  --hidden-import=desktop_app.screens \
#  --hidden-import=desktop_app.components \
#  --paths . \
#  --exclude-module=torch \
#  --exclude-module=tensorflow \
#  --exclude-module=scipy \
# --exclude-module=sklearn \
#  --exclude-module=matplotlib \
#  --exclude-module=pandas \
#  --exclude-module=cv2 \
#  --exclude-module=sympy \
#  --exclude-module=lxml \
#  --exclude-module=sqlalchemy \
#  --exclude-module=pydantic \
#  --exclude-module=pytest \
#  --exclude-module=rich \
#  --exclude-module=PIL.SpiderImagePlugin \
#  desktop_app/main.py