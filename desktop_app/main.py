"""
IsoCortex Desktop App — Entry Point
=====================================
Launches the native desktop GUI application.
No web server, no browser, no terminal commands required.

On first launch, downloads the AI model before showing the login screen.
"""

import sys
import os
import logging
import threading
from pathlib import Path

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

    # ── First-launch model check ──────────────────────────────────────
    from desktop_app.llm import model_exists, download_model_with_progress

    if not model_exists():
        logger.info("AI model not found. Showing download screen...")

        # Create a minimal download window
        dl_root = ctk.CTk()
        dl_root.title("IsoCortex — Downloading AI Model")
        dl_root.geometry("520x280")
        dl_root.resizable(False, False)
        dl_root.configure(fg_color="#0a0a12")

        # Set IsoCortex icon on download window
        _assets = Path(__file__).parent / "assets"
        for _icon_name in ("app_icon.png", "isocortex-logo.png", "favicon.png"):
            _icon_p = _assets / _icon_name
            if _icon_p.exists():
                try:
                    from PIL import Image
                    _img = Image.open(str(_icon_p)).convert("RGBA")
                    if _img.size[0] > 256:
                        _img = _img.resize((256, 256), Image.Resampling.LANCZOS)
                    dl_root.iconphoto(True, _img)
                except Exception:
                    pass
                break

        # Center on screen
        dl_root.update_idletasks()
        x = (dl_root.winfo_screenwidth() // 2) - 260
        y = (dl_root.winfo_screenheight() // 2) - 140
        dl_root.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(dl_root, fg_color="#0a0a12")
        frame.pack(fill="both", expand=True, padx=40, pady=30)

        ctk.CTkLabel(
            frame,
            text="IsoCortex AI Setup",
            font=("Segoe UI", 20, "bold"),
            text_color="#eaeaf2",
        ).pack(pady=(0, 8))

        ctk.CTkLabel(
            frame,
            text="Downloading the AI model for the first time.\n"
                 "This is a one-time step (~950 MB).",
            font=("Segoe UI", 12),
            text_color="#8888a8",
            justify="center",
        ).pack(pady=(0, 20))

        # Progress bar
        progress = ctk.CTkProgressBar(
            frame, width=440, height=8,
            fg_color="#181830", progress_color="#7c3aed",
        )
        progress.pack(pady=(0, 8))
        progress.set(0)

        status_label = ctk.CTkLabel(
            frame,
            text="Preparing download...",
            font=("Segoe UI", 11),
            text_color="#505068",
        )
        status_label.pack()

        download_done = threading.Event()
        download_error: list[str | None] = [None]
        retry_btn: list = [None]

        def _progress_cb(downloaded: int, total: int, text: str):
            if total > 1:
                pct = min(downloaded / total, 1.0)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                status_text = f"{downloaded_mb:.0f} / {total_mb:.0f} MB"
            else:
                pct = 0
                status_text = text

            dl_root.after(0, lambda: _update_progress(pct, status_text))

        def _update_progress(pct, text):
            try:
                progress.set(pct)
                status_label.configure(text=text)
            except Exception:
                pass

        def _start_download():
            if retry_btn[0] is not None:
                try:
                    retry_btn[0].pack_forget()
                except Exception:
                    pass
                retry_btn[0] = None
            progress.set(0)
            status_label.configure(text="Preparing download...", text_color="#505068")
            download_error[0] = None
            download_done.clear()
            t = threading.Thread(target=_download_thread, daemon=True)
            t.start()

        def _download_thread():
            try:
                download_model_with_progress(on_progress=_progress_cb)
                dl_root.after(0, _finish_download)
            except Exception as exc:
                captured_exc = str(exc)
                download_error[0] = captured_exc
                dl_root.after(0, lambda: _fail_download(captured_exc))

        def _finish_download():
            try:
                progress.set(1.0)
                status_label.configure(text="Download complete!", text_color="#22c55e")
            except Exception:
                pass
            download_done.set()
            dl_root.after(1500, dl_root.destroy)

        def _fail_download(error_msg):
            try:
                status_label.configure(
                    text=f"Download failed: {error_msg}",
                    text_color="#ef4444",
                )
                btn = ctk.CTkButton(
                    frame,
                    text="Retry Download",
                    font=("Segoe UI", 12),
                    width=200,
                    height=36,
                    fg_color="#7c3aed",
                    hover_color="#6d28d9",
                    command=_start_download,
                )
                btn.pack(pady=(16, 0))
                retry_btn[0] = btn
            except Exception:
                pass
            download_done.set()

        _start_download()

        dl_root.mainloop()

        if download_error[0]:
            logger.error("Model download failed: %s", download_error[0])
            # Continue anyway — user can still use search without AI
            logger.warning("Continuing without AI model. User can download later.")

    # ── First-launch: check embedding model ─────────────────────────
    from desktop_app.engine import embedding_model_exists, download_embedding_model

    if not embedding_model_exists():
        logger.info("Embedding model not found. Showing download screen...")

        emb_root = ctk.CTk()
        emb_root.title("IsoCortex — Downloading Embedding Model")
        emb_root.geometry("520x280")
        emb_root.resizable(False, False)
        emb_root.configure(fg_color="#0a0a12")

        # Set icon
        _assets = Path(__file__).parent / "assets"
        for _icon_name in ("app_icon.png", "isocortex-logo.png", "favicon.png"):
            _icon_p = _assets / _icon_name
            if _icon_p.exists():
                try:
                    from PIL import Image
                    _img = Image.open(str(_icon_p)).convert("RGBA")
                    if _img.size[0] > 256:
                        _img = _img.resize((256, 256), Image.Resampling.LANCZOS)
                    emb_root.iconphoto(True, _img)
                except Exception:
                    pass
                break

        emb_root.update_idletasks()
        x = (emb_root.winfo_screenwidth() // 2) - 260
        y = (emb_root.winfo_screenheight() // 2) - 140
        emb_root.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(emb_root, fg_color="#0a0a12")
        frame.pack(fill="both", expand=True, padx=40, pady=30)

        ctk.CTkLabel(
            frame,
            text="IsoCortex AI Setup",
            font=("Segoe UI", 20, "bold"),
            text_color="#eaeaf2",
        ).pack(pady=(0, 8))

        ctk.CTkLabel(
            frame,
            text="Downloading the embedding model (~90 MB).\n"
                 "This is needed for document search and is a one-time step.",
            font=("Segoe UI", 12),
            text_color="#8888a8",
            justify="center",
        ).pack(pady=(0, 20))

        emb_status = ctk.CTkLabel(
            frame,
            text="Preparing...",
            font=("Segoe UI", 11),
            text_color="#505068",
        )
        emb_status.pack(pady=(10, 0))

        # Simple spinner animation
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        _spin_idx = [0]

        def _spin():
            if not emb_done.is_set():
                try:
                    emb_status.configure(
                        text=f"{spinner_chars[_spin_idx[0] % len(spinner_chars)]} "
                             f"{emb_status.cget('text').split(' ', 1)[-1] if ' ' in emb_status.cget('text') else 'Downloading...'}"
                    )
                except Exception:
                    pass
                _spin_idx[0] += 1
                emb_root.after(80, _spin)

        emb_done = threading.Event()
        emb_error: list[str | None] = [None]
        emb_retry_btn: list = [None]

        def _start_emb_download():
            if emb_retry_btn[0] is not None:
                try:
                    emb_retry_btn[0].pack_forget()
                except Exception:
                    pass
                emb_retry_btn[0] = None
            emb_done.clear()
            emb_error[0] = None
            emb_status.configure(text="Downloading...", text_color="#505068")
            _spin()
            t = threading.Thread(target=_emb_download_thread, daemon=True)
            t.start()

        def _emb_download_thread():
            try:
                def _cb(status_text):
                    captured = status_text
                    emb_root.after(0, lambda: _update_emb(captured))
                download_embedding_model(on_progress=_cb)
                emb_root.after(0, _finish_emb)
            except Exception as exc:
                captured_exc = str(exc)
                emb_error[0] = captured_exc
                emb_root.after(0, lambda: _fail_emb(captured_exc))

        def _update_emb(text):
            try:
                if text:
                    emb_status.configure(text=text)
            except Exception:
                pass

        def _finish_emb():
            try:
                emb_status.configure(text="Embedding model ready!", text_color="#22c55e")
            except Exception:
                pass
            emb_done.set()
            emb_root.after(1500, emb_root.destroy)

        def _fail_emb(error_msg):
            try:
                emb_status.configure(
                    text=f"Embedding download failed: {error_msg}",
                    text_color="#ef4444",
                )
                btn = ctk.CTkButton(
                    frame,
                    text="Retry Download",
                    font=("Segoe UI", 12),
                    width=200,
                    height=36,
                    fg_color="#7c3aed",
                    hover_color="#6d28d9",
                    command=_start_emb_download,
                )
                btn.pack(pady=(16, 0))
                emb_retry_btn[0] = btn
            except Exception:
                pass
            emb_done.set()

        _start_emb_download()
        emb_root.mainloop()

        if emb_error[0]:
            logger.error("Embedding model download failed: %s", emb_error[0])
            logger.warning("Continuing without embedding model — search will not work.")

    # ── Launch main app ───────────────────────────────────────────────
    from desktop_app.app import IsoCortexApp

    logger.info("Starting IsoCortex Desktop App...")
    app = IsoCortexApp()
    app.mainloop()