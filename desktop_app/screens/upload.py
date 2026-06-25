"""
IsoCortex Desktop App — Upload Screen
======================================
Simple file upload interface: browse files, click Index, done.
All files go into a single default HNSW graph automatically.

Enhanced with theme animations: ShimmerBar, GlassCard, GradientDivider,
AnimatedPulseGlow, AnimatedGradientBG, FadeInFrame, create_badge.
"""

import customtkinter as ctk
import logging
from pathlib import Path
from tkinter import filedialog

logger = logging.getLogger("IsoCortex.upload")

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_TKDND = True
except ImportError:
    DND_FILES = ""  # type: ignore[assignment]
    _HAS_TKDND = False

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER,
    COLOR_PURPLE, COLOR_PURPLE_DARK, COLOR_PURPLE_LIGHT, COLOR_PURPLE_DEEP,
    COLOR_GOLD, COLOR_GOLD_LIGHT, COLOR_GOLD_BTN_TEXT,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR, COLOR_INFO,
    COLOR_SHADOW,
    FONT_FAMILY, FONT_FAMILY_DISPLAY, FONT_FAMILY_MONO,
    FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_MEDIUM, FONT_SIZE_NORMAL, FONT_SIZE_SMALL, FONT_SIZE_XXS,
    BORDER_RADIUS, BORDER_RADIUS_XS, BORDER_RADIUS_SM, BORDER_RADIUS_LG,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    GradientCanvas, GRADIENT_PURPLE_GOLD,
    ShimmerBar, GlassCard, GradientDivider,
    AnimatedGradientBG, FadeInFrame, create_badge,
    ANIM_DELAY_200, ANIM_DELAY_400, ANIM_DELAY_600,
    COLOR_GLASS_BG, COLOR_GLASS_BORDER,
)
from desktop_app.workers import IngestionWorker, WorkerThread

# ── Extension -> colour mapping for file-row left borders & badges ─────
_EXT_COLORS: dict[str, str] = {
    ".pdf":  COLOR_ERROR,       # red
    ".docx": COLOR_INFO,         # blue
    ".doc":  COLOR_INFO,
    ".xlsx": COLOR_SUCCESS,     # green
    ".xls":  COLOR_SUCCESS,
    ".csv":  COLOR_SUCCESS,
    ".py":   COLOR_INFO,
    ".md":   COLOR_PURPLE_LIGHT,
    ".pptx": COLOR_WARNING,
    ".json": COLOR_GOLD,
    ".html": COLOR_WARNING,
    ".txt":  COLOR_TEXT_SECONDARY,
}
_DEFAULT_EXT_COLOR = COLOR_TEXT_DIM

_EXT_ICONS: dict[str, str] = {
    ".pdf": "PDF", ".docx": "DOC", ".doc": "DOC",
    ".xlsx": "XLS", ".xls": "XLS", ".csv": "CSV",
    ".py": "PY", ".md": "MD", ".pptx": "PPT",
    ".json": "JSN", ".html": "HTM", ".txt": "TXT",
}


class UploadScreen(ctk.CTkFrame):
    """Upload files and index them into the default HNSW graph."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app = app
        self._selected_files: list[str] = []
        self._is_processing = False
        self._build_ui()

    # ══════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        self.configure(fg_color="transparent", corner_radius=0)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        master = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=COLOR_BG_ELEVATED,
            scrollbar_button_hover_color=COLOR_BG_HOVER,
        )
        master.grid(row=0, column=0, sticky="nsew")

        # Animated gradient background (subtle, behind everything)
        self._bg_gradient = AnimatedGradientBG(master)
        self._bg_gradient.place(x=0, y=0, relwidth=1, relheight=1)

        # ShimmerBar accent at top (replaces static GradientCanvas)
        ShimmerBar(master, height=4).pack(fill="x", pady=(0, PADDING_LG))

        self._build_header(master)
        self._build_files_card(master)
        self._build_status_bar(master)
        self._build_indexed_folders(master)
        self._setup_drag_and_drop()

    # ── Header ───────────────────────────────────────────────────────

    def _build_header(self, parent: ctk.CTkFrame) -> None:
        # Wrap header in FadeInFrame for staggered entrance animation
        header_wrapper = FadeInFrame(
            parent, fg_color="transparent",
            delay=ANIM_DELAY_200,
            corner_radius=0,
        )
        header_wrapper.pack(fill="x", pady=(0, PADDING))

        header = ctk.CTkFrame(header_wrapper, fg_color="transparent")
        header.pack(fill="x", pady=(0, 0))

        accent = ctk.CTkFrame(
            header, width=4, height=36,
            fg_color=COLOR_PURPLE, corner_radius=2,
        )
        accent.pack(side="left", padx=(0, PADDING_MD))
        accent.pack_propagate(False)

        title_col = ctk.CTkFrame(header, fg_color="transparent")
        title_col.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            title_col, text="Upload Files",
            font=(FONT_FAMILY_DISPLAY, FONT_SIZE_TITLE, "bold"),
            text_color=COLOR_TEXT, anchor="w",
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_col,
            text="Add your PDFs, documents, and text files — they'll be indexed into your search graph automatically",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM, anchor="w",
        ).pack(anchor="w")

    # ── Files Card ───────────────────────────────────────────────────

    def _build_files_card(self, parent: ctk.CTkFrame) -> None:
        # Gradient divider between header and files card
        GradientDivider(parent, height=1).pack(fill="x", pady=(0, PADDING_MD))

        # Wrap files card in FadeInFrame with staggered delay
        files_wrapper = FadeInFrame(
            parent, fg_color="transparent",
            delay=ANIM_DELAY_400,
            corner_radius=0,
        )
        files_wrapper.pack(fill="both", expand=True, pady=(0, PADDING_LG))

        # GlassCard replaces the manual glow + shadow + card stack
        card = GlassCard(
            files_wrapper,
            glow_color=COLOR_PURPLE,
            corner_radius=BORDER_RADIUS_LG,
        )
        card.pack(fill="both", expand=True)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING)

        # File count row
        count_row = ctk.CTkFrame(inner, fg_color="transparent")
        count_row.pack(fill="x", pady=(0, PADDING_SM))

        ctk.CTkLabel(
            count_row, text="Selected Files",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM),
            text_color=COLOR_TEXT, anchor="w",
        ).pack(side="left")

        self._file_count_label = ctk.CTkLabel(
            count_row, text="0 files",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
        )
        self._file_count_label.pack(side="right")

        # Drop zone
        self._drop_border = ctk.CTkFrame(
            inner, fg_color=COLOR_BG_ELEVATED,
            corner_radius=BORDER_RADIUS,
            border_width=2, border_color=COLOR_BORDER_LIGHT,
        )
        self._drop_border.pack(fill="x", pady=(0, PADDING_MD))

        drop_inner = ctk.CTkFrame(self._drop_border, fg_color="transparent")
        drop_inner.pack(fill="x", padx=PADDING_LG, pady=PADDING_MD)

        icon_text = ctk.CTkFrame(drop_inner, fg_color="transparent")
        icon_text.pack(anchor="center")

        # Static glow behind the upload icon (no animation timer needed)
        glow_canvas = ctk.CTkFrame(
            icon_text, fg_color=COLOR_PURPLE_DEEP,
            width=56, height=56, corner_radius=28,
        )
        glow_canvas.pack(pady=(0, 0))
        glow_canvas.pack_propagate(False)

        ctk.CTkLabel(
            icon_text, text="\u2B06",
            font=(FONT_FAMILY, 22),
            text_color=COLOR_PURPLE_LIGHT,
        ).place(in_=glow_canvas, relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            drop_inner,
            text="Drag & drop files here, or browse to select",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
        ).pack(pady=(PADDING_SM, 0))

        ctk.CTkButton(
            drop_inner, text="Browse Files",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color=COLOR_GOLD, hover_color=COLOR_GOLD_LIGHT,
            text_color=COLOR_GOLD_BTN_TEXT,
            height=34, width=140,
            corner_radius=BORDER_RADIUS_SM,
            command=self._browse_files,
        ).pack(pady=(PADDING_SM, 0))

        # File list (scrollable, grows with available space)
        list_container = ctk.CTkFrame(inner, fg_color="transparent")
        list_container.pack(fill="both", expand=True, pady=(0, PADDING))

        self._file_list = ctk.CTkScrollableFrame(
            list_container,
            fg_color="transparent",
            label_text="",
            scrollbar_button_color=COLOR_BG_ELEVATED,
            scrollbar_button_hover_color=COLOR_BG_HOVER,
        )
        self._file_list.pack(fill="both", expand=True)

        self._empty_label = ctk.CTkLabel(
            self._file_list,
            text="No files selected yet",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
        )
        self._empty_label.pack(pady=(PADDING, 0))

    # ── Status Bar ───────────────────────────────────────────────────

    def _build_status_bar(self, parent: ctk.CTkFrame) -> None:
        # Gradient divider between files card and status bar
        GradientDivider(parent, height=1).pack(fill="x", pady=(0, PADDING_MD))

        # Wrap status bar in FadeInFrame with staggered delay
        status_wrapper = FadeInFrame(
            parent, fg_color="transparent",
            delay=ANIM_DELAY_600,
            corner_radius=0,
        )
        status_wrapper.pack(fill="x")

        shadow = ctk.CTkFrame(
            status_wrapper, fg_color=COLOR_SHADOW,
            corner_radius=BORDER_RADIUS + 2,
        )
        shadow.pack(fill="x")

        bar = ctk.CTkFrame(
            shadow, fg_color=COLOR_BG_CARD,
            corner_radius=BORDER_RADIUS,
            border_width=1, border_color=COLOR_BORDER_LIGHT,
        )
        bar.pack(fill="x")

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=PADDING_LG, pady=PADDING_MD)

        self._status_label = ctk.CTkLabel(
            inner,
            text="Ready — add files and click Index",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM, anchor="w",
        )
        self._status_label.pack(side="left", fill="x", expand=True)

        self._progress_bar = ctk.CTkProgressBar(
            inner, height=6, width=180,
            fg_color=COLOR_BG_ELEVATED,
            progress_color=COLOR_PURPLE,
            corner_radius=3,
        )
        self._progress_bar.pack(side="left", padx=(0, PADDING))
        self._progress_bar.set(0)

        self._ingest_btn = ctk.CTkButton(
            inner, text="\u25B6  Index Files",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=COLOR_GOLD, hover_color=COLOR_GOLD_LIGHT,
            text_color=COLOR_GOLD_BTN_TEXT,
            height=40, width=150,
            corner_radius=BORDER_RADIUS_SM,
            command=self._start_indexing,
        )
        self._ingest_btn.pack(side="right")

    # ══════════════════════════════════════════════════════════════════
    #  INDEXED FOLDERS
    # ══════════════════════════════════════════════════════════════════

    def _build_indexed_folders(self, parent: ctk.CTkFrame) -> None:
        """Show watched folders with doc counts and Re-index buttons."""
        # GradientDivider
        GradientDivider(parent, height=1).pack(fill="x", pady=(PADDING_LG, PADDING_MD))

        folders_wrapper = FadeInFrame(
            parent, fg_color="transparent",
            delay=ANIM_DELAY_600,
            corner_radius=0,
        )
        folders_wrapper.pack(fill="x")

        header_row = ctk.CTkFrame(folders_wrapper, fg_color="transparent")
        header_row.pack(fill="x", pady=(0, PADDING_SM))

        ctk.CTkLabel(
            header_row, text="Indexed Folders",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM),
            text_color=COLOR_TEXT, anchor="w",
        ).pack(side="left")

        ctk.CTkLabel(
            header_row,
            text="Watched folders with their indexed document counts",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
        ).pack(side="left", padx=(PADDING_SM, 0))

        self._folder_list_frame = ctk.CTkFrame(
            folders_wrapper, fg_color="transparent",
        )
        self._folder_list_frame.pack(fill="x", pady=(0, PADDING))

        self._reindex_progress_label = ctk.CTkLabel(
            folders_wrapper,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
        )
        self._reindex_progress_label.pack(fill="x")

        self._reindex_progress_bar = ctk.CTkProgressBar(
            folders_wrapper, height=4,
            fg_color=COLOR_BG_ELEVATED,
            progress_color=COLOR_GOLD,
            corner_radius=2,
        )
        self._reindex_progress_bar.pack(fill="x")
        self._reindex_progress_bar.set(0)

        # Populate after a short delay so the screen renders first
        self.after(200, self._refresh_indexed_folders)

    def _refresh_indexed_folders(self) -> None:
        """Populate the indexed folders list."""
        for w in self._folder_list_frame.winfo_children():
            w.destroy()

        # Get watched folders
        watcher = None
        try:
            watcher = self._app._watcher
        except AttributeError:
            pass

        folders = []
        if watcher:
            folders = watcher.get_watched_folders()

        if not folders:
            ctk.CTkLabel(
                self._folder_list_frame,
                text="No watched folders configured — add one in Settings",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
            ).pack(pady=(PADDING_SM, 0))
            return

        for info in folders:
            folder_path = info["folder_path"]
            index_name = info["index_name"]
            self._build_folder_row(folder_path, index_name)

    def _get_folder_doc_count(self, folder_path: str) -> int:
        """Count documents in the DB for a given folder path."""
        try:
            conn = self._app.engine._get_db()
            row = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE file_path LIKE ?",
                (folder_path + "%",),
            ).fetchone()
            if row:
                return row[0]
        except Exception:
            pass
        return 0

    def _truncate_path(self, path: str, max_len: int = 50) -> str:
        """Truncate a path for display, keeping the end visible."""
        if len(path) <= max_len:
            return path
        return "\u2026" + path[-(max_len - 1):]

    def _build_folder_row(self, folder_path: str, index_name: str) -> None:
        """Build a single folder row with path, doc count, and Re-index button."""
        doc_count = self._get_folder_doc_count(folder_path)
        display_path = self._truncate_path(folder_path)

        row = ctk.CTkFrame(
            self._folder_list_frame,
            fg_color=COLOR_BG_ELEVATED,
            corner_radius=BORDER_RADIUS_SM,
            height=44,
        )
        row.pack(fill="x", pady=2)
        row.pack_propagate(False)

        # Folder icon + path
        ctk.CTkLabel(
            row, text="\U0001F4C1",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        ).pack(side="left", padx=(PADDING_MD, PADDING_SM))

        ctk.CTkLabel(
            row, text=display_path,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT, anchor="w",
        ).pack(side="left", fill="x", expand=True)

        # Doc count badge
        create_badge(
            row, text=f"{doc_count} docs", color=COLOR_PURPLE_LIGHT,
        ).pack(side="left", padx=(PADDING_SM, PADDING_SM))

        # Remove Watch button (Python 3.14 safe: assign btn before lambda)
        remove_btn = ctk.CTkButton(
            row, text="Remove Watch",
            font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
            fg_color=COLOR_BG_DARKEST,
            hover_color=COLOR_ERROR,
            text_color=COLOR_ERROR,
            height=26, width=90,
            corner_radius=BORDER_RADIUS_SM,
        )
        remove_btn.pack(side="right", padx=(0, PADDING_SM))
        captured_fp = folder_path
        captured_row = row
        captured_remove_btn = remove_btn
        remove_btn.configure(command=lambda: self._remove_watched_folder(captured_fp, captured_row, captured_remove_btn))

        # Re-index button
        btn = ctk.CTkButton(
            row, text="Re-index",
            font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
            fg_color=COLOR_PURPLE_DARK,
            hover_color=COLOR_PURPLE,
            text_color=COLOR_TEXT,
            height=26, width=70,
            corner_radius=BORDER_RADIUS_SM,
            command=lambda fp=folder_path, idx=index_name, b=btn: self._start_folder_reindex(fp, idx, b),
        )
        btn.pack(side="right", padx=(0, PADDING_SM))

    def _remove_watched_folder(self, folder_path: str, row, btn) -> None:
        """Remove a folder from the file watcher and update the UI."""
        try:
            watcher = self._app._watcher
            if not watcher:
                self._app.show_toast("No file watcher running", "error")
                return
            watcher.remove_watch(folder_path)
            row.destroy()
            self._app.show_toast(f"Stopped watching: {self._truncate_path(folder_path, 35)}", "success")
        except Exception as exc:
            captured_exc = exc
            try:
                btn.configure(state="normal")
            except Exception:
                pass
            self._app.show_toast(f"Failed to remove watch: {captured_exc}", "error")

    def _start_folder_reindex(self, folder_path: str, index_name: str, btn) -> None:
        """Kick off a folder re-index in a background thread."""
        btn.configure(state="disabled", text="Re-indexing\u2026")
        self._reindex_progress_bar.set(0)
        self._reindex_progress_label.configure(
            text=f"Re-indexing {self._truncate_path(folder_path, 40)}\u2026",
            text_color=COLOR_GOLD,
        )

        def _progress_cb(processed: int, total: int, filename: str):
            frac = processed / max(total, 1)
            self.after(0, lambda: self._reindex_progress_bar.set(frac))
            short = filename if len(filename) <= 30 else "\u2026" + filename[-29:]
            self.after(0, lambda: self._reindex_progress_label.configure(
                text=f"Re-indexing {processed}/{total}: {short}",
                text_color=COLOR_TEXT_SECONDARY,
            ))

        def _run():
            return self._app.engine.force_reindex_folder(
                folder_path, index_name, on_progress=_progress_cb,
            )

        def _on_done(stats):
            try:
                btn.configure(state="normal", text="Re-index")
                self._reindex_progress_bar.set(1.0)
                self._reindex_progress_label.configure(
                    text=(
                        f"\u2713  Re-indexed {stats['files_processed']} files \u00B7 "
                        f"{stats['total_vectors']} vectors ({stats['elapsed_seconds']:.1f}s)"
                    ),
                    text_color=COLOR_SUCCESS,
                )
                self._refresh_indexed_folders()
                self._app.show_toast(
                    f"Re-indexed {stats['total_vectors']} vectors",
                    "success",
                )
            except Exception:
                pass

        def _on_error(error: str):
            try:
                btn.configure(state="normal", text="Re-index")
                self._reindex_progress_bar.set(0)
                self._reindex_progress_label.configure(
                    text=f"\u2717  Re-index failed: {error}",
                    text_color=COLOR_ERROR,
                )
            except Exception:
                pass

        worker = WorkerThread(
            target=_run,
            on_success=_on_done,
            on_error=_on_error,
            name="FolderReindexWorker",
        )
        worker.set_app_ref(self._app)
        worker.start()

    # ══════════════════════════════════════════════════════════════════
    #  DRAG & DROP
    # ══════════════════════════════════════════════════════════════════

    def _setup_drag_and_drop(self) -> None:
        """Register drag-and-drop handlers on the drop zone.

        Uses tkinterdnd2 when available for native OS DnD.
        Falls back to basic Tk enter/leave visual feedback only.
        """
        if _HAS_TKDND and hasattr(self._drop_border, 'drop_target_register'):
            try:
                self._drop_border.drop_target_register(DND_FILES)
                self._drop_border.dnd_bind('<<DragEnter>>', self._on_drag_enter)
                self._drop_border.dnd_bind('<<DragLeave>>', self._on_drag_leave)
                self._drop_border.dnd_bind('<<Drop>>', self._on_drop)
                return
            except Exception:
                pass

        # Fallback: visual hover feedback even without tkdnd
        # (actual DnD requires tkdnd or OS-level integration)
        self._drop_border.bind('<Enter>', lambda e: self._on_drag_enter(None))
        self._drop_border.bind('<Leave>', lambda e: self._on_drag_leave(None))

    def _on_drag_enter(self, event) -> None:
        """Highlight drop zone when files are dragged over it."""
        try:
            self._drop_border.configure(
                border_color=COLOR_PURPLE,
                fg_color=COLOR_PURPLE_DARK + "15",  # subtle tint
            )
        except Exception:
            pass

    def _on_drag_leave(self, event) -> None:
        """Reset drop zone styling when drag leaves."""
        try:
            self._drop_border.configure(
                border_color=COLOR_BORDER_LIGHT,
                fg_color=COLOR_BG_ELEVATED,
            )
        except Exception:
            pass

    def _on_drop(self, event) -> None:
        """Handle file drop — parse paths from the DnD event data."""
        self._on_drag_leave(None)  # reset styling

        if event is None:
            return

        try:
            # tkinterdnd2 delivers paths as space-separated string;
            # paths with spaces are wrapped in {}
            raw = event.data.strip()
            paths = self._parse_dnd_paths(raw)
            if paths:
                self._selected_files.extend(paths)
                self._render_file_list()
                self._update_file_count()
        except Exception:
            pass

    @staticmethod
    def _parse_dnd_paths(raw: str) -> list[str]:
        """Parse file paths from a tkdnd drop event string.

        Handles both space-separated paths and {}-wrapped paths
        that contain spaces (Windows/Mac convention).
        """
        paths = []
        i = 0
        while i < len(raw):
            if raw[i] == '{':
                # Find matching closing brace
                end = raw.find('}', i)
                if end == -1:
                    end = len(raw)
                paths.append(raw[i + 1:end])
                i = end + 1
            elif raw[i] == ' ':
                i += 1
            else:
                # Regular token until next space
                end = raw.find(' ', i)
                if end == -1:
                    end = len(raw)
                token = raw[i:end]
                if token:  # skip empty tokens
                    paths.append(token)
                i = end + 1
        return [p for p in paths if Path(p).exists()]

    # ══════════════════════════════════════════════════════════════════
    #  FILE MANAGEMENT
    # ══════════════════════════════════════════════════════════════════

    def _browse_files(self) -> None:
        filetypes = [
            ("All Supported", "*.txt *.md *.pdf *.docx *.pptx *.xlsx *.csv *.json *.html *.eml"),
            ("Documents", "*.pdf *.docx *.pptx *.xlsx"),
            ("Text Files", "*.txt *.md *.log"),
            ("Code Files", "*.py *.cpp *.c *.h *.js *.ts"),
            ("Data Files", "*.csv *.json"),
            ("All Files", "*.*"),
        ]
        files = filedialog.askopenfilenames(
            title="Select Files to Index",
            filetypes=filetypes,
        )
        if files:
            self._selected_files.extend(files)
            self._render_file_list()
            self._update_file_count()

    def _render_file_list(self) -> None:
        for widget in self._file_list.winfo_children():
            widget.destroy()

        if not self._selected_files:
            ctk.CTkLabel(
                self._file_list,
                text="No files selected yet",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
            ).pack(pady=(PADDING, 0))
            return

        for i, filepath in enumerate(self._selected_files):
            self._build_file_row(i, filepath)

    def _build_file_row(self, index: int, filepath: str) -> None:
        p = Path(filepath)
        fname = p.name
        ext = p.suffix.lower()
        try:
            fsize = p.stat().st_size if p.exists() else 0
        except OSError:
            fsize = 0

        if fsize >= 1_048_576:
            size_str = f"{fsize / 1_048_576:.1f} MB"
        elif fsize >= 1_024:
            size_str = f"{fsize / 1_024:.1f} KB"
        else:
            size_str = f"{fsize} B"

        ext_color = _EXT_COLORS.get(ext, _DEFAULT_EXT_COLOR)
        ext_label = _EXT_ICONS.get(ext, ext.upper().lstrip(".") if ext else "?")

        row = ctk.CTkFrame(
            self._file_list,
            fg_color=COLOR_BG_ELEVATED,
            corner_radius=BORDER_RADIUS_SM,
            height=52,
        )
        row.pack(fill="x", pady=2)
        row.pack_propagate(False)

        left_border = ctk.CTkFrame(
            row, width=4,
            fg_color=ext_color,
            corner_radius=0,
        )
        left_border.pack(side="left", fill="y")
        left_border.pack_propagate(False)

        # Use create_badge for the file format badge instead of plain label
        create_badge(
            row, text=ext_label, color=ext_color,
        ).pack(side="left", padx=(PADDING_SM, PADDING_SM))

        ctk.CTkLabel(
            row, text=fname,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT, anchor="w",
        ).pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            row, text=size_str,
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
        ).pack(side="left", padx=(PADDING_SM, PADDING_SM))

        ctk.CTkButton(
            row, text="\u2715",
            font=(FONT_FAMILY, 10),
            fg_color="transparent",
            hover_color="#3a1515",
            text_color=COLOR_TEXT_DIM,
            width=28, height=28,
            corner_radius=BORDER_RADIUS_XS,
            command=lambda idx=index: self._remove_file(idx),
        ).pack(side="right", padx=(0, PADDING_SM))

        # OCR warning for image files when Tesseract is not available
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}:
            try:
                from desktop_app.ocr import get_ocr_status
                if not get_ocr_status()["available"]:
                    warn = ctk.CTkLabel(
                        self._file_list,
                        text="\u26A0 OCR required \u2014 install Tesseract to index this file",
                        font=(FONT_FAMILY, FONT_SIZE_XXS),
                        text_color=COLOR_WARNING,
                        anchor="w",
                    )
                    warn.pack(fill="x", padx=(PADDING_MD, 0))
            except Exception:
                pass

    def _remove_file(self, index: int) -> None:
        if 0 <= index < len(self._selected_files):
            self._selected_files.pop(index)
            self._render_file_list()
            self._update_file_count()

    def _update_file_count(self) -> None:
        n = len(self._selected_files)
        text = f"{n} file{'s' if n != 1 else ''} selected"
        try:
            self._file_count_label.configure(text=text)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════
    #  INDEXING PIPELINE
    # ══════════════════════════════════════════════════════════════════

    def _start_indexing(self) -> None:
        if self._is_processing:
            return

        if not self._selected_files:
            try:
                self._status_label.configure(
                    text="\u26A0  No files selected — browse to add files.",
                    text_color=COLOR_WARNING,
                )
            except Exception:
                pass
            return

        self._is_processing = True
        try:
            self._ingest_btn.configure(text="Processing\u2026", state="disabled")
            self._progress_bar.set(0)
        except Exception:
            pass

        # Use IngestionWorker for proper background thread handling
        index_name = self._app.engine.ensure_default_index()

        # Pre-load model check in a quick background step
        def _pre_check():
            if not self._app.engine.ensure_model():
                raise RuntimeError(
                    "Failed to load AI model. Make sure onnxruntime and tokenizers are installed: "
                    "pip install onnxruntime tokenizers"
                )

        def _on_pre_check_done(_result):
            try:
                # Model loaded, now start the actual ingestion worker
                self.after(0, lambda: self._safe_set_status(
                    "Model loaded. Indexing files\u2026",
                    COLOR_PURPLE_LIGHT,
                ))
                self.after(0, lambda: self._progress_bar.set(0.1))

                worker = IngestionWorker(
                    engine=self._app.engine,
                    index_name=index_name,
                    file_paths=self._selected_files[:],
                    on_progress=self._update_progress,
                    on_done=self._on_ingestion_complete,
                    on_error=self._on_ingestion_error,
                )
                worker.set_app_ref(self._app)
                worker.start()
            except Exception as exc:
                logger.error("Failed to start ingestion: %s", exc, exc_info=True)
                self._on_ingestion_error(f"Failed to start indexing: {exc}")

        def _on_pre_check_error(exc):
            captured = str(exc)
            self.after(0, lambda: self._on_ingestion_error(captured))

        # Show loading message
        self.after(0, lambda: self._safe_set_status(
            "Loading AI model (first time may download ~90MB)\u2026",
            COLOR_PURPLE_LIGHT,
        ))
        self.after(0, lambda: self._progress_bar.set(0.05))

        pre_worker = WorkerThread(
            target=_pre_check,
            on_success=_on_pre_check_done,
            on_error=_on_pre_check_error,
            name="PreCheckWorker",
        )
        pre_worker.set_app_ref(self._app)
        pre_worker.start()

    def _update_progress(self, processed: int, total: int, filename: str) -> None:
        frac = processed / max(total, 1)
        try:
            self._progress_bar.set(frac)
        except Exception:
            pass
        short_name = filename if len(filename) <= 40 else "\u2026" + filename[-39:]
        self.after(0, lambda: self._safe_set_status(
            f"Processing {processed}/{total}: {short_name}",
            COLOR_TEXT_SECONDARY,
        ))

    def _safe_set_status(self, text: str, color: str) -> None:
        try:
            self._status_label.configure(text=text, text_color=color)
        except Exception:
            pass

    def _on_ingestion_complete(self, stats) -> None:
        self._is_processing = False
        try:
            self._progress_bar.set(1.0)
            self._ingest_btn.configure(text="\u25B6  Index Files", state="normal")
            self._status_label.configure(
                text=(
                    f"\u2713  Done! {stats.files_processed} files \u00B7 "
                    f"{stats.total_chunks} chunks \u00B7 "
                    f"{stats.total_vectors} vectors  ({stats.elapsed_seconds:.1f}s)"
                ),
                text_color=COLOR_SUCCESS,
            )
        except Exception:
            pass

        self._selected_files.clear()
        self._render_file_list()
        self._update_file_count()
        self._app.show_toast(
            f"Indexed {stats.total_vectors} vectors in {stats.elapsed_seconds:.1f}s",
            "success",
        )

    def _on_ingestion_error(self, error: str) -> None:
        self._is_processing = False
        try:
            self._progress_bar.set(0)
            self._ingest_btn.configure(text="\u25B6  Index Files", state="normal")
            self._status_label.configure(
                text=f"\u2717  Error: {error}",
                text_color=COLOR_ERROR,
            )
        except Exception:
            pass
        self._app.show_toast(f"Indexing failed: {error}", "error")
