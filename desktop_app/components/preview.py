"""
IsoCortex Desktop App — Document Preview Modal
==============================================
A modal overlay that shows document content when the user clicks
on a source citation or search result.
"""

import customtkinter as ctk
import tkinter as tk

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER,
    COLOR_BG_DARKEST,
    COLOR_PURPLE, COLOR_PURPLE_DARK,
    COLOR_GOLD,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    FONT_FAMILY,
    FONT_SIZE_MEDIUM, FONT_SIZE_NORMAL,
    FONT_SIZE_XXS,
    BORDER_RADIUS_SM,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    _dim_hex,
)


class DocumentPreview(ctk.CTkToplevel):
    """
    Modal window that displays document content.

    Usage:
        preview = DocumentPreview(
            parent=root_widget,
            title="report.pdf — Page 3",
            content="Full text of the document chunk...",
            file_path="/path/to/file.pdf",
            page_number=3,
            highlight_text="search query",  # optional: highlights this text
        )
        preview.show()
    """

    def __init__(
        self,
        parent,
        title: str = "Document Preview",
        content: str = "",
        file_path: str = "",
        page_number: int = 0,
        highlight_text: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._parent = parent
        self._title_text = title
        self._content = content
        self._file_path = file_path
        self._page_number = page_number
        self._highlight_text = highlight_text

        # Configure as modal
        self.title("IsoCortex — Document Preview")
        self.geometry("750x550")
        self.minsize(500, 400)
        self.configure(fg_color=COLOR_BG_DARKEST)

        # Center on parent
        self.transient(parent)
        self.grab_set()

        # Build UI
        self._build_ui()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """Build the preview UI."""
        # Top bar
        top = ctk.CTkFrame(self, fg_color=COLOR_BG_CARD, height=52, corner_radius=0)
        top.pack(fill="x", side="top")
        top.pack_propagate(False)

        # Title area
        title_frame = ctk.CTkFrame(top, fg_color="transparent")
        title_frame.pack(side="left", fill="x", expand=True, padx=PADDING_LG)

        ctk.CTkLabel(
            title_frame,
            text=self._title_text,
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(side="left", pady=14)

        # Page badge (if applicable)
        if self._page_number:
            ctk.CTkLabel(
                title_frame,
                text=f"  Page {self._page_number}  ",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                text_color=COLOR_GOLD,
                fg_color=_dim_hex(COLOR_GOLD, 0.15),
                corner_radius=4,
            ).pack(side="left", padx=(PADDING_MD, 0), pady=14)

        # Close button
        close_btn = ctk.CTkButton(
            top,
            text="X",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color="transparent",
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_DIM,
            width=40,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
            command=self._on_close,
        )
        close_btn.pack(side="right", padx=PADDING_MD, pady=6)

        # Action bar
        action_bar = ctk.CTkFrame(self, fg_color=COLOR_BG, height=36, corner_radius=0)
        action_bar.pack(fill="x", side="top")
        action_bar.pack_propagate(False)

        # File path
        if self._file_path:
            ctk.CTkLabel(
                action_bar,
                text=self._file_path,
                font=(FONT_FAMILY, FONT_SIZE_XXS),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
            ).pack(side="left", padx=PADDING_LG, pady=8)

        # Copy button
        copy_btn = ctk.CTkButton(
            action_bar,
            text="Copy Text",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_SECONDARY,
            height=26,
            corner_radius=4,
            width=80,
            command=self._copy_content,
        )
        copy_btn.pack(side="right", padx=(0, PADDING_MD), pady=5)

        # Word count
        word_count = len(self._content.split()) if self._content else 0
        ctk.CTkLabel(
            action_bar,
            text=f"{word_count} words",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
        ).pack(side="right", padx=PADDING_SM, pady=8)

        # Content area (scrollable text widget)
        content_frame = ctk.CTkFrame(self, fg_color=COLOR_BG, corner_radius=0)
        content_frame.pack(fill="both", expand=True)

        # Use tk.Text for selection and copy support
        self._text_widget = tk.Text(
            content_frame,
            wrap="word",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            bd=0,
            highlightthickness=0,
            padx=PADDING_XL,
            pady=PADDING_LG,
            cursor="arrow",
            selectbackground=COLOR_PURPLE_DARK,
            selectforeground="#ffffff",
            relief="flat",
            spacing1=2,
            spacing3=4,
            insertbackground=COLOR_TEXT,
        )

        scrollbar = ctk.CTkScrollbar(
            content_frame,
            command=self._text_widget.yview,
        )
        self._text_widget.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._text_widget.pack(side="left", fill="both", expand=True)

        # Insert content with optional highlighting
        self._insert_content()

    def _insert_content(self):
        """Insert content into the text widget, with optional highlighting."""
        self._text_widget.insert("1.0", self._content)

        if self._highlight_text and len(self._highlight_text) >= 2:
            # Highlight all occurrences of the search term
            start = "1.0"
            search_term = self._highlight_text.lower()
            while True:
                pos = self._text_widget.search(
                    search_term, start, stopindex="end",
                    nocase=True,
                )
                if not pos:
                    break
                end = f"{pos}+{len(self._highlight_text)}c"
                self._text_widget.tag_add("highlight", pos, end)
                start = end

            self._text_widget.tag_configure(
                "highlight",
                background=_dim_hex(COLOR_GOLD, 0.3),
                foreground=COLOR_GOLD,
            )

        self._text_widget.configure(state="disabled")

    def _copy_content(self):
        """Copy all content to clipboard."""
        try:
            self._text_widget.configure(state="normal")
            text = self._text_widget.get("1.0", "end").strip()
            self._text_widget.configure(state="disabled")
            self.clipboard_clear()
            self.clipboard_append(text)
        except Exception:
            pass

    def _on_close(self):
        """Close the preview window."""
        try:
            self.grab_release()
            self.destroy()
        except Exception:
            pass

    def show(self):
        """Show the preview and focus it."""
        self.deiconify()
        self.focus()
        # Center on parent
        self.update_idletasks()
        try:
            pw = self._parent.winfo_width()
            ph = self._parent.winfo_height()
            px = self._parent.winfo_x()
            py = self._parent.winfo_y()
            w = self.winfo_width()
            h = self.winfo_height()
            x = px + (pw - w) // 2
            y = py + (ph - h) // 2
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass