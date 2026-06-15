"""
IsoCortex Desktop App — Unified Search & AI Chat Screen
=========================================================
Single screen that combines semantic search with AI-powered RAG chat.

Layout Vision: "Conversation-Centric"
  - Full-height scrollable chat area (not split-pane)
  - Input bar pinned to the bottom
  - When empty: shows welcome state with suggested prompts
  - User types a query -> it appears as a chat message
  - AI responds with streamed text + expandable source citations
  - Top bar has: title, "New Chat" button, mode indicator
  - Token speed shown inline during generation

Chat persistence:
  - Every conversation auto-saved to SQLite
  - New chat creates a fresh conversation
  - All chats always persisted (ALWAYS SAVE ALL CHATS)
"""

import customtkinter as ctk
import tkinter as tk
import os
import logging
from typing import Any, Callable

logger = logging.getLogger("IsoCortex.search")

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER,
    COLOR_BG_DARKEST,
    COLOR_PURPLE, COLOR_PURPLE_DARK, COLOR_PURPLE_LIGHT, COLOR_PURPLE_DEEP,
    COLOR_GOLD, COLOR_GOLD_LIGHT, COLOR_GOLD_BTN_TEXT,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR, COLOR_INFO,
    COLOR_SHADOW, COLOR_SURFACE_1,
    FONT_FAMILY, FONT_FAMILY_MONO,
    FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_MEDIUM, FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL, FONT_SIZE_XXS,
    BORDER_RADIUS, BORDER_RADIUS_XS, BORDER_RADIUS_SM, BORDER_RADIUS_LG,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    ShimmerBar, GlassCard, GradientDivider,
    _dim_hex,
)
from desktop_app.workers import RAGWorker


# ══════════════════════════════════════════════════════════════════════
#  Suggested Prompts (empty chat state)
# ══════════════════════════════════════════════════════════════════════

SUGGESTED_PROMPTS = [
    ("What documents have I indexed?", "Summarize your indexed files"),
    ("Explain the main topics in my files", "Get an overview of your knowledge base"),
    ("Find specific information about...", "Ask a targeted question"),
    ("Help me understand a concept from my documents", "Learn from your uploaded content"),
]


# ══════════════════════════════════════════════════════════════════════
#  Chat Message Bubble
# ══════════════════════════════════════════════════════════════════════

class ChatBubble(ctk.CTkFrame):
    """
    A single chat message bubble — user or AI.
    AI messages include: copy button, token speed badge, expandable sources.
    """

    def __init__(
        self,
        parent,
        role: str,
        content: str = "",
        sources: list | None = None,
        on_copy: Callable | None = None,
        on_regenerate: Callable | None = None,
        on_preview_source: Callable | None = None,
        on_bookmark_source: Callable | None = None,
        **kwargs,
    ):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._role = role
        self._content = content
        self._sources = sources or []
        self._on_copy = on_copy
        self._on_regenerate = on_regenerate
        self._on_preview_source = on_preview_source
        self._on_bookmark_source = on_bookmark_source
        self._sources_expanded = False
        self._build()

    def _build(self):
        is_user = self._role == "user"

        # Outer wrapper with alignment
        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.pack(fill="x", padx=PADDING_LG, pady=(PADDING_SM, 0))

        if is_user:
            # User messages: right-aligned, purple background
            inner = ctk.CTkFrame(
                wrapper,
                fg_color=COLOR_PURPLE_DARK,
                corner_radius=BORDER_RADIUS,
            )
            inner.pack(anchor="e")

            bubble_pad = ctk.CTkFrame(inner, fg_color="transparent")
            bubble_pad.pack(padx=PADDING_MD, pady=(PADDING_SM, PADDING_SM), fill="both", expand=True)

            self._text_widget = tk.Text(
                bubble_pad,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                fg=COLOR_TEXT,
                bg=COLOR_PURPLE_DARK,
                bd=0,
                highlightthickness=0,
                wrap="word",
                height=1,
                spacing1=0, spacing2=0, spacing3=0,
                padx=0, pady=0,
                cursor="arrow",
            )
            self._text_widget.insert("1.0", self._content)
            self._text_widget.pack(anchor="e", fill="x")
            self._text_widget.update_idletasks()
            self._refresh_text_height()
            self._text_widget.configure(state="disabled")
        else:
            # AI messages: left-aligned, card background
            # Avatar column + content column
            row = ctk.CTkFrame(wrapper, fg_color="transparent")
            row.pack(anchor="w", fill="x")

            # AI avatar (small purple circle)
            avatar_frame = ctk.CTkFrame(row, fg_color="transparent", width=36, height=36)
            avatar_frame.pack(side="left", padx=(0, PADDING_MD), pady=(PADDING_SM, 0))
            avatar_frame.pack_propagate(False)

            avatar = ctk.CTkFrame(
                avatar_frame,
                width=28, height=28,
                fg_color=COLOR_PURPLE,
                corner_radius=14,
            )
            avatar.place(relx=0.5, rely=0.0, anchor="n")

            ctk.CTkLabel(
                avatar, text="AI",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                text_color="#ffffff",
            ).place(relx=0.5, rely=0.5, anchor="center")

            # Content column
            content_col = ctk.CTkFrame(row, fg_color="transparent")
            content_col.pack(side="left", fill="x", expand=True)

            # Bubble card
            bubble = ctk.CTkFrame(
                content_col,
                fg_color=COLOR_BG_CARD,
                corner_radius=BORDER_RADIUS,
                border_width=1,
                border_color=COLOR_BORDER_LIGHT,
            )
            bubble.pack(anchor="w", fill="x")

            bubble_inner = ctk.CTkFrame(bubble, fg_color="transparent")
            bubble_inner.pack(fill="x", padx=PADDING_MD, pady=PADDING_MD)

            # AI response text (use tk.Text for selection/copy support)
            # height=1 initially, then _refresh_text_height() sizes it to fit content.
            # No scrollbar widget added — outer CTkScrollableFrame handles scrolling.
            self._text_widget = tk.Text(
                bubble_inner,
                wrap="word",
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                fg=COLOR_TEXT,
                bg=COLOR_BG_CARD,
                bd=0,
                highlightthickness=0,
                height=1,
                padx=0,
                pady=0,
                cursor="arrow",
                selectbackground=COLOR_PURPLE_DARK,
                selectforeground="#ffffff",
                relief="flat",
                spacing1=0,
                spacing3=0,
            )
            self._text_widget.pack(fill="x")
            if self._content:
                self._text_widget.insert("1.0", self._content)
            self._text_widget.update_idletasks()
            self._refresh_text_height()
            self._text_widget.configure(state="disabled")

            # Bottom action bar (copy, regenerate, token speed)
            self._action_bar = ctk.CTkFrame(content_col, fg_color="transparent", height=24)
            self._action_bar.pack(anchor="w", pady=(2, 0))
            self._action_bar.pack_propagate(False)

            # Export button
            if self._on_copy:
                export_btn = ctk.CTkButton(
                    self._action_bar,
                    text="Export",
                    font=(FONT_FAMILY, FONT_SIZE_XXS),
                    fg_color="transparent",
                    hover_color=COLOR_BG_HOVER,
                    text_color=COLOR_TEXT_DIM,
                    height=22,
                    corner_radius=BORDER_RADIUS_XS,
                    width=55,
                    command=self._handle_export,
                )
                export_btn.pack(side="left", padx=(0, PADDING_SM))

            # Copy button
            if self._on_copy:
                copy_btn = ctk.CTkButton(
                    self._action_bar,
                    text="Copy",
                    font=(FONT_FAMILY, FONT_SIZE_XXS),
                    fg_color="transparent",
                    hover_color=COLOR_BG_HOVER,
                    text_color=COLOR_TEXT_DIM,
                    height=22,
                    corner_radius=BORDER_RADIUS_XS,
                    width=50,
                    command=self._handle_copy,
                )
                copy_btn.pack(side="left", padx=(0, PADDING_SM))

            # Regenerate button
            if self._on_regenerate:
                regen_btn = ctk.CTkButton(
                    self._action_bar,
                    text="Regenerate",
                    font=(FONT_FAMILY, FONT_SIZE_XXS),
                    fg_color="transparent",
                    hover_color=COLOR_BG_HOVER,
                    text_color=COLOR_TEXT_DIM,
                    height=22,
                    corner_radius=BORDER_RADIUS_XS,
                    width=75,
                    command=self._on_regenerate,
                )
                regen_btn.pack(side="left", padx=(0, PADDING_SM))

            # Token speed label (hidden until generation)
            self._speed_label = ctk.CTkLabel(
                self._action_bar,
                text="",
                font=(FONT_FAMILY, FONT_SIZE_XXS),
                text_color=COLOR_TEXT_DIM,
            )
            self._speed_label.pack(side="left")

            # Sources section (expandable)
            if self._sources:
                self._build_sources_section(content_col)

    def inject_sources(self, sources: list[dict]):
        """Add source citations to this bubble after it was initially created.
        Called when sources become available after streaming completes."""
        if not sources:
            return
        self._sources = sources

        # Find the content column (parent of the bubble card and action bar)
        # The structure is: self > wrapper > row > content_col > [bubble, action_bar]
        # We need to add sources after the action bar
        if self._role != "assistant":
            return

        # Find the action bar's parent (content_col)
        try:
            content_col = self._action_bar.master
        except Exception:
            return

        source_count = len(sources)

        # Toggle button
        self._sources_toggle = ctk.CTkButton(
            content_col,
            text=f"  {source_count} source{'s' if source_count != 1 else ''} cited  +  ",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_PURPLE_LIGHT,
            height=24,
            corner_radius=BORDER_RADIUS_XS,
            command=self._toggle_sources,
        )
        self._sources_toggle.pack(anchor="w", pady=(PADDING_SM, 0))

        # Sources container (hidden by default)
        self._sources_container = ctk.CTkFrame(content_col, fg_color="transparent")

        for src in sources:
            src_card = ctk.CTkFrame(
                self._sources_container,
                fg_color=COLOR_BG_ELEVATED,
                corner_radius=BORDER_RADIUS_SM,
                border_width=1,
                border_color=COLOR_BORDER,
            )
            src_card.pack(fill="x", pady=2)

            src_inner = ctk.CTkFrame(src_card, fg_color="transparent")
            src_inner.pack(fill="x", padx=PADDING_MD, pady=(PADDING_SM, PADDING_SM))

            # Source number badge
            ctk.CTkLabel(
                src_inner,
                text=f" #{src.get('index', '?')} ",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                text_color=COLOR_GOLD,
                fg_color=_dim_hex(COLOR_GOLD, 0.15),
                corner_radius=BORDER_RADIUS_XS,
            ).pack(side="left", padx=(0, PADDING_SM))

            # File name (clickable to open preview)
            file_text = src.get("file", "unknown")
            page = src.get("page", 0)
            label_text = file_text
            if page:
                label_text += f"  |  page {page}"

            file_btn = ctk.CTkButton(
                src_inner,
                text=label_text,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                fg_color="transparent",
                hover_color=COLOR_BG_HOVER,
                text_color=COLOR_TEXT_SECONDARY,
                height=22,
                corner_radius=BORDER_RADIUS_XS,
                anchor="w",
                command=lambda s=src: self._on_preview_source(s) if self._on_preview_source else None,
            )
            file_btn.pack(side="left", fill="x", expand=True)

            # Relevance score + match type
            score = src.get("score", 0)
            kw_score = src.get("keyword_score", 0)

            # Determine match type
            if kw_score > 0.1 and score > 0.3:
                match_label = "hybrid"
                match_color = COLOR_GOLD
            elif kw_score > 0.1:
                match_label = "keyword"
                match_color = COLOR_WARNING
            else:
                match_label = "semantic"
                match_color = COLOR_PURPLE_LIGHT

            score_frame = ctk.CTkFrame(src_inner, fg_color="transparent")
            score_frame.pack(side="right")

            # Bookmark button
            bookmark_btn = ctk.CTkButton(
                src_inner,
                text="B",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                fg_color="transparent",
                hover_color=_dim_hex(COLOR_GOLD, 0.15),
                text_color=COLOR_TEXT_DIM,
                width=20, height=20, corner_radius=BORDER_RADIUS_XS,
                command=lambda s=src: self._on_bookmark_source(s) if self._on_bookmark_source else None,
            )
            bookmark_btn.pack(side="right", padx=(PADDING_SM, 0))

            ctk.CTkLabel(
                score_frame,
                text=match_label,
                font=(FONT_FAMILY, FONT_SIZE_XXS),
                text_color=match_color,
            ).pack(side="right", padx=(PADDING_SM, 0))

            ctk.CTkLabel(
                score_frame,
                text=f"{score:.0%}",
                font=(FONT_FAMILY, FONT_SIZE_XXS),
                text_color=COLOR_SUCCESS if score > 0.5 else COLOR_WARNING,
            ).pack(side="right")

    def _toggle_sources(self):
        """Expand or collapse the sources section."""
        if self._sources_expanded:
            self._sources_container.pack_forget()
            self._sources_toggle.configure(
                text=f"  {len(self._sources)} source{'s' if len(self._sources) != 1 else ''} cited  +  "
            )
        else:
            self._sources_container.pack(anchor="w", fill="x", pady=(2, 0))
            self._sources_toggle.configure(
                text=f"  {len(self._sources)} source{'s' if len(self._sources) != 1 else ''} cited  -  "
            )
        self._sources_expanded = not self._sources_expanded

    def _handle_copy(self):
        """Copy the AI response text to clipboard."""
        try:
            # Get text from the widget
            if isinstance(self._text_widget, tk.Text):
                text = self._text_widget.get("1.0", "end").strip()
            else:
                text = self._text_widget.cget("text")

            self._text_widget.master.clipboard_clear()
            self._text_widget.master.clipboard_append(text)
        except Exception:
            pass

    def _handle_export(self):
        """Export this AI response as a Markdown file."""
        try:
            from desktop_app.components.exporter import export_single_answer_markdown

            # Get the text content
            if isinstance(self._text_widget, tk.Text):
                text = self._text_widget.get("1.0", "end").strip()
            else:
                text = self._text_widget.cget("text")

            if not text:
                return

            # Find the parent query from context
            query = ""
            if hasattr(self.master, 'master'):
                screen = self.master.master
                if hasattr(screen, '_messages'):
                    msgs = screen._messages
                    # Find the last user message before this assistant message
                    for m in reversed(msgs):
                        if m["role"] == "user":
                            query = m["content"]
                            break

            path = export_single_answer_markdown(
                query=query,
                answer=text,
                sources=self._sources,
            )

            # Show success via toast
            if hasattr(self.master, 'master'):
                screen = self.master.master
                if hasattr(screen, '_app') and hasattr(screen._app, 'show_toast'):
                    from pathlib import Path
                    fname = Path(path).name
                    screen._app.show_toast(f"Exported as {fname}", "success")
        except Exception:
            pass

    def _refresh_text_height(self):
        """Recalculate tk.Text height to exactly fit its content.

        Uses update_idletasks() to ensure the widget has its real width
        (needed for word-wrap calculation), then counts display lines
        via the Tk count() API with a newline-count fallback.
        Must be called while the widget is in state='normal'.
        """
        if not isinstance(self._text_widget, tk.Text):
            return
        try:
            # Force layout so the widget knows its real width for wrap
            self._text_widget.update_idletasks()
            display_lines = 1
            # Primary: Tk count() API — most reliable for display lines
            try:
                display_lines = int(
                    self._text_widget.count("1.0", "end", "displaylines")
                )
            except Exception:
                # Fallback: parse index, then safety-check with newline count
                try:
                    idx = self._text_widget.index("end-1c")
                    display_lines = int(idx.split(".")[0])
                except Exception:
                    pass
            # Safety net: if widget still thinks 1 line but content has newlines,
            # count newlines manually — guarantees multi-line content is visible
            if display_lines <= 1 and "\n" in self._content:
                display_lines = self._content.count("\n") + 1
            self._text_widget.configure(height=max(display_lines, 1))
        except Exception:
            pass

    def append_token(self, token_text: str):
        """Append a streamed token to the AI response text widget."""
        self._content += token_text
        if isinstance(self._text_widget, tk.Text):
            self._text_widget.configure(state="normal")
            self._text_widget.insert("end", token_text)
            self._refresh_text_height()
            self._text_widget.configure(state="disabled")

    def update_speed(self, token_count: int, elapsed: float):
        """Update the token speed indicator."""
        if elapsed > 0:
            tps = token_count / elapsed
            try:
                self._speed_label.configure(text=f"{tps:.1f} tok/s  |  {token_count} tokens")
            except Exception:
                pass

    def finalize(self, token_count: int, elapsed: float, tps: float):
        """Called when generation is complete — update speed label and final height."""
        if isinstance(self._text_widget, tk.Text):
            self._text_widget.configure(state="normal")
            self._refresh_text_height()
            self._text_widget.configure(state="disabled")
        try:
            self._speed_label.configure(
                text=f"{tps:.1f} tok/s  |  {token_count} tokens  |  {elapsed:.1f}s"
            )
        except Exception:
            pass

    def set_content(self, text: str):
        """Set the full content (for regeneration)."""
        self._content = text
        if isinstance(self._text_widget, tk.Text):
            self._text_widget.configure(state="normal")
            self._text_widget.delete("1.0", "end")
            self._text_widget.insert("1.0", text)
            self._refresh_text_height()
            self._text_widget.configure(state="disabled")
        else:
            self._text_widget.configure(text=text)


# ══════════════════════════════════════════════════════════════════════
#  Main Search & Chat Screen
# ══════════════════════════════════════════════════════════════════════

class SearchScreen(ctk.CTkFrame):
    """
    Unified Search & AI Chat screen.

    Vision: A single conversation-centric view where the user types queries
    at the bottom and receives AI answers (with RAG context from their
    indexed documents) in a scrollable chat area above.
    """

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app = app
        self._engine = app.engine

        # Mode: "chat" or "search"
        self._mode = "chat"

        # Chat state
        self._conversation_id: str = ""
        self._messages: list[dict] = []  # In-memory: [{"role": ..., "content": ...}]
        self._bubbles: list[ChatBubble] = []
        self._current_ai_bubble: ChatBubble | None = None
        self._rag_worker: RAGWorker | None = None
        self._is_generating = False
        self._was_stopped = False
        self._last_sources: list[dict] = []
        self._llm: Any = None  # Lazy-loaded LLM instance
        self._llm_loaded = False

        self._build_ui()

        # Create a new conversation on screen init
        try:
            self._new_chat()
        except Exception:
            pass  # Chat will work once DB is available

    def destroy(self):
        if hasattr(self, '_history_panel') and self._history_panel:
            try:
                self._history_panel.destroy()
            except Exception:
                pass
            self._history_panel = None
        super().destroy()

    # ══════════════════════════════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        """Build the full screen layout."""
        # Full-screen container
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True)

        # ── Top bar ──────────────────────────────────────────────────
        self._build_top_bar(content)

        # Shimmer accent
        ShimmerBar(content, height=2).pack(fill="x")

        # ── Chat area (scrollable) ──────────────────────────────────
        self._build_chat_area(content)

        # ── Input bar (pinned to bottom) ────────────────────────────
        self._build_input_bar(content)

    def _build_top_bar(self, parent):
        """Build the top bar with title, new chat button, status."""
        top = ctk.CTkFrame(parent, fg_color="transparent", height=44)
        top.pack(fill="x", padx=PADDING_LG, pady=(PADDING_MD, PADDING_SM))
        top.pack_propagate(False)

        # Left: title + subtitle
        left = ctk.CTkFrame(top, fg_color="transparent")
        left.pack(side="left")

        # Purple accent bar
        accent = ctk.CTkFrame(left, width=4, corner_radius=2, fg_color=COLOR_PURPLE)
        accent.pack(side="left", padx=(0, PADDING_MD), pady=10)
        accent.pack_propagate(False)

        self._title_label = ctk.CTkLabel(
            left,
            text="AI Chat",
            font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        )
        self._title_label.pack(side="left", pady=8)

        # AI status badge (compact, next to title — replaces separate subtitle)
        self._ai_status = ctk.CTkLabel(
            left,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
        )
        self._ai_status.pack(side="left", padx=PADDING_MD, pady=12)

        # ── Mode toggle: Search | AI Chat ────────────────────────────
        mode_frame = ctk.CTkFrame(
            top,
            fg_color=COLOR_BG_ELEVATED,
            corner_radius=BORDER_RADIUS_SM,
            height=32,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        mode_frame.pack(side="right", padx=(0, PADDING_SM), pady=6)
        mode_frame.pack_propagate(False)

        self._search_mode_btn = ctk.CTkButton(
            mode_frame,
            text="Search",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color="transparent",
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_DIM,
            height=30,
            corner_radius=BORDER_RADIUS_XS,
            width=70,
            command=lambda: self._set_mode("search"),
        )
        self._search_mode_btn.pack(side="left", padx=(2, 0), pady=1)

        self._chat_mode_btn = ctk.CTkButton(
            mode_frame,
            text="AI Chat",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_DARK,
            text_color="#ffffff",
            height=30,
            corner_radius=BORDER_RADIUS_XS,
            width=70,
            command=lambda: self._set_mode("chat"),
        )
        self._chat_mode_btn.pack(side="left", padx=(0, 2), pady=1)

        # History button
        self._history_btn = ctk.CTkButton(
            top,
            text="H  History",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_SECONDARY,
            height=32,
            corner_radius=BORDER_RADIUS_SM,
            border_width=1,
            border_color=COLOR_BORDER,
            command=self._toggle_history_panel,
        )
        self._history_btn.pack(side="right", padx=(0, PADDING_SM), pady=6)

        # New Chat button
        self._new_chat_btn = ctk.CTkButton(
            top,
            text="+  New",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_PURPLE_LIGHT,
            height=32,
            corner_radius=BORDER_RADIUS_SM,
            border_width=1,
            border_color=COLOR_BORDER,
            command=self._new_chat,
        )
        self._new_chat_btn.pack(side="right", padx=(0, PADDING_SM), pady=6)

    def _build_chat_area(self, parent):
        """Build the scrollable chat message area."""
        # Scrollable frame for chat bubbles
        self._chat_scroll = ctk.CTkScrollableFrame(
            parent,
            fg_color="transparent",
        )
        self._chat_scroll.pack(fill="both", expand=True, padx=0, pady=0)

        # Show welcome state initially
        self._show_welcome()

    def _build_input_bar(self, parent):
        """Build the input bar pinned to the bottom."""
        # Divider line above input
        GradientDivider(parent, height=1).pack(fill="x", padx=PADDING_LG)

        input_frame = ctk.CTkFrame(parent, fg_color="transparent")
        input_frame.pack(fill="x", padx=PADDING_LG, pady=PADDING_MD)

        # GlassCard wrapper
        glass = GlassCard(input_frame, corner_radius=BORDER_RADIUS_LG)
        glass.pack(fill="x")

        bar = ctk.CTkFrame(glass, fg_color="transparent")
        bar.pack(fill="x", padx=PADDING_MD, pady=PADDING_MD)

        # Entry — NO placeholder_text (CTkEntry renders it as real selectable
        # text inside the tk Entry).  We use a separate overlay label instead.
        self._input_entry = ctk.CTkEntry(
            bar,
            placeholder_text="",  # intentionally empty
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=COLOR_BG_ELEVATED,
            border_color=COLOR_BORDER,
            text_color=COLOR_TEXT,
            height=42,
            corner_radius=BORDER_RADIUS_SM,
        )
        self._input_entry.pack(side="left", fill="x", expand=True, padx=(0, PADDING_SM))
        self._input_entry.bind("<Return>", self._on_enter)

        # ── Placeholder label overlay ─────────────────────────────
        # A real label sitting on top of the entry; never selectable,
        # never the entry's content.  Hidden as soon as the user focuses.
        self._placeholder_label = ctk.CTkLabel(
            bar,
            text="Ask anything about your documents...",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        # Place overlay on top of the entry area
        self._placeholder_label.place(
            relx=0.0, rely=0.5, x=14, anchor="w",  # 14px ≈ entry text left-pad
        )
        self._input_entry.bind("<FocusIn>", self._hide_placeholder)
        self._input_entry.bind("<FocusOut>", self._show_placeholder)
        self._input_entry.bind("<Key>", self._hide_placeholder_on_key)

        # Send button
        self._send_btn = ctk.CTkButton(
            bar,
            text="Send",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=COLOR_GOLD,
            hover_color=COLOR_GOLD_LIGHT,
            text_color=COLOR_GOLD_BTN_TEXT,
            height=42,
            corner_radius=BORDER_RADIUS_SM,
            width=80,
            command=self._handle_send,
        )
        self._send_btn.pack(side="right")

        # Stop button (hidden by default)
        self._stop_btn = ctk.CTkButton(
            bar,
            text="Stop",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=COLOR_ERROR,
            hover_color="#dc2626",
            text_color="#ffffff",
            height=42,
            corner_radius=BORDER_RADIUS_SM,
            width=80,
            command=self._handle_stop,
        )
        # Not packed — shown only during generation

    # ══════════════════════════════════════════════════════════════════
    #  Welcome / Empty State
    # ══════════════════════════════════════════════════════════════════

    def _clear_chat_area(self):
        """Remove all widgets from the chat scroll area."""
        for widget in self._chat_scroll.winfo_children():
            widget.destroy()

    def _show_welcome(self):
        """Show the welcome state with suggested prompts."""
        self._clear_chat_area()

        welcome_frame = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
        welcome_frame.pack(expand=True, fill="both")

        # Centered content
        center = ctk.CTkFrame(welcome_frame, fg_color="transparent")
        center.place(relx=0.5, rely=0.4, anchor="center")

        # Logo glow
        glow_bg = ctk.CTkFrame(center, fg_color=COLOR_PURPLE_DEEP, width=80, height=80, corner_radius=40)
        glow_bg.pack(pady=(0, PADDING_MD))
        glow_bg.pack_propagate(False)

        ctk.CTkLabel(
            glow_bg, text="AI",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_PURPLE_LIGHT,
        ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            center,
            text="IsoCortex AI",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_TEXT,
        ).pack()

        ctk.CTkLabel(
            center,
            text="Ask questions about your indexed documents.\n"
                 "AI will find relevant passages and answer with citations.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            justify="center",
        ).pack(pady=(PADDING_SM, PADDING_LG))

        # Suggested prompt chips
        chips_frame = ctk.CTkFrame(center, fg_color="transparent")
        chips_frame.pack()

        for i, (prompt_text, description) in enumerate(SUGGESTED_PROMPTS):
            chip = ctk.CTkButton(
                chips_frame,
                text=prompt_text,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                fg_color=COLOR_BG_ELEVATED,
                hover_color=COLOR_BG_HOVER,
                text_color=COLOR_TEXT_SECONDARY,
                height=36,
                corner_radius=BORDER_RADIUS_SM,
                border_width=1,
                border_color=COLOR_BORDER,
                width=260,
                anchor="w",
                command=lambda p=prompt_text: self._fill_and_send(p),
            )
            chip.pack(pady=2, fill="x")

    def _fill_and_send(self, text: str):
        """Fill the input with text and send."""
        self._input_entry.delete(0, "end")
        self._input_entry.insert(0, text)
        self._handle_send()

    # ══════════════════════════════════════════════════════════════════
    #  Chat Management
    # ══════════════════════════════════════════════════════════════════

    def _new_chat(self):
        """Start a new conversation."""
        # Stop any ongoing generation
        self._handle_stop()

        self._conversation_id = self._engine.create_conversation()
        self._messages = []
        self._bubbles = []
        self._last_sources = []
        self._current_ai_bubble = None
        self._was_stopped = False
        self._rag_worker = None
        self._is_generating = False

        # Reset UI
        try:
            self._title_label.configure(text="AI Chat")
        except Exception:
            pass

        # Show welcome if chat area exists
        try:
            self._show_welcome()
        except Exception:
            pass

        # Re-enable input and show send button
        try:
            self._swap_send_stop(show_stop=False)
        except Exception:
            pass

        # Focus input
        try:
            self._input_entry.focus_set()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════
    #  LLM Loading
    # ══════════════════════════════════════════════════════════════════

    def _ensure_llm(self) -> bool:
        """Check if LLM is ready. Returns True if ready, False otherwise.
        On first call, kicks off a background load and returns False (caller should not proceed).
        On subsequent calls after load completes, returns True."""
        if self._llm_loaded and self._llm is not None:
            return self._llm.is_loaded

        # Check if model file exists before attempting load
        from desktop_app.llm import LLM, model_exists

        if not model_exists():
            try:
                self._ai_status.configure(text="Model not found", text_color=COLOR_ERROR)
            except Exception:
                pass
            self._llm_loaded = True  # Don't retry
            return False

        # Show loading status
        try:
            self._ai_status.configure(text="Loading AI model...", text_color=COLOR_GOLD)
            self._swap_send_stop(show_stop=True)
        except Exception:
            pass

        # Load in background — GUI stays responsive
        self._llm = LLM()
        # Note: load_error is a read-only property; load_model() resets it internally
        self._pending_query = None  # Will be set if there's a queued query

        def _load_in_background():
            success = self._llm.load_model()
            self._llm_loaded = True
            if self._app:
                self._app.after(0, lambda: self._on_llm_loaded(success))

        import threading
        t = threading.Thread(target=_load_in_background, daemon=True)
        t.start()

        return False  # Not ready yet — _on_llm_loaded will handle continuation

    def _on_llm_loaded(self, success: bool):
        """Callback when LLM finishes loading in background."""
        try:
            self._swap_send_stop(show_stop=False)
        except Exception:
            pass

        if success:
            try:
                self._ai_status.configure(text="AI Ready", text_color=COLOR_SUCCESS)
            except Exception:
                pass
            # If there was a pending query, auto-send it now
            if hasattr(self, '_pending_query') and self._pending_query:
                query = self._pending_query
                self._pending_query = None
                self._do_generate(query)
        else:
            err_msg = self._llm.load_error or "Failed to load model"
            logger.error("LLM load failed: %s", err_msg)
            try:
                self._ai_status.configure(text="AI Error", text_color=COLOR_ERROR)
            except Exception:
                pass
            # Clean up the empty AI bubble if we added one
            self._cleanup_empty_bubble()
            self._pending_query = None

    # ══════════════════════════════════════════════════════════════════
    #  Mode Toggle
    # ══════════════════════════════════════════════════════════════════

    def _set_mode(self, mode: str):
        """Switch between 'search' and 'chat' modes."""
        if self._is_generating:
            return
        self._mode = mode

        if mode == "search":
            self._search_mode_btn.configure(
                fg_color=COLOR_PURPLE, hover_color=COLOR_PURPLE_DARK,
                text_color="#ffffff", font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            )
            self._chat_mode_btn.configure(
                fg_color="transparent", hover_color=COLOR_BG_HOVER,
                text_color=COLOR_TEXT_DIM, font=(FONT_FAMILY, FONT_SIZE_SMALL),
            )
            self._title_label.configure(text="Search")
            self._set_placeholder_text("Search your indexed documents...")
            self._send_btn.configure(text="Search")
            self._history_btn.pack_forget()
            self._new_chat_btn.pack_forget()
            self._show_search_welcome()
        else:
            self._chat_mode_btn.configure(
                fg_color=COLOR_PURPLE, hover_color=COLOR_PURPLE_DARK,
                text_color="#ffffff", font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            )
            self._search_mode_btn.configure(
                fg_color="transparent", hover_color=COLOR_BG_HOVER,
                text_color=COLOR_TEXT_DIM, font=(FONT_FAMILY, FONT_SIZE_SMALL),
            )
            self._title_label.configure(text="AI Chat")
            self._set_placeholder_text("Ask anything about your documents...")
            self._send_btn.configure(text="Send")
            self._new_chat_btn.pack(side="right", padx=(0, PADDING_SM), pady=6)
            self._history_btn.pack(side="right", padx=(0, PADDING_SM), pady=6)
            self._show_welcome()

    def _show_search_welcome(self):
        """Show the search mode welcome state."""
        self._clear_chat_area()

        welcome = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
        welcome.pack(expand=True, fill="both")

        center = ctk.CTkFrame(welcome, fg_color="transparent")
        center.place(relx=0.5, rely=0.4, anchor="center")

        glow_bg = ctk.CTkFrame(center, fg_color=COLOR_BG_ELEVATED, width=80, height=80, corner_radius=40,
                                border_width=1, border_color=COLOR_BORDER)
        glow_bg.pack(pady=(0, PADDING_MD))
        glow_bg.pack_propagate(False)

        ctk.CTkLabel(
            glow_bg, text="S",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_GOLD,
        ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            center,
            text="Semantic Search",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_TEXT,
        ).pack()

        ctk.CTkLabel(
            center,
            text="Find relevant passages from your indexed documents.\n"
                 "No AI generation — pure similarity-based results.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            justify="center",
        ).pack(pady=(PADDING_SM, 0))

    def _handle_pure_search(self, query: str):
        """Run a pure semantic search without AI — show results as cards."""
        self._clear_chat_area_if_welcome()

        # Query label
        q_label = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
        q_label.pack(fill="x", padx=PADDING_LG, pady=(PADDING_SM, PADDING_MD))
        ctk.CTkLabel(
            q_label,
            text="Results for:  ",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            q_label,
            text=query,
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(side="left")

        # Loading indicator
        loading_frame = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
        loading_frame.pack(fill="x", padx=PADDING_LG, pady=PADDING_MD)
        loading_label = ctk.CTkLabel(
            loading_frame,
            text="Searching documents...",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        loading_label.pack(fill="x")

        # Store query for the callback
        self._pending_search_query = query
        self._pending_search_loading = loading_frame

        # Run search in background thread to avoid freezing GUI
        import threading
        def _bg_search():
            try:
                results = self._engine.search_all_indexes(query, top_k=10)
                if self._app:
                    self._app.after(0, lambda: self._on_pure_search_done(results))
            except Exception as exc:
                captured_exc = exc  # Python 3.14: capture before lambda
                if self._app:
                    self._app.after(0, lambda: self._on_pure_search_error(str(captured_exc)))

        t = threading.Thread(target=_bg_search, daemon=True, name="PureSearch")
        t.start()

    def _on_pure_search_done(self, results):
        """Callback when pure search completes in background."""
        # Remove loading indicator
        try:
            self._pending_search_loading.destroy()
        except Exception:
            pass

        query = getattr(self, '_pending_search_query', '')

        if not results:
            empty = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
            empty.pack(fill="x", padx=PADDING_LG, pady=PADDING_LG)
            ctk.CTkLabel(
                empty,
                text="No results found. Try different keywords or upload more documents.",
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                text_color=COLOR_TEXT_DIM,
                justify="center",
            ).pack()
            self._scroll_to_bottom()
            return

        count_label = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
        count_label.pack(fill="x", padx=PADDING_LG, pady=(0, PADDING_SM))
        ctk.CTkLabel(
            count_label,
            text=f"{len(results)} result{'s' if len(results) != 1 else ''} found",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(side="left")

        for i, result in enumerate(results):
            score = getattr(result, "score", 0) or 0
            file_name = getattr(result, "source_file", "unknown") or "unknown"
            # Use just the filename, not full path
            if "/" in file_name or "\\" in file_name:
                file_name = os.path.basename(file_name)
            chunk_text = getattr(result, "text", "") or ""
            page = getattr(result, "page_number", 0) or 0

            card = ctk.CTkFrame(
                self._chat_scroll,
                fg_color=COLOR_BG_CARD,
                corner_radius=BORDER_RADIUS_SM,
                border_width=1,
                border_color=COLOR_BORDER,
            )
            card.pack(fill="x", padx=PADDING_LG, pady=2)

            # Header row: rank + filename + score
            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=PADDING_MD, pady=(PADDING_SM, PADDING_SM))

            ctk.CTkLabel(
                header,
                text=f" #{i + 1} ",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                text_color=COLOR_GOLD,
                fg_color=_dim_hex(COLOR_GOLD, 0.15),
                corner_radius=BORDER_RADIUS_XS,
            ).pack(side="left", padx=(0, PADDING_SM))

            name_text = file_name
            if page:
                name_text += f"  ·  page {page}"

            ctk.CTkLabel(
                header,
                text=name_text,
                font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
                text_color=COLOR_TEXT,
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

            # Score badge
            score_pct = f"{score:.0%}" if score < 1 else "100%"
            ctk.CTkLabel(
                header,
                text=f"  {score_pct}  ",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                text_color=COLOR_SUCCESS if score > 0.5 else COLOR_WARNING,
                fg_color=_dim_hex(COLOR_SUCCESS if score > 0.5 else COLOR_WARNING, 0.1),
                corner_radius=BORDER_RADIUS_XS,
            ).pack(side="right")

            # Chunk text
            # Truncate long chunks for display
            display_text = chunk_text[:500] + ("..." if len(chunk_text) > 500 else "")
            text_label = ctk.CTkLabel(
                card,
                text=display_text,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_SECONDARY,
                anchor="w",
                justify="left",
                wraplength=700,
            )
            text_label.pack(fill="x", padx=PADDING_MD, pady=(0, PADDING_SM))

        self._scroll_to_bottom()

    def _on_pure_search_error(self, error_msg: str):
        """Callback when pure search fails in background."""
        try:
            self._pending_search_loading.destroy()
        except Exception:
            pass
        err_frame = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
        err_frame.pack(fill="x", padx=PADDING_LG, pady=PADDING_MD)
        ctk.CTkLabel(
            err_frame,
            text=f"Search failed: {error_msg}",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_ERROR,
            anchor="w",
        ).pack(fill="x")
        self._scroll_to_bottom()

    # ══════════════════════════════════════════════════════════════════
    #  Send / Generate
    # ══════════════════════════════════════════════════════════════════

    def _hide_placeholder(self, event=None):
        """Hide the placeholder label overlay."""
        try:
            self._placeholder_label.place_forget()
        except Exception:
            pass

    def _show_placeholder(self, event=None):
        """Show the placeholder label overlay if the entry is empty."""
        try:
            if not self._input_entry.get():
                self._placeholder_label.place(
                    relx=0.0, rely=0.5, x=14, anchor="w",
                )
        except Exception:
            pass

    def _hide_placeholder_on_key(self, event=None):
        """Hide placeholder on any keystroke."""
        self._hide_placeholder()

    def _set_placeholder_text(self, text: str):
        """Update the placeholder label text (used when switching modes)."""
        try:
            self._placeholder_label.configure(text=text)
            # Re-evaluate visibility
            if not self._input_entry.get():
                self._placeholder_label.place(
                    relx=0.0, rely=0.5, x=14, anchor="w",
                )
        except Exception:
            pass

    def _on_enter(self, event):
        """Handle Enter key in input entry."""
        self._handle_send()

    def _handle_send(self):
        """Handle the user pressing Send."""
        if self._is_generating:
            return

        query = self._input_entry.get().strip()

        # Guard: reject empty or too-short text
        if not query or len(query) < 2:
            return

        # Clear input and re-show placeholder
        self._input_entry.delete(0, "end")
        self._show_placeholder()

        # Route to pure search mode if active
        if self._mode == "search":
            self._handle_pure_search(query)
            return

        # ── AI Chat mode below ─────────────────────────────────────

        # Add user message to UI
        self._add_user_bubble(query)

        # Add to in-memory messages
        self._messages.append({"role": "user", "content": query})

        # Save to DB
        self._engine.save_message(self._conversation_id, "user", query)

        # Update conversation title from first message
        if len(self._messages) == 1:
            title = query[:60] + ("..." if len(query) > 60 else "")
            self._engine.update_conversation_title(self._conversation_id, title)
            try:
                self._title_label.configure(text=title)
            except Exception:
                pass

        # Start RAG generation
        self._start_generation(query)

    def _add_user_bubble(self, text: str):
        """Add a user message bubble to the chat."""
        self._clear_chat_area_if_welcome()

        bubble = ChatBubble(
            self._chat_scroll,
            role="user",
            content=text,
        )
        bubble.pack(fill="x", pady=(PADDING_SM, 0))
        self._bubbles.append(bubble)

        # Scroll to bottom
        self._scroll_to_bottom()

    def _add_ai_bubble(self) -> ChatBubble:
        """Add an empty AI message bubble (will be filled by streaming)."""
        bubble = ChatBubble(
            self._chat_scroll,
            role="assistant",
            content="",
            sources=None,
            on_copy=lambda: None,
            on_regenerate=self._handle_regenerate,
            on_preview_source=self._open_source_preview,
            on_bookmark_source=self._bookmark_source,
        )
        bubble.pack(fill="x", pady=(PADDING_SM, 0))
        self._bubbles.append(bubble)
        self._current_ai_bubble = bubble
        self._scroll_to_bottom()
        return bubble

    def _clear_chat_area_if_welcome(self):
        """Clear welcome state if it's still showing (AI or Search welcome)."""
        children = self._chat_scroll.winfo_children()
        if not children:
            return
        first = children[0]
        welcome_texts = ("IsoCortex AI", "Semantic Search")
        try:
            for child in first.winfo_children():
                for sub in child.winfo_children():
                    try:
                        if hasattr(sub, 'cget') and sub.cget("text") in welcome_texts:
                            self._clear_chat_area()
                            return
                    except Exception:
                        continue
        except Exception:
            pass

    def _start_generation(self, query: str):
        """Start the RAG + LLM generation pipeline. Handles LLM loading."""
        # Check if LLM is ready
        if not self._ensure_llm():
            # LLM not ready yet — either loading in background or failed
            if self._llm_loaded and not (self._llm and self._llm.is_loaded):
                # LLM load already attempted and failed — show REAL error
                real_err = self._llm.load_error if self._llm else "Unknown error"
                self._add_error_bubble(
                    f"AI model could not be loaded.\n\n"
                    f"Error: {real_err}\n\n"
                    f"Make sure llama-cpp-python is installed and the model file "
                    f"exists in ~/.isocortex/models/"
                )
            else:
                # LLM is loading in background — queue the query
                self._pending_query = query
                self._add_ai_bubble()  # Show empty bubble as placeholder
                try:
                    ai_bubble = self._current_ai_bubble
                    if ai_bubble and isinstance(ai_bubble._text_widget, tk.Text):
                        ai_bubble._text_widget.configure(state="normal")
                        ai_bubble._text_widget.insert("1.0", "Loading AI model...")
                        ai_bubble._refresh_text_height()
                        ai_bubble._text_widget.configure(state="disabled")
                except Exception:
                    pass
            return

        # LLM is ready — proceed with generation
        self._do_generate(query)

    def _do_generate(self, query: str):
        """Actually start the RAG worker (LLM is confirmed loaded)."""
        self._is_generating = True
        self._was_stopped = False
        self._last_sources = []
        self._swap_send_stop(show_stop=True)

        try:
            self._ai_status.configure(text="Searching documents...", text_color=COLOR_PURPLE_LIGHT)
        except Exception:
            pass

        # If no AI bubble yet (wasn't created during LLM loading), create one
        if not self._current_ai_bubble or not self._bubbles or self._bubbles[-1]._role != "assistant":
            self._add_ai_bubble()
            # Show a typing indicator while RAG context is being retrieved
            try:
                ai_bubble = self._current_ai_bubble
                if ai_bubble and isinstance(ai_bubble._text_widget, tk.Text):
                    ai_bubble._text_widget.configure(state="normal")
                    ai_bubble._text_widget.insert("1.0", "Searching your documents...")
                    ai_bubble._refresh_text_height()
                    ai_bubble._text_widget.configure(state="disabled")
                    ai_bubble._content = ""
            except Exception:
                pass
        else:
            # Clear the "Loading AI model..." placeholder
            try:
                ai_bubble = self._current_ai_bubble
                if ai_bubble and isinstance(ai_bubble._text_widget, tk.Text):
                    ai_bubble._text_widget.configure(state="normal")
                    ai_bubble._text_widget.delete("1.0", "end")
                    ai_bubble._refresh_text_height()
                    ai_bubble._text_widget.configure(state="disabled")
                    ai_bubble._content = ""
            except Exception:
                pass

        # Build conversation history — keep last 8 messages (~4 exchanges)
        # Small context window (4096) means history must be minimal
        # to leave room for system prompt + RAG context
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in self._messages[-8:]
            if m["role"] in ("user", "assistant")
        ]

        # Get system prompt
        from desktop_app.llm import SYSTEM_PROMPT

        # Start RAG worker
        self._rag_worker = RAGWorker(
            engine=self._engine,
            llm=self._llm,
            query=query,
            conversation_messages=history,
            system_prompt=SYSTEM_PROMPT,
            on_context_ready=self._on_context_ready,
            on_token=self._on_token,
            on_complete=self._on_generation_complete,
            on_error=self._on_generation_error,
        )
        self._rag_worker.set_app_ref(self._app)
        self._rag_worker.start()

    def _on_context_ready(self, sources: list[dict]):
        """Called when RAG context has been retrieved (before LLM starts)."""
        self._last_sources = sources
        try:
            self._ai_status.configure(text="Generating...", text_color=COLOR_GOLD)
        except Exception:
            pass

    def _on_token(self, token_text: str, token_count: int, elapsed: float):
        """Called for each streamed token."""
        # If stopped, ignore late tokens
        if getattr(self, '_was_stopped', False):
            return

        if self._current_ai_bubble:
            # Clear placeholder text on first real token
            if token_count == 1 and self._current_ai_bubble._content == "":
                try:
                    self._current_ai_bubble._text_widget.configure(state="normal")
                    self._current_ai_bubble._text_widget.delete("1.0", "end")
                    self._current_ai_bubble._text_widget.configure(state="disabled")
                except Exception:
                    pass
            self._current_ai_bubble.append_token(token_text)
            self._current_ai_bubble.update_speed(token_count, elapsed)

        # Auto-scroll during generation (throttled)
        if token_count % 5 == 0:
            self._scroll_to_bottom()

    def _on_generation_complete(self, result: dict):
        """Called when LLM generation finishes."""
        # If the user stopped generation, don't process the late callback
        if getattr(self, '_was_stopped', False):
            self._was_stopped = False
            self._rag_worker = None
            return

        response_text = result["response"]
        token_count = result["token_count"]
        elapsed = result["elapsed"]
        tps = result.get("tokens_per_second", 0)
        sources = result.get("sources", [])

        # Handle empty response
        if not response_text.strip():
            response_text = "I couldn't generate a response. Please try rephrasing your question."
            if self._current_ai_bubble:
                self._current_ai_bubble.set_content(response_text)

        # Save assistant message to DB
        self._engine.save_message(
            self._conversation_id,
            "assistant",
            response_text,
            sources=sources,
            token_count=token_count,
        )

        # Add to in-memory messages
        self._messages.append({"role": "assistant", "content": response_text})

        # Finalize the AI bubble
        if self._current_ai_bubble:
            self._current_ai_bubble.finalize(token_count, elapsed, tps)
            # Inject sources into the existing bubble (no widget destruction)
            if sources:
                self._current_ai_bubble.inject_sources(sources)

        # Reset state
        self._is_generating = False
        self._rag_worker = None
        self._swap_send_stop(show_stop=False)

        try:
            self._ai_status.configure(text="AI Ready", text_color=COLOR_SUCCESS)
        except Exception:
            pass

        self._scroll_to_bottom()

        # Save to search history
        try:
            user_query = self._messages[-2]["content"] if len(self._messages) >= 2 else ""
            if user_query:
                self._engine.save_search(user_query, len(result.get("sources", [])))
        except Exception:
            pass

    def _on_generation_error(self, exc):
        """Called when generation fails."""
        # If the user stopped generation, ignore the error callback
        if getattr(self, '_was_stopped', False):
            self._was_stopped = False
            self._rag_worker = None
            return

        self._is_generating = False
        self._rag_worker = None
        self._swap_send_stop(show_stop=False)

        # Remove the empty AI bubble if it has no content
        self._cleanup_empty_bubble()

        self._add_error_bubble(f"Generation failed: {exc}")

        try:
            self._ai_status.configure(text="Error", text_color=COLOR_ERROR)
        except Exception:
            pass

    def _cleanup_empty_bubble(self):
        """Remove the current AI bubble if it has no meaningful content."""
        if self._current_ai_bubble and not self._current_ai_bubble._content:
            try:
                self._current_ai_bubble.destroy()
                if self._current_ai_bubble in self._bubbles:
                    self._bubbles.remove(self._current_ai_bubble)
            except Exception:
                pass
            self._current_ai_bubble = None

    def _add_error_bubble(self, error_text: str):
        """Add an error message as an AI bubble."""
        self._clear_chat_area_if_welcome()
        bubble = ChatBubble(
            self._chat_scroll,
            role="assistant",
            content=error_text,
        )
        bubble.pack(fill="x", pady=(PADDING_SM, 0))
        self._bubbles.append(bubble)
        self._scroll_to_bottom()

    # ══════════════════════════════════════════════════════════════════
    #  Stop / Regenerate
    # ══════════════════════════════════════════════════════════════════

    def _handle_stop(self):
        """Stop ongoing generation."""
        if not self._is_generating:
            return

        # Mark as not generating FIRST to prevent _on_generation_complete
        # from doing anything if it fires after we stop
        self._is_generating = False
        self._was_stopped = True

        if self._rag_worker and self._rag_worker.is_alive():
            self._rag_worker.stop()
            # Don't set to None yet — the thread may still be running
            # and _on_generation_complete/_on_generation_error may fire

        # Null out callbacks to prevent post-destruction UI updates
        if self._rag_worker:
            self._rag_worker._on_token = None
            self._rag_worker._on_complete = None
            self._rag_worker._on_error = None
            self._rag_worker._on_context_ready = None

        self._swap_send_stop(show_stop=False)

        try:
            self._ai_status.configure(text="Stopped", text_color=COLOR_WARNING)
        except Exception:
            pass

        # Save whatever was generated so far
        if self._current_ai_bubble and self._current_ai_bubble._content:
            partial = self._current_ai_bubble._content
            if partial:
                self._engine.save_message(
                    self._conversation_id,
                    "assistant",
                    partial + " [stopped]",
                    sources=self._last_sources,
                )
                self._messages.append({"role": "assistant", "content": partial + " [stopped]"})

    def _handle_regenerate(self):
        """Regenerate the last AI response using the same RAG context."""
        if self._is_generating:
            return

        # Find the last user message
        last_user_query = None
        for msg in reversed(self._messages):
            if msg["role"] == "user":
                last_user_query = msg["content"]
                break

        if not last_user_query:
            return

        # Remove the last AI bubble from UI
        if self._current_ai_bubble:
            try:
                self._current_ai_bubble.destroy()
                if self._current_ai_bubble in self._bubbles:
                    self._bubbles.remove(self._current_ai_bubble)
            except Exception:
                pass
            self._current_ai_bubble = None

        # Remove last assistant message from memory
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages.pop()

        # Start generation directly (LLM is already loaded)
        self._do_generate(last_user_query)

    def _export_conversation(self):
        """Export the entire conversation as a Markdown file."""
        try:
            from desktop_app.components.exporter import export_conversation_markdown

            if not self._conversation_id or not self._messages:
                return

            path = export_conversation_markdown(
                engine=self._engine,
                conversation_id=self._conversation_id,
            )
            from pathlib import Path
            fname = Path(path).name
            self._app.show_toast(f"Conversation exported as {fname}", "success")
        except Exception as exc:
            try:
                self._app.show_toast(f"Export failed: {exc}", "error")
            except Exception:
                pass

    def _bookmark_source(self, source: dict):
        """Bookmark a source citation."""
        try:
            self._engine.add_bookmark(
                query="",  # Will be filled from current context
                file_path=source.get("file", ""),
                chunk_text=source.get("full_text", source.get("chunk_text", ""))[:500],
                score=source.get("score", 0),
            )
            self._app.show_toast("Source bookmarked", "success")
        except Exception:
            pass

    def _open_source_preview(self, source: dict):
        """Open a document preview popup for a source citation."""
        try:
            from desktop_app.components.preview import DocumentPreview

            # Get the full chunk text from the engine if possible
            chunk_text = ""
            file_name = source.get("file", "unknown")

            # Try to find the full text in recent search results
            # The sources come from build_rag_context which has the chunk text
            # We'll use the source's full_text if available
            chunk_text = source.get("full_text", source.get("chunk_text", ""))

            # If we don't have full text, try to read the file directly
            if not chunk_text and file_name != "unknown":
                try:
                    from pathlib import Path
                    # Search for the file in the engine's data
                    conn = self._engine.get_db_connection()
                    row = conn.execute(
                        "SELECT file_path FROM documents WHERE file_path LIKE ? LIMIT 1",
                        (f"%{file_name}",),
                    ).fetchone()
                    if row:
                        file_path = Path(row[0])
                        if file_path.exists():
                            chunk_text = self._engine.extract_text(file_path)
                except Exception:
                    pass

            if not chunk_text:
                chunk_text = "Full document text is not available for preview."

            title = file_name
            page = source.get("page", 0)
            if page:
                title += f" — Page {page}"

            preview = DocumentPreview(
                parent=self._app,
                title=title,
                content=chunk_text,
                file_path=source.get("file_path", file_name),
                page_number=page,
                highlight_text=source.get("query", ""),
            )
            preview.show()
        except Exception as exc:
            logger.error("Failed to open preview: %s", exc)

    # ══════════════════════════════════════════════════════════════════
    #  Search History & Bookmarks Panel
    # ══════════════════════════════════════════════════════════════════

    def _toggle_history_panel(self):
        """Toggle the search history sidebar panel."""
        if hasattr(self, '_history_panel') and self._history_panel and self._history_panel.winfo_exists():
            self._history_panel.destroy()
            self._history_panel = None
            return
        self._show_history_panel()

    def _show_history_panel(self):
        """Show a floating panel with search history and bookmarks."""
        import tkinter as tk

        # Create a toplevel panel positioned to the right of the top bar
        self._history_panel = ctk.CTkToplevel(self._app)
        self._history_panel.title("Search History")
        self._history_panel.geometry("340x500")
        self._history_panel.minsize(280, 300)
        self._history_panel.configure(fg_color=COLOR_BG_DARKEST)
        self._history_panel.transient(self._app)
        self._history_panel.attributes("-topmost", True)

        # Position to the right of parent
        self._history_panel.update_idletasks()
        try:
            px = self._app.winfo_x() + self._app.winfo_width() - 350
            py = self._app.winfo_y() + 80
            self._history_panel.geometry(f"+{max(0, px)}+{py}")
        except Exception:
            pass

        self._history_panel.protocol("WM_DELETE_WINDOW", self._toggle_history_panel)

        # Content
        container = ctk.CTkFrame(self._history_panel, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=PADDING_MD, pady=PADDING_MD)

        # Tab selector
        tab_frame = ctk.CTkFrame(container, fg_color="transparent")
        tab_frame.pack(fill="x", pady=(0, PADDING_SM))

        self._hist_tab_var = "history"

        hist_tab = ctk.CTkButton(
            tab_frame, text="History",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_DARK,
            text_color="#ffffff",
            height=30, corner_radius=BORDER_RADIUS_SM,
            command=lambda: self._switch_hist_tab("history"),
        )
        hist_tab.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self._hist_tab_btn = hist_tab

        bm_tab = ctk.CTkButton(
            tab_frame, text="Bookmarks",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_SECONDARY,
            height=30, corner_radius=BORDER_RADIUS_SM,
            command=lambda: self._switch_hist_tab("bookmarks"),
        )
        bm_tab.pack(side="left", fill="x", expand=True, padx=(2, 0))
        self._bm_tab_btn = bm_tab

        # Scrollable content
        self._hist_scroll = ctk.CTkScrollableFrame(container, fg_color="transparent")
        self._hist_scroll.pack(fill="both", expand=True)

        # Clear button
        clear_btn = ctk.CTkButton(
            container,
            text="Clear History",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            fg_color="transparent",
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_DIM,
            height=24, corner_radius=BORDER_RADIUS_XS,
            command=self._clear_history,
        )
        clear_btn.pack(fill="x", pady=(PADDING_SM, 0))

        # Load history
        self._load_history_items()

    def _switch_hist_tab(self, tab: str):
        """Switch between history and bookmarks tabs."""
        self._hist_tab_var = tab
        if tab == "history":
            self._hist_tab_btn.configure(fg_color=COLOR_PURPLE, text_color="#ffffff")
            self._bm_tab_btn.configure(fg_color=COLOR_BG_ELEVATED, text_color=COLOR_TEXT_SECONDARY)
        else:
            self._bm_tab_btn.configure(fg_color=COLOR_PURPLE, text_color="#ffffff")
            self._hist_tab_btn.configure(fg_color=COLOR_BG_ELEVATED, text_color=COLOR_TEXT_SECONDARY)
        self._load_history_items()

    def _load_history_items(self):
        """Load and display history or bookmarks."""
        # Clear existing
        for w in self._hist_scroll.winfo_children():
            w.destroy()

        if self._hist_tab_var == "history":
            items = self._engine.get_search_history(limit=30)
            if not items:
                ctk.CTkLabel(
                    self._hist_scroll, text="No search history yet",
                    font=(FONT_FAMILY, FONT_SIZE_SMALL),
                    text_color=COLOR_TEXT_DIM,
                ).pack(pady=PADDING_LG)
                return

            for item in items:
                row = ctk.CTkFrame(self._hist_scroll, fg_color=COLOR_BG_ELEVATED, corner_radius=BORDER_RADIUS_SM)
                row.pack(fill="x", pady=1)
                inner = ctk.CTkFrame(row, fg_color="transparent")
                inner.pack(fill="x", padx=PADDING_MD, pady=(PADDING_SM, PADDING_SM))

                ctk.CTkLabel(
                    inner, text=item["query"],
                    font=(FONT_FAMILY, FONT_SIZE_SMALL),
                    text_color=COLOR_TEXT,
                    anchor="w", wraplength=260,
                ).pack(anchor="w")

                time_label = ctk.CTkLabel(
                    inner, text=item.get("created_at", "")[:16],
                    font=(FONT_FAMILY, FONT_SIZE_XXS),
                    text_color=COLOR_TEXT_DIM,
                    anchor="w",
                )
                time_label.pack(anchor="w")

                # Click to re-search
                row.configure(cursor="hand2")
                query = item["query"]
                row.bind("<Button-1>", lambda e, q=query: self._reuse_query(q))
                for child in row.winfo_children():
                    child.bind("<Button-1>", lambda e, q=query: self._reuse_query(q))
                    for sub in child.winfo_children():
                        sub.bind("<Button-1>", lambda e, q=query: self._reuse_query(q))
        else:
            items = self._engine.get_bookmarks(limit=30)
            if not items:
                ctk.CTkLabel(
                    self._hist_scroll, text="No bookmarks yet",
                    font=(FONT_FAMILY, FONT_SIZE_SMALL),
                    text_color=COLOR_TEXT_DIM,
                ).pack(pady=PADDING_LG)
                return

            for item in items:
                row = ctk.CTkFrame(self._hist_scroll, fg_color=COLOR_BG_ELEVATED, corner_radius=BORDER_RADIUS_SM, border_width=1, border_color=COLOR_BORDER)
                row.pack(fill="x", pady=2)
                inner = ctk.CTkFrame(row, fg_color="transparent")
                inner.pack(fill="x", padx=PADDING_MD, pady=(PADDING_SM, PADDING_SM))

                file_name = item.get("file_path", "")
                if file_name:
                    from pathlib import Path
                    file_name = Path(file_name).name

                header = item.get("query", "")[:60]
                if file_name:
                    header += f"  —  {file_name}"

                ctk.CTkLabel(
                    inner, text=header,
                    font=(FONT_FAMILY, FONT_SIZE_SMALL),
                    text_color=COLOR_TEXT,
                    anchor="w", wraplength=250,
                ).pack(anchor="w")

                note = item.get("note", "")
                if note:
                    ctk.CTkLabel(
                        inner, text=note[:80],
                        font=(FONT_FAMILY, FONT_SIZE_XXS),
                        text_color=COLOR_TEXT_DIM,
                        anchor="w", wraplength=250,
                    ).pack(anchor="w")

                # Delete button
                del_btn = ctk.CTkButton(
                    inner, text="x",
                    font=(FONT_FAMILY, FONT_SIZE_XXS),
                    fg_color="transparent",
                    hover_color=COLOR_BG_HOVER,
                    text_color=COLOR_TEXT_DIM,
                    width=20, height=20, corner_radius=BORDER_RADIUS_XS,
                    command=lambda bid=item["bookmark_id"]: self._delete_bookmark(bid),
                )
                del_btn.pack(anchor="e")

    def _reuse_query(self, query: str):
        """Fill input with a history query and send it."""
        if hasattr(self, '_history_panel') and self._history_panel and self._history_panel.winfo_exists():
            self._history_panel.destroy()
            self._history_panel = None
        self._input_entry.delete(0, "end")
        self._input_entry.insert(0, query)
        self._handle_send()

    def _clear_history(self):
        """Clear search history."""
        try:
            count = self._engine.clear_search_history()
            self._load_history_items()
        except Exception:
            pass

    def _delete_bookmark(self, bookmark_id: str):
        """Delete a bookmark and refresh the list."""
        try:
            self._engine.remove_bookmark(bookmark_id)
            self._load_history_items()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════
    #  UI Helpers
    # ══════════════════════════════════════════════════════════════════

    def _swap_send_stop(self, show_stop: bool):
        """Toggle between Send and Stop buttons."""
        try:
            if show_stop:
                self._send_btn.pack_forget()
                self._stop_btn.pack(side="right")
                self._input_entry.configure(state="disabled")
            else:
                self._stop_btn.pack_forget()
                self._send_btn.pack(side="right")
                self._input_entry.configure(state="normal")
                self._input_entry.focus_set()
        except Exception:
            pass

    def _scroll_to_bottom(self):
        """Scroll the chat area to the bottom."""
        try:
            self._chat_scroll._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass