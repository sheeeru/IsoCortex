"""
IsoCortex Desktop App — Search Screen
======================================
Compact search results with page numbers, context snippets,
and query-word highlighting. Everything in one default graph.

Enhanced with premium theme components:
  - AnimatedGradientBG: subtle animated radial gradient background
  - ShimmerBar: animated top accent bar
  - GlassCard: glassmorphism containers for search bar & empty states
  - GradientDivider: gradient dividers between sections
  - AnimatedPulseGlow: pulsing glow behind the search area
  - FadeInFrame: staggered fade-in entrance for result cards
  - create_badge: styled sample query chips
"""

import time
import tkinter as tk
import customtkinter as ctk

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER,
    COLOR_PURPLE, COLOR_PURPLE_DARK, COLOR_PURPLE_LIGHT, COLOR_PURPLE_DEEP,
    COLOR_GOLD, COLOR_GOLD_LIGHT,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    COLOR_SHADOW,
    FONT_FAMILY, FONT_FAMILY_MONO,
    FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_MEDIUM, FONT_SIZE_NORMAL, FONT_SIZE_SMALL, FONT_SIZE_XXS,
    BORDER_RADIUS, BORDER_RADIUS_SM, BORDER_RADIUS_LG,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    # Theme animation components
    ShimmerBar,
    GlassCard,
    GradientDivider,
    FadeInFrame,
    AnimatedGradientBG,
    create_badge,
    ANIM_DELAY_200, ANIM_DELAY_400, ANIM_DELAY_600, ANIM_DELAY_800,
    stagger_animation,
)
from desktop_app.engine import DEFAULT_INDEX_NAME, _HIGHLIGHT_MARKER


class SearchScreen(ctk.CTkFrame):
    """Compact semantic search across all indexed files."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app = app
        self._current_results = []
        self._last_query = ""
        self._build_ui()

    # ══════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Animated gradient background layer ──────────────────────
        self._bg_canvas = AnimatedGradientBG(self)
        self._bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.after(100, self._fit_bg)
        self.bind("<Configure>", lambda e: self.after_idle(self._fit_bg))

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING_LG)

        self._build_header(content)
        self._build_shimmer_bar(content)
        self._build_search_bar(content)
        self._build_sample_chips(content)
        GradientDivider(content, height=1).pack(fill="x", pady=(PADDING_SM, PADDING_MD))
        self._build_results_header(content)
        self._build_results_area(content)

    def _fit_bg(self):
        """Keep the gradient background canvas filling the entire screen."""
        try:
            self._bg_canvas.place(relx=0.5, rely=0.5, anchor="center",
                                  relwidth=1.0, relheight=1.0)
        except tk.TclError:
            pass

    # ── Header ───────────────────────────────────────────────────────

    def _build_header(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent", height=36)
        header.pack(fill="x", pady=(0, PADDING_MD))
        header.pack_propagate(False)

        accent = ctk.CTkFrame(header, width=4, corner_radius=2, fg_color=COLOR_PURPLE)
        accent.pack(side="left", padx=(0, PADDING_MD), pady=6)
        accent.pack_propagate(False)

        ctk.CTkLabel(
            header, text="Semantic Search",
            font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
            text_color=COLOR_TEXT, anchor="w",
        ).pack(side="left", pady=6)

        ctk.CTkLabel(
            header, text="Search across all your indexed files",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
        ).pack(side="right", pady=6)

    # ── Shimmer Bar Accent ───────────────────────────────────────────

    def _build_shimmer_bar(self, parent):
        """Animated shimmer accent bar below the header."""
        self._shimmer = ShimmerBar(parent, height=3)
        self._shimmer.pack(fill="x", pady=(0, PADDING_MD))

    # ── Search Bar (GlassCard container) ─────────────────────────────

    def _build_search_bar(self, parent):
        # GlassCard wraps the search bar for a glassmorphism look
        self._search_glass = GlassCard(
            parent,
            corner_radius=BORDER_RADIUS_LG,
        )
        self._search_glass.pack(fill="x", pady=(0, PADDING_MD))

        bar = ctk.CTkFrame(self._search_glass, fg_color="transparent")
        bar.pack(fill="x", padx=PADDING_MD, pady=PADDING_MD)

        # Static glow container behind the search icon
        glow_container = ctk.CTkFrame(bar, fg_color=COLOR_PURPLE_DEEP, width=44, height=40, corner_radius=20)
        glow_container.pack(side="left", padx=(0, PADDING_SM))
        glow_container.pack_propagate(False)

        # Search icon label on top of the glow
        ctk.CTkLabel(
            glow_container, text="\u23CE",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            text_color=COLOR_PURPLE_LIGHT, fg_color="transparent",
        ).place(relx=0.5, rely=0.5, anchor="center")

        self._query_entry = ctk.CTkEntry(
            bar,
            placeholder_text="Ask anything about your documents\u2026",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=COLOR_BG_ELEVATED,
            border_color=COLOR_BORDER,
            text_color=COLOR_TEXT,
            placeholder_text_color=COLOR_TEXT_DIM,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
        )
        self._query_entry.pack(side="left", fill="x", expand=True)
        self._query_entry.bind("<Return>", lambda e: self._handle_search())

        self._search_btn = ctk.CTkButton(
            bar, text="\u23CE  Search",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=COLOR_GOLD, hover_color=COLOR_GOLD_LIGHT,
            text_color="#0a0a0f",
            height=40, corner_radius=BORDER_RADIUS_SM, width=110,
            command=self._handle_search,
        )
        self._search_btn.pack(side="right")

    # ── Sample Chips (create_badge) ──────────────────────────────────

    def _build_sample_chips(self, parent):
        chips = ctk.CTkFrame(parent, fg_color="transparent")
        chips.pack(fill="x", pady=(0, PADDING_SM))

        ctk.CTkLabel(chips, text="Try:", font=(FONT_FAMILY, FONT_SIZE_XXS), text_color=COLOR_TEXT_DIM).pack(side="left", padx=(0, PADDING_SM))
        for query in ["Summarize key findings", "Main risks?", "Methodology", "Compare results"]:
            badge = create_badge(
                chips, text=query,
                color=COLOR_PURPLE_LIGHT,
            )
            badge.pack(side="left", padx=(0, PADDING_SM))
            badge.configure(cursor="hand2")
            badge.bind(
                "<Button-1>",
                lambda e, q=query: self._fill_query(q),
            )

    # ── Results Header ───────────────────────────────────────────────

    def _build_results_header(self, parent):
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.pack(fill="x", pady=(0, PADDING_SM))

        self._results_count_label = ctk.CTkLabel(bar, text="", font=(FONT_FAMILY, FONT_SIZE_XXS), text_color=COLOR_TEXT_DIM, anchor="w")
        self._results_count_label.pack(side="left")

        self._results_time_label = ctk.CTkLabel(bar, text="", font=(FONT_FAMILY, FONT_SIZE_XXS), text_color=COLOR_TEXT_DIM)
        self._results_time_label.pack(side="right")

    # ── Results Area ─────────────────────────────────────────────────

    def _build_results_area(self, parent):
        self._results_scroll = ctk.CTkScrollableFrame(
            parent, fg_color="transparent",
        )
        self._results_scroll.pack(fill="both", expand=True)
        self._show_empty_state()

    # ─────────────────────────────────────────────────────────────────
    #  Empty States
    # ─────────────────────────────────────────────────────────────────

    def _show_empty_state(self):
        self._clear_results()
        glass = GlassCard(self._results_scroll, corner_radius=BORDER_RADIUS_LG)
        glass.pack(fill="both", expand=True, padx=40, pady=PADDING_LG)

        # Static glow behind the icon (no animation timer)
        glow_frame = ctk.CTkFrame(glass, fg_color="transparent")
        glow_frame.pack(pady=(PADDING_XL, PADDING_MD))

        icon_bg = ctk.CTkFrame(glow_frame, fg_color=COLOR_PURPLE_DEEP, width=80, height=80, corner_radius=40)
        icon_bg.pack()
        icon_bg.pack_propagate(False)
        ctk.CTkLabel(
            icon_bg, text="\u2299",
            font=(FONT_FAMILY, 48), text_color=COLOR_TEXT_DIM,
        ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(glass, text="Search your indexed documents", font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"), text_color=COLOR_TEXT_SECONDARY).pack()
        ctk.CTkLabel(glass, text="Type a query and discover relevant passages\npowered by semantic vector search.", font=(FONT_FAMILY, FONT_SIZE_SMALL), text_color=COLOR_TEXT_DIM, justify="center").pack(pady=(PADDING_SM, PADDING_XL))

    def _show_no_results_state(self):
        self._clear_results()
        glass = GlassCard(self._results_scroll, corner_radius=BORDER_RADIUS_LG)
        glass.pack(fill="both", expand=True, padx=40, pady=PADDING_LG)

        # Static glow behind the icon (no animation timer)
        glow_frame = ctk.CTkFrame(glass, fg_color="transparent")
        glow_frame.pack(pady=(PADDING_XL, PADDING_MD))

        icon_bg = ctk.CTkFrame(glow_frame, fg_color="#3d2800", width=80, height=80, corner_radius=40)
        icon_bg.pack()
        icon_bg.pack_propagate(False)
        ctk.CTkLabel(
            icon_bg, text="\u2298",
            font=(FONT_FAMILY, 48), text_color=COLOR_TEXT_DIM,
        ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(glass, text="No results found", font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"), text_color=COLOR_TEXT_SECONDARY).pack()
        ctk.CTkLabel(glass, text="Try different keywords or rephrase your query.", font=(FONT_FAMILY, FONT_SIZE_SMALL), text_color=COLOR_TEXT_DIM, justify="center").pack(pady=(PADDING_SM, PADDING_XL))

    def _clear_results(self):
        for widget in self._results_scroll.winfo_children():
            widget.destroy()

    # ─────────────────────────────────────────────────────────────────
    #  Search Logic
    # ─────────────────────────────────────────────────────────────────

    def _fill_query(self, query: str):
        self._query_entry.delete(0, "end")
        self._query_entry.insert(0, query)
        self._handle_search()

    def _handle_search(self):
        query = self._query_entry.get().strip()
        if len(query) < 3:
            try:
                self._results_count_label.configure(text="\u26A0  Min 3 characters", text_color=COLOR_WARNING)
                self._results_time_label.configure(text="")
            except Exception:
                pass
            return

        try:
            self._search_btn.configure(text="\u2026", state="disabled")
            self._results_count_label.configure(text="Searching\u2026", text_color=COLOR_PURPLE)
            self._results_time_label.configure(text="")
        except Exception:
            pass

        t0 = time.perf_counter()
        try:
            results = self._app.engine.search(DEFAULT_INDEX_NAME, query, top_k=10)
            elapsed = time.perf_counter() - t0
            self._current_results = results
            self._last_query = query
            self._render_results(results, elapsed)
        except ValueError as exc:
            try:
                self._results_count_label.configure(text=f"\u26A0  {exc}", text_color=COLOR_ERROR)
            except Exception:
                pass
        except Exception as exc:
            try:
                self._results_count_label.configure(text=f"\u26A0  Error: {exc}", text_color=COLOR_ERROR)
            except Exception:
                pass
        finally:
            try:
                self._search_btn.configure(text="\u23CE  Search", state="normal")
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────
    #  Result Rendering — Compact Cards
    # ─────────────────────────────────────────────────────────────────

    def _render_results(self, results, elapsed: float):
        self._clear_results()
        count = len(results)
        try:
            self._results_count_label.configure(text=f"{count} result{'s' if count != 1 else ''}", text_color=COLOR_TEXT_SECONDARY)
            self._results_time_label.configure(text=f"{elapsed * 1000:.0f} ms", text_color=COLOR_TEXT_DIM)
        except Exception:
            pass

        if not results:
            self._show_no_results_state()
            return

        for idx, result in enumerate(results):
            # Staggered entrance: each card fades in with increasing delay
            delay = idx * 80  # 80ms stagger between cards
            stagger_animation(
                self._results_scroll,
                delay_ms=delay,
                callback=lambda r=result: self._create_compact_card(r),
            )

    def _create_compact_card(self, result):
        """Build a compact result card: rank, score, file, page, highlighted context."""
        score_pct = result.score * 100
        score_color = COLOR_SUCCESS if result.score > 0.5 else COLOR_WARNING

        # FadeInFrame wrapping each card for staggered entrance animation
        card = FadeInFrame(
            self._results_scroll,
            fg_color=COLOR_BG_CARD,
            corner_radius=BORDER_RADIUS,
            border_width=1,
            border_color=COLOR_BORDER_LIGHT,
        )
        card.pack(fill="x", pady=2)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=PADDING_MD, pady=(PADDING_SM, PADDING_SM))

        # ── Row 1: rank + score bar + % + badges ─────────────────
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x")

        # Rank
        ctk.CTkLabel(
            row1, text=f"#{result.rank}",
            font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
            text_color=COLOR_GOLD, anchor="w", width=26,
        ).pack(side="left")

        # Mini score bar
        bar_w = 50
        bar_frame = ctk.CTkFrame(row1, fg_color=COLOR_BG_ELEVATED, corner_radius=2, height=4, width=bar_w)
        bar_frame.pack(side="left", padx=(0, PADDING_SM), pady=(6, 0))
        fill_w = max(int(bar_w * min(result.score, 1.0)), 3)
        bar_fill = ctk.CTkFrame(bar_frame, fg_color=score_color, corner_radius=2, width=fill_w)
        bar_fill.pack(side="left")
        bar_fill.pack_propagate(False)

        # Score %
        ctk.CTkLabel(
            row1, text=f"{score_pct:.0f}%",
            font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
            text_color=score_color,
        ).pack(side="left", padx=(0, PADDING_SM))

        # Spacer — pushes badges to the right
        ctk.CTkLabel(row1, text="").pack(side="left", fill="x", expand=True)

        # Page number badge (only for PDFs / non-zero pages)
        if result.page_number and result.page_number > 0:
            ctk.CTkLabel(
                row1, text=f"  p.{result.page_number}  ",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                text_color=COLOR_PURPLE_LIGHT,
                fg_color=COLOR_BG_ELEVATED,
                corner_radius=4,
            ).pack(side="right")

        # Format badge
        if result.format_category:
            ctk.CTkLabel(
                row1, text=f" {result.format_category.upper()} ",
                font=(FONT_FAMILY, FONT_SIZE_XXS),
                text_color=COLOR_TEXT_DIM,
                fg_color=COLOR_BG_ELEVATED,
                corner_radius=4,
            ).pack(side="right", padx=(0, PADDING_SM))

        # ── Row 2: Full file name (no truncation) ───────────────
        src = result.source_file or "unknown"
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(
            row2, text=src,
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            text_color=COLOR_TEXT, anchor="w",
        ).pack(side="left")

        # ── Row 3: Selectable highlighted context snippet ───────
        snippet = result.matched_segment or ""
        if not snippet and result.text:
            snippet = result.text[:300]
            if len(result.text) > 300:
                snippet += "\u2026"

        if snippet:
            ctx_frame = ctk.CTkFrame(inner, fg_color="transparent")
            ctx_frame.pack(fill="x", pady=(3, 0))

            try:
                self._render_selectable_snippet(ctx_frame, snippet)
            except Exception:
                # Fallback: show raw snippet without highlighting
                try:
                    clean = snippet.replace(_HIGHLIGHT_MARKER, "")
                    ctk.CTkLabel(
                        ctx_frame, text=clean[:300],
                        font=(FONT_FAMILY, FONT_SIZE_SMALL),
                        text_color=COLOR_TEXT_SECONDARY,
                        anchor="w", wraplength=750, justify="left",
                    ).pack(anchor="w")
                except Exception:
                    pass

    def _render_selectable_snippet(self, parent: ctk.CTkFrame, text: str):
        """Render the snippet in a tk.Text widget so the user can select and copy text.

        Uses tk.Text (not CTkLabel) because tk.Text supports:
          - Mouse text selection
          - Ctrl+C / Cmd+C to copy
          - Tag-based styling for highlighted words
        The widget is set to disabled state after filling, so it acts as
        read-only while still allowing selection and copy.
        """
        # Calculate available width (rough estimate)
        try:
            avail_w = max(parent.winfo_width() - PADDING_MD * 2, 400)
        except Exception:
            avail_w = 700

        snippet_text = tk.Text(
            parent,
            height=3,
            wrap="word",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg=COLOR_TEXT_SECONDARY,
            bg=COLOR_BG_CARD,
            bd=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            cursor="arrow",
            selectbackground=COLOR_PURPLE_DARK,
            selectforeground="#ffffff",
            relief="flat",
            spacing1=0,
            spacing3=0,
        )
        snippet_text.pack(fill="x")

        # Configure text tags for styling
        snippet_text.tag_configure(
            "highlight",
            foreground="#c4b5fd",
            background="#2d1f5e",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
        )
        snippet_text.tag_configure(
            "normal",
            foreground=COLOR_TEXT_SECONDARY,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
        )
        snippet_text.tag_configure(
            "ellipsis",
            foreground=COLOR_TEXT_DIM,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
        )

        # Parse highlighted segments and insert with tags
        parts = text.split(_HIGHLIGHT_MARKER)
        is_highlight = False
        for part in parts:
            if not part:
                is_highlight = not is_highlight
                continue

            tag = "highlight" if is_highlight else "normal"
            snippet_text.insert("end", part, tag)
            is_highlight = not is_highlight

        # Make read-only (still selectable and copyable)
        snippet_text.configure(state="disabled")
