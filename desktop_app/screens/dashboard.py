"""
IsoCortex Desktop App — Dashboard Screen (Enhanced Premium Redesign)
====================================================================
A polished, depth-rich dashboard showing system stats, quick actions,
embedding model status, and data storage information.

Design language:
  - Animated ShimmerBar accent (replaces static gradient)
  - GlassCard glassmorphism for Quick Actions, Model, Storage
  - Animated stat cards with per-card shimmer bars
  - AnimatedPulseGlow behind model status indicator
  - GradientDivider between all major sections
  - FadeInFrame wrappers with staggered entrance delays
  - AnimatedGradientBG for living/breathing background
  - Large 32px bold stat values with per-card colour coding
  - Generous whitespace and visual hierarchy
  - Pulsing status dots for model health
  - Selectable text labels for copy support
"""

import customtkinter as ctk

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER,
    COLOR_PURPLE, COLOR_PURPLE_DARK, COLOR_PURPLE_LIGHT,
    COLOR_PURPLE_DEEP,
    COLOR_GOLD, COLOR_GOLD_LIGHT, COLOR_GOLD_BTN_TEXT,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_SUCCESS, COLOR_INFO, COLOR_WARNING,
    COLOR_SHADOW, COLOR_SURFACE_1,
    COLOR_GLASS_BG,
    FONT_FAMILY, FONT_FAMILY_MONO,
    FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_MEDIUM,
    FONT_SIZE_NORMAL, FONT_SIZE_SMALL, FONT_SIZE_XXS,
    BORDER_RADIUS, BORDER_RADIUS_SM, BORDER_RADIUS_LG,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG,
    PulseIndicator, make_selectable_label,
    GRADIENT_PURPLE_GOLD,
    # ── Enhanced animated visual components ──
    ShimmerBar,
    GradientDivider,
    GlassCard,
    create_animated_stat_card,
    FadeInFrame,
    AnimatedGradientBG,
)


def _update_text_widget(widget, new_text):
    """Helper to update a tk.Text selectable label with new text."""
    try:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", new_text)
        widget.configure(state="disabled")
    except Exception:
        pass


class DashboardScreen(ctk.CTkFrame):
    """Premium dashboard with animated visual effects and staggered entrances."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app = app
        self._build_ui()
        # Delayed refresh — waits for all sections to finish building
        self.after(100, self._refresh_stats)

    # ────────────────────────────────────────────────────────────────
    # UI Construction
    # ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build all dashboard UI sections with animated visual effects."""
        # Outer scrollable content area
        content = ctk.CTkScrollableFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING_LG)

        # ── AnimatedGradientBG: living/breathing gradient background ──
        self._gradient_bg = AnimatedGradientBG(content)
        self._gradient_bg.place(x=0, y=0, relwidth=1, relheight=1)

        # ── ShimmerBar: animated top accent bar (replaces GradientCanvas) ──
        ShimmerBar(content, height=4).pack(fill="x", pady=(0, PADDING_LG))

        # ── Sequential section build (minimal stagger for render order) ──
        self.after(10, lambda: self._build_header(content))
        self.after(20, lambda: self._build_stat_cards(content))
        self.after(30, lambda: self._build_bottom_panels(content))

    # ── Helper: GlassCard with corrected pack behaviour ───────────

    def _make_glass_card(self, parent, glow_color=None,
                         fill="both", expand=True, **pack_kwargs):
        """Create a GlassCard with natural content sizing.

        The stock GlassCard constructor forces ``pack_propagate(False)``
        and ``expand=True`` which breaks vertical-flow layouts.  This
        helper re-packs the glow frame so it sizes to its children.
        """
        card = GlassCard(parent, glow_color=glow_color)
        # Undo the constructor's automatic pack, then re-pack correctly
        card._glow_frame.pack_forget()
        card._glow_frame.pack_propagate(True)
        # Use caller's padx/pady or sensible defaults
        padx = pack_kwargs.pop("padx", 0)
        pady = pack_kwargs.pop("pady", 0)
        card._glow_frame.pack(fill=fill, expand=expand,
                              padx=padx, pady=pady, **pack_kwargs)
        return card

    # ── Header ──────────────────────────────────────────────────────

    def _build_header(self, parent):
        """Page header wrapped in FadeInFrame with GradientDivider beneath."""
        # FadeInFrame wrapper — stagger delay already handled by .after(200)
        fade = FadeInFrame(parent, fg_color=COLOR_BG)
        fade.pack(fill="x")

        header = ctk.CTkFrame(fade, fg_color="transparent")
        header.pack(fill="x", pady=(0, PADDING_LG))

        # Left: purple accent bar + title area
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True)

        accent_bar = ctk.CTkFrame(
            left,
            width=4,
            height=32,
            corner_radius=2,
            fg_color=COLOR_PURPLE,
        )
        accent_bar.pack(side="left", padx=(0, PADDING), pady=6)
        accent_bar.pack_propagate(False)

        title_block = ctk.CTkFrame(left, fg_color="transparent")
        title_block.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            title_block,
            text="Dashboard",
            font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_block,
            text="System overview and quick actions",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(anchor="w")

        # Right: refresh button
        refresh_btn = ctk.CTkButton(
            header,
            text="↻  Refresh",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_SECONDARY,
            height=36,
            corner_radius=BORDER_RADIUS_SM,
            width=100,
            command=self._refresh_stats,
        )
        refresh_btn.pack(side="right", padx=(PADDING, 0))

        # GradientDivider after header section
        GradientDivider(fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

    # ── Stat Cards ─────────────────────────────────────────────────

    def _build_stat_cards(self, parent):
        """Four animated stat cards in a row, wrapped in FadeInFrame.

        Uses ``create_animated_stat_card`` which adds a ShimmerBar
        on top of each card instead of the static GradientCanvas.
        """
        # FadeInFrame wrapper — stagger delay already handled by .after(400)
        fade = FadeInFrame(parent, fg_color=COLOR_BG)
        fade.pack(fill="x")

        # Cards row container
        cards_row = ctk.CTkFrame(fade, fg_color="transparent")
        cards_row.pack(fill="x", pady=(PADDING_SM, PADDING_SM))

        stat_configs = [
            ("indexes",   "⬡  Indexes",   "0", COLOR_PURPLE),
            ("documents", "⊞  Documents", "0", COLOR_INFO),
            ("vectors",   "◈  Vectors",   "0", COLOR_GOLD),
            ("searches",  "⊙  Searches",  "0", COLOR_SUCCESS),
        ]

        self._stat_values = {}

        for i, (key, label, default, color) in enumerate(stat_configs):
            # create_animated_stat_card returns (card, value_label, shimmer)
            card, value_label, shimmer = create_animated_stat_card(
                cards_row, "", label, default, color,
            )
            card.pack(side="left", fill="both", expand=True, padx=(0, 10))
            if i == len(stat_configs) - 1:
                card.pack_configure(padx=0)
            self._stat_values[key] = value_label

        # GradientDivider after stat cards section
        GradientDivider(fade, height=1).pack(fill="x", pady=(PADDING_SM, 0))

    # ── Bottom Panels (Quick Actions + Info Cards) ─────────────────

    def _build_bottom_panels(self, parent):
        """Two-column layout: left = quick actions, right = model + storage.

        Right column sections are further staggered with internal delays
        so model appears at ~800 ms and storage at ~1 000 ms total.
        """
        bottom = ctk.CTkFrame(parent, fg_color="transparent")
        bottom.pack(fill="both", expand=True, pady=(PADDING_SM, 0))

        # ── Left column: Quick Actions (delay ≈ 600 ms) ───────────
        left_col = ctk.CTkFrame(bottom, fg_color="transparent")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, PADDING_MD))

        self._build_quick_actions(left_col)

        # ── Right column: Model + Storage (further staggered) ──────
        right_col = ctk.CTkFrame(bottom, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True, padx=(PADDING_MD, 0))

        # Model + Storage build immediately (no sub-stagger needed)
        self._build_model_section(right_col)
        self._build_storage_section(right_col)

    # ── Quick Actions ──────────────────────────────────────────────

    def _build_quick_actions(self, parent):
        """Quick Actions section inside a GlassCard with GradientDivider."""
        # FadeInFrame wrapper (stagger ≈ 600 ms via parent .after)
        fade = FadeInFrame(parent, fg_color=COLOR_BG)
        fade.pack(fill="both", expand=True, pady=(0, PADDING_MD))

        # GlassCard replaces the old shadow + CTkFrame + accent combo
        card = self._make_glass_card(fade, glow_color=COLOR_PURPLE)

        # Card content — GlassCard already has its own top gradient accent
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=PADDING, pady=PADDING)

        # Section title with GradientDivider replacing plain CTkFrame
        section_label = ctk.CTkFrame(inner, fg_color="transparent")
        section_label.pack(fill="x", pady=(0, PADDING_MD))

        ctk.CTkLabel(
            section_label,
            text="QUICK ACTIONS",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_PURPLE,
            anchor="w",
        ).pack(side="left")

        GradientDivider(section_label, height=1).pack(
            side="right", fill="x", expand=True, padx=(PADDING_MD, 0),
        )

        # Action buttons container
        actions = ctk.CTkFrame(inner, fg_color="transparent")
        actions.pack(fill="both", expand=True)

        # Upload Files — gold bg
        ctk.CTkButton(
            actions,
            text="⊕   Upload Files",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=COLOR_GOLD,
            hover_color=COLOR_GOLD_LIGHT,
            text_color=COLOR_GOLD_BTN_TEXT,
            height=44,
            corner_radius=BORDER_RADIUS_SM,
            anchor="w",
            command=lambda: self._app.show_screen("upload"),
        ).pack(fill="x", pady=(0, 8))

        # Semantic Search — purple bg
        ctk.CTkButton(
            actions,
            text="⊙   Semantic Search",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_DARK,
            text_color=COLOR_TEXT,
            height=44,
            corner_radius=BORDER_RADIUS_SM,
            anchor="w",
            command=lambda: self._app.show_screen("search"),
        ).pack(fill="x", pady=(0, 8))

        # Manage Indexes — elevated with purple text
        ctk.CTkButton(
            actions,
            text="▦   Manage Indexes",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_PURPLE,
            height=44,
            corner_radius=BORDER_RADIUS_SM,
            anchor="w",
            command=lambda: self._app.show_screen("indexes"),
        ).pack(fill="x")

    # ── Embedding Model ────────────────────────────────────────────

    def _build_model_section(self, parent):
        """Embedding model in a GlassCard with AnimatedPulseGlow behind status."""
        # GlassCard with purple-light glow (stagger ≈ 800 ms via parent .after)
        card = self._make_glass_card(
            parent,
            glow_color=COLOR_PURPLE_LIGHT,
            fill="x",
            expand=False,
            pady=(0, PADDING_MD),
        )

        # Section label row with GradientDivider
        label_row = ctk.CTkFrame(card, fg_color="transparent")
        label_row.pack(fill="x", padx=PADDING, pady=(PADDING_MD, PADDING_SM))

        ctk.CTkLabel(
            label_row,
            text="EMBEDDING MODEL",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_PURPLE,
            anchor="w",
        ).pack(side="left")

        GradientDivider(label_row, height=1).pack(
            side="right", fill="x", expand=True, padx=(PADDING_MD, 0),
        )

        # Model status content with AnimatedPulseGlow behind the dot
        status_row = ctk.CTkFrame(card, fg_color="transparent")
        status_row.pack(fill="x", padx=PADDING, pady=(0, PADDING_SM))

        # Static glow behind status indicator (no animation timer)
        status_glow_bg = ctk.CTkFrame(
            status_row, fg_color=COLOR_PURPLE_DEEP,
            width=36, height=36, corner_radius=18,
        )
        status_glow_bg.pack(side="left", padx=(0, PADDING_SM))
        status_glow_bg.pack_propagate(False)

        # Animated status dot — placed on top of the glow
        try:
            self._model_status_dot = PulseIndicator(
                status_glow_bg, color=COLOR_TEXT_DIM, size=10,
            )
            self._model_status_dot.place(
                relx=0.5, rely=0.5, anchor="center",
            )
        except Exception:
            self._model_status_dot = ctk.CTkLabel(
                status_row, text="●", font=(FONT_FAMILY, 12),
                text_color=COLOR_TEXT_DIM, anchor="w",
            )
            self._model_status_dot.pack(side="left", padx=(0, PADDING_SM))

        # Status text
        self._model_status_label = ctk.CTkLabel(
            status_row,
            text="Checking model status…",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        self._model_status_label.pack(side="left", fill="x", expand=True)

        # Model detail line (selectable mono font)
        try:
            self._model_detail_label = make_selectable_label(
                card, text="", font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM, bg_color=COLOR_GLASS_BG,
            )
            self._model_detail_label.pack(fill="x", padx=PADDING, pady=(0, PADDING))
        except Exception:
            self._model_detail_label = ctk.CTkLabel(
                card, text="", font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM, anchor="w",
            )
            self._model_detail_label.pack(fill="x", padx=PADDING, pady=(0, PADDING))

        # Deferred check
        self.after(200, self._check_model_status)

    # ── Data Storage ───────────────────────────────────────────────

    def _build_storage_section(self, parent):
        """Data storage in a GlassCard with GradientDivider."""
        # GlassCard with gold glow (stagger ≈ 1 000 ms via parent .after)
        card = self._make_glass_card(
            parent,
            glow_color=COLOR_GOLD,
            fill="x",
            expand=False,
        )

        # Section label row with GradientDivider
        label_row = ctk.CTkFrame(card, fg_color="transparent")
        label_row.pack(fill="x", padx=PADDING, pady=(PADDING_MD, PADDING_SM))

        ctk.CTkLabel(
            label_row,
            text="DATA STORAGE",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_GOLD,
            anchor="w",
        ).pack(side="left")

        GradientDivider(label_row, height=1).pack(
            side="right", fill="x", expand=True, padx=(PADDING_MD, 0),
        )

        # Storage detail (selectable mono font)
        try:
            self._storage_path_label = make_selectable_label(
                card, text="Calculating…", font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_SECONDARY, bg_color=COLOR_GLASS_BG, wrap="word",
            )
            self._storage_path_label.pack(fill="x", padx=PADDING, pady=(0, PADDING_SM))
        except Exception:
            self._storage_path_label = ctk.CTkLabel(
                card, text="Calculating…", font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_SECONDARY, anchor="w", wraplength=500,
            )
            self._storage_path_label.pack(fill="x", padx=PADDING, pady=(0, PADDING_SM))

        self._storage_size_label = ctk.CTkLabel(
            card,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        )
        self._storage_size_label.pack(fill="x", padx=PADDING, pady=(0, PADDING))

    # ────────────────────────────────────────────────────────────────
    # Data / Refresh  (business logic — completely unchanged)
    # ────────────────────────────────────────────────────────────────

    def _refresh_stats(self):
        """Refresh dashboard statistics from the engine."""
        try:
            stats = self._app.engine.get_system_stats()

            try:
                self._stat_values["indexes"].configure(
                    text=str(stats["index_count"])
                )
            except Exception:
                pass

            try:
                self._stat_values["documents"].configure(
                    text=str(stats["total_documents"])
                )
            except Exception:
                pass

            try:
                self._stat_values["vectors"].configure(
                    text=str(stats["total_vectors"])
                )
            except Exception:
                pass

            try:
                self._stat_values["searches"].configure(
                    text=str(stats["total_searches"])
                )
            except Exception:
                pass

            # Update storage path label (may be a selectable tk.Text widget)
            path_text = str(self._app.engine.data_dir)
            try:
                path_text = str(self._app.engine.data_dir)
                _update_text_widget(self._storage_path_label, path_text)
            except Exception:
                try:
                    self._storage_path_label.configure(text=path_text)
                except Exception:
                    pass

            try:
                size_mb = stats.get("data_size_mb", 0)
                self._storage_size_label.configure(
                    text=f"{size_mb:.1f} MB used"
                )
            except Exception:
                pass

        except Exception:
            pass

    def _check_model_status(self):
        """Check and display embedding model status."""
        try:
            status = self._app.engine.get_model_status()

            if status["loaded"]:
                # Update status dot color
                try:
                    self._model_status_dot.set_color(COLOR_SUCCESS)
                except Exception:
                    try:
                        self._model_status_dot.configure(text_color=COLOR_SUCCESS)
                    except Exception:
                        pass

                try:
                    self._model_status_label.configure(
                        text="Model Ready",
                        text_color=COLOR_SUCCESS,
                    )
                except Exception:
                    pass

                # Update model detail (may be a selectable tk.Text widget)
                detail_text = ""
                try:
                    detail_text = f"{status['model_name']}  ·  {status['dimension']}d  ·  {status['device']}"
                    _update_text_widget(self._model_detail_label, detail_text)
                except Exception:
                    try:
                        self._model_detail_label.configure(
                            text=detail_text,
                            text_color=COLOR_TEXT_SECONDARY,
                        )
                    except Exception:
                        pass
            else:
                # Update status dot color
                try:
                    self._model_status_dot.set_color(COLOR_WARNING)
                except Exception:
                    try:
                        self._model_status_dot.configure(text_color=COLOR_WARNING)
                    except Exception:
                        pass

                try:
                    self._model_status_label.configure(
                        text="Model Not Loaded",
                        text_color=COLOR_WARNING,
                    )
                except Exception:
                    pass

                # Update model detail (may be a selectable tk.Text widget)
                detail_text = ""
                try:
                    detail_text = f"{status['model_name']}  —  Will auto-load on first use"
                    _update_text_widget(self._model_detail_label, detail_text)
                except Exception:
                    try:
                        self._model_detail_label.configure(
                            text=detail_text,
                            text_color=COLOR_TEXT_DIM,
                        )
                    except Exception:
                        pass

        except Exception:
            # Fallback: status unknown
            try:
                self._model_status_dot.set_color(COLOR_TEXT_DIM)
            except Exception:
                try:
                    self._model_status_dot.configure(text_color=COLOR_TEXT_DIM)
                except Exception:
                    pass

            try:
                self._model_status_label.configure(
                    text="Status Unknown",
                    text_color=COLOR_TEXT_DIM,
                )
            except Exception:
                pass

            try:
                _update_text_widget(self._model_detail_label, "")
            except Exception:
                try:
                    self._model_detail_label.configure(text="")
                except Exception:
                    pass
