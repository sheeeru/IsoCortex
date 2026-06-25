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
    FONT_FAMILY, FONT_FAMILY_DISPLAY, FONT_FAMILY_MONO,
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
        self.after(15, lambda: self._build_welcome_banner(content))
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
            font=(FONT_FAMILY_DISPLAY, FONT_SIZE_TITLE, "bold"),
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

    # ── Welcome Banner ─────────────────────────────────────────────

    def _build_welcome_banner(self, parent):
        """Welcome banner with username and system status."""
        from desktop_app.theme import (
            COLOR_PURPLE, COLOR_PURPLE_DEEP, COLOR_GOLD, COLOR_TEXT,
            COLOR_TEXT_SECONDARY, FONT_FAMILY, FONT_SIZE_LARGE, FONT_SIZE_NORMAL,
            PADDING, PADDING_SM, PADDING_MD, BORDER_RADIUS_LG,
            _blend_colors, GradientCanvas
        )
        banner = ctk.CTkFrame(
            parent,
            fg_color=_blend_colors("#000000", COLOR_PURPLE_DEEP, 0.6),
            corner_radius=BORDER_RADIUS_LG,
            border_width=1,
            border_color=_blend_colors("#000000", COLOR_PURPLE, 0.25),
        )
        banner.pack(fill="x", padx=PADDING, pady=(PADDING, PADDING_SM))

        inner = ctk.CTkFrame(banner, fg_color="transparent")
        inner.pack(fill="x", padx=PADDING_MD + 4, pady=PADDING_MD)

        # Get username
        username = "Admin"
        try:
            user = self._app.engine.current_user
            if user:
                username = user.get("username", "Admin").title()
        except Exception:
            pass

        ctk.CTkLabel(
            inner,
            text=f"Welcome back, {username}",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x")

        ctk.CTkLabel(
            inner,
            text="Your knowledge base is ready. Here's what's happening today.",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

    # ── Stat Cards ─────────────────────────────────────────────────

    def _build_stat_cards(self, parent):
        """2×2 grid of stat cards with large numbers."""
        from desktop_app.theme import (
            COLOR_PURPLE, COLOR_GOLD, COLOR_SUCCESS, COLOR_INFO,
            COLOR_BG_CARD, COLOR_TEXT, COLOR_TEXT_SECONDARY,
            FONT_FAMILY, FONT_SIZE_TITLE, FONT_SIZE_NORMAL, FONT_SIZE_SMALL,
            PADDING, PADDING_SM, PADDING_MD, BORDER_RADIUS_LG,
            COLOR_BG_ELEVATED, _blend_colors, COLOR_BORDER
        )
        stat_configs = [
            ("indexes",   "Indexes",   "0", COLOR_PURPLE, "◈"),
            ("documents", "Documents", "0", COLOR_INFO,   "⊞"),
            ("vectors",   "Vectors",   "0", COLOR_GOLD,   "◈"),
            ("searches",  "Searches",  "0", COLOR_SUCCESS, "⊙"),
        ]
        self._stat_values = {}

        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(fill="x", padx=PADDING, pady=(0, PADDING_SM))
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        for i, (key, label, default_val, color, icon) in enumerate(stat_configs):
            row_idx = i // 2
            col_idx = i % 2

            # Outer card frame
            card = ctk.CTkFrame(
                grid,
                fg_color=_blend_colors(COLOR_BG_CARD, color, 0.06),
                corner_radius=BORDER_RADIUS_LG,
                border_width=1,
                border_color=_blend_colors(COLOR_BORDER, color, 0.3),
            )
            card.grid(
                row=row_idx, column=col_idx,
                sticky="nsew",
                padx=(0 if col_idx == 0 else PADDING_SM, PADDING_SM if col_idx == 0 else 0),
                pady=(0 if row_idx == 0 else PADDING_SM, PADDING_SM if row_idx == 0 else 0),
            )
            card.grid_propagate(False)
            card.configure(height=130)

            # Left accent strip
            strip = ctk.CTkFrame(card, width=4, fg_color=color, corner_radius=2)
            strip.pack(side="left", fill="y", padx=(PADDING_SM, 0), pady=PADDING_MD)
            strip.pack_propagate(False)

            # Card content
            content = ctk.CTkFrame(card, fg_color="transparent")
            content.pack(side="left", fill="both", expand=True, padx=PADDING_MD, pady=PADDING_MD)

            # Big number
            val_lbl = ctk.CTkLabel(
                content,
                text=default_val,
                font=(FONT_FAMILY_DISPLAY, FONT_SIZE_TITLE, "bold"),
                text_color=color,
                anchor="w",
            )
            val_lbl.pack(fill="x")
            self._stat_values[key] = val_lbl

            # Label
            ctk.CTkLabel(
                content,
                text=f"{icon}  {label}",
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                text_color=COLOR_TEXT_SECONDARY,
                anchor="w",
            ).pack(fill="x")

    # ── Bottom Panels (Analytics: Recent Activity + Document Breakdown) ──

    def _build_bottom_panels(self, parent):
        """Two-column analytics layout: Recent Activity (left) + Document Breakdown (right)."""
        bottom = ctk.CTkFrame(parent, fg_color="transparent")
        bottom.pack(fill="both", expand=True, pady=(PADDING_SM, 0))
        bottom.grid_columnconfigure(0, weight=55)
        bottom.grid_columnconfigure(1, weight=45)

        # ── Left column: Recent Activity (55%) ─────────────────────
        left_col = ctk.CTkFrame(bottom, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, PADDING_SM))

        self._build_recent_activity(left_col)

        # ── Right column: Document Breakdown (45%) ──────────────────
        right_col = ctk.CTkFrame(bottom, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(PADDING_SM, 0))

        self._build_document_breakdown(right_col)

    def _build_recent_activity(self, parent):
        """Recent Activity panel showing last 10 search queries."""
        fade = FadeInFrame(parent, fg_color=COLOR_BG)
        fade.pack(fill="both", expand=True)

        card = self._make_glass_card(fade, glow_color=COLOR_PURPLE)

        # Section header
        label_row = ctk.CTkFrame(card, fg_color="transparent")
        label_row.pack(fill="x", padx=PADDING, pady=(PADDING_MD, PADDING_SM))

        ctk.CTkLabel(
            label_row,
            text="RECENT ACTIVITY",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_PURPLE,
            anchor="w",
        ).pack(side="left")

        GradientDivider(label_row, height=1).pack(
            side="right", fill="x", expand=True, padx=(PADDING_MD, 0),
        )

        # Scrollable search list
        self._activity_container = ctk.CTkFrame(card, fg_color="transparent")
        self._activity_container.pack(fill="both", expand=True, padx=PADDING, pady=(0, PADDING))

        # Load data after UI is ready
        self.after(150, self._populate_recent_activity)

    def _populate_recent_activity(self):
        """Fetch and display recent searches."""
        # Clear placeholder
        for w in self._activity_container.winfo_children():
            w.destroy()

        try:
            searches = self._app.engine.get_recent_searches(10)
        except Exception:
            searches = []

        if not searches:
            ctk.CTkLabel(
                self._activity_container,
                text="No searches yet",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
            ).pack(fill="x", pady=PADDING_SM)
            return

        for s in searches:
            row = ctk.CTkFrame(self._activity_container, fg_color="transparent")
            row.pack(fill="x", pady=2)

            # Query text (clickable)
            query_text = s.get("query", "")
            if len(query_text) > 60:
                display_query = query_text[:57] + "..."
            else:
                display_query = query_text

            # Time ago
            time_str = self._time_ago(s.get("created_at", ""))

            # Result count
            count = s.get("results_count", 0)

            lbl = ctk.CTkLabel(
                row,
                text=display_query,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT,
                anchor="w",
                cursor="hand2",
            )
            lbl.pack(side="left", fill="x", expand=True)

            # Click to search
            captured_query = query_text
            lbl.bind("<Button-1>", lambda e, q=captured_query: self._run_past_search(q))

            # Hover effects
            lbl.bind("<Enter>", lambda e, l=lbl: l.configure(text_color=COLOR_PURPLE))
            lbl.bind("<Leave>", lambda e, l=lbl: l.configure(text_color=COLOR_TEXT))

            # Meta: count + time
            meta_text = f"{count} results  ·  {time_str}"
            ctk.CTkLabel(
                row,
                text=meta_text,
                font=(FONT_FAMILY, FONT_SIZE_XXS),
                text_color=COLOR_TEXT_DIM,
                anchor="e",
                width=140,
            ).pack(side="right")

    def _run_past_search(self, query: str):
        """Navigate to search screen and run the query."""
        try:
            self._app.show_screen("search")
            # The SearchScreen will be created fresh, so we need to pass
            # the query via a small delay
            self._app.after(300, lambda: self._inject_search_query(query))
        except Exception:
            pass

    def _inject_search_query(self, query: str):
        """Inject a query into the search screen's input field and submit."""
        try:
            screen = self._app._screen_container.winfo_children()[0]
            if hasattr(screen, '_input_entry'):
                screen._input_entry.delete(0, "end")
                screen._input_entry.insert(0, query)
                if hasattr(screen, '_on_send'):
                    screen._on_send()
        except Exception:
            pass

    @staticmethod
    def _time_ago(date_str: str) -> str:
        """Convert an ISO date string to a human-readable 'time ago' format."""
        if not date_str:
            return ""
        try:
            from datetime import datetime, timezone
            # Handle various date formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                return date_str

            # Make timezone-aware if naive
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            diff = now - dt
            seconds = int(diff.total_seconds())

            if seconds < 60:
                return "just now"
            elif seconds < 3600:
                return f"{seconds // 60}m ago"
            elif seconds < 86400:
                return f"{seconds // 3600}h ago"
            elif seconds < 172800:
                return "Yesterday"
            elif seconds < 604800:
                return f"{seconds // 86400}d ago"
            else:
                return dt.strftime("%b %d")
        except Exception:
            return date_str

    def _build_document_breakdown(self, parent):
        """Document Breakdown panel showing file type distribution as colored bars."""
        fade = FadeInFrame(parent, fg_color=COLOR_BG)
        fade.pack(fill="both", expand=True)

        card = self._make_glass_card(fade, glow_color=COLOR_GOLD)

        # Section header
        label_row = ctk.CTkFrame(card, fg_color="transparent")
        label_row.pack(fill="x", padx=PADDING, pady=(PADDING_MD, PADDING_SM))

        ctk.CTkLabel(
            label_row,
            text="DOCUMENT BREAKDOWN",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_GOLD,
            anchor="w",
        ).pack(side="left")

        GradientDivider(label_row, height=1).pack(
            side="right", fill="x", expand=True, padx=(PADDING_MD, 0),
        )

        # Content area
        self._breakdown_container = ctk.CTkFrame(card, fg_color="transparent")
        self._breakdown_container.pack(fill="both", expand=True, padx=PADDING, pady=(0, PADDING))

        self._breakdown_total_label = ctk.CTkLabel(
            self._breakdown_container,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        )
        self._breakdown_total_label.pack(fill="x", pady=(PADDING_SM, 0))

        # Load data after UI is ready
        self.after(150, self._populate_document_breakdown)

    def _populate_document_breakdown(self):
        """Fetch and display document type breakdown."""
        # Clear placeholder
        for w in self._breakdown_container.winfo_children():
            w.destroy()

        try:
            breakdown = self._app.engine.get_document_breakdown()
        except Exception:
            breakdown = {}

        if not breakdown:
            ctk.CTkLabel(
                self._breakdown_container,
                text="No indexed documents",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
            ).pack(fill="x", pady=PADDING_SM)
            return

        # Category display mapping
        category_map = {
            "document": ("Documents", COLOR_PURPLE),
            "code": ("Code", COLOR_SUCCESS),
            "spreadsheet": ("Spreadsheets", COLOR_GOLD),
            "presentation": ("Presentations", COLOR_INFO),
            "text": ("Text", COLOR_TEXT_SECONDARY),
            "email": ("Email", COLOR_WARNING),
            "image": ("Images", "#e07c9a"),
        }

        total_docs = 0
        total_words = 0
        max_count = 0
        category_data = []

        for cat, info in breakdown.items():
            count = info["count"]
            words = info["words"]
            total_docs += count
            total_words += words
            if count > max_count:
                max_count = count
            display_name, color = category_map.get(cat, (cat.title(), COLOR_TEXT_DIM))
            category_data.append((display_name, count, words, color))

        if max_count == 0:
            max_count = 1  # avoid division by zero

        # Sort by count descending
        category_data.sort(key=lambda x: x[1], reverse=True)

        # Build bars
        for display_name, count, words, color in category_data:
            row = ctk.CTkFrame(self._breakdown_container, fg_color="transparent")
            row.pack(fill="x", pady=3)

            # Label
            label_col = ctk.CTkFrame(row, fg_color="transparent", width=100)
            label_col.pack(side="left", fill="y")
            label_col.pack_propagate(False)

            ctk.CTkLabel(
                label_col,
                text=display_name,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_SECONDARY,
                anchor="w",
            ).pack(side="left", pady=(2, 0))

            # Bar area
            bar_area = ctk.CTkFrame(row, fg_color="transparent")
            bar_area.pack(side="left", fill="x", expand=True, padx=(PADDING_SM, PADDING_SM))

            # Background bar
            bg_bar = ctk.CTkFrame(bar_area, height=20, fg_color=COLOR_BG_ELEVATED, corner_radius=4)
            bg_bar.pack(fill="x")
            bg_bar.pack_propagate(False)

            # Filled bar (proportional width)
            bar_width = max(count / max_count, 0.05)  # minimum 5% width for visibility
            # We use place to position the filled bar inside the bg_bar
            filled = ctk.CTkFrame(bg_bar, fg_color=color, corner_radius=4)
            # Calculate pixel width approximation — use a fixed reference
            filled.place(relx=0, rely=0, relwidth=bar_width, relheight=1)

            # Count label
            ctk.CTkLabel(
                row,
                text=str(count),
                font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL, "bold"),
                text_color=COLOR_TEXT,
                width=40,
                anchor="e",
            ).pack(side="right")

        # Total summary at bottom
        total_frame = ctk.CTkFrame(self._breakdown_container, fg_color="transparent")
        total_frame.pack(fill="x", pady=(PADDING_MD, 0))

        ctk.CTkLabel(
            total_frame,
            text=f"{total_docs} files  ·  {total_words:,} words",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(side="left")

    # ────────────────────────────────────────────────────────────────
    # Data / Refresh
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

        # Refresh analytics panels
        try:
            self._populate_recent_activity()
        except Exception:
            pass
        try:
            self._populate_document_breakdown()
        except Exception:
            pass
