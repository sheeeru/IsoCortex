"""
IsoCortex Desktop App — Indexes Screen (Premium Redesign)
==========================================================
Browse, manage, load, unload, and delete search indexes.

Design language:
  - Animated ShimmerBar top accent (purple → gold sweep)
  - AnimatedGradientBG subtle layered radial glow background
  - Page header with purple accent pillar + title + refresh button
  - GradientDivider between header, section label, and index list
  - GlassCard index cards with glassmorphism borders + glow
  - FadeInFrame wrapping each card with staggered entrance delays
  - Health badges via create_badge ("Healthy" / "Issues")
  - Stat tags via create_tag_chip (vectors, deleted, size)
  - Gold "Load" button, dim "Unload" button, red "Delete" button
  - Premium delete confirmation dialog with red accent bar
  - Empty state with large icon and guidance text inside a GlassCard
"""

import customtkinter as ctk

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER,
    COLOR_PURPLE, COLOR_PURPLE_DARK, COLOR_PURPLE_LIGHT,
    COLOR_GOLD, COLOR_GOLD_LIGHT, COLOR_GOLD_BTN_TEXT,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    COLOR_SHADOW, COLOR_SURFACE_1,
    FONT_FAMILY, FONT_FAMILY_DISPLAY, FONT_FAMILY_MONO,
    FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_MEDIUM, FONT_SIZE_NORMAL, FONT_SIZE_SMALL, FONT_SIZE_XXS,
    BORDER_RADIUS, BORDER_RADIUS_SM, BORDER_RADIUS_LG,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    ShimmerBar, GlassCard, GradientDivider, AnimatedGradientBG,
    FadeInFrame, create_badge, create_tag_chip,
    ANIM_DELAY_200, ANIM_DELAY_400, ANIM_DELAY_600,
)


class IndexesScreen(ctk.CTkFrame):
    """Premium screen for browsing and managing search indexes."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app = app

        self._build_ui()
        self.after(150, self._refresh)

    # ────────────────────────────────────────────────────────────────
    # UI Construction
    # ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build all sections of the indexes screen."""

        # ── Subtle animated gradient background ─────────────────────
        self._bg_canvas = AnimatedGradientBG(self)
        self._bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # Outer scrollable content wrapper
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING_LG)

        # ── Animated shimmer accent bar ─────────────────────────────
        self._build_shimmer_bar(content)

        # ── Page header ────────────────────────────────────────────
        self._build_header(content)

        # ── Gradient divider after header ────────────────────────
        GradientDivider(content, height=1).pack(fill="x", pady=(PADDING_SM, PADDING_MD))

        # ── Section label ──────────────────────────────────────────
        self._build_section_label(content)

        # ── Gradient divider before list ─────────────────────────
        GradientDivider(content, height=1).pack(fill="x", pady=(PADDING_SM, PADDING_MD))

        # ── Scrollable index list ────────────────────────────────
        self._list_frame = ctk.CTkScrollableFrame(
            content,
            fg_color="transparent",
        )
        self._list_frame.pack(fill="both", expand=True)

    # ── Shimmer Bar ───────────────────────────────────────────────

    def _build_shimmer_bar(self, parent):
        """Animated shimmer bar at the top (replaces static gradient strip)."""
        ShimmerBar(
            parent,
            height=4,
            colors=[COLOR_BG_ELEVATED, COLOR_PURPLE, COLOR_GOLD, COLOR_BG_ELEVATED],
        ).pack(fill="x", pady=(0, PADDING_LG))

    # ── Page Header ────────────────────────────────────────────────

    def _build_header(self, parent):
        """Purple accent pillar + 'Indexes' title + refresh button."""
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", pady=(0, PADDING))

        # Left: accent + title block
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True)

        accent = ctk.CTkFrame(
            left,
            width=4,
            height=36,
            corner_radius=2,
            fg_color=COLOR_PURPLE,
        )
        accent.pack(side="left", padx=(0, PADDING), pady=4)
        accent.pack_propagate(False)

        title_block = ctk.CTkFrame(left, fg_color="transparent")
        title_block.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            title_block,
            text="Indexes",
            font=(FONT_FAMILY_DISPLAY, FONT_SIZE_TITLE, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_block,
            text="Browse and manage your search indexes",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(anchor="w")

        # Right: refresh button
        refresh_btn = ctk.CTkButton(
            header,
            text="\u21BB  Refresh",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_SECONDARY,
            height=36,
            corner_radius=BORDER_RADIUS_SM,
            width=100,
            command=self._refresh,
        )
        refresh_btn.pack(side="right", padx=(PADDING, 0))

    # ── Section Label ──────────────────────────────────────────────

    def _build_section_label(self, parent):
        """Uppercase 'ALL INDEXES' label in purple."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", pady=(PADDING_SM, 0))

        ctk.CTkLabel(
            section,
            text="ALL INDEXES",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_PURPLE,
            anchor="w",
        ).pack(side="left")

    # ────────────────────────────────────────────────────────────────
    # Data Management
    # ────────────────────────────────────────────────────────────────

    def _refresh(self):
        """Refresh the index list from the engine."""
        for widget in self._list_frame.winfo_children():
            widget.destroy()

        try:
            indexes = self._app.engine.list_indexes()
        except Exception as exc:
            ctk.CTkLabel(
                self._list_frame,
                text=f"Error loading indexes: {exc}",
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                text_color=COLOR_ERROR,
            ).pack(pady=40)
            return

        if not indexes:
            self._show_empty_state()
            return

        for i, idx in enumerate(indexes):
            try:
                delay = (i % 3) * ANIM_DELAY_200  # stagger: 0, 200, 400, 0, 200, 400, …
                self._create_index_card(idx, fade_delay=delay)
            except Exception as exc:
                ctk.CTkLabel(
                    self._list_frame,
                    text=f"Error rendering index '{getattr(idx, 'name', '?')}': {exc}",
                    font=(FONT_FAMILY, FONT_SIZE_SMALL),
                    text_color=COLOR_ERROR,
                ).pack(fill="x", pady=2)

    # ── Empty State ────────────────────────────────────────────────

    def _show_empty_state(self):
        """Premium empty state when no indexes exist — using GlassCard."""
        glass = GlassCard(self._list_frame, glow_color=COLOR_PURPLE)
        glass.pack(fill="x", padx=20, pady=PADDING_LG)

        inner = ctk.CTkFrame(glass, fg_color="transparent")
        inner.pack(fill="x", padx=PADDING_LG, pady=PADDING_XL)

        # Large icon
        ctk.CTkLabel(
            inner,
            text="\u25A6",
            font=(FONT_FAMILY, 40),
            text_color=COLOR_TEXT_DIM,
        ).pack(pady=(0, PADDING))

        # Title text
        ctk.CTkLabel(
            inner,
            text="No indexes yet",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_TEXT_SECONDARY,
        ).pack(pady=(0, PADDING_SM))

        # Guidance text
        ctk.CTkLabel(
            inner,
            text="Go to Upload to create your first index",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
        ).pack(pady=(0, 0))

        # Go-to-upload button
        go_btn = ctk.CTkButton(
            inner,
            text="\u2295  Go to Upload",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=COLOR_GOLD,
            hover_color=COLOR_GOLD_LIGHT,
            text_color=COLOR_GOLD_BTN_TEXT,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
            width=160,
            command=lambda: self._app.show_screen("upload"),
        )
        go_btn.pack(pady=(PADDING, 0))

    # ── Index Card ─────────────────────────────────────────────────

    def _create_index_card(self, idx, fade_delay=0):
        """Build a GlassCard with FadeInFrame for a single index."""
        # FadeInFrame wrapper for staggered entrance animation
        fade_wrapper = FadeInFrame(
            self._list_frame,
            delay=fade_delay,
            fg_color="transparent",
        )
        fade_wrapper.pack(fill="x", pady=(0, PADDING_SM))

        # GlassCard (replaces shadow + CTkFrame combo)
        card = GlassCard(fade_wrapper, glow_color=COLOR_PURPLE)
        card.pack(fill="x")

        # Inner content wrapper
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=PADDING_LG, pady=(PADDING, PADDING_MD))

        # ── Top row: index name + health badge ─────────────────────
        top_row = ctk.CTkFrame(inner, fg_color="transparent")
        top_row.pack(fill="x")

        ctk.CTkLabel(
            top_row,
            text=idx.name,
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"),
            text_color=COLOR_PURPLE,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        # Health badge via create_badge
        if idx.healthy:
            create_badge(top_row, "Healthy", color=COLOR_SUCCESS)
        else:
            create_badge(top_row, "Issues", color=COLOR_WARNING)

        # ── Description ────────────────────────────────────────────
        if idx.description:
            desc_label = ctk.CTkLabel(
                inner,
                text=idx.description,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
                wraplength=620,
            )
            desc_label.pack(fill="x", pady=(4, PADDING_SM))

        # ── Stats row with tag chips ──────────────────────────────
        stats_row = ctk.CTkFrame(inner, fg_color="transparent")
        stats_row.pack(fill="x", pady=(0, 0))

        # Vectors count tag chip
        create_tag_chip(stats_row, f"\u2B21 {idx.vector_count:,} vectors").pack(
            side="left", padx=(0, PADDING_SM),
        )

        # Deleted count tag chip
        create_tag_chip(stats_row, f"\u2715 {idx.deleted_count:,} deleted").pack(
            side="left", padx=(0, PADDING_SM),
        )

        # Size tag chip
        size_text = f"{idx.index_size_mb:.1f} MB" if idx.index_size_mb > 0 else "0 MB"
        create_tag_chip(stats_row, size_text).pack(
            side="left",
        )

        # Created date (right-aligned, dim)
        date_str = idx.created_at[:10] if idx.created_at else "N/A"
        ctk.CTkLabel(
            stats_row,
            text=f"Created {date_str}",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="e",
        ).pack(side="right")

        # ── Gradient divider between stats and actions ─────────────
        GradientDivider(inner, height=1).pack(fill="x", pady=(PADDING_SM, PADDING_SM))

        # ── Action buttons row ─────────────────────────────────────
        actions = ctk.CTkFrame(inner, fg_color="transparent")
        actions.pack(fill="x")

        # Load button — elevated bg, gold text, bold
        ctk.CTkButton(
            actions,
            text="\u25B6  Load",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_GOLD,
            height=32,
            corner_radius=BORDER_RADIUS_SM,
            width=80,
            command=lambda n=idx.name: self._load_index(n),
        ).pack(side="left", padx=(0, PADDING_SM))

        # Unload button — elevated bg, dim text
        ctk.CTkButton(
            actions,
            text="\u25A0  Unload",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_DIM,
            height=32,
            corner_radius=BORDER_RADIUS_SM,
            width=90,
            command=lambda n=idx.name: self._unload_index(n),
        ).pack(side="left", padx=(0, PADDING_SM))

        # Delete button — right-aligned, elevated bg, red text
        ctk.CTkButton(
            actions,
            text="\u2716  Delete",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=COLOR_BG_ELEVATED,
            hover_color="#3a1515",
            text_color=COLOR_ERROR,
            height=32,
            corner_radius=BORDER_RADIUS_SM,
            width=86,
            command=lambda n=idx.name: self._delete_index(n),
        ).pack(side="right")

    # ────────────────────────────────────────────────────────────────
    # Index Actions
    # ────────────────────────────────────────────────────────────────

    def _load_index(self, name: str):
        """Load an index into memory and show success toast."""
        try:
            self._app.engine.load_index(name)
            self._app.show_toast(f"Index '{name}' loaded into memory", "success")
            self.after(300, self._refresh)
        except Exception as exc:
            self._app.show_toast(f"Error loading index: {exc}", "error")

    def _unload_index(self, name: str):
        """Unload an index from memory and show info toast."""
        try:
            self._app.engine.unload_index(name)
            self._app.show_toast(f"Index '{name}' unloaded", "info")
            self.after(300, self._refresh)
        except Exception as exc:
            self._app.show_toast(f"Error unloading index: {exc}", "error")

    def _delete_index(self, name: str):
        """Show a premium delete confirmation dialog, then delete if confirmed."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Delete Index")
        dialog.geometry("460x260")
        try:
            dialog.configure(fg_color=COLOR_BG)
        except Exception:
            pass
        dialog.transient(self)
        dialog.grab_set()
        # Center on parent
        dialog.update_idletasks()
        try:
            x = self.winfo_rootx() + (self.winfo_width() - 460) // 2
            y = self.winfo_rooty() + (self.winfo_height() - 260) // 2
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        # ── Red accent bar at top ──────────────────────────────────
        red_bar = ctk.CTkFrame(dialog, height=3, fg_color=COLOR_ERROR, corner_radius=0)
        red_bar.pack(fill="x")
        red_bar.pack_propagate(False)

        # ── Dialog header ──────────────────────────────────────────
        dlg_header = ctk.CTkFrame(dialog, fg_color="transparent")
        dlg_header.pack(fill="x", padx=PADDING_LG, pady=(PADDING_LG, PADDING_SM))

        # Red accent pillar
        red_pillar = ctk.CTkFrame(
            dlg_header,
            width=4,
            height=28,
            corner_radius=2,
            fg_color=COLOR_ERROR,
        )
        red_pillar.pack(side="left", padx=(0, PADDING))
        red_pillar.pack_propagate(False)

        ctk.CTkLabel(
            dlg_header,
            text="Delete Index",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_ERROR,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        # ── Warning text ───────────────────────────────────────────
        body = ctk.CTkFrame(dialog, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=PADDING_LG)

        # Warning icon + name
        warn_row = ctk.CTkFrame(body, fg_color="transparent")
        warn_row.pack(fill="x", pady=(PADDING_SM, PADDING_SM))

        ctk.CTkLabel(
            warn_row,
            text="\u26A0",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
            text_color=COLOR_WARNING,
            anchor="w",
        ).pack(side="left", padx=(0, PADDING_SM))

        ctk.CTkLabel(
            warn_row,
            text=f'"{name}" will be permanently removed.',
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            body,
            text="This action cannot be undone. All indexed vectors and "
                 "associated documents for this index will be deleted "
                 "from disk and memory.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
            wraplength=400,
            justify="left",
        ).pack(fill="x", pady=(0, PADDING_LG))

        # ── Buttons ────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.pack(fill="x")

        def do_cancel():
            dialog.destroy()

        def do_delete():
            dialog.destroy()
            try:
                self._app.engine.delete_index(name)
                self.after(200, self._refresh)
                self._app.show_toast(f"Index '{name}' deleted", "warning")
            except FileNotFoundError:
                self._app.show_toast(f"Index '{name}' not found", "error")
            except Exception as exc:
                self._app.show_toast(f"Error: {exc}", "error")

        # Cancel button
        ctk.CTkButton(
            btn_row,
            text="Cancel",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
            command=do_cancel,
        ).pack(side="left", fill="x", expand=True, padx=(0, PADDING_SM))

        # Delete button — red bg
        ctk.CTkButton(
            btn_row,
            text="\u2716  Delete",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            text_color="#fca5a5",
            height=40,
            corner_radius=BORDER_RADIUS_SM,
            width=120,
            command=do_delete,
        ).pack(side="right")
