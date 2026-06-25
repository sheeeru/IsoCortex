"""
IsoCortex Desktop App — Settings Screen (Premium Redesign)
===========================================================
Application settings with a polished, depth-rich aesthetic.

Sections:
  Appearance        — Theme toggle (dark / light)
  System Info       — Version, data directory, embedding model, HNSW params
  Index Management  — Index file breakdown
  Security          — Security guarantees checklist
  Change Password   — Current / new / confirm entry fields
  Danger Zone       — Nuclear-reset with confirmation dialog

Design language:
  - ShimmerBar animated accent at page top
  - GlassCard containers with glassmorphism effect
  - GradientDivider between major sections
  - FadeInFrame staggered entrance on every section
  - AnimatedPulseGlow behind the theme toggle
  - create_badge pills for security items & system-info labels
  - AnimatedGradientBG subtle living background
"""

import customtkinter as ctk
import logging
import threading
import tkinter as tk

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER,
    COLOR_PURPLE, COLOR_PURPLE_DARK, COLOR_PURPLE_LIGHT, COLOR_PURPLE_DEEP,
    COLOR_GOLD, COLOR_GOLD_LIGHT, COLOR_GOLD_BTN_TEXT,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    COLOR_SHADOW, COLOR_SURFACE_1,
    FONT_FAMILY, FONT_FAMILY_DISPLAY, FONT_FAMILY_MONO,
    FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_MEDIUM, FONT_SIZE_NORMAL, FONT_SIZE_SMALL, FONT_SIZE_XXS,
    BORDER_RADIUS, BORDER_RADIUS_SM, BORDER_RADIUS_LG,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    ThemeMode,
    ShimmerBar, GlassCard, GradientDivider, AnimatedGradientBG,
    FadeInFrame, create_badge,
    ANIM_DELAY_200, ANIM_DELAY_400, ANIM_DELAY_600, ANIM_DELAY_800,
)
from desktop_app.workers import WorkerThread


class SettingsScreen(ctk.CTkFrame):
    """Premium settings and configuration screen."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app = app

        self._build_ui()
        self.after(120, self._load_settings)

    # ────────────────────────────────────────────────────────────────
    # Reusable helpers
    # ────────────────────────────────────────────────────────────────

    def _section_header(self, parent, title: str):
        """Uppercase XXS label in purple + thin separator line."""
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", pady=(PADDING_LG, PADDING_SM))

        ctk.CTkLabel(
            header,
            text=title.upper(),
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_PURPLE,
            anchor="w",
        ).pack(side="left")

        sep = ctk.CTkFrame(header, height=1, fg_color=COLOR_BORDER)
        sep.pack(side="right", fill="x", expand=True, padx=(PADDING_MD, 0))
        sep.pack_propagate(False)

    def _content_card(self, parent, glow_color=None, **card_kw):
        """GlassCard-based content card with inner transparent frame.

        Returns the *inner* transparent frame (pack your widgets there).
        """
        card = GlassCard(
            parent,
            glow_color=glow_color,
            **card_kw,
        )
        card.pack(fill="x", pady=(0, PADDING))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=PADDING_LG, pady=PADDING_LG)

        return inner

    # ────────────────────────────────────────────────────────────────
    # UI Construction
    # ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build the full settings page."""

        # ── Animated gradient background ────────────────────────────
        self._bg_canvas = AnimatedGradientBG(self)
        self._bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # ── Scrollable content area ──────────────────────────────────
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING_LG)

        # Top animated shimmer bar (purple → gold accent)
        ShimmerBar(
            scroll, height=4,
            colors=[COLOR_BG_ELEVATED, COLOR_PURPLE, COLOR_GOLD, COLOR_BG_ELEVATED],
        ).pack(fill="x", pady=(0, PADDING_LG))

        # ── Page header ──────────────────────────────────────────────
        header = ctk.CTkFrame(scroll, fg_color="transparent")
        header.pack(fill="x", pady=(0, PADDING_MD))

        accent_bar = ctk.CTkFrame(
            header,
            width=4,
            height=32,
            corner_radius=2,
            fg_color=COLOR_PURPLE,
        )
        accent_bar.pack(side="left", padx=(0, PADDING), pady=6)
        accent_bar.pack_propagate(False)

        title_block = ctk.CTkFrame(header, fg_color="transparent")
        title_block.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            title_block,
            text="Settings",
            font=(FONT_FAMILY_DISPLAY, FONT_SIZE_TITLE, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_block,
            text="Configure your IsoCortex experience",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(anchor="w")

        # ════════════════════════════════════════════════════════════
        # 1 · APPEARANCE  (FadeInFrame delay=ANIM_DELAY_200)
        # ════════════════════════════════════════════════════════════
        appearance_fade = FadeInFrame(
            scroll, fg_color="transparent", delay=ANIM_DELAY_200,
        )
        appearance_fade.pack(fill="x")

        self._section_header(appearance_fade, "Appearance")

        appearance_inner = self._content_card(appearance_fade, glow_color=COLOR_PURPLE)

        # Purple accent strip at top of card content
        accent_strip = ctk.CTkFrame(appearance_inner, height=3, fg_color=COLOR_PURPLE, corner_radius=0)
        accent_strip.pack(fill="x", pady=(0, PADDING_MD))
        accent_strip.pack_propagate(False)

        # Theme label row with AnimatedPulseGlow behind it
        theme_label_row = ctk.CTkFrame(appearance_inner, fg_color="transparent")
        theme_label_row.pack(fill="x", pady=(0, PADDING_SM))

        # Static glow behind the theme toggle area (no animation timer)
        glow_container = ctk.CTkFrame(
            theme_label_row, fg_color=COLOR_PURPLE_DEEP,
            width=48, height=48, corner_radius=24,
        )
        glow_container.pack(side="left", padx=(0, PADDING_SM))
        glow_container.pack_propagate(False)

        ctk.CTkLabel(
            glow_container,
            text="T",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_PURPLE_LIGHT,
        ).place(relx=0.5, rely=0.5, anchor="center")

        title_sub = ctk.CTkFrame(theme_label_row, fg_color="transparent")
        title_sub.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            title_sub,
            text="Theme Mode",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM),
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(fill="x")

        self._mode_label = ctk.CTkLabel(
            title_sub,
            text="Dark mode is the only supported theme.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        self._mode_label.pack(fill="x")

        self._theme_segment = ctk.CTkSegmentedButton(
            appearance_inner,
            values=["Dark"],
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            height=40,
            corner_radius=BORDER_RADIUS_SM,
        )
        self._theme_segment.pack(fill="x", pady=(PADDING_SM, 0))
        self._theme_segment.set("Dark")

        ctk.CTkLabel(
            appearance_inner,
            text="Light mode is under development and will be available in a future update.",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
            wraplength=560,
        ).pack(fill="x", pady=(PADDING_SM, 0))

        # GradientDivider after Appearance
        GradientDivider(appearance_fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

        # ════════════════════════════════════════════════════════════
        # 2 · SYSTEM INFORMATION  (FadeInFrame delay=ANIM_DELAY_400)
        # ════════════════════════════════════════════════════════════
        sysinfo_fade = FadeInFrame(
            scroll, fg_color="transparent", delay=ANIM_DELAY_400,
        )
        sysinfo_fade.pack(fill="x")

        self._section_header(sysinfo_fade, "System Information")

        info_inner = self._content_card(sysinfo_fade, glow_color=COLOR_GOLD)

        # Gold accent strip
        info_accent = ctk.CTkFrame(info_inner, height=3, fg_color=COLOR_GOLD, corner_radius=0)
        info_accent.pack(fill="x", pady=(0, PADDING_MD))
        info_accent.pack_propagate(False)

        self._settings_labels: dict[str, ctk.CTkLabel] = {}

        info_rows = [
            ("version",       "Version"),
            ("data_dir",      "Data Directory"),
            ("model",         "Embedding Model"),
            ("dimension",     "Vector Dimension"),
            ("hnsw_m",        "HNSW M"),
            ("hnsw_efc",      "HNSW ef_construction"),
            ("hnsw_efs",      "HNSW ef_search"),
            ("chunk_size",    "Chunk Size"),
            ("chunk_overlap", "Chunk Overlap"),
        ]

        row_count = len(info_rows)
        for idx, (key, label) in enumerate(info_rows):
            row = ctk.CTkFrame(info_inner, fg_color="transparent")
            row.pack(fill="x", pady=3)

            ctk.CTkLabel(
                row,
                text=label,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
                width=200,
            ).pack(side="left")

            val_label = ctk.CTkLabel(
                row,
                text="Loading…",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_SECONDARY,
                anchor="w",
            )
            val_label.pack(side="left", fill="x")
            self._settings_labels[key] = val_label

            # Thin separator between rows — skip after the last one
            if idx < row_count - 1:
                sep = ctk.CTkFrame(info_inner, height=1, fg_color=COLOR_BORDER)
                sep.pack(fill="x")
                sep.pack_propagate(False)

        # OCR status row
        try:
            from desktop_app.ocr import get_ocr_status
            ocr_st = get_ocr_status()
        except Exception:
            ocr_st = {"available": False, "install_cmd": ""}

        ocr_row = ctk.CTkFrame(info_inner, fg_color=COLOR_BG_ELEVATED, corner_radius=BORDER_RADIUS_SM)
        ocr_row.pack(fill="x", pady=3)

        ocr_row_content = ctk.CTkFrame(ocr_row, fg_color="transparent")
        ocr_row_content.pack(fill="x", padx=PADDING_MD, pady=PADDING_SM)

        if ocr_st["available"]:
            ocr_dot = ctk.CTkLabel(ocr_row_content, text="\u25cf", font=(FONT_FAMILY, FONT_SIZE_SMALL),
                                    text_color=COLOR_SUCCESS, width=20)
            ocr_dot.pack(side="left")
            ocr_text = "OCR: Active \u2014 image indexing enabled"
            ocr_color = COLOR_SUCCESS
        else:
            ocr_dot = ctk.CTkLabel(ocr_row_content, text="\u25cf", font=(FONT_FAMILY, FONT_SIZE_SMALL),
                                    text_color=COLOR_WARNING, width=20)
            ocr_dot.pack(side="left")
            ocr_text = "OCR: Not installed \u2014 images will be skipped"
            ocr_color = COLOR_WARNING

        ctk.CTkLabel(
            ocr_row_content, text=ocr_text,
            font=(FONT_FAMILY, FONT_SIZE_SMALL), text_color=ocr_color, anchor="w",
        ).pack(side="left")

        if not ocr_st["available"] and ocr_st["install_cmd"]:
            def _show_ocr_help():
                dialog = ctk.CTkToplevel(self)
                dialog.title("Install OCR")
                dialog.geometry("440x180")
                try:
                    dialog.configure(fg_color=COLOR_BG)
                except Exception:
                    pass
                dialog.transient(self)
                dialog.grab_set()

                content = ctk.CTkFrame(dialog, fg_color="transparent")
                content.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING_LG)

                ctk.CTkLabel(
                    content,
                    text="Install Tesseract OCR",
                    font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
                    text_color=COLOR_WARNING, anchor="w",
                ).pack(fill="x", pady=(0, PADDING))

                ctk.CTkLabel(
                    content,
                    text=f"Run this command to enable image indexing:\n\n{ocr_st['install_cmd']}",
                    font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL),
                    text_color=COLOR_TEXT_SECONDARY, anchor="w", justify="left",
                ).pack(fill="x", pady=(0, PADDING_LG))

                ctk.CTkButton(
                    content, text="Close",
                    font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                    fg_color=COLOR_BG_ELEVATED, hover_color=COLOR_BG_HOVER,
                    text_color=COLOR_TEXT, height=36, corner_radius=BORDER_RADIUS_SM,
                    command=dialog.destroy,
                ).pack(anchor="e")

            help_btn = ctk.CTkButton(
                ocr_row_content, text="?",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                fg_color=COLOR_WARNING, hover_color="#d4a017",
                text_color="#1a1a2e", width=24, height=24,
                corner_radius=12, command=_show_ocr_help,
            )
            help_btn.pack(side="right")

        # GradientDivider after System Information
        GradientDivider(sysinfo_fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

        # ════════════════════════════════════════════════════════════
        # 3 · INDEX MANAGEMENT  (FadeInFrame delay=ANIM_DELAY_600)
        # ════════════════════════════════════════════════════════════
        index_fade = FadeInFrame(
            scroll, fg_color="transparent", delay=ANIM_DELAY_600,
        )
        index_fade.pack(fill="x")

        self._section_header(index_fade, "Index Management")

        idx_inner = self._content_card(index_fade, glow_color=COLOR_PURPLE_LIGHT)

        ctk.CTkLabel(
            idx_inner,
            text="Vector indexes are stored locally in ~/.isocortex/indices/  and consist of the following files:",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
            wraplength=580,
        ).pack(fill="x", pady=(0, PADDING))

        file_items = [
            ("index_info.json", "Index metadata — dimension, model name, creation timestamp"),
            ("vectors.bin",     "Raw embedding vectors stored as float32 binary"),
            ("metadata.json",   "Chunk text, source references, and document mappings"),
        ]

        for filename, description in file_items:
            item_row = ctk.CTkFrame(
                idx_inner,
                fg_color=COLOR_BG_ELEVATED,
                corner_radius=BORDER_RADIUS_SM,
            )
            item_row.pack(fill="x", pady=3)

            item_content = ctk.CTkFrame(item_row, fg_color="transparent")
            item_content.pack(fill="x", padx=PADDING_MD, pady=PADDING_SM)

            ctk.CTkLabel(
                item_content,
                text=filename,
                font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL),
                text_color=COLOR_GOLD,
                anchor="w",
            ).pack(side="left")

            ctk.CTkLabel(
                item_content,
                text=f"  —  {description}",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
                wraplength=400,
            ).pack(side="left", fill="x")

        # ── Index list with Re-index buttons ─────────────────────
        self._idx_reindex_status = ctk.CTkLabel(
            idx_inner, text="",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
        )
        self._idx_reindex_status.pack(fill="x", pady=(PADDING_SM, 0))

        self._index_list_frame = ctk.CTkFrame(idx_inner, fg_color="transparent")
        self._index_list_frame.pack(fill="x", pady=(PADDING_SM, 0))

        self.after(200, self._refresh_index_list)

        # GradientDivider after Index Management
        GradientDivider(index_fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

        # ════════════════════════════════════════════════════════════
        # 4 · WATCH FOLDERS  (FadeInFrame delay=ANIM_DELAY_800)
        # ════════════════════════════════════════════════════════════
        watch_fade = FadeInFrame(
            scroll, fg_color="transparent", delay=ANIM_DELAY_800,
        )
        watch_fade.pack(fill="x")

        self._section_header(watch_fade, "Watch Folders")

        watch_inner = self._content_card(watch_fade, glow_color=COLOR_GOLD)

        ctk.CTkLabel(
            watch_inner,
            text="Automatically index new files when they appear in watched folders.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
            wraplength=580,
        ).pack(fill="x", pady=(0, PADDING))

        self._watch_list_frame = ctk.CTkFrame(watch_inner, fg_color="transparent")
        self._watch_list_frame.pack(fill="x")
        self._no_watch_label = ctk.CTkLabel(
            self._watch_list_frame,
            text="No watch folders configured",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        self._no_watch_label.pack(fill="x", pady=(0, PADDING_SM))

        watch_btn_row = ctk.CTkFrame(watch_inner, fg_color="transparent")
        watch_btn_row.pack(fill="x", pady=(PADDING_SM, 0))

        ctk.CTkButton(
            watch_btn_row,
            text="+ Add Folder",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color=COLOR_GOLD, hover_color=COLOR_GOLD_LIGHT,
            text_color=COLOR_GOLD_BTN_TEXT,
            height=32, width=120,
            corner_radius=BORDER_RADIUS_SM,
            command=self._add_watch_folder,
        ).pack(side="left")

        self._refresh_watch_list()

        # GradientDivider after Watch Folders
        GradientDivider(watch_fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

        # ════════════════════════════════════════════════════════════
        # 4b · EXCLUSION RULES  (FadeInFrame delay=ANIM_DELAY_800)
        # ════════════════════════════════════════════════════════════
        excl_fade = FadeInFrame(
            scroll, fg_color="transparent", delay=ANIM_DELAY_800,
        )
        excl_fade.pack(fill="x")

        self._section_header(excl_fade, "Exclusion Rules")

        excl_inner = self._content_card(excl_fade, glow_color=COLOR_WARNING)

        ctk.CTkLabel(
            excl_inner,
            text="Files and folders matching these patterns will be skipped during indexing and folder watching.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
            wraplength=580,
        ).pack(fill="x", pady=(0, PADDING))

        try:
            current_patterns = self._app.engine.get_exclusion_patterns()
        except Exception:
            current_patterns = []

        self._excl_text_widget = tk.Text(
            excl_inner,
            height=8,
            font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL),
            fg=COLOR_TEXT,
            bg=COLOR_BG_ELEVATED,
            insertbackground=COLOR_TEXT,
            selectbackground=COLOR_PURPLE,
            selectforeground=COLOR_TEXT,
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            highlightcolor=COLOR_PURPLE,
            wrap="word",
            padx=PADDING_SM,
            pady=PADDING_SM,
        )
        self._excl_text_widget.pack(fill="x", pady=(0, PADDING_SM))
        if current_patterns:
            self._excl_text_widget.insert("1.0", "\n".join(current_patterns))

        ctk.CTkLabel(
            excl_inner,
            text="One glob pattern per line. Examples: .git, node_modules, *.tmp, **/.env",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(fill="x", pady=(0, PADDING))

        excl_btn_row = ctk.CTkFrame(excl_inner, fg_color="transparent")
        excl_btn_row.pack(fill="x", pady=(PADDING_SM, 0))

        ctk.CTkButton(
            excl_btn_row,
            text="Save Rules",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color=COLOR_PURPLE, hover_color=COLOR_PURPLE_DARK,
            text_color=COLOR_TEXT,
            height=32, width=120,
            corner_radius=BORDER_RADIUS_SM,
            command=self._save_exclusion_rules,
        ).pack(side="left")

        ctk.CTkButton(
            excl_btn_row,
            text="Reset to Defaults",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color="transparent",
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_SECONDARY,
            height=32, width=140,
            corner_radius=BORDER_RADIUS_SM,
            border_width=1, border_color=COLOR_BORDER,
            command=self._reset_exclusion_rules,
        ).pack(side="left", padx=(PADDING_SM, 0))

        self._excl_status_label = ctk.CTkLabel(
            excl_inner, text="",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        self._excl_status_label.pack(fill="x", pady=(PADDING_SM, 0))

        # GradientDivider after Exclusion Rules
        GradientDivider(excl_fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

        # ════════════════════════════════════════════════════════════
        # 5 · SECURITY  (FadeInFrame delay=ANIM_DELAY_800)
        # ════════════════════════════════════════════════════════════
        security_fade = FadeInFrame(
            scroll, fg_color="transparent", delay=ANIM_DELAY_800,
        )
        security_fade.pack(fill="x")

        self._section_header(security_fade, "Security")

        sec_inner = self._content_card(security_fade, glow_color=COLOR_SUCCESS)

        # Green accent strip
        sec_accent = ctk.CTkFrame(sec_inner, height=3, fg_color=COLOR_SUCCESS, corner_radius=0)
        sec_accent.pack(fill="x", pady=(0, PADDING_MD))
        sec_accent.pack_propagate(False)

        ctk.CTkLabel(
            sec_inner,
            text="IsoCortex is designed with a security-first, local-first philosophy.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
            wraplength=580,
        ).pack(fill="x", pady=(0, PADDING))

        security_items = [
            "Passwords hashed with bcrypt (12 rounds) — never stored in plain text",
            "All data stored locally on your machine — nothing is sent to external servers",
            "JWT tokens with short expiry used for secure session management",
            "Account lockout after 5 failed login attempts with a 30-minute cooldown",
        ]

        security_badge_labels = [
            "bcrypt 12r", "local-first", "JWT short-lived", "lockout 5/30m",
        ]

        for item, badge_text in zip(security_items, security_badge_labels):
            item_row = ctk.CTkFrame(sec_inner, fg_color="transparent")
            item_row.pack(fill="x", pady=3)

            # Checkmark
            checkmark = ctk.CTkLabel(
                item_row,
                text="✓",
                font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
                text_color=COLOR_SUCCESS,
                width=20,
            )
            checkmark.pack(side="left")

            ctk.CTkLabel(
                item_row,
                text=item,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_SECONDARY,
                anchor="w",
                wraplength=460,
            ).pack(side="left", fill="x")

            # Badge pill for the security feature tag
            create_badge(
                item_row,
                text=badge_text,
                color=COLOR_SUCCESS,
            ).pack(side="right", padx=(PADDING_SM, 0))

        # GradientDivider after Security
        GradientDivider(security_fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

        # ════════════════════════════════════════════════════════════
        # 6 · CHANGE PASSWORD  (FadeInFrame delay=ANIM_DELAY_800)
        # ════════════════════════════════════════════════════════════
        password_fade = FadeInFrame(
            scroll, fg_color="transparent", delay=ANIM_DELAY_800,
        )
        password_fade.pack(fill="x")

        self._section_header(password_fade, "Change Password")

        pw_inner = self._content_card(password_fade, glow_color=COLOR_PURPLE)

        entry_opts = dict(
            height=40,
            border_width=1,
            border_color=COLOR_BORDER,
            fg_color=COLOR_BG_ELEVATED,
            text_color=COLOR_TEXT,
            placeholder_text_color=COLOR_TEXT_DIM,
            corner_radius=BORDER_RADIUS_SM,
        )

        # Current Password
        ctk.CTkLabel(
            pw_inner,
            text="Current Password",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, PADDING_SM))

        self._current_pw_entry = ctk.CTkEntry(
            pw_inner,
            placeholder_text="Enter current password",
            show="*",
            **entry_opts,
        )
        self._current_pw_entry.pack(fill="x", pady=(0, PADDING))

        # New Password
        ctk.CTkLabel(
            pw_inner,
            text="New Password",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, PADDING_SM))

        self._new_pw_entry = ctk.CTkEntry(
            pw_inner,
            placeholder_text="Min 8 chars — A-Z, a-z, 0-9, !@#$…",
            show="*",
            **entry_opts,
        )
        self._new_pw_entry.pack(fill="x", pady=(0, PADDING))

        # Confirm New Password
        ctk.CTkLabel(
            pw_inner,
            text="Confirm New Password",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, PADDING_SM))

        self._confirm_pw_entry = ctk.CTkEntry(
            pw_inner,
            placeholder_text="Re-enter new password",
            show="*",
            **entry_opts,
        )
        self._confirm_pw_entry.pack(fill="x", pady=(0, PADDING_SM))

        # Error / success label
        self._pw_error_label = ctk.CTkLabel(
            pw_inner,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_ERROR,
            wraplength=460,
            anchor="w",
        )
        self._pw_error_label.pack(fill="x", pady=(0, PADDING))

        # Update Password button
        ctk.CTkButton(
            pw_inner,
            text="Update Password",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_DARK,
            text_color=COLOR_TEXT,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
            command=self._handle_change_password,
        ).pack(fill="x")

        # GradientDivider after Change Password
        GradientDivider(password_fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

        # ════════════════════════════════════════════════════════════
        # 7 · PLUGINS
        # ════════════════════════════════════════════════════════════
        plugins_fade = FadeInFrame(
            scroll, fg_color="transparent", delay=ANIM_DELAY_200 * 5,
        )
        plugins_fade.pack(fill="x")

        self._section_header(plugins_fade, "Plugins")

        plugins_inner = self._content_card(plugins_fade, glow_color=COLOR_GOLD)

        ctk.CTkLabel(
            plugins_inner,
            text="Extend IsoCortex with custom Python plugins.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, PADDING_SM))

        # Plugin list
        self._plugins_container = ctk.CTkFrame(plugins_inner, fg_color="transparent")
        self._plugins_container.pack(fill="x")

        self._refresh_plugins_list()

        # GradientDivider after Plugins
        GradientDivider(plugins_fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

        # ════════════════════════════════════════════════════════════
        # 7 · DANGER ZONE  (FadeInFrame delay=ANIM_DELAY_800)
        # ════════════════════════════════════════════════════════════
        danger_fade = FadeInFrame(
            scroll, fg_color="transparent", delay=ANIM_DELAY_800,
        )
        danger_fade.pack(fill="x")

        self._section_header(danger_fade, "Danger Zone")

        danger_inner = self._content_card(
            danger_fade,
            glow_color=COLOR_ERROR,
        )

        # Top red accent bar
        red_accent = ctk.CTkFrame(danger_inner, height=3, fg_color=COLOR_ERROR, corner_radius=0)
        red_accent.pack(fill="x", pady=(0, PADDING_MD))
        red_accent.pack_propagate(False)

        # Title row with warning icon
        danger_title_row = ctk.CTkFrame(danger_inner, fg_color="transparent")
        danger_title_row.pack(fill="x", pady=(0, PADDING_SM))

        ctk.CTkLabel(
            danger_title_row,
            text="!",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_WARNING,
            anchor="w",
            width=24,
        ).pack(side="left")

        ctk.CTkLabel(
            danger_title_row,
            text="Reset All Data",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"),
            text_color=COLOR_WARNING,
            anchor="w",
        ).pack(side="left")

        # Danger badge
        create_badge(
            danger_title_row,
            text="IRREVERSIBLE",
            color=COLOR_ERROR,
        ).pack(side="right")

        ctk.CTkLabel(
            danger_inner,
            text="This will permanently delete all indexes, documents, embeddings, and account data. "
                 "This action is irreversible — proceed with extreme caution.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
            wraplength=540,
        ).pack(fill="x", pady=(0, PADDING))

        ctk.CTkButton(
            danger_inner,
            text="Reset All Data",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color="#3a1515",
            hover_color="#5a2020",
            text_color=COLOR_ERROR,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
            command=self._handle_reset,
        ).pack(fill="x")

    # ────────────────────────────────────────────────────────────────
    # Theme Toggle
    # ────────────────────────────────────────────────────────────────

    def _on_theme_change(self, value: str):
        """Switch theme and rebuild the entire app to propagate colors everywhere."""
        ThemeMode.set(value.lower())
        try:
            # Rebuild current screen within the same app instance
            # so sidebar and all UI elements pick up new theme colors
            current = self._app._current_screen
            if current:
                self._app.show_screen(current, force=True)
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────
    # Load Settings Data
    # ────────────────────────────────────────────────────────────────

    def _load_settings(self):
        """Populate system-information fields from the engine."""
        try:
            settings = self._app.engine.get_settings()

            mappings = {
                "version":       "2.0.0",
                "data_dir":      str(settings.get("data_dir", "N/A")),
                "model":         settings.get("model_name", "N/A"),
                "dimension":     str(settings.get("vector_dim", "N/A")),
                "hnsw_m":        str(settings.get("hnsw", {}).get("M", "N/A")),
                "hnsw_efc":      str(settings.get("hnsw", {}).get("ef_construction", "N/A")),
                "hnsw_efs":      str(settings.get("hnsw", {}).get("ef_search", "N/A")),
                "chunk_size":    str(settings.get("chunking", {}).get("chunk_size", "N/A")),
                "chunk_overlap": str(settings.get("chunking", {}).get("overlap", "N/A")),
            }

            for key, value in mappings.items():
                if key in self._settings_labels:
                    try:
                        self._settings_labels[key].configure(text=value)
                    except Exception:
                        pass

        except Exception as exc:
            for label in self._settings_labels.values():
                try:
                    label.configure(text=f"Error: {exc}", text_color=COLOR_ERROR)
                except Exception:
                    pass

    # ────────────────────────────────────────────────────────────────
    # Change Password
    # ────────────────────────────────────────────────────────────────

    def _handle_change_password(self):
        """Validate inputs and call the engine to change the password."""
        current = self._current_pw_entry.get()
        new_pw = self._new_pw_entry.get()
        confirm = self._confirm_pw_entry.get()

        # All fields required
        if not current or not new_pw or not confirm:
            try:
                self._pw_error_label.configure(
                    text="All fields are required.",
                    text_color=COLOR_ERROR,
                )
            except Exception:
                pass
            return

        # Passwords must match
        if new_pw != confirm:
            try:
                self._pw_error_label.configure(
                    text="New passwords do not match.",
                    text_color=COLOR_ERROR,
                )
            except Exception:
                pass
            return

        try:
            self._app.engine.change_password(current, new_pw)
            try:
                self._pw_error_label.configure(
                    text="Password updated successfully!",
                    text_color=COLOR_SUCCESS,
                )
            except Exception:
                pass
            # Clear fields
            self._current_pw_entry.delete(0, "end")
            self._new_pw_entry.delete(0, "end")
            self._confirm_pw_entry.delete(0, "end")
            try:
                self._app.show_toast("Password updated successfully", "success")
            except Exception:
                pass

        except ValueError as exc:
            try:
                self._pw_error_label.configure(
                    text=str(exc),
                    text_color=COLOR_ERROR,
                )
            except Exception:
                pass

        except Exception as exc:
            try:
                self._pw_error_label.configure(
                    text=f"Error: {exc}",
                    text_color=COLOR_ERROR,
                )
            except Exception:
                pass

    # ────────────────────────────────────────────────────────────────
    # Watch Folders
    # ────────────────────────────────────────────────────────────────

    def _refresh_watch_list(self):
        """Re-render the list of currently watched folders."""
        for w in self._watch_list_frame.winfo_children():
            w.destroy()

        watcher = self._get_watcher()
        if not watcher:
            ctk.CTkLabel(
                self._watch_list_frame,
                text="Watch folder service is not available",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
            ).pack(fill="x", pady=(0, PADDING_SM))
            return

        folders = watcher.get_watched_folders()
        if not folders:
            ctk.CTkLabel(
                self._watch_list_frame,
                text="No watch folders configured",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
            ).pack(fill="x", pady=(0, PADDING_SM))
            return

        for info in folders:
            folder_path = info["folder_path"]
            index_name = info["index_name"]

            row = ctk.CTkFrame(self._watch_list_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)

            ctk.CTkLabel(
                row,
                text=folder_path,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT,
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

            idx_badge = create_badge(row, text=index_name, color=COLOR_GOLD)
            idx_badge.pack(side="left", padx=(PADDING, PADDING_SM))

            remove_btn = ctk.CTkButton(
                row,
                text="Remove",
                font=(FONT_FAMILY, FONT_SIZE_XXS),
                fg_color="transparent",
                hover_color="#3a1515",
                text_color=COLOR_ERROR,
                height=26, width=60,
                corner_radius=BORDER_RADIUS_SM,
                border_width=1, border_color=COLOR_BORDER,
                command=lambda fp=folder_path: self._remove_watch_folder(fp),
            )
            remove_btn.pack(side="right")

    def _add_watch_folder(self):
        """Open a folder picker and start watching the selected folder."""
        from tkinter import filedialog
        chosen = filedialog.askdirectory(title="Select Folder to Watch")
        if not chosen:
            return

        watcher = self._get_watcher()
        if not watcher:
            try:
                self._app.show_toast("Watch service not available", "error")
            except Exception:
                pass
            return

        try:
            watcher.add_watch(chosen, "default")
            self._refresh_watch_list()
            try:
                self._app.show_toast(f"Now watching: {chosen}", "success")
            except Exception:
                pass
        except ValueError as exc:
            try:
                self._app.show_toast(str(exc), "error")
            except Exception:
                pass
        except Exception as exc:
            try:
                self._app.show_toast(f"Failed to watch folder: {exc}", "error")
            except Exception:
                pass

    def _remove_watch_folder(self, folder_path: str):
        """Stop watching a folder and refresh the list."""
        watcher = self._get_watcher()
        if not watcher:
            return

        try:
            watcher.remove_watch(folder_path)
            self._refresh_watch_list()
            try:
                self._app.show_toast(f"Stopped watching: {folder_path}", "info")
            except Exception:
                pass
        except Exception:
            pass

    def _get_watcher(self):
        """Get the app's FolderWatcher instance, or None."""
        try:
            return self._app._watcher
        except AttributeError:
            return None

    # ────────────────────────────────────────────────────────────────
    # Exclusion Rules
    # ────────────────────────────────────────────────────────────────

    def _save_exclusion_rules(self):
        text = self._excl_text_widget.get("1.0", "end").strip()
        patterns = [p.strip() for p in text.split("\n") if p.strip()]
        self._app.engine.set_exclusion_patterns(patterns)
        try:
            self._excl_status_label.configure(text=f"Saved {len(patterns)} rules", text_color=COLOR_SUCCESS)
            self._app.show_toast(f"Exclusion rules saved ({len(patterns)} patterns)", "success")
        except Exception:
            pass

    def _reset_exclusion_rules(self):
        defaults = [
            ".git", ".git/**", "node_modules", "node_modules/**",
            "__pycache__", "*.pyc", ".DS_Store", "*.tmp", "*.log",
            ".env", "*.env", "venv/**", ".venv/**",
        ]
        self._excl_text_widget.delete("1.0", "end")
        self._excl_text_widget.insert("1.0", "\n".join(defaults))
        self._app.engine.set_exclusion_patterns(defaults)
        try:
            self._excl_status_label.configure(text="Reset to default rules", text_color=COLOR_SUCCESS)
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────
    # Index Re-indexing
    # ────────────────────────────────────────────────────────────────

    _logger = logging.getLogger("IsoCortex.settings")

    def _refresh_index_list(self):
        """Populate the index list with name, vector count, and Re-index button."""
        for w in self._index_list_frame.winfo_children():
            w.destroy()

        try:
            indexes = self._app.engine.list_indexes()
        except Exception:
            indexes = []

        if not indexes:
            ctk.CTkLabel(
                self._index_list_frame,
                text="No indexes found",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
            ).pack(fill="x", pady=(PADDING_SM, 0))
            return

        for idx_info in indexes:
            row = ctk.CTkFrame(
                self._index_list_frame,
                fg_color=COLOR_BG_ELEVATED,
                corner_radius=BORDER_RADIUS_SM,
            )
            row.pack(fill="x", pady=2)

            row_content = ctk.CTkFrame(row, fg_color="transparent")
            row_content.pack(fill="x", padx=PADDING_MD, pady=PADDING_SM)

            ctk.CTkLabel(
                row_content, text=idx_info.name,
                font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL),
                text_color=COLOR_GOLD,
                anchor="w",
            ).pack(side="left")

            ctk.CTkLabel(
                row_content,
                text=f"  \u00B7  {idx_info.vector_count} vectors  \u00B7  {idx_info.index_size_mb:.1f} MB",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                anchor="w",
            ).pack(side="left")

            btn = ctk.CTkButton(
                row_content, text="Re-index",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                fg_color=COLOR_PURPLE_DARK,
                hover_color=COLOR_PURPLE,
                text_color=COLOR_TEXT,
                height=26, width=70,
                corner_radius=BORDER_RADIUS_SM,
            )
            btn.pack(side="right")
            # Bind click after btn is assigned (Python 3.14 safe)
            captured_name = idx_info.name
            captured_btn = btn
            btn.configure(command=lambda: self._reindex_index(captured_name, captured_btn))

    def _reindex_index(self, index_name: str, btn):
        """Re-index all files in a given index from scratch."""
        btn.configure(state="disabled", text="Working\u2026")
        try:
            self._idx_reindex_status.configure(
                text=f"Collecting files for '{index_name}'\u2026",
                text_color=COLOR_GOLD,
            )
        except Exception:
            pass

        def _collect_and_reindex():
            engine = self._app.engine
            conn = engine._get_db()
            rows = conn.execute(
                "SELECT DISTINCT file_path FROM documents WHERE index_name = ?",
                (index_name,),
            ).fetchall()
            file_paths = [r[0] for r in rows if r[0]]
            if not file_paths:
                return {"files_processed": 0, "total_vectors": 0, "elapsed_seconds": 0, "total_chunks": 0}

            # Delete all document records for this index
            conn.execute(
                "DELETE FROM documents WHERE index_name = ?",
                (index_name,),
            )
            conn.commit()

            # Re-ingest
            result = engine.ingest_files(index_name, file_paths)
            return {
                "files_processed": result.files_processed,
                "total_chunks": result.total_chunks,
                "total_vectors": result.total_vectors,
                "elapsed_seconds": result.elapsed_seconds,
            }

        def _on_done(stats):
            try:
                btn.configure(state="normal", text="Re-index")
                self._idx_reindex_status.configure(
                    text=(
                        f"\u2713  Re-indexed '{index_name}': "
                        f"{stats['files_processed']} files \u00B7 "
                        f"{stats['total_vectors']} vectors ({stats['elapsed_seconds']:.1f}s)"
                    ),
                    text_color=COLOR_SUCCESS,
                )
                self._refresh_index_list()
                self._app.show_toast(
                    f"Re-indexed '{index_name}': {stats['total_vectors']} vectors",
                    "success",
                )
            except Exception:
                pass

        def _on_error(error: str):
            try:
                btn.configure(state="normal", text="Re-index")
                self._idx_reindex_status.configure(
                    text=f"\u2717  Re-index failed: {error}",
                    text_color=COLOR_ERROR,
                )
            except Exception:
                pass

        worker = WorkerThread(
            target=_collect_and_reindex,
            on_success=_on_done,
            on_error=_on_error,
            name="IndexReindexWorker",
        )
        worker.set_app_ref(self._app)
        worker.start()

    def _refresh_plugins_list(self):
        """Refresh the plugin list in settings."""
        # Clear existing
        for w in self._plugins_container.winfo_children():
            w.destroy()

        pm = self._app.engine.plugin_manager
        if not pm:
            ctk.CTkLabel(
                self._plugins_container,
                text="Plugin system not available",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
            ).pack(anchor="w")
            return

        plugins = pm.get_plugin_list()

        if not plugins:
            ctk.CTkLabel(
                self._plugins_container,
                text="No plugins loaded. Place .py files in ~/.isocortex/plugins/",
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                text_color=COLOR_TEXT_DIM,
                wraplength=400,
            ).pack(anchor="w", pady=PADDING_SM)
            return

        for plugin in plugins:
            row = ctk.CTkFrame(self._plugins_container, fg_color=COLOR_BG_ELEVATED, corner_radius=BORDER_RADIUS_SM)
            row.pack(fill="x", pady=2)

            row_inner = ctk.CTkFrame(row, fg_color="transparent")
            row_inner.pack(fill="x", padx=PADDING_MD, pady=(PADDING_SM, PADDING_SM))

            # Name + version
            name_text = f"{plugin['name']}  v{plugin['version']}"
            ctk.CTkLabel(
                row_inner, text=name_text,
                font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
                text_color=COLOR_TEXT,
                anchor="w",
            ).pack(anchor="w")

            # Description
            if plugin.get("description"):
                ctk.CTkLabel(
                    row_inner, text=plugin["description"],
                    font=(FONT_FAMILY, FONT_SIZE_XXS),
                    text_color=COLOR_TEXT_DIM,
                    anchor="w", wraplength=350,
                ).pack(anchor="w")

            # Hooks
            hooks = plugin.get("hooks", [])
            if hooks:
                ctk.CTkLabel(
                    row_inner, text=f"Hooks: {', '.join(hooks)}",
                    font=(FONT_FAMILY, FONT_SIZE_XXS),
                    text_color=COLOR_PURPLE_LIGHT,
                    anchor="w",
                ).pack(anchor="w")

            # Author
            if plugin.get("author"):
                ctk.CTkLabel(
                    row_inner, text=f"by {plugin['author']}",
                    font=(FONT_FAMILY, FONT_SIZE_XXS),
                    text_color=COLOR_TEXT_DIM,
                    anchor="w",
                ).pack(anchor="w")

    # ────────────────────────────────────────────────────────────────
    # Reset All Data (Danger Zone)
    # ────────────────────────────────────────────────────────────────

    def _handle_reset(self):
        """Show a confirmation dialog before performing a full data reset."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Confirm Reset")
        dialog.geometry("440x260")
        try:
            dialog.configure(fg_color=COLOR_BG)
        except Exception:
            pass
        dialog.transient(self)
        dialog.grab_set()

        # ShimmerBar as top accent instead of plain red bar
        ShimmerBar(
            dialog, height=4,
            colors=["#3a1515", COLOR_ERROR, "#5a2020", "#3a1515"],
        ).pack(fill="x")

        # Content area
        content = ctk.CTkFrame(dialog, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING)

        # Header row
        dlg_header = ctk.CTkFrame(content, fg_color="transparent")
        dlg_header.pack(fill="x", pady=(0, PADDING))

        accent = ctk.CTkFrame(
            dlg_header,
            width=4,
            height=24,
            corner_radius=2,
            fg_color=COLOR_ERROR,
        )
        accent.pack(side="left", padx=(0, PADDING_SM))
        accent.pack_propagate(False)

        ctk.CTkLabel(
            dlg_header,
            text="Are you sure?",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=COLOR_ERROR,
            anchor="w",
        ).pack(side="left")

        # Warning text
        ctk.CTkLabel(
            content,
            text="This will permanently delete all indexes, documents, embeddings, "
                 "and your account. This action cannot be undone.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            wraplength=380,
            justify="left",
        ).pack(fill="x", pady=(0, PADDING_LG))

        # Button row
        btn_row = ctk.CTkFrame(content, fg_color="transparent")
        btn_row.pack(fill="x", pady=(PADDING_SM, 0))

        def do_cancel():
            dialog.destroy()

        def do_reset():
            dialog.destroy()

            def _reset_thread():
                try:
                    success = self._app.engine.reset_data()
                except Exception:
                    success = False
                self.after(0, lambda: _reset_done(success))

            def _reset_done(success):
                if success:
                    self._app.show_screen("setup")
                    try:
                        self._app.show_toast("All data has been reset", "warning")
                    except Exception:
                        pass
                else:
                    try:
                        self._app.show_toast("Failed to reset data", "error")
                    except Exception:
                        pass

            threading.Thread(target=_reset_thread, daemon=True).start()

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

        ctk.CTkButton(
            btn_row,
            text="Reset Everything",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color="#3a1515",
            hover_color="#5a2020",
            text_color=COLOR_ERROR,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
            width=160,
            command=do_reset,
        ).pack(side="right")
