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

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER,
    COLOR_PURPLE, COLOR_PURPLE_DARK, COLOR_PURPLE_LIGHT, COLOR_PURPLE_DEEP,
    COLOR_GOLD, COLOR_GOLD_LIGHT,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    COLOR_SHADOW, COLOR_SURFACE_1,
    FONT_FAMILY, FONT_FAMILY_MONO,
    FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_MEDIUM, FONT_SIZE_NORMAL, FONT_SIZE_SMALL, FONT_SIZE_XXS,
    BORDER_RADIUS, BORDER_RADIUS_SM, BORDER_RADIUS_LG,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    ThemeMode,
    GradientCanvas, GRADIENT_PURPLE_GOLD,
    ShimmerBar, GlassCard, GradientDivider, AnimatedGradientBG,
    FadeInFrame, create_badge,
    ANIM_DELAY_200, ANIM_DELAY_400, ANIM_DELAY_600, ANIM_DELAY_800,
)


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
            font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
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
            text="\U0001f3a8",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
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
            text=f"Currently: {ThemeMode.get().title()} Mode",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        self._mode_label.pack(fill="x")

        self._theme_segment = ctk.CTkSegmentedButton(
            appearance_inner,
            values=["Dark", "Light"],
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            command=self._on_theme_change,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
        )
        self._theme_segment.pack(fill="x", pady=(PADDING_SM, 0))

        # Set initial selection from current theme mode
        try:
            self._theme_segment.set(ThemeMode.get().title())
        except Exception:
            pass

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
            text="Vector indexes are stored locally in  ~/.isortex/indices/  and consist of the following files:",
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

        # GradientDivider after Index Management
        GradientDivider(index_fade, height=1).pack(fill="x", pady=(PADDING_MD, 0))

        # ════════════════════════════════════════════════════════════
        # 4 · SECURITY  (FadeInFrame delay=ANIM_DELAY_800)
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
        # 5 · CHANGE PASSWORD  (FadeInFrame delay=ANIM_DELAY_800)
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
        # 6 · DANGER ZONE  (FadeInFrame delay=ANIM_DELAY_800)
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
            text="⚠",
            font=(FONT_FAMILY, FONT_SIZE_LARGE),
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
        """Switch theme and rebuild the settings screen."""
        ThemeMode.set(value.lower())
        try:
            self._app.show_screen("settings", force=True)
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
                "version":       "1.0.0",
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
            try:
                success = self._app.engine.reset_data()
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
            except Exception:
                try:
                    self._app.show_toast("Failed to reset data", "error")
                except Exception:
                    pass

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
