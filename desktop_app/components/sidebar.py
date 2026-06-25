"""
IsoCortex Desktop App — Sidebar Component
==========================================
Reusable navigation sidebar widget.

Note: The sidebar is currently integrated directly into app.py's _build_sidebar()
method. This module provides a standalone reusable version that can be
substituted in the future for modular screen layouts (e.g., login screen
without sidebar, settings in a dialog, etc.).
"""

import customtkinter as ctk

from desktop_app.theme import (
    COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_PURPLE,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM, COLOR_ERROR, COLOR_BORDER,
    FONT_FAMILY, FONT_FAMILY_DISPLAY, FONT_SIZE_SMALL, SIDEBAR_WIDTH,
    COLOR_PURPLE_LIGHT, _blend_colors,
    PADDING, PADDING_SM, BORDER_RADIUS_SM,
)


class Sidebar(ctk.CTkFrame):
    """
    Navigation sidebar with logo, nav buttons, and user info.
    """

    def __init__(
        self,
        parent,
        app,
        on_navigate=None,
        on_logout=None,
        **kwargs,
    ):
        super().__init__(
            parent,
            width=SIDEBAR_WIDTH,
            fg_color=COLOR_BG_CARD,
            corner_radius=0,
            **kwargs,
        )
        self._app = app
        self._on_navigate = on_navigate
        self._on_logout = on_logout
        self._nav_buttons = {}

        self._build()

    def _build(self):
        """Build the sidebar contents."""
        # ── Logo area ───────────────────────────────────────────────
        logo_frame = ctk.CTkFrame(
            self,
            fg_color=COLOR_BG_CARD,
            height=60,
            corner_radius=0,
        )
        logo_frame.pack(fill="x", padx=PADDING, pady=(PADDING, 0))
        logo_frame.pack_propagate(False)

        ctk.CTkLabel(
            logo_frame,
            text="IsoCortex",
            font=(FONT_FAMILY_DISPLAY, 19, "bold"),
            text_color=COLOR_PURPLE_LIGHT,
            anchor="w",
        ).pack(side="left", padx=(4, 0))

        ctk.CTkLabel(
            logo_frame,
            text="v1.0",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(side="left", padx=(8, 0), pady=(4, 0))

        # ── Separator ──────────────────────────────────────────────
        sep = ctk.CTkFrame(self, height=1, fg_color=COLOR_BORDER)
        sep.pack(fill="x", padx=PADDING, pady=PADDING_SM)

        # ── Navigation buttons ─────────────────────────────────────
        nav_items = [
            ("dashboard", "Dashboard", "⬡"),
            ("upload", "Upload Files", "↑"),
            ("indexes", "Indexes", "▦"),
            ("search", "Search", "⊙"),
            ("settings", "Settings", "⚙"),
        ]

        nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        nav_frame.pack(fill="x", padx=PADDING_SM, pady=PADDING_SM)

        for screen_id, label, icon in nav_items:
            btn = ctk.CTkButton(
                nav_frame,
                text=f"  {icon}  {label}",
                font=(FONT_FAMILY, FONT_SIZE_SMALL + 1),
                fg_color="transparent",
                hover_color=COLOR_BG_ELEVATED,
                text_color=COLOR_TEXT_SECONDARY,
                anchor="w",
                height=36,
                corner_radius=BORDER_RADIUS_SM,
                command=lambda s=screen_id: self._handle_nav(s),
            )
            btn.pack(fill="x", pady=1)
            self._nav_buttons[screen_id] = btn

        # ── Spacer ──────────────────────────────────────────────────
        spacer = ctk.CTkFrame(nav_frame, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        # ── User info / logout at bottom ────────────────────────────
        user_frame = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        user_frame.pack(fill="x", padx=PADDING, pady=PADDING)

        sep2 = ctk.CTkFrame(self, height=1, fg_color=COLOR_BORDER)
        sep2.pack(fill="x", padx=PADDING, pady=(0, PADDING_SM))

        self._user_label = ctk.CTkLabel(
            user_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        )
        self._user_label.pack(fill="x")

        self._logout_btn = ctk.CTkButton(
            user_frame,
            text="Sign Out",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=COLOR_BG_ELEVATED,
            hover_color=_blend_colors(COLOR_BG_ELEVATED, COLOR_ERROR, 0.22),
            text_color=COLOR_ERROR,
            height=30,
            corner_radius=BORDER_RADIUS_SM,
            command=self._handle_logout,
        )
        self._logout_btn.pack(fill="x", pady=(PADDING_SM, 0))

    def _handle_nav(self, screen_id: str):
        """Handle navigation button click."""
        if self._on_navigate:
            self._on_navigate(screen_id)

    def _handle_logout(self):
        """Handle logout button click."""
        if self._on_logout:
            self._on_logout()

    def set_active(self, screen_id: str):
        """Update the active state of sidebar buttons."""
        for sid, btn in self._nav_buttons.items():
            if sid == screen_id:
                btn.configure(
                    fg_color=COLOR_BG_ELEVATED,
                    text_color=COLOR_PURPLE,
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=COLOR_TEXT_SECONDARY,
                )

    def update_user_info(self, username: str = "", is_authenticated: bool = False):
        """Update the user info display in the sidebar."""
        if is_authenticated and username:
            self._user_label.configure(text=f"  {username}")
            self._logout_btn.pack(fill="x", pady=(PADDING_SM, 0))
        else:
            self._user_label.configure(text="  Not signed in")
            self._logout_btn.pack_forget()
