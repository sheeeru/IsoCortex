"""
IsoCortex Desktop App — Main Application Controller
=====================================================
Premium UI with depth-layered sidebar, gradient accents, glow effects,
and the IsoCortex purple-gold brand palette.

Website-inspired enhancements:
  - Animated logo with pulsing glow in sidebar
  - Hero-gradient background on login/setup screens
  - Glassmorphism-style user info card in sidebar
  - Animated gradient accents throughout
  - Staggered entrance animations on screen transitions
"""

import os
import sys
import customtkinter as ctk
from pathlib import Path

from desktop_app.theme import (
    CTK_THEME, FONT_FAMILY, FONT_SIZE_TITLE, FONT_SIZE_HERO,
    FONT_SIZE_MEDIUM, FONT_SIZE_NORMAL, FONT_SIZE_SMALL, FONT_SIZE_XXS,
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER,
    COLOR_BG_DARKEST, COLOR_SHADOW, COLOR_SURFACE_1, COLOR_SURFACE_2,
    COLOR_PURPLE, COLOR_PURPLE_DARK, COLOR_PURPLE_DEEP, COLOR_PURPLE_LIGHT,
    COLOR_GOLD, COLOR_GOLD_LIGHT,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    COLOR_BORDER, COLOR_BORDER_LIGHT,
    COLOR_SUCCESS, COLOR_ERROR,
    COLOR_SIDEBAR_BG,
    GRADIENT_PURPLE_GOLD, GRADIENT_SIDEBAR,
    SIDEBAR_WIDTH, BORDER_RADIUS, BORDER_RADIUS_SM, BORDER_RADIUS_LG,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    ThemeMode, GradientCanvas, create_gradient_bar,
    AnimatedLogo, GlassCard, GradientDivider,
    ShimmerBar, create_badge, _dim_hex,
)
from desktop_app.engine import IsoCortexEngine


class IsoCortexApp(ctk.CTk):
    """
    Main IsoCortex desktop application window.
    Premium layout with depth-layered sidebar and content area.
    """

    def __init__(self):
        super().__init__(fg_color=COLOR_BG_DARKEST)

        # ── Window configuration ──────────────────────────────────────
        self.title("IsoCortex — Offline Semantic Search Engine")
        self.geometry("1120x720")
        self.minsize(1060, 700)
        self.configure(fg_color=COLOR_BG_DARKEST)

        # Set icon if available
        icon_path = Path(__file__).parent / "assets" / "favicon.png"
        if not icon_path.exists():
            icon_path = Path(__file__).parent.parent / "website" / "public" / "favicon.png"
        if icon_path.exists():
            try:
                from PIL import Image
                icon = Image.open(str(icon_path))
                self.iconphoto(True, icon)
            except Exception:
                pass

        # ── Initialize engine ─────────────────────────────────────────
        self.engine = IsoCortexEngine()

        # ── State ─────────────────────────────────────────────────────
        self._current_screen = None
        self._sidebar_buttons = {}

        # ── Apply custom theme (once) ─────────────────────────────────
        try:
            ctk.set_default_color_theme("blue")
        except Exception:
            pass

        # ── Build UI ─────────────────────────────────────────────────
        self._build_layout()

        # ── Show initial screen ──────────────────────────────────────
        if self.engine.is_first_run():
            self.show_screen("setup")
        else:
            self.show_screen("login")

    # ─────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        """Build the main application layout."""
        # Root frame
        self._main_frame = ctk.CTkFrame(self, fg_color=COLOR_BG_DARKEST, corner_radius=0)
        self._main_frame.pack(fill="both", expand=True)

        # ── Sidebar ──────────────────────────────────────────────────
        self._build_sidebar()

        # ── Content Area ────────────────────────────────────────────
        self._content_frame = ctk.CTkFrame(
            self._main_frame,
            fg_color=COLOR_BG,
            corner_radius=0,
        )
        self._content_frame.pack(side="left", fill="both", expand=True)

        # Screen container
        self._screen_container = ctk.CTkFrame(
            self._content_frame,
            fg_color="transparent",
            corner_radius=0,
        )
        self._screen_container.pack(fill="both", expand=True, padx=PADDING_XL, pady=PADDING_XL)

    # ─────────────────────────────────────────────────────────────────
    # Sidebar — Premium Depth-Layered Design
    # ─────────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> None:
        """Build a premium sidebar with depth layers, animated logo, and gradient accents."""
        self._sidebar = ctk.CTkFrame(
            self._main_frame,
            width=SIDEBAR_WIDTH,
            fg_color=COLOR_SIDEBAR_BG,
            corner_radius=0,
        )
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)
        sidebar = self._sidebar

        # ── Top animated shimmer gradient bar ────────────────────────
        ShimmerBar(sidebar, height=3, duration=4000).pack(fill="x", side="top")

        # ── Logo area with animated glow ─────────────────────────────
        logo_container = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_container.pack(fill="x", padx=PADDING, pady=(PADDING_LG, PADDING_MD))

        # Logo card with subtle glow border
        logo_glow_wrapper = ctk.CTkFrame(
            logo_container,
            fg_color=COLOR_SHADOW,
            corner_radius=BORDER_RADIUS_LG + 3,
        )
        logo_glow_wrapper.pack(fill="x")

        logo_card = ctk.CTkFrame(
            logo_glow_wrapper,
            fg_color=COLOR_SURFACE_1,
            corner_radius=BORDER_RADIUS_LG,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        logo_card.pack(fill="x", padx=2, pady=2)
        logo_card.pack_propagate(False)
        logo_card.configure(height=56)

        logo_inner = ctk.CTkFrame(logo_card, fg_color="transparent")
        logo_inner.pack(fill="both", expand=True)

        # Animated logo image with pulsing glow (website-inspired)
        try:
            self._animated_logo = AnimatedLogo(logo_inner, logo_size=36)
            self._animated_logo.pack(side="left", padx=(PADDING, PADDING_SM))
        except Exception:
            # Fallback to text logo
            pass

        # Iso in purple
        ctk.CTkLabel(
            logo_inner,
            text="Iso",
            font=(FONT_FAMILY, FONT_SIZE_HERO, "bold"),
            text_color=COLOR_PURPLE,
            anchor="w",
        ).pack(side="left")

        # Cortex in gold
        ctk.CTkLabel(
            logo_inner,
            text="Cortex",
            font=(FONT_FAMILY, FONT_SIZE_HERO, "bold"),
            text_color=COLOR_GOLD,
            anchor="w",
        ).pack(side="left")

        # Version badge (pill-shaped, website-style)
        version_badge = ctk.CTkFrame(
            logo_inner,
            fg_color=_dim_hex(COLOR_PURPLE, 0.15),
            corner_radius=10,
            border_width=1,
            border_color=_dim_hex(COLOR_PURPLE, 0.3),
            height=18,
        )
        version_badge.pack(side="right", padx=(0, PADDING))
        version_badge.pack_propagate(False)

        ctk.CTkLabel(
            version_badge,
            text=" v1.0 ",
            font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
            text_color=COLOR_PURPLE_LIGHT,
        )

        # ── Gradient divider (website: .section-divider) ─────────────
        GradientDivider(sidebar, height=1).pack(fill="x", padx=PADDING)

        # ── Navigation section label with badge ──────────────────────
        nav_label = ctk.CTkFrame(sidebar, fg_color="transparent")
        nav_label.pack(fill="x", padx=(PADDING_LG, PADDING), pady=(PADDING_SM, PADDING_SM))

        ctk.CTkLabel(
            nav_label,
            text="NAVIGATION",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(side="left")

        # ── Navigation buttons ──────────────────────────────────────
        self._nav_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        self._nav_frame.pack(fill="x", padx=PADDING, pady=(0, PADDING_SM))

        nav_items = [
            ("dashboard", "Dashboard",   "D"),
            ("upload",    "Upload Files", "U"),
            ("search",    "Search",      "S"),
            ("settings",  "Settings",    "G"),
        ]

        for i, (screen_id, label, icon_letter) in enumerate(nav_items):
            btn = self._build_nav_button(self._nav_frame, screen_id, label, icon_letter)
            btn.pack(fill="x", pady=2)
            # Staggered entrance animation delay (website pattern)
            self._nav_frame.after(i * 80, lambda b=btn: self._nav_button_entrance(b))
            self._sidebar_buttons[screen_id] = btn

        # ── Spacer ──────────────────────────────────────────────────
        spacer = ctk.CTkFrame(self._nav_frame, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        # ── Bottom gradient divider (static — reduced animation load) ─────
        GradientDivider(sidebar, height=1).pack(fill="x", padx=PADDING, pady=(0, PADDING_SM))

        # ── User info section (GlassCard style) ────────────────────
        self._user_glow = ctk.CTkFrame(
            sidebar,
            fg_color=COLOR_SHADOW,
            corner_radius=BORDER_RADIUS_LG + 2,
        )
        self._user_glow.pack(fill="x", padx=PADDING, pady=(0, PADDING))

        self._user_frame = ctk.CTkFrame(
            self._user_glow,
            fg_color=COLOR_SURFACE_1,
            corner_radius=BORDER_RADIUS_LG,
            border_width=1,
            border_color=COLOR_BORDER_LIGHT,
        )
        self._user_frame.pack(fill="x", padx=2, pady=2)

        user_inner = ctk.CTkFrame(self._user_frame, fg_color="transparent")
        user_inner.pack(fill="x", padx=PADDING, pady=PADDING_MD)

        # Avatar circle with static purple glow
        avatar_container = ctk.CTkFrame(user_inner, fg_color="transparent", width=44, height=44)
        avatar_container.pack(side="left", padx=(0, PADDING_MD))
        avatar_container.pack_propagate(False)

        # Static glow behind avatar (no animation timer needed)
        try:
            avatar_glow_bg = ctk.CTkFrame(
                avatar_container, fg_color=_dim_hex(COLOR_PURPLE, 0.3),
                corner_radius=22,
            )
            avatar_glow_bg.place(relx=0.5, rely=0.5, anchor="center")
        except Exception:
            pass

        avatar = ctk.CTkFrame(
            avatar_container,
            width=36,
            height=36,
            fg_color=COLOR_PURPLE,
            corner_radius=18,
        )
        avatar.place(relx=0.5, rely=0.5, anchor="center")

        self._avatar_label = ctk.CTkLabel(
            avatar,
            text="?",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"),
            text_color="#ffffff",
        )
        self._avatar_label.place(relx=0.5, rely=0.5, anchor="center")

        # User name + role
        user_text = ctk.CTkFrame(user_inner, fg_color="transparent")
        user_text.pack(side="left", fill="x", expand=True)

        self._user_label = ctk.CTkLabel(
            user_text,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        )
        self._user_label.pack(fill="x")

        self._user_role_label = ctk.CTkLabel(
            user_text,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        self._user_role_label.pack(fill="x")

        # Sign out button
        self._logout_btn = ctk.CTkButton(
            self._user_frame,
            text="Sign Out",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color="transparent",
            hover_color="#2a1520",
            text_color=COLOR_ERROR,
            height=32,
            corner_radius=BORDER_RADIUS_SM,
            anchor="center",
            command=self._handle_logout,
        )
        # Sign out packed inside user frame, below user info
        self._logout_btn_inner = ctk.CTkFrame(self._user_frame, fg_color="transparent")
        self._logout_btn_inner.pack(fill="x", padx=PADDING_SM, pady=(0, PADDING_SM))
        self._logout_btn.pack(fill="x", in_=self._logout_btn_inner)

    def _nav_button_entrance(self, btn):
        """Simulate a fade-in entrance for nav buttons."""
        try:
            btn.configure(fg_color=_dim_hex(COLOR_PURPLE, 0.1))
            self.after(150, lambda: btn.configure(fg_color="transparent"))
        except Exception:
            pass

    def _build_nav_button(self, parent, screen_id: str, label: str, icon_letter: str) -> ctk.CTkButton:
        """Create a premium navigation button with icon letter + label."""
        btn = ctk.CTkButton(
            parent,
            text=f"  {label}",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color="transparent",
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
            height=42,
            corner_radius=BORDER_RADIUS_SM,
            command=lambda s=screen_id: self.show_screen(s),
        )
        return btn

    def _build_separator(self, parent) -> None:
        """Build a gradient divider line (website-inspired)."""
        GradientDivider(parent, height=1).pack(fill="x", padx=PADDING)

    # ─────────────────────────────────────────────────────────────────
    # Sidebar State Updates
    # ─────────────────────────────────────────────────────────────────

    def _update_sidebar_active(self, screen_id: str) -> None:
        """Highlight the active navigation button with purple accent."""
        for sid, btn in self._sidebar_buttons.items():
            if sid == screen_id:
                btn.configure(
                    fg_color=COLOR_PURPLE,
                    hover_color=COLOR_PURPLE_DARK,
                    text_color="#ffffff",
                    font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    hover_color=COLOR_BG_HOVER,
                    text_color=COLOR_TEXT_SECONDARY,
                    font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                )

    def _update_user_info(self) -> None:
        """Update the user info section in the sidebar."""
        if self.engine.is_authenticated:
            user = self.engine.current_user
            username = user.get("username", "User")
            initial = username[0].upper() if username else "?"
            try:
                self._user_label.configure(text=username)
                self._avatar_label.configure(text=initial)
                self._user_role_label.configure(text="Offline Mode")
                self._logout_btn_inner.pack(fill="x", padx=PADDING_SM, pady=(0, PADDING_SM))
            except Exception:
                pass
        else:
            try:
                self._user_label.configure(text="Not signed in")
                self._avatar_label.configure(text="?")
                self._user_role_label.configure(text="")
                self._logout_btn_inner.pack_forget()
            except Exception:
                pass

    def _rebuild_sidebar_theme(self) -> None:
        """Rebuild sidebar colors for the current theme mode."""
        try:
            self._sidebar.configure(fg_color=COLOR_SIDEBAR_BG)
            if self._current_screen and self._current_screen in self._sidebar_buttons:
                self._update_sidebar_active(self._current_screen)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    # Screen Management
    # ─────────────────────────────────────────────────────────────────

    def show_screen(self, screen_id: str, force: bool = False) -> None:
        """Switch to a different screen."""
        # Sync theme
        ThemeMode.get()
        self._rebuild_sidebar_theme()

        # Auth guard
        auth_free_screens = {"setup", "login"}
        if not force and screen_id not in auth_free_screens and not self.engine.is_authenticated:
            screen_id = "login"

        # Clear current screen
        for widget in self._screen_container.winfo_children():
            widget.destroy()

        self._current_screen = screen_id

        # Toggle sidebar visibility
        if screen_id in auth_free_screens:
            self._sidebar.pack_forget()
        else:
            self._sidebar.pack(side="left", fill="y", before=self._content_frame)

        # Create the screen
        try:
            if screen_id == "setup":
                from desktop_app.screens.login import SetupScreen
                screen = SetupScreen(self._screen_container, self, fg_color=COLOR_BG)
            elif screen_id == "login":
                from desktop_app.screens.login import LoginScreen
                screen = LoginScreen(self._screen_container, self, fg_color=COLOR_BG)
            elif screen_id == "dashboard":
                from desktop_app.screens.dashboard import DashboardScreen
                screen = DashboardScreen(self._screen_container, self, fg_color=COLOR_BG)
            elif screen_id == "upload":
                from desktop_app.screens.upload import UploadScreen
                screen = UploadScreen(self._screen_container, self, fg_color=COLOR_BG)
            elif screen_id == "indexes":
                from desktop_app.screens.indexes import IndexesScreen
                screen = IndexesScreen(self._screen_container, self, fg_color=COLOR_BG)
            elif screen_id == "search":
                from desktop_app.screens.search import SearchScreen
                screen = SearchScreen(self._screen_container, self, fg_color=COLOR_BG)
            elif screen_id == "settings":
                from desktop_app.screens.settings import SettingsScreen
                screen = SettingsScreen(self._screen_container, self, fg_color=COLOR_BG)
            else:
                ctk.CTkLabel(
                    self._screen_container,
                    text=f"Unknown screen: {screen_id}",
                    text_color=COLOR_TEXT,
                ).pack(expand=True)
                return

            screen.pack(fill="both", expand=True)

            if screen_id in self._sidebar_buttons:
                self._update_sidebar_active(screen_id)

            self._update_user_info()

        except Exception as exc:
            import traceback
            traceback.print_exc()
            ctk.CTkLabel(
                self._screen_container,
                text=f"Error loading screen: {exc}",
                text_color=COLOR_ERROR,
            ).pack(expand=True)

    # ─────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────

    def _handle_logout(self) -> None:
        """Handle user logout."""
        self.engine.logout()
        self.show_screen("login")

    # ─────────────────────────────────────────────────────────────────
    # Toast Notifications
    # ─────────────────────────────────────────────────────────────────

    def show_toast(self, message: str, toast_type: str = "info") -> None:
        """Show a temporary notification toast."""
        colors = {
            "info": "#3b82f6",
            "success": COLOR_SUCCESS,
            "warning": "#f59e0b",
            "error": COLOR_ERROR,
        }
        color = colors.get(toast_type, "#3b82f6")
        from desktop_app.components.toast import ToastNotification
        toast = ToastNotification(
            self._content_frame,
            message=message,
            color=color,
            corner_radius=BORDER_RADIUS_LG,
        )
