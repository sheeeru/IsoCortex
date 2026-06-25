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

import logging
import os
import sys
import webbrowser
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
    COLOR_SUCCESS, COLOR_ERROR, COLOR_INFO, COLOR_WARNING,
    COLOR_SIDEBAR_BG,
    GRADIENT_PURPLE_GOLD, GRADIENT_SIDEBAR,
    SIDEBAR_WIDTH, BORDER_RADIUS, BORDER_RADIUS_SM, BORDER_RADIUS_LG,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    ThemeMode, GradientCanvas, create_gradient_bar,
    AnimatedLogo, GlassCard, GradientDivider,
    create_badge, _dim_hex, _blend_colors,
    apply_ctk_theme, FONT_FAMILY_DISPLAY,
)
from desktop_app.engine import IsoCortexEngine

logger = logging.getLogger("IsoCortex.app")


class IsoCortexApp(ctk.CTk):
    """
    Main IsoCortex desktop application window.
    Premium layout with depth-layered sidebar and content area.
    """

    def __init__(self):
        super().__init__(fg_color=COLOR_BG_DARKEST)

        # ── Window configuration ──────────────────────────────────────
        self.title("IsoCortex")
        self.geometry("1120x720")
        self.minsize(960, 640)
        self.maxsize(3840, 2160)
        self.configure(fg_color=COLOR_BG_DARKEST)

        # ── 4K / HiDPI awareness (Windows + Linux) ────────────────────
        try:
            # Windows DPI awareness
            if sys.platform == "win32":
                try:
                    from ctypes import windll
                    windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE_V2
                except Exception:
                    try:
                        from ctypes import windll
                        windll.user32.SetProcessDPIAware()
                    except Exception:
                        pass
            # Linux: respect GDK_SCALE / GDK_DPI_SCALE env vars (set by user or DE)
            # macOS: handled natively by Tk
        except Exception:
            pass

        # Set window icon (IsoCortex logo)
        self._set_app_icon(self)

        # ── Initialize engine ─────────────────────────────────────────
        self.engine = IsoCortexEngine()

        # --- Folder Watcher (auto-index) ---
        self._watcher = None
        self._init_watcher()

        # ── State ─────────────────────────────────────────────────────
        self._current_screen = None
        self._sidebar_buttons = {}

        # ── Apply custom theme (once) ─────────────────────────────────
        try:
            ctk.set_default_color_theme("blue")
        except Exception:
            pass
        # Recolor CustomTkinter's default widgets to the IsoCortex palette so
        # switches, sliders, dropdowns, progress bars, etc. never fall back to
        # the stock blue accent. Cosmetic only — overwrites existing keys.
        try:
            apply_ctk_theme()
        except Exception:
            pass

        # ── Build UI ─────────────────────────────────────────────────
        self._build_layout()

        # ── Update check (delayed 3 seconds) ──────────────────────────
        self._update_banner = None  # will hold the banner frame
        self.after(3000, self._check_for_updates)

        # ── Show initial screen ──────────────────────────────────────
        if self.engine.is_first_run():
            self.show_screen("setup")
        else:
            self.show_screen("login")

    # ── Window icon helper ──────────────────────────────────────────

    @staticmethod
    def _set_app_icon(window):
        """Set the IsoCortex logo as the window/dock icon on all platforms."""
        assets_dir = Path(__file__).parent / "assets"
        icon_path = assets_dir / "app_icon.png"
        if not icon_path.exists():
            icon_path = assets_dir / "isocortex-logo.png"
        if not icon_path.exists():
            icon_path = assets_dir / "favicon.png"
        if not icon_path.exists():
            return

        # macOS: use AppKit to set the dock icon reliably
        if sys.platform == "darwin":
            try:
                from PIL import Image as _Img
                _pil_img = _Img.open(str(icon_path)).convert("RGBA")
                # macOS needs a .icns or NSImage from TIFF data
                import io
                _buf = io.BytesIO()
                _pil_img.save(_buf, format="TIFF")
                _tiff_data = _buf.getvalue()
                try:
                    from AppKit import NSImage, NSData
                    nsdata = NSData.dataWithBytes_length_(_tiff_data, len(_tiff_data))
                    nsimage = NSImage.alloc().initWithData_(nsdata)
                    if nsimage:
                        from Foundation import NSApplication
                        NSApplication.sharedApplication().setApplicationIconImage_(nsimage)
                except ImportError:
                    pass
            except Exception:
                pass
            # Also set via iconphoto for the title bar
            try:
                from PIL import Image
                img = Image.open(str(icon_path)).convert("RGBA")
                if img.size[0] > 256:
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                window.iconphoto(True, img)
            except Exception:
                pass
            return

        # Windows / Linux: iconphoto works fine
        try:
            from PIL import Image
            img = Image.open(str(icon_path)).convert("RGBA")
            if img.size[0] > 256:
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
            window.iconphoto(True, img)
        except Exception:
            pass

    def _init_watcher(self):
        try:
            from desktop_app.watcher import FolderWatcher
            watched = self.engine.get_watched_folders()
            active = [f for f in watched if f.get("is_active")]
            if active:
                self._watcher = FolderWatcher(self.engine)
                for f in active:
                    try:
                        self._watcher.add_watch(f["folder_path"], f.get("index_name", "default"))
                    except Exception:
                        pass
                self._watcher.start()
                logger.info("Started folder watcher with %d active folders", len(active))
        except ImportError:
            logger.warning("watchdog not installed — watch folders disabled")
        except Exception as exc:
            logger.warning("Failed to start folder watcher: %s", exc)

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
        self._screen_container.pack(fill="both", expand=True)

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

        # ── Top accent gradient bar (purple → gold, static) ─────────
        from desktop_app.theme import create_gradient_bar
        create_gradient_bar(sidebar, height=2).pack(fill="x", side="top")

        # ── Logo area — clean wordmark, no heavy card ────────────────
        logo_container = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_container.pack(fill="x", padx=PADDING, pady=(PADDING_LG, PADDING_MD))

        logo_inner = ctk.CTkFrame(logo_container, fg_color="transparent")
        logo_inner.pack(fill="x")

        # Animated logo image
        try:
            self._animated_logo = AnimatedLogo(logo_inner, logo_size=28)
            self._animated_logo.pack(side="left", padx=(0, PADDING_SM))
        except Exception:
            pass

        # "Iso" in amethyst + "Cortex" in champagne — display cut, editorial
        ctk.CTkLabel(
            logo_inner,
            text="Iso",
            font=(FONT_FAMILY_DISPLAY, 22, "bold"),
            text_color=COLOR_PURPLE_LIGHT,
            anchor="w",
        ).pack(side="left")

        ctk.CTkLabel(
            logo_inner,
            text="Cortex",
            font=(FONT_FAMILY_DISPLAY, 22, "bold"),
            text_color=COLOR_GOLD,
            anchor="w",
        ).pack(side="left")

        # Version pill — right-aligned, very subtle
        version_badge = ctk.CTkFrame(
            logo_inner,
            fg_color=_blend_colors(COLOR_BG_ELEVATED, COLOR_PURPLE, 0.12),
            corner_radius=8,
            border_width=1,
            border_color=_blend_colors(COLOR_BORDER, COLOR_PURPLE, 0.3),
            height=16,
        )
        version_badge.pack(side="right")
        version_badge.pack_propagate(False)

        ctk.CTkLabel(
            version_badge,
            text=" v2 ",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_PURPLE_LIGHT,
        ).place(relx=0.5, rely=0.5, anchor="center")

        # ── Gradient divider (website: .section-divider) ─────────────
        GradientDivider(sidebar, height=1).pack(fill="x", padx=PADDING)


        # ── User info + bottom divider — packed side="bottom" FIRST ───
        # (tkinter requirement: bottom-anchored items must be packed before
        #  any expanding widget, otherwise the expander claims all space first)

        self._user_frame = ctk.CTkFrame(
            sidebar,
            fg_color=COLOR_SURFACE_1,
            corner_radius=BORDER_RADIUS_LG,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        self._user_frame.pack(side="bottom", fill="x", padx=PADDING, pady=(0, PADDING))

        user_inner = ctk.CTkFrame(self._user_frame, fg_color="transparent")
        user_inner.pack(fill="x", padx=PADDING_MD, pady=PADDING_MD)

        # Avatar circle
        avatar = ctk.CTkFrame(
            user_inner,
            width=32,
            height=32,
            fg_color=_blend_colors(COLOR_BG_ELEVATED, COLOR_PURPLE, 0.4),
            corner_radius=16,
        )
        avatar.pack(side="left", padx=(0, PADDING_SM))
        avatar.pack_propagate(False)

        self._avatar_label = ctk.CTkLabel(
            avatar,
            text="?",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            text_color=COLOR_PURPLE_LIGHT,
        )
        self._avatar_label.place(relx=0.5, rely=0.5, anchor="center")

        # User name + role
        user_text = ctk.CTkFrame(user_inner, fg_color="transparent")
        user_text.pack(side="left", fill="x", expand=True)

        self._user_label = ctk.CTkLabel(
            user_text,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
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
            user_inner,
            text="↩",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM),
            fg_color="transparent",
            hover_color=_blend_colors(COLOR_BG_ELEVATED, COLOR_ERROR, 0.2),
            text_color=COLOR_TEXT_DIM,
            width=28,
            height=28,
            corner_radius=BORDER_RADIUS_SM,
            anchor="center",
            command=self._handle_logout,
        )
        self._logout_btn.pack(side="right")
        self._logout_btn_inner = None

        # Bottom gradient divider (above user card)
        GradientDivider(sidebar, height=1).pack(side="bottom", fill="x", padx=PADDING, pady=(0, PADDING_SM))

        # ── Navigation frame ──────────────────────────────────────────
        self._nav_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        self._nav_frame.pack(fill="x", padx=PADDING_SM, pady=(0, PADDING_SM))

        workspace_items = [
            ("dashboard", "Dashboard",   "◈"),
            ("upload",    "Upload",      "⊕"),
            ("search",    "AI Chat",     "⊘"),
            ("indexes",   "Indexes",     "▦"),
        ]
        system_items = [
            ("settings",  "Settings",   "◎"),
        ]

        self._sidebar_indicators: dict[str, ctk.CTkFrame] = {}
        self._sidebar_icon_labels: dict[str, ctk.CTkLabel] = {}

        # ── WORKSPACE section ──────────────────────────────────────────
        ctk.CTkLabel(
            self._nav_frame,
            text="WORKSPACE",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(fill="x", padx=(PADDING_SM, PADDING_SM), pady=(PADDING_SM, 0))

        for i, (screen_id, label, icon) in enumerate(workspace_items):
            btn, indicator, icon_lbl = self._build_nav_button(self._nav_frame, screen_id, label, icon)
            self._nav_frame.after(i * 70, lambda b=btn: self._nav_button_entrance(b))
            self._sidebar_buttons[screen_id] = btn
            self._sidebar_indicators[screen_id] = indicator
            self._sidebar_icon_labels[screen_id] = icon_lbl

        # ── SYSTEM section ─────────────────────────────────────────────
        ctk.CTkLabel(
            self._nav_frame,
            text="SYSTEM",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        ).pack(fill="x", padx=(PADDING_SM, PADDING_SM), pady=(PADDING_SM, 0))

        for i, (screen_id, label, icon) in enumerate(system_items):
            btn, indicator, icon_lbl = self._build_nav_button(self._nav_frame, screen_id, label, icon)
            self._nav_frame.after(i * 70, lambda b=btn: self._nav_button_entrance(b))
            self._sidebar_buttons[screen_id] = btn
            self._sidebar_indicators[screen_id] = indicator
            self._sidebar_icon_labels[screen_id] = icon_lbl


    def _nav_button_entrance(self, btn):
        """Simulate a subtle flash entrance for nav buttons."""
        try:
            btn.configure(text_color=COLOR_TEXT)
            self.after(200, lambda: btn.configure(text_color=COLOR_TEXT_SECONDARY))
        except Exception:
            pass

    def _build_nav_button(self, parent, screen_id: str, label: str, icon: str = "•"):
        """Fixed-height nav row using place() — bypasses CTK button height inflation on macOS.
        Returns (text_label, indicator_frame, icon_label).
        """
        ROW_H = 36

        # pack_propagate(False) locks the row at exactly ROW_H — no child can stretch it
        row = ctk.CTkFrame(parent, fg_color="transparent", height=ROW_H, corner_radius=BORDER_RADIUS_SM)
        row.pack(fill="x", pady=1)
        row.pack_propagate(False)

        # Indicator bar — width/height in constructor (CTK blocks them in place())
        indicator = ctk.CTkFrame(row, width=3, height=ROW_H - 10, fg_color="transparent", corner_radius=2)
        indicator.place(x=3, y=5)

        # Icon pill — absolutely placed
        icon_pill = ctk.CTkFrame(row, width=26, height=26, fg_color=COLOR_BG_ELEVATED, corner_radius=BORDER_RADIUS_SM)
        icon_pill.place(x=10, y=5)

        icon_lbl = ctk.CTkLabel(
            icon_pill,
            text=icon,
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            text_color=COLOR_TEXT_SECONDARY,
        )
        icon_lbl.place(relx=0.5, rely=0.5, anchor="center")

        # Text label — height in constructor, relwidth fills remaining row space
        btn = ctk.CTkLabel(
            row,
            text=f"  {label}",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
            fg_color="transparent",
            height=ROW_H - 8,
        )
        btn.place(x=42, y=4, relwidth=1.0)

        # Hover + click on every part of the row
        def _enter(e, r=row): r.configure(fg_color=COLOR_BG_HOVER)
        def _leave(e, r=row): r.configure(fg_color="transparent")
        for w in (row, icon_pill, icon_lbl, btn):
            w.bind("<Button-1>", lambda e, s=screen_id: self.show_screen(s))
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

        return btn, indicator, icon_lbl

    def _build_separator(self, parent) -> None:
        """Build a gradient divider line (website-inspired)."""
        GradientDivider(parent, height=1).pack(fill="x", padx=PADDING)

    # ─────────────────────────────────────────────────────────────────
    # Sidebar State Updates
    # ─────────────────────────────────────────────────────────────────

    def _update_sidebar_active(self, screen_id: str) -> None:
        """Highlight the active nav row — pill turns purple, label bold white, row tinted."""
        for sid, btn in self._sidebar_buttons.items():
            indicator = self._sidebar_indicators.get(sid)
            icon_lbl = self._sidebar_icon_labels.get(sid)
            row = getattr(btn, "master", None)
            if sid == screen_id:
                try:
                    if row:
                        row.configure(fg_color=_blend_colors(COLOR_SIDEBAR_BG, COLOR_PURPLE, 0.22))
                    btn.configure(text_color=COLOR_TEXT, font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"))
                except Exception:
                    pass
                if indicator:
                    try:
                        indicator.configure(fg_color=COLOR_PURPLE)
                    except Exception:
                        pass
                if icon_lbl:
                    try:
                        icon_lbl.master.configure(fg_color=COLOR_PURPLE)
                        icon_lbl.configure(text_color="#ffffff", font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"))
                    except Exception:
                        pass
            else:
                try:
                    if row:
                        row.configure(fg_color="transparent")
                    btn.configure(text_color=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, FONT_SIZE_NORMAL))
                except Exception:
                    pass
                if indicator:
                    try:
                        indicator.configure(fg_color="transparent")
                    except Exception:
                        pass
                if icon_lbl:
                    try:
                        icon_lbl.master.configure(fg_color=COLOR_BG_ELEVATED)
                        icon_lbl.configure(text_color=COLOR_TEXT_SECONDARY, font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"))
                    except Exception:
                        pass

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
                self._logout_btn.pack(side="right")
            except Exception:
                pass
        else:
            try:
                self._user_label.configure(text="Not signed in")
                self._avatar_label.configure(text="?")
                self._user_role_label.configure(text="")
                self._logout_btn.pack_forget()
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

        # Clear current screen — use _safe_destroy to avoid Python 3.14 / CTkButton _font bug
        def _safe_destroy(w):
            """Patch CTkButton children lacking _font before destroying."""
            if w is None:
                return
            def _patch(widget):
                try:
                    if not hasattr(widget, '_font'):
                        widget._font = None
                except Exception:
                    pass
                try:
                    for child in widget.winfo_children():
                        _patch(child)
                except Exception:
                    pass
            _patch(w)
            try:
                w.destroy()
            except Exception:
                pass

        for widget in self._screen_container.winfo_children():
            _safe_destroy(widget)

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
    # Update Notifier
    # ─────────────────────────────────────────────────────────────────

    def _check_for_updates(self):
        """Check for updates in the background and show banner if available."""
        try:
            from desktop_app.updater import check_for_updates
            check_for_updates(callback=self._on_update_check_done)
        except Exception as exc:
            logger.debug("Update check init failed: %s", exc)

    def _on_update_check_done(self, latest_version: str | None):
        """Called on the main thread when update check completes."""
        if not latest_version:
            return

        try:
            self._show_update_banner(latest_version)
        except Exception as exc:
            logger.debug("Failed to show update banner: %s", exc)

    def _show_update_banner(self, latest_version: str):
        """Show a dismissable update notification banner at the top of the content area."""
        if self._update_banner is not None:
            return  # already showing

        try:
            from desktop_app.theme import (
                COLOR_GOLD, COLOR_GOLD_LIGHT, COLOR_TEXT,
                COLOR_TEXT_SECONDARY, COLOR_BG_ELEVATED,
                FONT_FAMILY, FONT_SIZE_SMALL, FONT_SIZE_XXS,
                BORDER_RADIUS_SM, PADDING, PADDING_SM,
            )
        except Exception:
            return

        self._update_banner = ctk.CTkFrame(
            self._content_frame,
            fg_color=COLOR_BG_ELEVATED,
            corner_radius=BORDER_RADIUS_SM,
            height=44,
        )
        self._update_banner.pack_propagate(False)

        # Pack at the TOP of content_frame, BEFORE the screen container
        self._update_banner.pack(
            side="top", fill="x",
            padx=PADDING, pady=(PADDING_SM, 0),
            before=self._screen_container,
        )

        # Banner content
        inner = ctk.CTkFrame(self._update_banner, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=PADDING_SM)

        # Gold accent dot
        ctk.CTkLabel(
            inner, text="●",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_GOLD,
        ).pack(side="left", padx=(0, PADDING_SM))

        # Update text
        ctk.CTkLabel(
            inner,
            text=f"IsoCortex v{latest_version} is available",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(side="left")

        # View Release button
        release_btn = ctk.CTkButton(
            inner,
            text="View Release",
            font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
            fg_color=COLOR_GOLD,
            hover_color=COLOR_GOLD_LIGHT,
            text_color="#1a1a2e",
            height=26,
            width=90,
            corner_radius=BORDER_RADIUS_SM,
            command=lambda: webbrowser.open(
                f"https://github.com/shaheerdev/isocortex/releases/tag/v{latest_version}"
            ),
        )
        release_btn.pack(side="right", padx=(PADDING_SM, 0))

        # Dismiss button
        def _dismiss():
            try:
                if self._update_banner:
                    self._update_banner.pack_forget()
                    self._update_banner.destroy()
                    self._update_banner = None
            except Exception:
                pass

        dismiss_btn = ctk.CTkButton(
            inner,
            text="✕",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color="transparent",
            hover_color="#3a1515",
            text_color=COLOR_TEXT_SECONDARY,
            width=28,
            height=28,
            corner_radius=BORDER_RADIUS_SM,
            command=_dismiss,
        )
        dismiss_btn.pack(side="right")

    def _dismiss_update_banner(self):
        """Dismiss the update banner (for the rest of the session)."""
        if self._update_banner:
            try:
                self._update_banner.pack_forget()
                self._update_banner.destroy()
            except Exception:
                pass
            self._update_banner = None

    # ─────────────────────────────────────────────────────────────────
    # Toast Notifications
    # ─────────────────────────────────────────────────────────────────

    def show_toast(self, message: str, toast_type: str = "info") -> None:
        """Show a temporary notification toast."""
        colors = {
            "info": COLOR_INFO,
            "success": COLOR_SUCCESS,
            "warning": COLOR_WARNING,
            "error": COLOR_ERROR,
        }
        color = colors.get(toast_type, COLOR_INFO)
        from desktop_app.components.toast import ToastNotification
        toast = ToastNotification(
            self._content_frame,
            message=message,
            color=color,
            corner_radius=BORDER_RADIUS_LG,
        )