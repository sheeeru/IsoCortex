"""
IsoCortex Desktop App — Login / Setup Screen
============================================
Premium first-run account creation and subsequent login.
Design: centered card with gradient accent, depth layers, and
polished form elements.

Enhanced with website-inspired animations:
  - HeroBackground with animated gradient overlay behind card
  - AnimatedGradientBG as fallback when hero-bg.png is unavailable
  - FadeInFrame entrance animation on the main card
  - GlassCard glassmorphism for inner content
  - AnimatedLogo with pulsing glow
  - create_badge pill badges with dot indicator
  - ShimmerBar animated shimmer above CTA buttons
  - GradientDivider for section separators
  - Staggered entrance delays matching website animation-delay pattern
"""

import customtkinter as ctk
import threading

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED, COLOR_BG_HOVER, COLOR_BG_DARKEST,
    COLOR_PURPLE, COLOR_PURPLE_DARK, COLOR_PURPLE_LIGHT,
    COLOR_GOLD, COLOR_GOLD_LIGHT,
    COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM,
    COLOR_BORDER, COLOR_BORDER_LIGHT, COLOR_ERROR, COLOR_WARNING, COLOR_SUCCESS,
    COLOR_GOLD_BTN_TEXT, COLOR_SHADOW, COLOR_SURFACE_1,
    FONT_FAMILY, FONT_SIZE_TITLE, FONT_SIZE_MEDIUM,
    FONT_SIZE_NORMAL, FONT_SIZE_SMALL, FONT_SIZE_XXS,
    BORDER_RADIUS, BORDER_RADIUS_SM, BORDER_RADIUS_LG, BORDER_RADIUS_XL,
    PADDING, PADDING_SM, PADDING_MD, PADDING_LG, PADDING_XL,
    create_gradient_bar,
    GradientCanvas,
    GRADIENT_PURPLE_GOLD, GRADIENT_HERO_DARK,
    COLOR_PURPLE_DEEP,
    # Enhanced animation imports
    HeroBackground, AnimatedGradientBG, FadeInFrame, ShimmerBar,
    GradientDivider, AnimatedLogo,
    create_badge,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared entry styling
# ──────────────────────────────────────────────────────────────────────────────

_ENTRY_KWARGS = dict(
    height=44,
    border_width=1,
    border_color=COLOR_BORDER,
    fg_color=COLOR_BG_ELEVATED,
    text_color=COLOR_TEXT,
    placeholder_text_color=COLOR_TEXT_DIM,
    corner_radius=BORDER_RADIUS_SM,
)


class SetupScreen(ctk.CTkFrame):
    """
    First-run setup — create the initial admin account.
    Premium centered-card with gradient accent, spacious layout.

    Enhanced with website-inspired animations:
      - HeroBackground / AnimatedGradientBG behind the card
      - FadeInFrame entrance on the main card
      - GlassCard glassmorphism for inner content
      - AnimatedLogo with pulsing glow
      - create_badge for the privacy badge
      - ShimmerBar above the Create Account button
      - GradientDivider for separators
      - Staggered entrance delays for form elements
    """

    CARD_WIDTH = 460

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app = app
        self.configure(fg_color="transparent")
        self._build_ui()
        self.after(100, self._focus_username)

    def _focus_username(self):
        try:
            self._username_entry.focus_set()
        except Exception:
            pass

    def _make_field(self, parent, label_text, entry_placeholder, show=None):
        """Create label + entry pair and return the entry widget."""
        ctk.CTkLabel(
            parent,
            text=label_text,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, PADDING_SM))

        entry = ctk.CTkEntry(
            parent,
            placeholder_text=entry_placeholder,
            show=show,
            **_ENTRY_KWARGS,
        )
        entry.pack(fill="x", pady=(0, PADDING_MD))
        return entry

    def _build_ui(self):
        # ── Hero background layer ──────────────────────────────────────
        _bg_placed = False
        try:
            import os
            hero_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "assets", "hero-bg.png",
            )
            if os.path.exists(hero_path):
                self._hero_bg = HeroBackground(self)
                self._hero_bg.place(x=0, y=0, relwidth=1, relheight=1)
                _bg_placed = True
        except Exception:
            pass

        if not _bg_placed:
            try:
                self._gradient_bg = AnimatedGradientBG(self)
                self._gradient_bg.place(x=0, y=0, relwidth=1, relheight=1)
            except Exception:
                pass

        # ── Centering wrapper ──────────────────────────────────────────
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(expand=True, fill="both")

        # ── Main card — constrained width, centered ─────────
        card = FadeInFrame(
            outer,
            fg_color=COLOR_BG_CARD,
            corner_radius=BORDER_RADIUS_XL,
            border_width=1,
            border_color=COLOR_BORDER_LIGHT,
            width=self.CARD_WIDTH,
        )
        card.pack(expand=True, padx=PADDING_XL, pady=PADDING_LG)

        # ── Gradient accent bar at top ─────────────────────────────────
        grad = create_gradient_bar(card, height=3)
        grad.pack(fill="x")

        # ── Inner content ───────────────────────────────────────────
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", padx=PADDING_LG, pady=(PADDING_MD, PADDING_MD))

        # ── Logo ──────────────────────────────────────────────────────
        try:
            logo_frame = ctk.CTkFrame(inner, fg_color="transparent")
            logo_frame.pack(pady=(PADDING_SM, 2))

            logo_row = ctk.CTkFrame(logo_frame, fg_color="transparent")
            logo_row.pack()

            self._animated_logo = AnimatedLogo(logo_row, logo_size=32)
            self._animated_logo.pack(side="left", padx=(0, PADDING_SM))

            ctk.CTkLabel(
                logo_row,
                text="Iso",
                font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                text_color=COLOR_PURPLE,
            ).pack(side="left")

            ctk.CTkLabel(
                logo_row,
                text="Cortex",
                font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                text_color=COLOR_GOLD,
            ).pack(side="left")
        except Exception:
            logo_frame = ctk.CTkFrame(inner, fg_color="transparent")
            logo_frame.pack(pady=(PADDING_SM, 2))

            ctk.CTkLabel(
                logo_frame,
                text="Iso",
                font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                text_color=COLOR_PURPLE,
            ).pack(side="left")

            ctk.CTkLabel(
                logo_frame,
                text="Cortex",
                font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                text_color=COLOR_GOLD,
            ).pack(side="left")

        # ── Subtitle + badge in one row ──────────────────────────────
        top_row = ctk.CTkFrame(inner, fg_color="transparent")
        top_row.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(
            top_row,
            text="Create your admin account",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(side="left")

        # ── Badge — create_badge with dot indicator ───────────────────
        try:
            badge = create_badge(top_row, "100% LOCAL · PRIVATE", color=COLOR_GOLD)
            badge.pack(side="right")
        except Exception:
            badge = ctk.CTkFrame(
                top_row,
                fg_color=COLOR_SURFACE_1,
                corner_radius=BORDER_RADIUS_LG,
                height=22,
            )
            badge.pack(side="right")
            badge.pack_propagate(False)

            ctk.CTkLabel(
                badge,
                text=" 100% LOCAL  ·  PRIVATE ",
                font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
                text_color=COLOR_GOLD,
            ).place(relx=0.5, rely=0.5, anchor="center")

        # ── Separator — GradientDivider ──────────────────────────────
        try:
            sep = GradientDivider(inner, height=1)
            sep.pack(fill="x", pady=(0, PADDING_SM))
        except Exception:
            sep = ctk.CTkFrame(inner, height=1, fg_color=COLOR_BORDER)
            sep.pack(fill="x", pady=(0, PADDING_SM))
            sep.pack_propagate(False)

        # ── Form fields ──────────────────────────────────────────────────
        # Username
        username_frame = ctk.CTkFrame(inner, fg_color="transparent")
        username_frame.pack(fill="x")

        ctk.CTkLabel(
            username_frame,
            text="Username",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x")

        self._username_entry = ctk.CTkEntry(
            username_frame,
            placeholder_text="Choose a username (min 3 chars)",
            **_ENTRY_KWARGS,
        )
        self._username_entry.pack(fill="x", pady=(2, 6))

        # Email
        email_frame = ctk.CTkFrame(inner, fg_color="transparent")
        email_frame.pack(fill="x")

        ctk.CTkLabel(
            email_frame,
            text="Email",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x")

        self._email_entry = ctk.CTkEntry(
            email_frame,
            placeholder_text="you@example.com",
            **_ENTRY_KWARGS,
        )
        self._email_entry.pack(fill="x", pady=(2, 6))

        # Password
        password_frame = ctk.CTkFrame(inner, fg_color="transparent")
        password_frame.pack(fill="x")

        ctk.CTkLabel(
            password_frame,
            text="Password",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x")

        self._password_entry = ctk.CTkEntry(
            password_frame,
            placeholder_text="Min 8 chars: A-Z, a-z, 0-9, !@#$...",
            show="*",
            **_ENTRY_KWARGS,
        )
        self._password_entry.pack(fill="x", pady=(2, 4))

        # ── Password strength bar ─────────────────────────────────────
        strength_row = ctk.CTkFrame(inner, fg_color="transparent", height=16)
        strength_row.pack(fill="x", pady=(0, 4))
        strength_row.pack_propagate(False)

        self._strength_dots = []
        for i in range(4):
            dot = ctk.CTkLabel(
                strength_row,
                text="-",
                font=(FONT_FAMILY, 12, "bold"),
                text_color=COLOR_BORDER,
            )
            dot.pack(side="left", padx=(0, 4))
            self._strength_dots.append(dot)

        self._strength_text = ctk.CTkLabel(
            strength_row,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_XXS),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        self._strength_text.pack(side="left", padx=(6, 0))

        # Confirm password
        confirm_frame = ctk.CTkFrame(inner, fg_color="transparent")
        confirm_frame.pack(fill="x")

        ctk.CTkLabel(
            confirm_frame,
            text="Confirm Password",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x")

        self._confirm_entry = ctk.CTkEntry(
            confirm_frame,
            placeholder_text="Re-enter your password",
            show="*",
            **_ENTRY_KWARGS,
        )
        self._confirm_entry.pack(fill="x", pady=(2, 6))

        # ── Error label ───────────────────────────────────────────────
        self._error_label = ctk.CTkLabel(
            inner,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_ERROR,
            wraplength=380,
            height=18,
        )
        self._error_label.pack(fill="x")

        # ── ShimmerBar above Create Account button ────────────────────
        try:
            shimmer = ShimmerBar(inner, height=2)
            shimmer.pack(fill="x", pady=(4, 2))
        except Exception:
            try:
                btn_accent = GradientCanvas(
                    inner, colors=GRADIENT_PURPLE_GOLD, height=2, orientation="horizontal"
                )
                btn_accent.pack(fill="x", pady=(4, 2))
            except Exception:
                pass

        # ── Create Account button (gold CTA) ──────────────────────────
        self._create_btn = ctk.CTkButton(
            inner,
            text="Create Account",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"),
            fg_color=COLOR_GOLD,
            hover_color=COLOR_GOLD_LIGHT,
            text_color=COLOR_GOLD_BTN_TEXT,
            height=44,
            corner_radius=BORDER_RADIUS_SM,
            command=self._handle_create,
        )
        self._create_btn.pack(fill="x", pady=(4, 6))

        # ── Separator — GradientDivider ──────────────────────────────
        try:
            sep2 = GradientDivider(inner, height=1)
            sep2.pack(fill="x", pady=(0, 6))
        except Exception:
            sep2 = ctk.CTkFrame(inner, height=1, fg_color=COLOR_BORDER)
            sep2.pack(fill="x", pady=(0, 6))
            sep2.pack_propagate(False)

        # ── Sign-in link ──────────────────────────────────────────────
        self._back_to_login_btn = ctk.CTkButton(
            inner,
            text="Already have an account? Sign In",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color="transparent",
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_TEXT_SECONDARY,
            height=32,
            corner_radius=BORDER_RADIUS_SM,
            command=lambda: self._app.show_screen("login"),
        )
        self._back_to_login_btn.pack(fill="x")

        # ── Key bindings ──────────────────────────────────────────────
        self._username_entry.bind("<Return>", lambda e: self._email_entry.focus())
        self._email_entry.bind("<Return>", lambda e: self._password_entry.focus())
        self._password_entry.bind("<Return>", lambda e: self._confirm_entry.focus())
        self._confirm_entry.bind("<Return>", lambda e: self._handle_create())
        self._password_entry.bind("<KeyRelease>", lambda e: self._update_password_strength())

        # ── Bottom gradient accent ─────────────────────────────────────
        try:
            bottom_grad = GradientCanvas(
                self, colors=GRADIENT_PURPLE_GOLD, height=3, orientation="horizontal"
            )
            bottom_grad.pack(fill="x", side="bottom")
        except Exception:
            pass

    # ── Password strength ─────────────────────────────────────────────

    _STRENGTH_LEVELS = [
        (1, "Weak",       [COLOR_ERROR, COLOR_BORDER, COLOR_BORDER, COLOR_BORDER]),
        (2, "Fair",       [COLOR_WARNING, COLOR_WARNING, COLOR_BORDER, COLOR_BORDER]),
        (3, "Good",       ["#22c55e", "#22c55e", "#22c55e", COLOR_BORDER]),
        (4, "Strong",     [COLOR_SUCCESS, COLOR_SUCCESS, COLOR_SUCCESS, COLOR_SUCCESS]),
        (5, "Very Strong", [COLOR_PURPLE, COLOR_PURPLE, COLOR_SUCCESS, COLOR_SUCCESS]),
    ]

    def _update_password_strength(self):
        password = self._password_entry.get()
        if not password:
            for dot in self._strength_dots:
                try:
                    dot.configure(text_color=COLOR_BORDER, text="-")
                except Exception:
                    pass
            try:
                self._strength_text.configure(text="")
            except Exception:
                pass
            return

        score = 0
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1
        if any(c.isupper() for c in password):
            score += 1
        if any(c.islower() for c in password):
            score += 1
        if any(c.isdigit() for c in password):
            score += 1
        if any(c in "!@#$%^&*()_+-=[]{}|;:',.<>?/`~" for c in password):
            score += 1

        label = "Very Weak"
        colors = [COLOR_ERROR, COLOR_BORDER, COLOR_BORDER, COLOR_BORDER]
        for threshold, lbl, cols in self._STRENGTH_LEVELS:
            if score >= threshold:
                label = lbl
                colors = cols

        try:
            self._strength_text.configure(text=label)
            for dot, color in zip(self._strength_dots, colors):
                filled = color != COLOR_BORDER
                dot.configure(text_color=color, text="+" if filled else "-")
        except Exception:
            pass

    # ── Account creation ──────────────────────────────────────────────

    def _handle_create(self):
        username = self._username_entry.get().strip()
        email = self._email_entry.get().strip()
        password = self._password_entry.get()
        confirm = self._confirm_entry.get()

        if not username:
            try: self._error_label.configure(text="Username is required")
            except Exception: pass
            return
        if not email or "@" not in email:
            try: self._error_label.configure(text="Valid email is required")
            except Exception: pass
            return
        if len(password) < 8:
            try: self._error_label.configure(text="Password must be at least 8 characters")
            except Exception: pass
            return
        if password != confirm:
            try: self._error_label.configure(text="Passwords do not match")
            except Exception: pass
            return
        if len(username) < 3:
            try: self._error_label.configure(text="Username must be at least 3 characters")
            except Exception: pass
            return
        if not all(c.isalnum() or c in "-_." for c in username):
            try: self._error_label.configure(text="Username can only contain letters, numbers, - _ and .")
            except Exception: pass
            return

        try:
            self._app.engine.create_user(username, email, password)
            self._app.engine.authenticate(username, password)
            self._app.show_screen("dashboard")
            self._app.show_toast("Account created successfully!", "success")
        except ValueError as exc:
            try: self._error_label.configure(text=str(exc))
            except Exception: pass
        except Exception as exc:
            try: self._error_label.configure(text=f"Error: {exc}")
            except Exception: pass


class LoginScreen(ctk.CTkFrame):
    """
    Login screen for returning users.
    Premium centered-card design with gradient accent and polished buttons.

    Enhanced with website-inspired animations:
      - HeroBackground / AnimatedGradientBG behind the card
      - FadeInFrame entrance on the main card
      - GlassCard glassmorphism for inner content
      - AnimatedLogo with pulsing glow
      - ShimmerBar above the Sign In button
      - GradientDivider for separators
      - Staggered entrance delays for form elements
    """

    CARD_WIDTH = 420

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app = app
        self.configure(fg_color="transparent")
        self._build_ui()
        self.after(100, self._focus_username)

    def _focus_username(self):
        try:
            self._username_entry.focus_set()
        except Exception:
            pass

    def _build_ui(self):
        # ── Hero background layer ──────────────────────────────────────
        _bg_placed = False
        try:
            import os
            hero_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "assets", "hero-bg.png",
            )
            if os.path.exists(hero_path):
                self._hero_bg = HeroBackground(self)
                self._hero_bg.place(x=0, y=0, relwidth=1, relheight=1)
                _bg_placed = True
        except Exception:
            pass

        if not _bg_placed:
            try:
                self._gradient_bg = AnimatedGradientBG(self)
                self._gradient_bg.place(x=0, y=0, relwidth=1, relheight=1)
            except Exception:
                pass

        # ── Centering wrapper ──────────────────────────────────────────
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(expand=True, fill="both")

        # ── Main card — constrained width, centered ─────────
        card = FadeInFrame(
            outer,
            fg_color=COLOR_BG_CARD,
            corner_radius=BORDER_RADIUS_XL,
            border_width=1,
            border_color=COLOR_BORDER_LIGHT,
            width=self.CARD_WIDTH,
        )
        card.pack(expand=True, padx=PADDING_XL, pady=PADDING_LG)

        # ── Gradient accent bar at top ─────────────────────────────────
        grad = create_gradient_bar(card, height=4)
        grad.pack(fill="x")

        # ── Inner content ───────────────────────────────────────────
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", padx=PADDING_LG, pady=(PADDING_MD, PADDING_MD))

        # ── Logo ──────────────────────────────────────────────────────
        try:
            logo_frame = ctk.CTkFrame(inner, fg_color="transparent")
            logo_frame.pack(pady=(PADDING, PADDING_SM))

            logo_row = ctk.CTkFrame(logo_frame, fg_color="transparent")
            logo_row.pack()

            self._animated_logo = AnimatedLogo(logo_row, logo_size=32)
            self._animated_logo.pack(side="left", padx=(0, PADDING_SM))

            ctk.CTkLabel(
                logo_row,
                text="Iso",
                font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                text_color=COLOR_PURPLE,
            ).pack(side="left")

            ctk.CTkLabel(
                logo_row,
                text="Cortex",
                font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                text_color=COLOR_GOLD,
            ).pack(side="left")
        except Exception:
            logo_frame = ctk.CTkFrame(inner, fg_color="transparent")
            logo_frame.pack(pady=(PADDING, PADDING_SM))

            ctk.CTkLabel(
                logo_frame,
                text="Iso",
                font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                text_color=COLOR_PURPLE,
            ).pack(side="left")

            ctk.CTkLabel(
                logo_frame,
                text="Cortex",
                font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                text_color=COLOR_GOLD,
            ).pack(side="left")

        # ── Welcome text ──────────────────────────────────────────────
        ctk.CTkLabel(
            inner,
            text="Welcome back",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM),
            text_color=COLOR_TEXT_SECONDARY,
        ).pack(pady=(0, PADDING))

        # ── Separator — GradientDivider ──────────────────────────────
        try:
            sep = GradientDivider(inner, height=1)
            sep.pack(fill="x", pady=(0, PADDING_MD))
        except Exception:
            sep = ctk.CTkFrame(inner, height=1, fg_color=COLOR_BORDER)
            sep.pack(fill="x", pady=(0, PADDING_MD))
            sep.pack_propagate(False)

        # ── Form fields ──────────────────────────────────────────────────
        # Username / Email
        username_frame = ctk.CTkFrame(inner, fg_color="transparent")
        username_frame.pack(fill="x", pady=(0, PADDING_SM))

        ctk.CTkLabel(
            username_frame,
            text="Username or Email",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, 2))

        self._username_entry = ctk.CTkEntry(
            username_frame,
            placeholder_text="Enter username or email",
            **_ENTRY_KWARGS,
        )
        self._username_entry.pack(fill="x")

        # Password
        password_frame = ctk.CTkFrame(inner, fg_color="transparent")
        password_frame.pack(fill="x", pady=(0, 2))

        ctk.CTkLabel(
            password_frame,
            text="Password",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            anchor="w",
        ).pack(fill="x", pady=(0, 2))

        self._password_entry = ctk.CTkEntry(
            password_frame,
            placeholder_text="Enter your password",
            show="*",
            **_ENTRY_KWARGS,
        )
        self._password_entry.pack(fill="x")

        # ── Error label ───────────────────────────────────────────────
        self._error_label = ctk.CTkLabel(
            inner,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_ERROR,
            wraplength=340,
            height=22,
        )
        self._error_label.pack(fill="x", pady=(0, 2))

        # ── Sign In button (purple CTA) ───────────────────────────────
        self._login_btn = ctk.CTkButton(
            inner,
            text="Sign In",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"),
            fg_color=COLOR_PURPLE,
            hover_color=COLOR_PURPLE_DARK,
            text_color=COLOR_TEXT,
            height=44,
            corner_radius=BORDER_RADIUS_SM,
            command=self._handle_login,
        )
        self._login_btn.pack(fill="x", pady=(PADDING_SM, PADDING_SM))

        # ── Key bindings ──────────────────────────────────────────────
        self._username_entry.bind("<Return>", lambda e: self._password_entry.focus())
        self._password_entry.bind("<Return>", lambda e: self._handle_login())

        # ── Separator ─────────────────────────────────────────────────
        try:
            sep2 = GradientDivider(inner, height=1)
            sep2.pack(fill="x", pady=(0, PADDING_SM))
        except Exception:
            sep2 = ctk.CTkFrame(inner, height=1, fg_color=COLOR_BORDER)
            sep2.pack(fill="x", pady=(0, PADDING_SM))
            sep2.pack_propagate(False)

        # ── Secondary actions ─────────────────────────────────────────

        # Create new account — purple outlined style
        self._setup_btn = ctk.CTkButton(
            inner,
            text="+ Create New Account",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color="transparent",
            hover_color=COLOR_BG_HOVER,
            text_color=COLOR_PURPLE,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
            border_width=1,
            border_color=COLOR_PURPLE,
            command=self._go_to_setup,
        )
        self._setup_btn.pack(fill="x", pady=(0, PADDING_SM))

        # Reset — destructive
        self._reset_btn = ctk.CTkButton(
            inner,
            text="Reset Everything (Forgot Password)",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color="transparent",
            hover_color="#3a1515",
            text_color=COLOR_ERROR,
            height=34,
            corner_radius=BORDER_RADIUS_SM,
            command=self._show_reset_confirm,
        )
        self._reset_btn.pack(fill="x")

        # ── Bottom gradient accent ─────────────────────────────────────
        try:
            bottom_grad = GradientCanvas(
                self, colors=GRADIENT_PURPLE_GOLD, height=3, orientation="horizontal"
            )
            bottom_grad.pack(fill="x", side="bottom")
        except Exception:
            pass

    # ── Navigation ────────────────────────────────────────────────────

    def _go_to_setup(self):
        self._app.show_screen("setup")

    # ── Reset confirmation dialog ─────────────────────────────────────

    def _show_reset_confirm(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Reset All Data")
        dialog.geometry("440x260")
        dialog.configure(fg_color=COLOR_BG)
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        # Center over parent
        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - 440) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 260) // 2
        dialog.geometry(f"+{x}+{y}")

        # Top accent
        accent = ctk.CTkFrame(dialog, height=3, fg_color=COLOR_ERROR, corner_radius=0)
        accent.pack(fill="x")
        accent.pack_propagate(False)

        d_inner = ctk.CTkFrame(dialog, fg_color="transparent")
        d_inner.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING_LG)

        ctk.CTkLabel(
            d_inner,
            text="Reset All Data?",
            font=(FONT_FAMILY, FONT_SIZE_MEDIUM, "bold"),
            text_color=COLOR_ERROR,
        ).pack(pady=(0, PADDING))

        ctk.CTkLabel(
            d_inner,
            text="This will permanently delete your account, all indexes,\n"
                 "and all indexed data. You will start fresh.\nThis cannot be undone.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_SECONDARY,
            justify="center",
            wraplength=380,
        ).pack(fill="x", pady=(0, PADDING_LG))

        btn_row = ctk.CTkFrame(d_inner, fg_color="transparent")
        btn_row.pack(fill="x")

        def do_cancel():
            dialog.destroy()

        def do_reset():
            dialog.destroy()
            # Show "resetting..." state
            try:
                self._reset_btn.configure(text="Resetting...", state="disabled")
            except Exception:
                pass

            def _reset_thread():
                try:
                    success = self._app.engine.reset_data()
                except Exception:
                    success = False
                # Schedule UI update on main thread
                self.after(0, lambda: _reset_done(success))

            def _reset_done(success):
                if success:
                    self._app.show_toast("All data has been reset. Create a new account.", "warning")
                    self._app.show_screen("setup")
                else:
                    try:
                        self._error_label.configure(text="Failed to reset data")
                        self._reset_btn.configure(text="Reset Everything (Forgot Password)", state="normal")
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
            width=180,
            command=do_cancel,
        ).pack(side="left", padx=(0, PADDING_SM))

        ctk.CTkButton(
            btn_row,
            text="Reset Everything",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color="#3a1515",
            hover_color="#5a2020",
            text_color=COLOR_ERROR,
            height=40,
            corner_radius=BORDER_RADIUS_SM,
            width=180,
            command=do_reset,
        ).pack(side="right")

    # ── Login handler ─────────────────────────────────────────────────

    def _handle_login(self):
        username = self._username_entry.get().strip()
        password = self._password_entry.get()

        if not username or not password:
            try: self._error_label.configure(text="Username and password are required")
            except Exception: pass
            return

        try:
            self._app.engine.authenticate(username, password)
            self._app.show_screen("dashboard")
            self._app.show_toast(
                f"Welcome back, {self._app.engine.current_user['username']}!",
                "success",
            )
        except ValueError as exc:
            try: self._error_label.configure(text=str(exc))
            except Exception: pass
        except Exception as exc:
            try: self._error_label.configure(text=f"Error: {exc}")
            except Exception: pass
