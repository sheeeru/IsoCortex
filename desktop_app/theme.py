"""
IsoCortex Desktop App — Theme Configuration
=============================================
Premium dark & light theme with Canvas gradients, glow effects,
selectable text helpers, and a purple-gold brand palette.

Website-inspired enhancements:
  - FloatingWidget: smooth vertical float animation (6s infinite)
  - FadeInFrame: gradual opacity fade-in on widget creation
  - ShimmerBar: animated shimmer effect (3s linear infinite)
  - GradientDivider: transparent → purple → gold → transparent divider
  - AnimatedGradientBG: multi-layered radial gradient background canvas
  - GlassCard: glassmorphism card with translucent borders + glow
  - AnimatedPulseGlow: pulsing glow ring around any widget
  - AnimatedLogo: logo image with hover glow pulse
  - stagger_animation: utility for cascading entrance delays

Performance optimizations:
  - GradientCanvas: uses PIL PhotoImage instead of w canvas lines per resize
  - ShimmerBar: uses PIL PhotoImage instead of w create_line calls 33x/sec
  - AnimatedGradientBG: uses PIL ImageDraw for radial gradients, 200ms refresh
  - HeroBackground: caches base blurred image, 250ms refresh with ImageEnhance
  - AnimatedPulseGlow: reduced refresh to 150ms
  - AnimatedLogo: replaced AnimatedPulseGlow with static glow frame
"""

import math
import os
import tkinter as tk

try:
    import customtkinter as ctk
except ImportError:
    ctk = None  # type: ignore

# ══════════════════════════════════════════════════════════════════════
# PIL Availability Check
# ══════════════════════════════════════════════════════════════════════

_HAS_PIL = False
try:
    from PIL import Image as _PILImage
    from PIL import ImageDraw as _PILImageDraw
    from PIL import ImageTk as _PILImageTk
    from PIL import ImageFilter as _PILImageFilter
    from PIL import ImageEnhance as _PILImageEnhance
    _HAS_PIL = True
except ImportError:
    _PILImage = None
    _PILImageDraw = None
    _PILImageTk = None
    _PILImageFilter = None
    _PILImageEnhance = None


# ══════════════════════════════════════════════════════════════════════
# Asset Paths
# ══════════════════════════════════════════════════════════════════════

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

def get_asset_path(filename: str) -> str:
    """Return the absolute path to an asset file."""
    return os.path.join(_ASSETS_DIR, filename)


# ══════════════════════════════════════════════════════════════════════
# Theme Mode Manager
# ══════════════════════════════════════════════════════════════════════

class ThemeMode:
    """Manages light/dark/system theme switching."""

    _current = "dark"  # "dark" or "light"

    DARK = "dark"
    LIGHT = "light"

    @classmethod
    def get(cls) -> str:
        return cls._current

    @classmethod
    def set(cls, mode: str) -> None:
        if mode in (cls.DARK, cls.LIGHT):
            cls._current = mode
            _apply_colors(mode)
            try:
                import customtkinter as ctk
                ctk.set_appearance_mode(mode)
            except Exception:
                pass

    @classmethod
    def toggle(cls) -> str:
        cls._current = cls.LIGHT if cls._current == cls.DARK else cls.DARK
        _apply_colors(cls._current)
        try:
            import customtkinter as ctk
            ctk.set_appearance_mode(cls._current)
        except Exception:
            pass
        return cls._current


def _apply_colors(mode: str):
    """Apply theme colors globally by updating module-level variables."""
    global COLOR_BG_DARKEST, COLOR_BG, COLOR_BG_CARD, COLOR_BG_ELEVATED
    global COLOR_BG_HOVER, COLOR_TEXT, COLOR_TEXT_SECONDARY, COLOR_TEXT_DIM
    global COLOR_BORDER, COLOR_BORDER_LIGHT, COLOR_GOLD_BTN_TEXT
    global COLOR_SIDEBAR_BG, COLOR_CARD_GLASS, COLOR_INPUT_BG
    global COLOR_SHADOW, COLOR_SHADOW_LIGHT, COLOR_SURFACE_1, COLOR_SURFACE_2
    global COLOR_GLASS_BG, COLOR_GLASS_BORDER

    if mode == "light":
        COLOR_BG_DARKEST   = "#d8d8e8"
        COLOR_BG            = "#eeeef6"
        COLOR_BG_CARD       = "#ffffff"
        COLOR_BG_ELEVATED   = "#e8e8f2"
        COLOR_BG_HOVER      = "#dddde8"
        COLOR_TEXT           = "#1a1a2e"
        COLOR_TEXT_SECONDARY = "#5c5c7a"
        COLOR_TEXT_DIM      = "#9898b0"
        COLOR_BORDER        = "#c8c8d8"
        COLOR_BORDER_LIGHT  = "#d8d8e8"
        COLOR_GOLD_BTN_TEXT = "#1a1a2e"
        COLOR_SIDEBAR_BG    = "#e4e4f0"
        COLOR_CARD_GLASS    = "#f8f8ff"
        COLOR_INPUT_BG      = "#f0f0f8"
        COLOR_SHADOW        = "#c0c0d4"
        COLOR_SHADOW_LIGHT  = "#d8d8e8"
        COLOR_SURFACE_1     = "#f4f4fa"
        COLOR_SURFACE_2     = "#eaeaf4"
        COLOR_GLASS_BG      = "#e8e0f8"
        COLOR_GLASS_BORDER  = "#c8b8e8"
    else:
        COLOR_BG_DARKEST   = "#04040a"
        COLOR_BG            = "#0a0a12"
        COLOR_BG_CARD       = "#111120"
        COLOR_BG_ELEVATED   = "#181830"
        COLOR_BG_HOVER      = "#20203c"
        COLOR_TEXT           = "#eaeaf2"
        COLOR_TEXT_SECONDARY = "#8888a8"
        COLOR_TEXT_DIM      = "#505068"
        COLOR_BORDER        = "#1c1c34"
        COLOR_BORDER_LIGHT  = "#242440"
        COLOR_GOLD_BTN_TEXT = "#0a0a12"
        COLOR_SIDEBAR_BG    = "#0c0c18"
        COLOR_CARD_GLASS    = "#141428"
        COLOR_INPUT_BG      = "#161630"
        COLOR_SHADOW        = "#060610"
        COLOR_SHADOW_LIGHT  = "#0e0e1c"
        COLOR_SURFACE_1     = "#0f0f1e"
        COLOR_SURFACE_2     = "#14142a"
        COLOR_GLASS_BG      = "#1a1838"
        COLOR_GLASS_BORDER  = "#2a2858"


# ══════════════════════════════════════════════════════════════════════
# Brand Colors (mode-independent)
# ══════════════════════════════════════════════════════════════════════

COLOR_PURPLE        = "#7c3aed"
COLOR_PURPLE_DARK   = "#5b21b6"
COLOR_PURPLE_LIGHT  = "#a78bfa"
COLOR_PURPLE_GLOW   = "#7c3aed"
COLOR_PURPLE_DEEP   = "#4c1d95"
COLOR_GOLD          = "#d4a017"
COLOR_GOLD_LIGHT    = "#eab308"
COLOR_GOLD_DIM      = "#a17e12"

COLOR_TEXT_PURPLE   = "#a78bfa"
COLOR_TEXT_GOLD     = "#d4a017"

COLOR_SUCCESS       = "#22c55e"
COLOR_WARNING       = "#f59e0b"
COLOR_ERROR         = "#ef4444"
COLOR_INFO          = "#3b82f6"

COLOR_BORDER_FOCUS  = "#7c3aed"

GRADIENT_PURPLE_GOLD = ["#7c3aed", "#d4a017"]
GRADIENT_SIDEBAR     = ["#0a0520", "#1a0a3e", "#0a0520"]
GRADIENT_HERO_DARK   = ["#4c1d95", "#7c3aed", "#d4a017"]

# Animation timing constants (from website globals.css)
ANIM_FLOAT_DURATION    = 6000  # 6s float animation
ANIM_PULSE_DURATION    = 3000  # 3s pulse-glow animation
ANIM_SHIMMER_DURATION  = 3000  # 3s shimmer animation
ANIM_FADEIN_DURATION   = 1000  # 1s fade-in

# Staggered entrance delay intervals (from website)
ANIM_DELAY_200 = 200
ANIM_DELAY_400 = 400
ANIM_DELAY_600 = 600
ANIM_DELAY_800 = 800


# ══════════════════════════════════════════════════════════════════════
# Typography
# ══════════════════════════════════════════════════════════════════════

FONT_FAMILY         = "Segoe UI"
FONT_FAMILY_MONO    = "Consolas"
FONT_SIZE_XXS       = 9
FONT_SIZE_SMALL     = 11
FONT_SIZE_NORMAL    = 12
FONT_SIZE_MEDIUM    = 13
FONT_SIZE_LARGE     = 15
FONT_SIZE_TITLE     = 22
FONT_SIZE_HERO      = 28


# ══════════════════════════════════════════════════════════════════════
# Layout & Spacing
# ══════════════════════════════════════════════════════════════════════

SIDEBAR_WIDTH       = 260
WINDOW_MIN_WIDTH    = 1060
WINDOW_MIN_HEIGHT   = 700
BORDER_RADIUS       = 12
BORDER_RADIUS_SM    = 8
BORDER_RADIUS_LG    = 16
BORDER_RADIUS_XL    = 20
PADDING             = 16
PADDING_SM          = 8
PADDING_MD          = 12
PADDING_LG          = 24
PADDING_XL          = 36
SHADOW_OFFSET       = 2


# ══════════════════════════════════════════════════════════════════════
# Initialize default colors (dark mode)
# ══════════════════════════════════════════════════════════════════════

COLOR_BG_DARKEST   = "#04040a"
COLOR_BG            = "#0a0a12"
COLOR_BG_CARD       = "#111120"
COLOR_BG_ELEVATED   = "#181830"
COLOR_BG_HOVER      = "#20203c"
COLOR_TEXT           = "#eaeaf2"
COLOR_TEXT_SECONDARY = "#8888a8"
COLOR_TEXT_DIM      = "#505068"
COLOR_BORDER        = "#1c1c34"
COLOR_BORDER_LIGHT  = "#242440"
COLOR_GOLD_BTN_TEXT = "#0a0a12"
COLOR_SIDEBAR_BG    = "#0c0c18"
COLOR_CARD_GLASS    = "#141428"
COLOR_INPUT_BG      = "#161630"
COLOR_SHADOW        = "#060610"
COLOR_SHADOW_LIGHT  = "#0e0e1c"
COLOR_SURFACE_1     = "#0f0f1e"
COLOR_SURFACE_2     = "#14142a"
COLOR_GLASS_BG      = "#1a1838"
COLOR_GLASS_BORDER  = "#2a2858"


# ══════════════════════════════════════════════════════════════════════
# Color Utilities
# ══════════════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to (r, g, b) tuple.
    Returns None for non-hex values like 'transparent'.
    """
    if not hex_color or not hex_color.startswith("#"):
        return None
    h = hex_color.lstrip("#")
    if len(h) < 6:
        return None
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return None


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert (r, g, b) to hex color string."""
    return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"


def _interpolate_colors(colors: list, t: float) -> str:
    """Interpolate between a list of hex colors at position t (0.0 - 1.0)."""
    if len(colors) == 1:
        return colors[0]
    t = max(0.0, min(1.0, t))
    segment = t * (len(colors) - 1)
    idx = min(int(segment), len(colors) - 2)
    local_t = segment - idx
    c1 = _hex_to_rgb(colors[idx])
    c2 = _hex_to_rgb(colors[idx + 1])
    r = int(c1[0] + (c2[0] - c1[0]) * local_t)
    g = int(c1[1] + (c2[1] - c1[1]) * local_t)
    b = int(c1[2] + (c2[2] - c1[2]) * local_t)
    return _rgb_to_hex(r, g, b)


def _interpolate_colors_rgb(colors: list, t: float) -> tuple:
    """Interpolate between a list of hex colors at position t (0.0 - 1.0).
    Returns (r, g, b) tuple directly, avoiding hex conversion overhead.
    Used by PIL-based rendering for performance.
    """
    if len(colors) == 1:
        return _hex_to_rgb(colors[0])
    t = max(0.0, min(1.0, t))
    segment = t * (len(colors) - 1)
    idx = min(int(segment), len(colors) - 2)
    local_t = segment - idx
    c1 = _hex_to_rgb(colors[idx])
    c2 = _hex_to_rgb(colors[idx + 1])
    return (
        int(c1[0] + (c2[0] - c1[0]) * local_t),
        int(c1[1] + (c2[1] - c1[1]) * local_t),
        int(c1[2] + (c2[2] - c1[2]) * local_t),
    )


def _dim_hex(hex_color: str, factor: float = 0.4) -> str:
    """Darken a hex color by a factor (0 = black, 1 = unchanged)."""
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(int(r * factor), int(g * factor), int(b * factor))


def _brighten_hex(hex_color: str, factor: float = 0.3) -> str:
    """Brighten a hex color toward white by a factor."""
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )


def _blend_colors(hex1: str, hex2: str, t: float) -> str:
    """Blend two hex colors. t=0 returns hex1, t=1 returns hex2.
    Returns hex2 unchanged if either color is not a valid hex.
    """
    c1 = _hex_to_rgb(hex1)
    c2 = _hex_to_rgb(hex2)
    if c1 is None or c2 is None:
        return hex2 if c2 is not None else (hex1 if c1 is not None else "#000000")
    return _rgb_to_hex(
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _blend_rgb(rgb1: tuple, rgb2: tuple, t: float) -> tuple:
    """Blend two (r, g, b) tuples. t=0 returns rgb1, t=1 returns rgb2.
    Used by PIL-based rendering for performance.
    Returns rgb2 unchanged if either input is None.
    """
    if rgb1 is None:
        return rgb2 or (0, 0, 0)
    if rgb2 is None:
        return rgb1
    return (
        int(rgb1[0] + (rgb2[0] - rgb1[0]) * t),
        int(rgb1[1] + (rgb2[1] - rgb1[1]) * t),
        int(rgb1[2] + (rgb2[2] - rgb1[2]) * t),
    )


def stagger_animation(widget, delay_ms: int = 0, callback=None):
    """
    Schedule a callback after a delay to create staggered entrance animations.
    Used to mimic the website's animation-delay-200/400/600/800 pattern.
    """
    if callback:
        widget.after(delay_ms, callback)
    return widget


# ══════════════════════════════════════════════════════════════════════
# GradientCanvas — True Smooth Gradient Using Canvas
# ══════════════════════════════════════════════════════════════════════

class GradientCanvas(ctk.CTkCanvas):
    """
    A Canvas widget that renders a smooth gradient fill.
    Supports horizontal and vertical orientations.
    Auto-redraws on resize.

    Performance: Uses PIL to generate the gradient as a single PhotoImage
    instead of creating w individual canvas lines per resize.
    Falls back to canvas lines if PIL is not available.
    """

    def __init__(self, master, colors=None, height=4, orientation="horizontal", **kwargs):
        super().__init__(
            master,
            height=height,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._colors = colors or GRADIENT_PURPLE_GOLD
        self._orientation = orientation
        self._draw_scheduled = False
        self._photo = None  # prevent GC of PhotoImage
        self._use_pil = _HAS_PIL
        self.bind("<Configure>", self._on_configure)
        self.after(50, self._draw_gradient)

    def _on_configure(self, event=None):
        if not self._draw_scheduled:
            self._draw_scheduled = True
            self.after_idle(self._deferred_draw)

    def _deferred_draw(self):
        self._draw_scheduled = False
        try:
            self._draw_gradient()
        except tk.TclError:
            pass

    def _draw_gradient(self):
        try:
            self.delete("all")
        except tk.TclError:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2 or h < 2:
            return

        # PIL-optimized path: single PhotoImage instead of w canvas lines
        if self._use_pil:
            try:
                self._draw_gradient_pil(w, h)
                return
            except Exception:
                self._use_pil = False  # fall back to canvas lines

        # Fallback: canvas lines (original method)
        steps = max(w if self._orientation == "horizontal" else h, 4)
        for i in range(steps):
            t = i / max(steps - 1, 1)
            color = _interpolate_colors(self._colors, t)
            if self._orientation == "horizontal":
                x = i * w / steps
                self.create_line(x, 0, x, h, fill=color)
            else:
                y = i * h / steps
                self.create_line(0, y, w, y, fill=color)

    def _draw_gradient_pil(self, w: int, h: int):
        """Render gradient using PIL as a single PhotoImage."""
        img = _PILImage.new("RGB", (w, h))
        draw = _PILImageDraw.Draw(img)

        if self._orientation == "horizontal":
            for x in range(w):
                t = x / max(w - 1, 1)
                color = _interpolate_colors_rgb(self._colors, t)
                draw.line([(x, 0), (x, h - 1)], fill=color)
        else:
            for y in range(h):
                t = y / max(h - 1, 1)
                color = _interpolate_colors_rgb(self._colors, t)
                draw.line([(0, y), (w - 1, y)], fill=color)

        self._photo = _PILImageTk.PhotoImage(img)
        self.create_image(0, 0, anchor="nw", image=self._photo)

    def update_colors(self, colors):
        self._colors = colors
        self._draw_gradient()


# ══════════════════════════════════════════════════════════════════════
# PulseIndicator — Animated Pulsing Status Dot
# ══════════════════════════════════════════════════════════════════════

class PulseIndicator(ctk.CTkCanvas):
    """
    An animated pulsing dot for status indicators.
    Draws a solid dot with an expanding/fading ring.
    Inspired by the website's pulse-glow animation.
    """

    def __init__(self, master, color=COLOR_SUCCESS, size=10, **kwargs):
        super().__init__(
            master,
            width=size + 12,
            height=size + 12,
            highlightthickness=0,
            bd=0,
        )
        self._color = color
        self._size = size
        self._running = True
        self._phase = 0.0
        self._draw()

    def _draw(self):
        if not self._running:
            return
        try:
            self.delete("all")
            s = self._size + 12
            cx, cy = s // 2, s // 2
            r = self._size // 2

            # Static ring (no animation)
            ring_r = r + 3
            try:
                self.create_oval(
                    cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
                    outline=self._color, width=1,
                )
            except tk.TclError:
                self._running = False
                return

            # Solid center dot
            try:
                self.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    fill=self._color, outline="",
                )
            except tk.TclError:
                self._running = False
                return
            # Draw once, no loop
        except tk.TclError:
            self._running = False

    def set_color(self, color):
        self._color = color

    def destroy(self):
        self._running = False
        try:
            super().destroy()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# NEW: FloatingWidget — Smooth Vertical Float Animation
# Inspired by the website's @keyframes float (6s ease-in-out infinite)
# ══════════════════════════════════════════════════════════════════════

class FloatingWidget:
    """
    Mixin that adds a smooth vertical float animation to any widget.
    The widget gently bobs up and down by `amplitude` pixels over
    `duration` ms, mimicking the website's CSS `float` animation.
    """

    def enable_float(self, amplitude=5, duration=ANIM_FLOAT_DURATION):
        """Start the floating animation on this widget."""
        self._float_amplitude = amplitude
        self._float_duration = duration
        self._float_phase = 0.0
        self._float_running = True
        self._float_offset_y = 0
        self._do_float()

    def _do_float(self):
        if not getattr(self, "_float_running", False):
            return
        try:
            import math as _m
            # Smooth sine wave: 0 → amplitude → 0 → -amplitude → 0
            new_offset = _m.sin(self._float_phase) * self._float_amplitude
            delta = new_offset - self._float_offset_y
            self._float_offset_y = new_offset
            # Move using place or pack offset
            try:
                info = self.place_info()
                if info and info.get("y"):
                    current_y = int(info.get("y", 0))
                    self.place(y=current_y + int(delta))
                else:
                    # Fallback: use pack
                    current_y = self.winfo_y()
                    self.place(x=self.winfo_x(), y=current_y + int(delta))
            except Exception:
                pass
            self._float_phase += (2 * _m.pi) / (self._float_duration / 50)
            self.after(50, self._do_float)
        except (tk.TclError, AttributeError):
            self._float_running = False

    def stop_float(self):
        self._float_running = False


class FloatingCanvas(ctk.CTkCanvas):
    """
    A canvas container that makes its single child widget float
    up and down smoothly, matching the website's float animation.
    """

    def __init__(self, master, amplitude=5, duration=ANIM_FLOAT_DURATION, **kwargs):
        super().__init__(
            master,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._amplitude = amplitude
        self._duration = duration
        self._phase = 0.0
        self._running = True
        self._child = None
        self._offset_y = 0
        self.bind("<Configure>", lambda e: self.after(50, self._draw))
        self.after(100, self._animate)

    def set_child(self, widget):
        """Set the widget to float."""
        self._child = widget
        self._draw()

    def _draw(self):
        try:
            self.delete("all")
        except tk.TclError:
            return
        if self._child is None:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        cw = self._child.winfo_reqwidth()
        ch = self._child.winfo_reqheight()
        if w < 2 or h < 2:
            return
        x = (w - cw) // 2
        y = int((h - ch) // 2 + self._offset_y)
        self._child.place_forget()
        try:
            self._child.place(in_=self, x=x, y=y)
        except tk.TclError:
            pass

    def _animate(self):
        if not self._running:
            return
        try:
            self._offset_y = math.sin(self._phase) * self._amplitude
            self._phase += (2 * math.pi) / (self._duration / 50)
            self._draw()
            self.after(50, self._animate)
        except tk.TclError:
            self._running = False

    def destroy(self):
        self._running = False
        try:
            super().destroy()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# NEW: FadeInFrame — Simulated Fade-In on Widget Creation
# Inspired by the website's @keyframes fade-in (1s ease-out)
# Since Tkinter doesn't support true alpha, we simulate by
# transitioning the fg_color from background to target.
# ══════════════════════════════════════════════════════════════════════

class FadeInFrame(ctk.CTkFrame):
    """
    A frame that performs a simulated fade-in effect on creation
    by transitioning its background color from the parent bg
    to the target color over ~1 second.

    Skips animation entirely if target_fg is 'transparent' or
    not a valid hex color (avoids ValueError spam).
    """

    def __init__(self, master, fade_duration=ANIM_FADEIN_DURATION, delay=0, **kwargs):
        self._target_fg = kwargs.pop("fg_color", COLOR_BG_CARD)
        super().__init__(master, fg_color=COLOR_BG, **kwargs)
        self._fade_duration = fade_duration
        self._fade_steps = 20
        self._fade_delay = delay
        self._current_step = 0

        # If target is 'transparent' or not a valid hex, just set it immediately
        if not self._target_fg or not self._target_fg.startswith("#") or len(self._target_fg) < 7:
            try:
                self.configure(fg_color=self._target_fg)
            except Exception:
                pass
            self._fading = False
            return

        self._fading = True
        # Start fade after delay
        if delay > 0:
            self.after(delay, self._start_fade)
        else:
            self.after(10, self._start_fade)

    def _start_fade(self):
        self._current_step = 0
        self._do_fade_step()

    def _do_fade_step(self):
        if not self._fading:
            return
        try:
            if self._current_step >= self._fade_steps:
                self.configure(fg_color=self._target_fg)
                return
            t = self._current_step / self._fade_steps
            # Ease-out curve
            t = 1 - (1 - t) ** 3
            new_color = _blend_colors(COLOR_BG, self._target_fg, t)
            self.configure(fg_color=new_color)
            self._current_step += 1
            step_delay = max(int(self._fade_duration / self._fade_steps), 15)
            self.after(step_delay, self._do_fade_step)
        except (tk.TclError, ValueError, Exception):
            # On any error, just snap to target color and stop
            try:
                self.configure(fg_color=self._target_fg)
            except Exception:
                pass
            self._fading = False


# ══════════════════════════════════════════════════════════════════════
# NEW: ShimmerBar — Animated Shimmer Effect
# Inspired by the website's @keyframes shimmer (3s linear infinite)
# Creates a moving highlight streak across the bar.
# ══════════════════════════════════════════════════════════════════════

class ShimmerBar(tk.Canvas):
    """
    An animated shimmer bar that shows a moving gradient highlight.
    Mimics the website's shimmer CSS animation.

    Performance: Uses PIL to generate the shimmer strip as a PhotoImage
    instead of creating w canvas lines every 30ms.
    Falls back to canvas lines if PIL is not available.
    """

    def __init__(self, master, height=3, colors=None, duration=ANIM_SHIMMER_DURATION, **kwargs):
        super().__init__(
            master,
            height=height,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._height = height
        self._colors = colors or [COLOR_BG_ELEVATED, COLOR_PURPLE, COLOR_GOLD, COLOR_BG_ELEVATED]
        self._duration = duration
        self._phase = 0.0
        self._running = True
        self._photo = None  # prevent GC of PhotoImage
        self._use_pil = _HAS_PIL
        self.bind("<Configure>", lambda e: self.after(50, self._draw))
        self.after(100, self._animate)

    def _draw(self):
        try:
            self.delete("all")
        except tk.TclError:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 1:
            return

        # PIL-optimized path: single PhotoImage per frame
        if self._use_pil:
            try:
                self._draw_pil(w, h)
                return
            except Exception:
                self._use_pil = False  # fall back

        # Fallback: canvas lines (original method)
        spot_width = w // 4
        spot_center = int(self._phase * w)

        for x in range(w):
            # Distance from the shimmer spot center
            dist = abs(x - spot_center)
            # Gaussian-like falloff
            intensity = max(0, 1.0 - (dist / (spot_width * 1.2)) ** 2)
            # Map intensity to color
            if intensity > 0.01:
                color = _blend_colors(COLOR_BG_ELEVATED, COLOR_PURPLE_LIGHT, intensity * 0.6)
            else:
                color = COLOR_BG_ELEVATED
            try:
                self.create_line(x, 0, x, h, fill=color)
            except tk.TclError:
                break

    def _draw_pil(self, w: int, h: int):
        """Render shimmer using PIL as a single PhotoImage per frame.
        Much faster than w create_line calls because PIL drawing is C-optimized
        and we only place 1 canvas item instead of w.
        """
        img = _PILImage.new("RGB", (w, h))
        pixels = img.load()

        base_rgb = _hex_to_rgb(COLOR_BG_ELEVATED)
        highlight_rgb = _hex_to_rgb(COLOR_PURPLE_LIGHT)
        spot_width = max(w // 4, 8)
        spot_center = int(self._phase * w)

        # Build pixel data row by row
        for x in range(w):
            dist = abs(x - spot_center)
            intensity = max(0.0, 1.0 - (dist / (spot_width * 1.2)) ** 2)
            if intensity > 0.01:
                r = int(base_rgb[0] + (highlight_rgb[0] - base_rgb[0]) * intensity * 0.6)
                g = int(base_rgb[1] + (highlight_rgb[1] - base_rgb[1]) * intensity * 0.6)
                b = int(base_rgb[2] + (highlight_rgb[2] - base_rgb[2]) * intensity * 0.6)
                color = (r, g, b)
            else:
                color = base_rgb
            for y in range(h):
                pixels[x, y] = color

        self._photo = _PILImageTk.PhotoImage(img)
        self.create_image(0, 0, anchor="nw", image=self._photo)

    def _animate(self):
        if not self._running:
            return
        try:
            self._draw()
            self._running = False  # render once, then stop
        except tk.TclError:
            self._running = False

    def destroy(self):
        self._running = False
        try:
            super().destroy()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# NEW: GradientDivider — Section Divider with Gradient
# Inspired by the website's .section-divider (transparent → purple → gold → transparent)
# ══════════════════════════════════════════════════════════════════════

class GradientDivider(GradientCanvas):
    """
    A horizontal divider line that fades from transparent at the edges
    to purple then gold in the center, matching the website's section-divider.
    """

    def __init__(self, master, height=1, **kwargs):
        colors = [
            "transparent",
            COLOR_PURPLE_GLOW + "40",
            COLOR_PURPLE,
            COLOR_GOLD,
            COLOR_GOLD + "40",
            "transparent",
        ]
        # Replace transparent with bg color for canvas rendering
        bg = COLOR_BG_DARKEST
        colors = [bg, _dim_hex(COLOR_PURPLE, 0.5), COLOR_PURPLE, COLOR_GOLD, _dim_hex(COLOR_GOLD, 0.5), bg]
        super().__init__(
            master,
            colors=colors,
            height=height,
            orientation="horizontal",
            **kwargs,
        )


# ══════════════════════════════════════════════════════════════════════
# NEW: AnimatedGradientBG — Multi-Layered Radial Gradient Background
# Inspired by the website's .hero-gradient (multi-layer radial gradients)
# ══════════════════════════════════════════════════════════════════════

class AnimatedGradientBG(tk.Canvas):
    """
    A full-size background canvas with animated multi-layered radial
    gradients, matching the website's hero-gradient effect.

    The gradient slowly shifts colors over time for a living,
    breathing feel.

    Performance: Uses PIL ImageDraw to render all radial gradient layers
    into a single PhotoImage instead of creating 135+ canvas ovals
    per frame. Updates every 200ms (still smooth for subtle gradients).
    Falls back to canvas ovals if PIL is not available.
    """

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._phase = 0.0
        self._running = True
        self._photo = None  # prevent GC of PhotoImage
        self._use_pil = _HAS_PIL
        self.bind("<Configure>", self._on_configure)
        self.after(100, self._animate)

    def _on_configure(self, event=None):
        self.after_idle(self._draw_bg)

    def _draw_bg(self):
        try:
            self.delete("all")
        except tk.TclError:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4:
            return

        # PIL-optimized path: single PhotoImage instead of 135+ canvas ovals
        if self._use_pil:
            try:
                self._draw_bg_pil(w, h)
                return
            except Exception:
                self._use_pil = False  # fall back

        # Fallback: canvas ovals (original method)
        self._draw_bg_canvas(w, h)

    def _draw_bg_canvas(self, w: int, h: int):
        """Original canvas oval method (fallback when PIL is unavailable)."""
        # Layer 1: Large top-center purple glow (like website's 80% 50% at 50% -20%)
        cx1, cy1 = w // 2, -h // 5
        r1 = int(w * 0.5)
        steps = min(r1, 60)
        for i in range(steps, 0, -1):
            t = i / steps
            alpha = (1 - t) * 0.5
            intensity = 0.15 + 0.15 * math.sin(self._phase + t * 2)
            color = _blend_colors(COLOR_BG_DARKEST, COLOR_PURPLE_DEEP, intensity * alpha)
            ri = int(r1 * t)
            try:
                self.create_oval(cx1 - ri, cy1 - ri, cx1 + ri, cy1 + ri, fill=color, outline="")
            except tk.TclError:
                break

        # Layer 2: Right-side gold glow (like website's 60% 40% at 80% 50%)
        cx2, cy2 = int(w * 0.8), h // 2
        r2 = int(w * 0.35)
        steps2 = min(r2, 40)
        for i in range(steps2, 0, -1):
            t = i / steps2
            alpha = (1 - t) * 0.15
            intensity = 0.1 + 0.08 * math.sin(self._phase * 0.7 + t * 3)
            color = _blend_colors(COLOR_BG_DARKEST, COLOR_GOLD_DIM, intensity * alpha)
            ri = int(r2 * t)
            try:
                self.create_oval(cx2 - ri, cy2 - ri, cx2 + ri, cy2 + ri, fill=color, outline="")
            except tk.TclError:
                break

        # Layer 3: Bottom-left purple glow (like website's 60% 40% at 20% 80%)
        cx3, cy3 = int(w * 0.2), int(h * 0.8)
        r3 = int(w * 0.3)
        steps3 = min(r3, 35)
        for i in range(steps3, 0, -1):
            t = i / steps3
            alpha = (1 - t) * 0.2
            intensity = 0.1 + 0.08 * math.sin(self._phase * 1.3 + t * 2.5)
            color = _blend_colors(COLOR_BG_DARKEST, COLOR_PURPLE_DEEP, intensity * alpha)
            ri = int(r3 * t)
            try:
                self.create_oval(cx3 - ri, cy3 - ri, cx3 + ri, cy3 + ri, fill=color, outline="")
            except tk.TclError:
                break

    def _draw_bg_pil(self, w: int, h: int):
        """Render multi-layered radial gradient using PIL ImageDraw.
        PIL drawing is C-optimized, so drawing filled ellipses is much
        faster than creating Tk canvas oval items.
        """
        bg_rgb = _hex_to_rgb(COLOR_BG_DARKEST)
        purple_deep_rgb = _hex_to_rgb(COLOR_PURPLE_DEEP)
        gold_dim_rgb = _hex_to_rgb(COLOR_GOLD_DIM)

        img = _PILImage.new("RGB", (w, h), bg_rgb)
        draw = _PILImageDraw.Draw(img)

        # Layer 1: Large top-center purple glow
        cx1, cy1 = w // 2, -h // 5
        r1 = int(w * 0.5)
        steps = min(r1, 60)
        for i in range(steps, 0, -1):
            t = i / steps
            alpha = (1 - t) * 0.5
            intensity = 0.15 + 0.15 * math.sin(self._phase + t * 2)
            blend_t = intensity * alpha
            color = _blend_rgb(bg_rgb, purple_deep_rgb, blend_t)
            ri = int(r1 * t)
            draw.ellipse([cx1 - ri, cy1 - ri, cx1 + ri, cy1 + ri], fill=color)

        # Layer 2: Right-side gold glow
        cx2, cy2 = int(w * 0.8), h // 2
        r2 = int(w * 0.35)
        steps2 = min(r2, 40)
        for i in range(steps2, 0, -1):
            t = i / steps2
            alpha = (1 - t) * 0.15
            intensity = 0.1 + 0.08 * math.sin(self._phase * 0.7 + t * 3)
            blend_t = intensity * alpha
            color = _blend_rgb(bg_rgb, gold_dim_rgb, blend_t)
            ri = int(r2 * t)
            draw.ellipse([cx2 - ri, cy2 - ri, cx2 + ri, cy2 + ri], fill=color)

        # Layer 3: Bottom-left purple glow
        cx3, cy3 = int(w * 0.2), int(h * 0.8)
        r3 = int(w * 0.3)
        steps3 = min(r3, 35)
        for i in range(steps3, 0, -1):
            t = i / steps3
            alpha = (1 - t) * 0.2
            intensity = 0.1 + 0.08 * math.sin(self._phase * 1.3 + t * 2.5)
            blend_t = intensity * alpha
            color = _blend_rgb(bg_rgb, purple_deep_rgb, blend_t)
            ri = int(r3 * t)
            draw.ellipse([cx3 - ri, cy3 - ri, cx3 + ri, cy3 + ri], fill=color)

        self._photo = _PILImageTk.PhotoImage(img)
        self.create_image(0, 0, anchor="nw", image=self._photo)

    def _animate(self):
        if not self._running:
            return
        try:
            self._draw_bg()
            self._running = False  # render once, then stop
        except tk.TclError:
            self._running = False

    def destroy(self):
        self._running = False
        try:
            super().destroy()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# NEW: GlassCard — Glassmorphism Card
# Inspired by the website's .glass-card:
#   background: oklch(0.16 0.02 280 / 0.6)
#   backdrop-filter: blur(20px)
#   border: 1px solid oklch(0.35 0.05 280 / 0.3)
# Since Tkinter doesn't support true backdrop blur, we simulate
# the glass effect with a translucent-like bg color and a
# subtle gradient border glow.
# ══════════════════════════════════════════════════════════════════════

class GlassCard(ctk.CTkFrame):
    """
    A frame that simulates the website's glass-card glassmorphism effect.
    Uses a subtly brighter background with gradient border accent to
    create the illusion of frosted glass.
    """

    def __init__(self, master, glow_color=None, **kwargs):
        self._glow_color = glow_color or COLOR_PURPLE
        corner_radius = kwargs.pop("corner_radius", BORDER_RADIUS_LG)

        # Outer glow frame — blend glow with background so the
        # outline is essentially invisible (95% bg + 5% glow color).
        self._glow_frame = ctk.CTkFrame(
            master,
            fg_color=_blend_colors(COLOR_BG, self._glow_color, 0.05),
            corner_radius=corner_radius + 1,
        )

        # Main card with glass-like background
        super().__init__(
            self._glow_frame,
            fg_color=COLOR_GLASS_BG,
            corner_radius=corner_radius,
            border_width=1,
            border_color=COLOR_GLASS_BORDER,
            **kwargs,
        )
        # Use super().pack() to pack THIS frame inside _glow_frame.
        # Do NOT use self.pack() here because the overridden pack()
        # redirects to _glow_frame.pack() which would be wrong at init time.
        super().pack(fill="both", expand=True, padx=1, pady=1)

        # Top gradient accent (subtle)
        try:
            self._accent_canvas = GradientCanvas(
                self,
                colors=[_dim_hex(self._glow_color, 0.6), _dim_hex(COLOR_GOLD, 0.4)],
                height=2,
                orientation="horizontal",
            )
            self._accent_canvas.pack(fill="x")
        except Exception:
            pass

    def pack(self, **kwargs):
        """Override pack to apply to the outer glow frame."""
        self._glow_frame.pack(**kwargs)

    def pack_forget(self):
        """Override pack_forget to apply to the outer glow frame."""
        self._glow_frame.pack_forget()

    def pack_propagate(self, flag):
        """Override pack_propagate to apply to the outer glow frame."""
        self._glow_frame.pack_propagate(flag)

    def grid(self, **kwargs):
        """Override grid to apply to the outer glow frame."""
        self._glow_frame.grid(**kwargs)

    def place(self, **kwargs):
        """Override place to apply to the outer glow frame."""
        self._glow_frame.place(**kwargs)


# ══════════════════════════════════════════════════════════════════════
# NEW: AnimatedPulseGlow — Pulsing Glow Ring Around a Widget
# Inspired by the website's pulse-glow animation + the navbar logo glow
# ══════════════════════════════════════════════════════════════════════

class AnimatedPulseGlow(tk.Canvas):
    """
    A canvas that draws a pulsing glow ring effect behind a widget.
    Mimics the website's logo glow: `bg-iso-purple/20 blur-lg group-hover:bg-iso-purple/30`.

    Performance: Refresh reduced to 150ms (from 80ms) since the visual
    difference is imperceptible at this scale.
    """

    def __init__(self, master, color=COLOR_PURPLE, size=60, **kwargs):
        super().__init__(
            master,
            width=size,
            height=size,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._color = color
        self._size = size
        self._phase = 0.0
        self._running = True
        self.after(50, self._animate)

    def _animate(self):
        if not self._running:
            return
        try:
            self.delete("all")
            cx = cy = self._size // 2
            pulse = (math.sin(self._phase) + 1.0) / 2.0

            # Outer glow circle
            glow_r = int(self._size * 0.42 + pulse * 4)
            glow_color = _dim_hex(self._color, 0.2 + pulse * 0.15)
            self.create_oval(
                cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r,
                fill=glow_color, outline="",
            )

            self._phase += 0.1
            # 150ms instead of 80ms — visual difference is imperceptible
            self.after(250, self._animate)
        except tk.TclError:
            self._running = False

    def destroy(self):
        self._running = False
        try:
            super().destroy()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# NEW: AnimatedLogo — Logo Image with Static Glow Effect
# Uses the isocortex-logo.png from website/public/
# ══════════════════════════════════════════════════════════════════════

class AnimatedLogo(ctk.CTkFrame):
    """
    A logo display widget that loads isocortex-logo.png and adds
    a static glow behind it, matching the website's navbar logo effect.

    Performance: Replaced AnimatedPulseGlow with a simple static CTkFrame
    glow. For a 36px logo, the continuous animation loop was unnecessary
    overhead. The static glow looks identical at this small size.
    """

    def __init__(self, master, logo_size=36, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._logo_size = logo_size

        # Static glow frame behind logo (replaces AnimatedPulseGlow animation loop)
        glow_size = logo_size + 20
        self._glow = ctk.CTkFrame(
            self,
            width=glow_size,
            height=glow_size,
            fg_color=_dim_hex(COLOR_PURPLE, 0.25),
            corner_radius=glow_size // 2,
        )
        self._glow.pack_propagate(False)
        self._glow.pack()

        # Inner glow ring for depth
        inner_glow_size = logo_size + 8
        self._inner_glow = ctk.CTkFrame(
            self._glow,
            width=inner_glow_size,
            height=inner_glow_size,
            fg_color=_dim_hex(COLOR_PURPLE, 0.18),
            corner_radius=inner_glow_size // 2,
        )
        self._inner_glow.pack_propagate(False)
        self._inner_glow.place(relx=0.5, rely=0.5, anchor="center")

        # Try to load the logo image
        self._logo_label = None
        logo_path = get_asset_path("isocortex-logo.png")
        if os.path.exists(logo_path):
            try:
                from PIL import Image as PILImage
                img = PILImage.open(logo_path)
                img = img.resize((logo_size, logo_size), PILImage.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(logo_size, logo_size))
                self._logo_label = ctk.CTkLabel(
                    self, image=ctk_img, text="",
                    fg_color="transparent",
                )
                self._logo_label.place(in_=self._glow, relx=0.5, rely=0.5, anchor="center")
                return
            except Exception:
                pass

        # Fallback: text logo
        self._logo_label = ctk.CTkLabel(
            self, text="IC", font=(FONT_FAMILY, logo_size // 2, "bold"),
            text_color=COLOR_GOLD, fg_color="transparent",
        )
        self._logo_label.place(in_=self._glow, relx=0.5, rely=0.5, anchor="center")


# ══════════════════════════════════════════════════════════════════════
# NEW: HeroBackground — Background Image with Gradient Overlay
# Inspired by the website hero section:
#   - hero-bg.png at 30% opacity with pulse-glow animation
#   - Gradient overlay from top (40% transparent) via mid (80%) to bottom (solid)
# ══════════════════════════════════════════════════════════════════════

class HeroBackground(tk.Canvas):
    """
    A full-size background canvas that displays hero-bg.png
    with a dark gradient overlay, matching the website hero section.
    The image subtly pulses in opacity (website's animate-pulse-glow on hero-bg).

    Performance: Caches the base darkened/blurred image on resize.
    Only applies slight brightness variation using ImageEnhance.Brightness
    on the cached image. Updates every 250ms instead of 100ms.
    Falls back to full recomputation if PIL is not available.
    """

    def __init__(self, master, overlay_opacity=0.70, **kwargs):
        super().__init__(
            master,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._overlay_opacity = overlay_opacity
        self._phase = 0.0
        self._running = True
        self._photo = None  # prevent GC of PhotoImage
        self._img_id = None
        self._overlay_ids = []
        self._cached_base = None  # cached darkened/blurred PIL image
        self._cache_size = (0, 0)  # dimensions of cached image
        self._use_pil = _HAS_PIL

        self._load_image()
        self.bind("<Configure>", lambda e: self.after(50, self._draw))
        self.after(200, self._animate)

    def _load_image(self):
        """Load the hero background image."""
        hero_path = get_asset_path("hero-bg.png")
        if not os.path.exists(hero_path):
            return
        try:
            from PIL import Image as PILImage, ImageTk as PILImageTk
            self._pil_img = PILImage.open(hero_path)
            self._has_image = True
        except Exception:
            self._has_image = False

    def _draw(self):
        try:
            self.delete("all")
        except tk.TclError:
            return
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4:
            return

        # Draw background image if available
        if getattr(self, "_has_image", False):
            if self._use_pil:
                try:
                    self._draw_pil(w, h)
                    return
                except Exception:
                    self._use_pil = False  # fall back

            # Fallback: full recomputation (original method)
            self._draw_canvas(w, h)

        # Gradient overlay: top transparent → middle semi-opaque → bottom solid
        steps = min(h, 100)
        for i in range(steps):
            t = i / max(steps - 1, 1)
            # Website: from-background/40 via-background/80 to-background
            if t < 0.3:
                opacity = 0.4
            elif t < 0.7:
                opacity = 0.4 + (t - 0.3) / 0.4 * 0.4
            else:
                opacity = 0.8 + (t - 0.7) / 0.3 * 0.2

            y = int(t * h)
            next_y = int((t + 1.0 / steps) * h) + 1
            r, g, b = _hex_to_rgb(COLOR_BG_DARKEST)
            bg_r, bg_g, bg_b = 0, 0, 0
            final_r = int(bg_r + (r - bg_r) * opacity)
            final_g = int(bg_g + (g - bg_g) * opacity)
            final_b = int(bg_b + (b - bg_b) * opacity)
            line_color = _rgb_to_hex(final_r, final_g, final_b)
            try:
                self.create_line(0, y, w, next_y, fill=line_color)
            except tk.TclError:
                break

    def _draw_canvas(self, w: int, h: int):
        """Original canvas method (full PIL recomputation each frame)."""
        try:
            from PIL import Image as PILImage, ImageTk as PILImageTk, ImageFilter
            # Resize to fill canvas
            img = self._pil_img.copy()
            img = img.resize((w, h), PILImage.LANCZOS)

            # Apply pulse-glow: vary brightness slightly
            pulse = (math.sin(self._phase) + 1.0) / 2.0
            brightness_factor = 0.7 + pulse * 0.3  # 0.7 to 1.0

            # Darken the image to match website's opacity-30 look
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor * 0.35)

            # Add slight blur for the bg effect
            img = img.filter(ImageFilter.GaussianBlur(radius=2))

            self._photo = PILImageTk.PhotoImage(img)
            self.create_image(0, 0, anchor="nw", image=self._photo)
        except Exception:
            pass

    def _draw_pil(self, w: int, h: int):
        """Optimized PIL rendering with cached base image.

        Caches the darkened+blurred base image on resize. Each frame
        only applies ImageEnhance.Brightness variation, which is much
        cheaper than a full resize+blur+darken cycle.
        """
        # Rebuild cache if canvas size changed
        if self._cache_size != (w, h):
            img = self._pil_img.copy()
            img = img.resize((w, h), _PILImage.LANCZOS)

            # Apply base darkening (website's opacity-30 look)
            enhancer = _PILImageEnhance.Brightness(img)
            img = enhancer.enhance(0.35)

            # Apply blur once and cache
            img = img.filter(_PILImageFilter.GaussianBlur(radius=2))
            self._cached_base = img
            self._cache_size = (w, h)

        # Apply pulse-glow: slight brightness variation on cached image
        pulse = (math.sin(self._phase) + 1.0) / 2.0
        brightness_factor = 0.7 + pulse * 0.3  # 0.7 to 1.0

        enhancer = _PILImageEnhance.Brightness(self._cached_base)
        img = enhancer.enhance(brightness_factor)

        self._photo = _PILImageTk.PhotoImage(img)
        self.create_image(0, 0, anchor="nw", image=self._photo)

    def _animate(self):
        if not self._running:
            return
        try:
            self._phase += 0.08
            self._draw()
            # 250ms instead of 100ms — still smooth for subtle brightness pulse
            self.after(250, self._animate)
        except tk.TclError:
            self._running = False

    def destroy(self):
        self._running = False
        try:
            super().destroy()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# NEW: Badge — Pill-Shaped Badge (Website .glass-card badges)
# ══════════════════════════════════════════════════════════════════════

def create_badge(parent, text, color=COLOR_GOLD, bg_color=None, **kwargs):
    """
    Create a pill-shaped badge matching the website's badge style.
    Website badges: 'px-3 py-1 rounded-full bg-iso-purple/10 border border-iso-purple/20'
    """
    if bg_color is None:
        bg_color = _dim_hex(color, 0.12)
    border_color = _dim_hex(color, 0.3)

    badge = ctk.CTkFrame(
        parent,
        fg_color=bg_color,
        corner_radius=20,
        border_width=1,
        border_color=border_color,
        height=24,
        **kwargs,
    )
    badge.pack_propagate(False)

    # Small dot indicator (like website's `w-1.5 h-1.5 rounded-full bg-iso-purple`)
    dot = ctk.CTkFrame(badge, width=6, height=6, fg_color=color, corner_radius=3)
    dot.pack(side="left", padx=(8, 4), pady=(9, 9))
    dot.pack_propagate(False)

    ctk.CTkLabel(
        badge,
        text=text,
        font=(FONT_FAMILY, FONT_SIZE_XXS, "bold"),
        text_color=color,
    ).pack(side="left", padx=(0, 8))

    return badge


# ══════════════════════════════════════════════════════════════════════
# NEW: TagChip — Small Tag Chip (Website feature tags)
# ══════════════════════════════════════════════════════════════════════

def create_tag_chip(parent, text):
    """
    Create a small tag chip matching the website's feature tag style.
    Website tags: 'px-2 py-0.5 text-[10px] font-medium rounded-md bg-secondary text-muted-foreground'
    """
    chip = ctk.CTkFrame(
        parent,
        fg_color=COLOR_BG_ELEVATED,
        corner_radius=4,
        height=20,
    )
    chip.pack_propagate(False)

    ctk.CTkLabel(
        chip,
        text=f" {text} ",
        font=(FONT_FAMILY, FONT_SIZE_XXS),
        text_color=COLOR_TEXT_DIM,
    ).pack(padx=2, pady=1)

    return chip


# ══════════════════════════════════════════════════════════════════════
# SelectableLabel — Read-Only Text Widget for Copy Support
# ══════════════════════════════════════════════════════════════════════

def make_selectable_label(
    parent,
    text="",
    font=None,
    text_color=None,
    bg_color=None,
    height=1,
    wrap="none",
    width=None,
    anchor="w",
    **kwargs,
):
    """
    Create a selectable, read-only text widget that looks like a label.
    Uses tk.Text in disabled state so users can select and copy text.
    Returns the tk.Text widget.
    """
    if bg_color is None:
        bg_color = COLOR_BG_CARD
    if text_color is None:
        text_color = COLOR_TEXT_SECONDARY
    if font is None:
        font = (FONT_FAMILY, FONT_SIZE_NORMAL)

    widget = tk.Text(
        parent,
        height=height,
        wrap=wrap,
        font=font,
        fg=text_color,
        bg=bg_color,
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
        takefocus=False,
    )
    if width:
        widget.configure(width=width)
    widget.insert("1.0", text)
    widget.configure(state="disabled")
    return widget


# ══════════════════════════════════════════════════════════════════════
# Premium UI Helper Functions
# ══════════════════════════════════════════════════════════════════════

def create_shadow_card(parent, corner_radius=BORDER_RADIUS_LG, shadow_size=SHADOW_OFFSET):
    """
    Create a premium card with a subtle shadow effect.
    Returns (shadow_frame, card_frame).
    """
    shadow = ctk.CTkFrame(
        parent,
        fg_color=COLOR_SHADOW,
        corner_radius=corner_radius + 2,
    )
    card = ctk.CTkFrame(
        shadow,
        fg_color=COLOR_BG_CARD,
        corner_radius=corner_radius,
        border_width=1,
        border_color=COLOR_BORDER_LIGHT,
    )
    card.pack(fill="both", expand=True, padx=2, pady=2)
    return shadow, card


def create_gradient_bar(parent, height=3, colors=None):
    """
    Create a horizontal gradient bar using Canvas for smooth blending.
    Returns the GradientCanvas widget (pack it yourself).
    """
    if colors is None:
        colors = GRADIENT_PURPLE_GOLD
    bar = GradientCanvas(parent, colors=colors, height=height, orientation="horizontal")
    return bar


def create_accent_strip(parent, color=COLOR_PURPLE, height=4, corner_radius=2):
    """Create a vertical accent strip (for card left borders)."""
    strip = ctk.CTkFrame(
        parent, width=height, fg_color=color, corner_radius=corner_radius,
    )
    strip.pack_propagate(False)
    return strip


def create_section_header(parent, title, accent_color=COLOR_PURPLE):
    """
    Create a section header with accent label + separator line.
    Returns the header frame (already packed into parent).
    """
    header = ctk.CTkFrame(parent, fg_color="transparent")
    header.pack(fill="x", pady=(PADDING_LG, PADDING_SM))

    ctk.CTkLabel(
        header, text=title,
        font=(FONT_FAMILY, FONT_SIZE_XXS),
        text_color=accent_color, anchor="w",
    ).pack(side="left")

    sep = ctk.CTkFrame(header, height=1, fg_color=COLOR_BORDER)
    sep.pack(side="right", fill="x", expand=True, padx=(PADDING, 0))
    sep.pack_propagate(False)
    return header


def create_page_header(parent, title, subtitle=None):
    """
    Create a page header with accent bar + title + optional subtitle.
    Returns the header frame (already packed into parent).
    """
    header = ctk.CTkFrame(parent, fg_color="transparent")
    header.pack(fill="x", pady=(0, PADDING_LG))

    accent = ctk.CTkFrame(
        header, width=4, corner_radius=2, fg_color=COLOR_PURPLE,
    )
    accent.pack(side="left", padx=(0, PADDING))
    accent.pack_propagate(False)

    ctk.CTkLabel(
        header, text=title,
        font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
        text_color=COLOR_TEXT, anchor="w",
    ).pack(side="left", fill="x", expand=True)

    if subtitle:
        ctk.CTkLabel(
            header, text=subtitle,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=COLOR_TEXT_DIM,
        ).pack(side="right", padx=(PADDING, 0))
    return header


def create_stat_card(parent, icon, label, default_value, color, card_width=None):
    """
    Create a premium stat card with gradient accent and large value.
    Returns (card, value_label) tuple.
    """
    card = ctk.CTkFrame(
        parent, fg_color=COLOR_BG_CARD,
        corner_radius=BORDER_RADIUS_LG,
        border_width=1, border_color=COLOR_BORDER_LIGHT,
    )

    # Top gradient accent line using Canvas
    grad = GradientCanvas(card, colors=[color, _dim_hex(color)], height=3, orientation="horizontal")
    grad.pack(fill="x")

    # Content
    inner = ctk.CTkFrame(card, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=PADDING, pady=(PADDING_MD, PADDING))

    top = ctk.CTkFrame(inner, fg_color="transparent")
    top.pack(fill="x")

    ctk.CTkLabel(
        top, text=icon, font=(FONT_FAMILY, 14),
        text_color=COLOR_TEXT_DIM, anchor="w",
    ).pack(side="left", padx=(0, 6))

    ctk.CTkLabel(
        top, text=label, font=(FONT_FAMILY, FONT_SIZE_SMALL),
        text_color=COLOR_TEXT_DIM, anchor="w",
    ).pack(side="left")

    value_label = ctk.CTkLabel(
        inner, text=default_value, font=(FONT_FAMILY, 32, "bold"),
        text_color=color, anchor="w",
    )
    value_label.pack(fill="x", pady=(4, 0))

    return card, value_label


# ══════════════════════════════════════════════════════════════════════
# NEW: create_animated_stat_card — Stat Card with Shimmer Bar
# ══════════════════════════════════════════════════════════════════════

def create_animated_stat_card(parent, icon, label, default_value, color, card_width=None):
    """
    Create a premium stat card with animated shimmer bar, matching
    the website's glass-card hover effects.
    Returns (card, value_label, shimmer_bar) tuple.
    """
    card = ctk.CTkFrame(
        parent, fg_color=COLOR_BG_CARD,
        corner_radius=BORDER_RADIUS_LG,
        border_width=1, border_color=COLOR_BORDER_LIGHT,
    )

    # Static gradient bar at top (saves 1 animation timer per card;
    # dashboard had 5 ShimmerBars — now just 1 on the page header)
    shimmer = GradientCanvas(card, colors=GRADIENT_PURPLE_GOLD, height=3, orientation="horizontal")
    shimmer.pack(fill="x")

    # Content
    inner = ctk.CTkFrame(card, fg_color="transparent")
    inner.pack(fill="both", expand=True, padx=PADDING, pady=(PADDING_MD, PADDING))

    top = ctk.CTkFrame(inner, fg_color="transparent")
    top.pack(fill="x")

    ctk.CTkLabel(
        top, text=icon, font=(FONT_FAMILY, 14),
        text_color=COLOR_TEXT_DIM, anchor="w",
    ).pack(side="left", padx=(0, 6))

    ctk.CTkLabel(
        top, text=label, font=(FONT_FAMILY, FONT_SIZE_SMALL),
        text_color=COLOR_TEXT_DIM, anchor="w",
    ).pack(side="left")

    value_label = ctk.CTkLabel(
        inner, text=default_value, font=(FONT_FAMILY, 32, "bold"),
        text_color=color, anchor="w",
    )
    value_label.pack(fill="x", pady=(4, 0))

    return card, value_label, shimmer


# ══════════════════════════════════════════════════════════════════════
# GlowFrame — Frame with a subtle colored glow border effect
# ══════════════════════════════════════════════════════════════════════

class GlowFrame(ctk.CTkFrame):
    """
    A frame with a simulated glow border effect using layered frames.
    The outer layer is a slightly transparent version of the glow color.
    """

    def __init__(self, master, glow_color=COLOR_PURPLE, glow_width=2, **kwargs):
        super().__init__(master, fg_color=_dim_hex(glow_color, 0.3), corner_radius=kwargs.get("corner_radius", BORDER_RADIUS_LG) + glow_width)
        self._inner = ctk.CTkFrame(
            self,
            fg_color=kwargs.pop("fg_color", COLOR_BG_CARD),
            corner_radius=kwargs.pop("corner_radius", BORDER_RADIUS_LG),
            border_width=1,
            border_color=glow_color,
        )
        self._inner.pack(fill="both", expand=True, padx=glow_width, pady=glow_width)

    @property
    def inner(self):
        return self._inner


# ══════════════════════════════════════════════════════════════════════
# CTK Theme Builder
# ══════════════════════════════════════════════════════════════════════

def _build_ctk_theme() -> dict:
    """Build the CustomTkinter theme dict from current mode colors."""
    return {
        "CTk": {
            "fg_color": [COLOR_BG, COLOR_BG],
            "bg_color": [COLOR_BG_DARKEST, COLOR_BG_DARKEST],
        },
        "CTkToplevel": {
            "fg_color": [COLOR_BG, COLOR_BG],
            "bg_color": [COLOR_BG_DARKEST, COLOR_BG_DARKEST],
        },
        "CTkFrame": {
            "corner_radius": BORDER_RADIUS,
            "border_width": 0,
            "fg_color": [COLOR_BG_CARD, COLOR_BG_CARD],
            "top_fg_color": [COLOR_BG_CARD, COLOR_BG_CARD],
        },
        "CTkLabel": {
            "corner_radius": 0,
            "fg_color": "transparent",
            "text_color": [COLOR_TEXT, COLOR_TEXT],
            "font": (FONT_FAMILY, FONT_SIZE_NORMAL),
        },
        "CTkEntry": {
            "corner_radius": BORDER_RADIUS_SM,
            "border_width": 1,
            "fg_color": [COLOR_INPUT_BG, COLOR_INPUT_BG],
            "border_color": [COLOR_BORDER, COLOR_BORDER],
            "text_color": [COLOR_TEXT, COLOR_TEXT],
            "placeholder_text_color": [COLOR_TEXT_DIM, COLOR_TEXT_DIM],
            "font": (FONT_FAMILY, FONT_SIZE_NORMAL),
        },
        "CTkButton": {
            "corner_radius": BORDER_RADIUS_SM,
            "border_width": 0,
            "fg_color": [COLOR_PURPLE, COLOR_PURPLE],
            "hover_color": [COLOR_PURPLE_DARK, COLOR_PURPLE_DARK],
            "text_color": [COLOR_TEXT, COLOR_TEXT],
            "font": (FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
        },
        "CTkTextbox": {
            "corner_radius": BORDER_RADIUS_SM,
            "border_width": 1,
            "fg_color": [COLOR_INPUT_BG, COLOR_INPUT_BG],
            "border_color": [COLOR_BORDER, COLOR_BORDER],
            "text_color": [COLOR_TEXT, COLOR_TEXT],
            "font": (FONT_FAMILY_MONO, FONT_SIZE_NORMAL),
        },
        "CTkScrollableFrame": {
            "corner_radius": 0,
            "fg_color": "transparent",
        },
        "CTkSegmentedButton": {
            "corner_radius": BORDER_RADIUS_SM,
            "border_width": 1,
            "fg_color": [COLOR_INPUT_BG, COLOR_INPUT_BG],
            "selected_color": [COLOR_PURPLE, COLOR_PURPLE],
            "selected_hover_color": [COLOR_PURPLE_DARK, COLOR_PURPLE_DARK],
            "unselected_color": [COLOR_INPUT_BG, COLOR_INPUT_BG],
            "unselected_hover_color": [COLOR_BG_HOVER, COLOR_BG_HOVER],
            "text_color": [COLOR_TEXT_DIM, COLOR_TEXT_DIM],
            "selected_text_color": [COLOR_TEXT, COLOR_TEXT],
            "font": (FONT_FAMILY, FONT_SIZE_NORMAL),
        },
    }


CTK_THEME = _build_ctk_theme()
