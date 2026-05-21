"""
IsoCortex Desktop App — Toast Notifications
==========================================
Premium temporary popup notifications with shadow and fade-out.
"""

import customtkinter as ctk

from desktop_app.theme import (
    COLOR_BG, COLOR_BG_ELEVATED, COLOR_BORDER,
    FONT_FAMILY, FONT_SIZE_NORMAL, FONT_SIZE_SMALL,
    PADDING, PADDING_SM, PADDING_MD,
    BORDER_RADIUS_LG,
)


class ToastNotification:
    """
    A premium notification that slides in from the top-right
    and auto-dismisses with a smooth experience.
    """

    def __init__(self, parent, message: str, color: str = "#3b82f6",
                 duration: int = 3000, corner_radius=BORDER_RADIUS_LG):
        self._parent = parent
        self._duration = duration

        # Shadow layer (Tkinter doesn't support alpha, use dark color)
        shadow = ctk.CTkFrame(
            parent,
            fg_color="#1a1a2e",
            corner_radius=corner_radius + 2,
        )

        # Main toast frame
        self._frame = ctk.CTkFrame(
            shadow,
            fg_color=COLOR_BG_ELEVATED,
            corner_radius=corner_radius,
            border_width=1,
            border_color=color,
        )

        # Colored left accent bar
        accent = ctk.CTkFrame(
            self._frame,
            width=4,
            fg_color=color,
            corner_radius=2,
        )
        accent.pack(side="left", fill="y", padx=(PADDING_SM, 0), pady=PADDING_MD)
        accent.pack_propagate(False)

        # Message text
        self._label = ctk.CTkLabel(
            self._frame,
            text=message,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color="#eaeaf2",
            anchor="w",
            wraplength=320,
        )
        self._label.pack(padx=(PADDING_SM, PADDING), pady=PADDING_MD)

        # Position at top-right with offset
        shadow.place(relx=1.0, x=-20, y=20, anchor="ne")

        # Auto-dismiss
        self._frame.after(duration, self._dismiss)

    def _dismiss(self):
        try:
            self._frame.master.destroy()
        except Exception:
            pass
