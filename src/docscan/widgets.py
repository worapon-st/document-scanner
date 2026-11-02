import tkinter as tk

# ─── Color Palette ────────────────────────────────────────────────────────────
BG        = "#0f0f0f"
PANEL     = "#1a1a1a"
CARD      = "#222222"
BORDER    = "#2e2e2e"
ACCENT    = "#c8f562"
TEXT      = "#f0f0f0"
SUBTEXT   = "#888888"
SLIDER_BG = "#2a2a2a"
SLIDER_FG = ACCENT

# ─── Custom Slider ─────────────────────────────────────────────────────────────
class CustomSlider(tk.Canvas):
    def __init__(
        self,
        parent: tk.Widget,
        from_: int,
        to: int,
        initial: int = 0,
        label: str = "",
        command=None,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=PANEL, highlightthickness=0, height=54, **kwargs)
        self.from_     = from_
        self.to        = to
        self.value     = float(initial)
        self.label     = label
        self.command   = command
        self._dragging = False

        self.bind("<Configure>",       self._draw)
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ── Internal geometry ──────────────────────────────────────────────────────
    def _track_x1_x2_y(self) -> tuple[float, float, float]:
        """Return (x1, x2, y) of the slider track in current canvas pixels."""
        w  = max(self.winfo_width(), 40)
        return 12.0, float(w - 12), 38.0

    def _knob_x(self) -> float:
        x1, x2, _ = self._track_x1_x2_y()
        span       = self.to - self.from_
        ratio      = (self.value - self.from_) / span if span else 0.0
        return x1 + ratio * (x2 - x1)

    # ── Drawing ────────────────────────────────────────────────────────────────
    def _draw(self, *_) -> None:
        self.delete("all")
        w          = max(self.winfo_width(), 40)
        x1, x2, y = self._track_x1_x2_y()
        kx         = self._knob_x()

        # Label (left) + numeric value (right)
        self.create_text(12, 14, text=self.label, anchor="w",
                         fill=SUBTEXT, font=("Courier New", 8, "bold"))
        self.create_text(w - 12, 14, text=str(int(self.value)), anchor="e",
                         fill=ACCENT, font=("Courier New", 9, "bold"))

        # Track background
        self.create_line(x1, y, x2, y, fill=SLIDER_BG, width=3, capstyle="round")

        # Bi-directional (from_ < 0): fill from centre toward knob
        if self.from_ < 0:
            cx = x1 + (x2 - x1) * 0.5
            self.create_line(cx, y - 5, cx, y + 5, fill=BORDER, width=1)
            if kx != cx:
                self.create_line(cx, y, kx, y, fill=SLIDER_FG, width=3, capstyle="round")
        else:
            if kx > x1:
                self.create_line(x1, y, kx, y, fill=SLIDER_FG, width=3, capstyle="round")

        # Knob outer ring + centre dot
        r = 7
        self.create_oval(kx - r, y - r, kx + r, y + r,
                         fill=BG, outline=ACCENT, width=2)
        self.create_oval(kx - 3, y - 3, kx + 3, y + 3,
                         fill=ACCENT, outline="")

    # ── Value ↔ pixel conversion ───────────────────────────────────────────────
    def _x_to_value(self, x: float) -> int:
        x1, x2, _ = self._track_x1_x2_y()
        span       = x2 - x1
        ratio      = max(0.0, min(1.0, (x - x1) / span)) if span else 0.0
        return round(self.from_ + ratio * (self.to - self.from_))

    # ── Mouse events ──────────────────────────────────────────────────────────
    def _on_press(self, e: tk.Event) -> None:
        self._dragging = True
        self._update_from_x(e.x)

    def _on_drag(self, e: tk.Event) -> None:
        if self._dragging:
            self._update_from_x(e.x)

    def _on_release(self, e: tk.Event) -> None:
        self._dragging = False

    def _update_from_x(self, x: float) -> None:
        new_val = self._x_to_value(x)
        if new_val != int(self.value):
            self.value = float(new_val)
            self._draw()
            if self.command:
                self.command(new_val)

    # ── Public API ─────────────────────────────────────────────────────────────
    def get(self) -> int:
        return int(self.value)

    def set(self, v: int) -> None:
        self.value = float(max(self.from_, min(self.to, v)))
        self._draw()
