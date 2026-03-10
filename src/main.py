import tkinter as tk 
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional
import os
import docScanner

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

CORNER_RADIUS   = 10           # px — draggable handle radius
CORNER_COLOR    = "white"
CORNER_OUTLINE  = "#aaaaaa"
CORNER_LINE_TAG = "corner_line"
CORNER_TAG      = "corner"

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


# ─── Corner Handle (selective mode) ───────────────────────────────────────────
class CornerHandle:
    """One draggable white circle on the original-image canvas."""

    LABELS: list[str] = ["TL", "TR", "BR", "BL"]

    def __init__(self, canvas: tk.Canvas, x: float, y: float, index: int) -> None:
        self.canvas    = canvas
        self.x         = x
        self.y         = y
        self.index     = index     # 0=TL  1=TR  2=BR  3=BL
        self._oval_id  = -1
        self._label_id = -1

    def draw(self) -> None:
        """Draw (or redraw) this handle on its canvas."""
        r = CORNER_RADIUS
        if self._oval_id  != -1:
            self.canvas.delete(self._oval_id)
        if self._label_id != -1:
            self.canvas.delete(self._label_id)

        self._oval_id = self.canvas.create_oval(
            self.x - r, self.y - r, self.x + r, self.y + r,
            fill=CORNER_COLOR, outline=CORNER_OUTLINE, width=2,
            tags=(CORNER_TAG, f"corner_{self.index}"),
        )
        self._label_id = self.canvas.create_text(
            self.x, self.y,
            text=self.LABELS[self.index],
            fill="#333333", font=("Courier New", 6, "bold"),
            tags=("corner_label", f"corner_label_{self.index}"),
        )

    def move_to(self, x: float, y: float) -> None:
        """Update position and redraw."""
        self.x = x
        self.y = y
        self.draw()

    def hit_test(self, x: float, y: float) -> bool:
        """True if (x, y) is within the clickable radius."""
        return (self.x - x) ** 2 + (self.y - y) ** 2 <= CORNER_RADIUS ** 2


# ─── Main App ──────────────────────────────────────────────────────────────────
class App(tk.Tk):
    MODE_AUTO      = "auto"
    MODE_SELECTIVE = "selective"

    def __init__(self) -> None:
        super().__init__()
        self.title("DocScan")
        self.configure(bg=BG)
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.resizable(True, True)

        self.original_cv:  Optional[np.ndarray] = None
        self.scanned_cv:   Optional[np.ndarray] = None
        self.processed_cv: Optional[np.ndarray] = None

        # Selective-mode state
        self._scan_mode:      str                    = self.MODE_AUTO
        self._corners:        list[CornerHandle]     = []
        self._drag_corner:    Optional[CornerHandle] = None
        self._display_scale:  float                  = 1.0
        self._display_offset: tuple[int, int]        = (0, 0)

        # Debounce token — prevents crash during rapid window resize
        self._resize_after_id: Optional[str] = None

        self._build_ui()

        # Single Configure binding on the root window (not individual widgets)
        self.bind("<Configure>", self._on_window_configure)

    # ── UI Construction ────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # Top bar
        topbar = tk.Frame(self, bg=PANEL, height=52)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="DOCSCAN", bg=PANEL, fg=ACCENT,
                 font=("Courier New", 13, "bold")).pack(side="left", padx=20, pady=14)
        tk.Label(topbar, text="document scanner", bg=PANEL, fg=SUBTEXT,
                 font=("Courier New", 9)).pack(side="left", pady=14)

        self.btn_export = self._topbtn(topbar, "EXPORT", self._export, side="right", padx=(0, 20))
        self.btn_scan   = self._topbtn(topbar, "SCAN",   self._scan,   side="right")
        self.btn_import = self._topbtn(topbar, "IMPORT", self._import, side="right")

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=PANEL, width=240)
        left.pack(fill="y", side="left")
        left.pack_propagate(False)
        self._build_controls(left)

        tk.Frame(body, bg=BORDER, width=1).pack(fill="y", side="left")

        right = tk.Frame(body, bg=BG)
        right.pack(fill="both", expand=True)
        self._build_canvas_area(right)

    def _topbtn(
        self,
        parent: tk.Widget,
        text: str,
        cmd,
        side: str = "left",
        padx: tuple[int, int] = (0, 0),
    ) -> tk.Button:
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=CARD, fg=TEXT, activebackground=ACCENT, activeforeground=BG,
            font=("Courier New", 8, "bold"), relief="flat", cursor="hand2",
            padx=14, pady=6, bd=0,
        )
        btn.pack(side=side, pady=10, padx=padx)
        btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT, fg=BG))
        btn.bind("<Leave>", lambda e: btn.config(bg=CARD,   fg=TEXT))
        return btn

    def _build_controls(self, parent: tk.Widget) -> None:
        # ── Scan mode ────────────────────────────────────────────────────────
        tk.Label(parent, text="SCAN MODE", bg=PANEL, fg=SUBTEXT,
                 font=("Courier New", 8, "bold")).pack(anchor="w", padx=20, pady=(20, 6))

        mode_row = tk.Frame(parent, bg=PANEL)
        mode_row.pack(fill="x", padx=20, pady=(0, 4))

        self.btn_mode_auto = self._mode_btn(mode_row, "AUTO",      self.MODE_AUTO)
        self.btn_mode_sel  = self._mode_btn(mode_row, "SELECTIVE",  self.MODE_SELECTIVE)
        self._refresh_mode_buttons()

        self._hint_var = tk.StringVar(value="Auto-detects document edges.")
        tk.Label(parent, textvariable=self._hint_var, bg=PANEL, fg=SUBTEXT,
                 font=("Courier New", 7), wraplength=200, justify="left"
                 ).pack(anchor="w", padx=20, pady=(0, 4))

        # ── Adjustments ──────────────────────────────────────────────────────
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(8, 0))
        tk.Label(parent, text="ADJUSTMENTS", bg=PANEL, fg=SUBTEXT,
                 font=("Courier New", 8, "bold")).pack(anchor="w", padx=20, pady=(12, 4))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(0, 12))

        # Sliders: brightness (bi-dir), contrast (0–100), saturation (bi-dir), sharpness (0–100)
        self.sl_brightness = self._slider(parent, "BRIGHTNESS", -100, 100,  0)
        self.sl_contrast   = self._slider(parent, "CONTRAST",      0, 100,  0)
        self.sl_saturation = self._slider(parent, "SATURATION", -100, 100,  0)
        self.sl_sharpness  = self._slider(parent, "SHARPNESS",     0, 100,  0)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(16, 12))
        reset = tk.Button(parent, text="RESET", command=self._reset_sliders,
                          bg=CARD, fg=SUBTEXT, activebackground=BORDER,
                          font=("Courier New", 8), relief="flat", cursor="hand2",
                          padx=10, pady=5, bd=0)
        reset.pack(anchor="w", padx=20)
        reset.bind("<Enter>", lambda e: reset.config(fg=TEXT))
        reset.bind("<Leave>", lambda e: reset.config(fg=SUBTEXT))

        self.status_var = tk.StringVar(value="No image loaded")
        tk.Label(parent, textvariable=self.status_var, bg=PANEL, fg=SUBTEXT,
                 font=("Courier New", 7), wraplength=200, justify="left"
                 ).pack(side="bottom", anchor="w", padx=20, pady=16)

    def _mode_btn(self, parent: tk.Widget, label: str, mode: str) -> tk.Button:
        btn = tk.Button(
            parent, text=label,
            command=lambda m=mode: self._set_mode(m),
            font=("Courier New", 7, "bold"), relief="flat", cursor="hand2",
            padx=8, pady=4, bd=0,
        )
        btn.pack(side="left", padx=(0, 6))
        return btn

    def _refresh_mode_buttons(self) -> None:
        """Highlight the currently active mode button."""
        m = self._scan_mode
        self.btn_mode_auto.config(
            bg=ACCENT if m == self.MODE_AUTO else CARD,
            fg=BG     if m == self.MODE_AUTO else SUBTEXT,
        )
        self.btn_mode_sel.config(
            bg=ACCENT if m == self.MODE_SELECTIVE else CARD,
            fg=BG     if m == self.MODE_SELECTIVE else SUBTEXT,
        )

    def _set_mode(self, mode: str) -> None:
        """Switch between AUTO and SELECTIVE scan modes."""
        self._scan_mode = mode
        self._refresh_mode_buttons()

        if mode == self.MODE_AUTO:
            self._hint_var.set("Auto-detects document edges.")
            self._clear_corners()
        else:
            self._hint_var.set("Drag the 4 white corners to set crop area, then press SCAN.")
            if self.original_cv is not None:
                self._place_default_corners()

    def _slider(
        self,
        parent: tk.Widget,
        label: str,
        from_: int,
        to: int,
        initial: int,
    ) -> CustomSlider:
        sl = CustomSlider(
            parent, from_=from_, to=to, initial=initial,
            label=label, command=lambda v: self._apply_adjustments(),
        )
        sl.pack(fill="x", padx=20, pady=4)
        return sl

    def _build_canvas_area(self, parent: tk.Widget) -> None:
        panes = tk.Frame(parent, bg=BG)
        panes.pack(fill="both", expand=True, padx=20, pady=20)
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)
        panes.rowconfigure(0, weight=1)

        self.canvas_orig   = self._img_card(panes, "ORIGINAL", 0, interactive=True)
        self.canvas_result = self._img_card(panes, "SCANNED",  1, interactive=False)

    def _img_card(
        self,
        parent: tk.Widget,
        title: str,
        col: int,
        interactive: bool = False,
    ) -> tk.Canvas:
        frame = tk.Frame(parent, bg=CARD, bd=0)
        frame.grid(row=0, column=col, sticky="nsew",
                   padx=(0, 8) if col == 0 else (0, 0))

        header = tk.Frame(frame, bg=CARD)
        header.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(header, text=title, bg=CARD, fg=SUBTEXT,
                 font=("Courier New", 8, "bold")).pack(side="left")

        canvas = tk.Canvas(frame, bg="#111111", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        if interactive:
            canvas.bind("<ButtonPress-1>",   self._corner_press)
            canvas.bind("<B1-Motion>",       self._corner_drag)
            canvas.bind("<ButtonRelease-1>", self._corner_release)

        return canvas

    # ── Resize handling ────────────────────────────────────────────────────────
    def _on_window_configure(self, event: tk.Event) -> None:
        """
        Debounce <Configure> events fired during window resize.
        Only fires _on_resize_done once the user stops dragging (150 ms gap).
        Ignores sub-widget configure events by checking event.widget.
        """
        if event.widget is not self:
            return   # ignore child-widget configure noise
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(150, self._on_resize_done)

    def _on_resize_done(self) -> None:
        """Redraw both canvases once the window is stable after resizing."""
        self._resize_after_id = None
        try:
            self._redraw_original()
            self._redraw_result()
        except Exception:
            pass   # silently ignore any transient geometry errors mid-resize

    def _redraw_original(self) -> None:
        """
        Re-render original image and remap corner handle positions to the
        new canvas scale/offset so they remain at the same image-space point.
        """
        if self.original_cv is None:
            return

        old_scale:  float          = self._display_scale
        old_offset: tuple[int,int] = self._display_offset

        scale, offset        = self._render_image(self.original_cv, self.canvas_orig)
        self._display_scale  = scale
        self._display_offset = offset

        # Remap each corner from old canvas coords → image coords → new canvas coords
        if self._corners and old_scale > 0:
            for c in self._corners:
                ix: float = (c.x - old_offset[0]) / old_scale
                iy: float = (c.y - old_offset[1]) / old_scale
                c.x = ix * scale + offset[0]
                c.y = iy * scale + offset[1]
                c.draw()
            self._draw_corner_lines()

    def _redraw_result(self) -> None:
        """Re-render the processed (scanned) result image."""
        if self.processed_cv is not None:
            self._render_image(self.processed_cv, self.canvas_result)

    # ── Display helpers ────────────────────────────────────────────────────────
    def _render_image(
        self,
        img_cv: np.ndarray,
        canvas: tk.Canvas,
    ) -> tuple[float, tuple[int, int]]:
        """
        Blit img_cv onto canvas, centered and aspect-ratio preserved.
        Returns (scale_factor, (x_offset, y_offset)) for coord conversion.
        Safe to call at any time — returns sensible defaults if canvas has no size yet.
        """
        canvas.update_idletasks()
        cw: int = max(canvas.winfo_width(),  2)
        ch: int = max(canvas.winfo_height(), 2)

        h, w    = img_cv.shape[:2]
        if w < 1 or h < 1:
            return 1.0, (0, 0)

        scale: float = min(cw / w, ch / h)
        nw: int      = max(1, int(w * scale))
        nh: int      = max(1, int(h * scale))
        x_off: int   = (cw - nw) // 2
        y_off: int   = (ch - nh) // 2

        resized: np.ndarray = cv2.resize(img_cv, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb: np.ndarray     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo: ImageTk.PhotoImage = ImageTk.PhotoImage(Image.fromarray(rgb))

        canvas.delete("all")
        canvas.create_image(x_off, y_off, anchor="nw", image=photo)
        canvas._photo = photo   # keep reference — prevents garbage collection
        return scale, (x_off, y_off)

    def _show_original(self) -> None:
        """First display of the original image; initialises scale/offset state."""
        if self.original_cv is None:
            return
        scale, offset        = self._render_image(self.original_cv, self.canvas_orig)
        self._display_scale  = scale
        self._display_offset = offset
        for c in self._corners:
            c.draw()
        self._draw_corner_lines()

    def _show_result(self) -> None:
        if self.processed_cv is not None:
            self._render_image(self.processed_cv, self.canvas_result)

    # ── Corner connecting lines ────────────────────────────────────────────────
    def _draw_corner_lines(self) -> None:
        """
        Draw a closed dashed white quadrilateral connecting the 4 corner handles
        in order TL → TR → BR → BL → TL.  Lines are kept behind the circles.
        Only drawn when exactly 4 corners exist (selective mode).
        """
        self.canvas_orig.delete(CORNER_LINE_TAG)
        if len(self._corners) != 4:
            return

        # Sort by index so order is always correct regardless of insertion order
        by_idx: dict[int, CornerHandle] = {c.index: c for c in self._corners}
        loop: list[int] = [0, 1, 2, 3, 0]   # closed quad

        for i in range(len(loop) - 1):
            a: CornerHandle = by_idx[loop[i]]
            b: CornerHandle = by_idx[loop[i + 1]]
            self.canvas_orig.create_line(
                a.x, a.y, b.x, b.y,
                fill="white", width=1, dash=(5, 4),
                tags=CORNER_LINE_TAG,
            )

        # Ensure lines render behind the white circles
        self.canvas_orig.tag_lower(CORNER_LINE_TAG, CORNER_TAG)

    # ── Coordinate conversion ──────────────────────────────────────────────────
    def _img_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        """Original-image pixel coords → canvas pixel coords."""
        ox, oy = self._display_offset
        s      = self._display_scale
        return ix * s + ox, iy * s + oy

    def _canvas_to_img(self, cx: float, cy: float) -> tuple[float, float]:
        """Canvas pixel coords → original-image pixel coords."""
        ox, oy = self._display_offset
        s      = self._display_scale if self._display_scale > 0 else 1.0
        return (cx - ox) / s, (cy - oy) / s

    # ── Corner management ──────────────────────────────────────────────────────
    def _place_default_corners(self) -> None:
        """
        Place 4 corner handles at a 10 % inset rectangle in image space,
        mapped to current canvas coords.
        """
        if self.original_cv is None:
            return
        self._clear_corners()

        h, w   = self.original_cv.shape[:2]
        mx, my = w * 0.10, h * 0.10

        # Image-space positions: TL, TR, BR, BL
        img_pts: list[tuple[float, float]] = [
            (mx,      my),
            (w - mx,  my),
            (w - mx,  h - my),
            (mx,      h - my),
        ]

        for idx, (ix, iy) in enumerate(img_pts):
            cx, cy = self._img_to_canvas(ix, iy)
            handle = CornerHandle(self.canvas_orig, cx, cy, idx)
            handle.draw()
            self._corners.append(handle)

        self._draw_corner_lines()

    def _clear_corners(self) -> None:
        """Remove all corner handles and connecting lines from the canvas."""
        self.canvas_orig.delete(CORNER_LINE_TAG)
        for c in self._corners:
            self.canvas_orig.delete(f"corner_{c.index}")
            self.canvas_orig.delete(f"corner_label_{c.index}")
        self._corners.clear()

    def _corners_to_contour(self) -> np.ndarray:
        """
        Convert 4 corner handle canvas coords → (4, 1, 2) int32 ndarray
        in image-pixel space, matching the OpenCV contour format expected by
        DocScanner.transform_perspective().
        """
        pts: list[list[list[int]]] = []
        for c in self._corners:
            ix, iy = self._canvas_to_img(c.x, c.y)
            pts.append([[int(ix), int(iy)]])
        return np.array(pts, dtype=np.int32)

    # ── Corner drag events ─────────────────────────────────────────────────────
    def _corner_press(self, e: tk.Event) -> None:
        if self._scan_mode != self.MODE_SELECTIVE:
            return
        for c in self._corners:
            if c.hit_test(e.x, e.y):
                self._drag_corner = c
                break

    def _corner_drag(self, e: tk.Event) -> None:
        if self._drag_corner is None:
            return
        cw: int   = max(self.canvas_orig.winfo_width(),  1)
        ch: int   = max(self.canvas_orig.winfo_height(), 1)
        nx: float = max(0.0, min(float(cw), float(e.x)))
        ny: float = max(0.0, min(float(ch), float(e.y)))
        self._drag_corner.move_to(nx, ny)
        self._draw_corner_lines()   # live-update connecting lines while dragging

    def _corner_release(self, e: tk.Event) -> None:
        self._drag_corner = None

    # ── Adjustments ────────────────────────────────────────────────────────────
    def _apply_adjustments(self) -> None:
        if self.scanned_cv is None:
            return

        img: np.ndarray = self.scanned_cv.copy().astype(np.float32)

        # Brightness: -100…100 → pixel offset ±255
        b: int = self.sl_brightness.get()
        if b != 0:
            img = np.clip(img + b * 2.55, 0, 255)

        # Contrast: 0…100 — scales pixel values around the midpoint (128).
        # 0 = no change; 100 ≈ doubles contrast.
        c: int = self.sl_contrast.get()
        if c > 0:
            factor: float = 1.0 + c / 100.0
            img = np.clip((img - 128.0) * factor + 128.0, 0, 255)

        # Saturation: -100…100 — scale the HSV S channel.
        # +100 doubles saturation; -100 removes colour entirely.
        sat_val: int = self.sl_saturation.get()
        if sat_val != 0:
            hsv: np.ndarray   = cv2.cvtColor(
                img.astype(np.uint8), cv2.COLOR_BGR2HSV
            ).astype(np.float32)
            sat_factor: float = 1.0 + sat_val / 100.0
            hsv[:, :, 1]      = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            img = cv2.cvtColor(
                hsv.astype(np.uint8), cv2.COLOR_HSV2BGR
            ).astype(np.float32)

        # Sharpness: 0…100 — unsharp mask with adaptive Gaussian kernel.
        s: int = self.sl_sharpness.get()
        if s > 0:
            uint: np.ndarray    = img.astype(np.uint8)
            k: int              = max(1, int(s / 10) * 2 + 1)   # must be odd
            blurred: np.ndarray = cv2.GaussianBlur(uint, (k, k), 0)
            strength: float     = s / 40.0
            img = cv2.addWeighted(
                uint, 1.0 + strength, blurred, -strength, 0
            ).astype(np.float32)

        self.processed_cv = np.clip(img, 0, 255).astype(np.uint8)
        self._show_result()

    def _reset_sliders(self) -> None:
        self.sl_brightness.set(0)
        self.sl_contrast.set(0)
        self.sl_saturation.set(0)
        self.sl_sharpness.set(0)
        if self.scanned_cv is not None:
            self.processed_cv = self.scanned_cv.copy()
            self._show_result()

    # ── Actions ────────────────────────────────────────────────────────────────
    def _import(self) -> None:
        path: str = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        img: Optional[np.ndarray] = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return

        self.original_cv  = img
        self.scanned_cv   = None
        self.processed_cv = None
        self._clear_corners()
        self.canvas_result.delete("all")

        self._show_original()
        self.status_var.set(f"Loaded: {os.path.basename(path)}")
        self._reset_sliders()

        if self._scan_mode == self.MODE_SELECTIVE:
            self._place_default_corners()

    def _scan(self) -> None:
        if self.original_cv is None:
            messagebox.showwarning("No Image", "Import an image first.")
            return

        if self._scan_mode == self.MODE_SELECTIVE:
            self._scan_selective()
        else:
            self._scan_auto()

    def _scan_auto(self) -> None:
        """Detect document contour automatically using DocScanner and warp."""
        scanner: docScanner.DocScanner               = docScanner.DocScanner(self.original_cv)
        edge: np.ndarray                  = scanner.preprocess_image()
        contours: list                    = scanner.find_contour(edge)
        doc_contour: Optional[np.ndarray] = scanner.get_document_contour(contours)

        if doc_contour is not None:
            self.scanned_cv   = scanner.transform_perspective(doc_contour)
            self.processed_cv = self.scanned_cv.copy()
            self._apply_adjustments()
            self.status_var.set("Auto: document detected & scanned.")
        else:
            self.scanned_cv   = self.original_cv.copy()
            self.processed_cv = self.scanned_cv.copy()
            self._show_result()
            self.status_var.set("Auto: no contour found — showing original.")
            messagebox.showinfo("Info", "Document boundary not detected.\nShowing original image.")

    def _scan_selective(self) -> None:
        """Use the 4 user-placed corner handles to warp perspective manually."""
        if len(self._corners) != 4:
            messagebox.showwarning("Selective Mode", "4 corner handles are required.")
            return

        contour: np.ndarray = self._corners_to_contour()
        scanner: docScanner.DocScanner = docScanner.DocScanner(self.original_cv)

        self.scanned_cv   = scanner.transform_perspective(contour)
        self.processed_cv = self.scanned_cv.copy()
        self._apply_adjustments()
        self.status_var.set("Selective: scanned with manual corners.")

    def _export(self) -> None:
        if self.processed_cv is None:
            messagebox.showwarning("Nothing to Export", "Scan a document first.")
            return

        path: str = filedialog.asksaveasfilename(
            title="Save Scanned Image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")],
        )
        if not path:
            return

        ok: bool = cv2.imwrite(path, self.processed_cv)
        if ok:
            self.status_var.set(f"Exported: {os.path.basename(path)}")
            messagebox.showinfo("Exported", f"Saved to:\n{path}")
        else:
            messagebox.showerror("Error", "Could not save file.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass