import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from docscan import core
from docscan.widgets import ACCENT, BG, BORDER, CARD, PANEL, SUBTEXT, TEXT, CustomSlider

CORNER_RADIUS  = 10           # px — draggable handle radius
CORNER_COLOR   = "white"
CORNER_OUTLINE = "#aaaaaa"
CORNER_TAG     = "corner"
CORNER_LABELS  = ["TL", "TR", "BR", "BL"]


# ─── Main App ──────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DocScan")
        self.configure(bg=BG)
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.resizable(True, True)

        self.original_cv: Optional[np.ndarray] = None
        self.scanned_cv:  Optional[np.ndarray] = None
        self._preview_cv: Optional[np.ndarray] = None   # scanned_cv at display size
        self._quarter_turns: int               = 0      # clockwise, applied after warp

        # Crop corners in original-image pixels, ordered TL, TR, BR, BL
        self._quad:           Optional[np.ndarray] = None
        self._drag_index:     Optional[int]        = None
        self._display_scale:  float                = 1.0
        self._display_offset: tuple[int, int]      = (0, 0)

        # Debounce token — prevents crash during rapid window resize
        self._resize_after_id: Optional[str] = None

        self._build_ui()

        # Single Configure binding on the root window (not individual widgets)
        self.bind("<Configure>", self._on_window_configure)

    def report_callback_exception(self, exc, val, tb) -> None:
        """Show callback errors on screen — the built app has no console."""
        super().report_callback_exception(exc, val, tb)
        messagebox.showerror("Error", str(val))

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

        self._topbtn(topbar, "EXPORT", self._export, side="right", padx=(0, 20))
        self._topbtn(topbar, "ROTATE", self._rotate, side="right")
        self._topbtn(topbar, "IMPORT", self._import, side="right")

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
    ) -> None:
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=CARD, fg=TEXT, activebackground=ACCENT, activeforeground=BG,
            font=("Courier New", 8, "bold"), relief="flat", cursor="hand2",
            padx=14, pady=6, bd=0,
        )
        btn.pack(side=side, pady=10, padx=padx)
        btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT, fg=BG))
        btn.bind("<Leave>", lambda e: btn.config(bg=CARD,   fg=TEXT))

    def _build_controls(self, parent: tk.Widget) -> None:
        # ── Adjustments ──────────────────────────────────────────────────────
        tk.Label(parent, text="ADJUSTMENTS", bg=PANEL, fg=SUBTEXT,
                 font=("Courier New", 8, "bold")).pack(anchor="w", padx=20, pady=(20, 4))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(0, 12))

        # Sliders: brightness, contrast and saturation (bi-dir), sharpness (0–100)
        self.sl_brightness = self._slider(parent, "BRIGHTNESS", -100, 100,  0)
        self.sl_contrast   = self._slider(parent, "CONTRAST",   -100, 100,  0)
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
        self._show_original()
        self._update_preview()

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
        """Draw the original image and its corners; updates scale/offset state."""
        if self.original_cv is None:
            return
        scale, offset        = self._render_image(self.original_cv, self.canvas_orig)
        self._display_scale  = scale
        self._display_offset = offset
        self._draw_corners()

    def _update_preview(self) -> None:
        """Shrink the scan to the result canvas so slider moves stay fast."""
        if self.scanned_cv is None:
            return
        h, w    = self.scanned_cv.shape[:2]
        s: float = min(1.0,
                       max(self.canvas_result.winfo_width(),  2) / w,
                       max(self.canvas_result.winfo_height(), 2) / h)
        self._preview_cv = cv2.resize(self.scanned_cv,
                                      (max(1, int(w * s)), max(1, int(h * s))),
                                      interpolation=cv2.INTER_AREA)
        self._apply_adjustments()

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

    # ── Corners ────────────────────────────────────────────────────────────────
    def _default_quad(self) -> np.ndarray:
        """A 10 % inset rectangle, used when no page is detected."""
        h, w   = self.original_cv.shape[:2]
        mx, my = w * 0.10, h * 0.10
        return np.array(
            [[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]],
            dtype=np.float32,
        )

    def _draw_corners(self) -> None:
        """Draw the dashed crop outline with a labelled handle on each corner."""
        canvas = self.canvas_orig
        canvas.delete(CORNER_TAG)
        if self._quad is None:
            return

        points: list[tuple[float, float]] = [self._img_to_canvas(x, y) for x, y in self._quad]

        for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1]):
            canvas.create_line(ax, ay, bx, by, fill="white", width=1, dash=(5, 4),
                               tags=CORNER_TAG)

        r = CORNER_RADIUS
        for (x, y), label in zip(points, CORNER_LABELS):
            canvas.create_oval(x - r, y - r, x + r, y + r,
                               fill=CORNER_COLOR, outline=CORNER_OUTLINE, width=2,
                               tags=CORNER_TAG)
            canvas.create_text(x, y, text=label, fill="#333333",
                               font=("Courier New", 6, "bold"), tags=CORNER_TAG)

    def _corner_press(self, e: tk.Event) -> None:
        if self._quad is None:
            return
        for i, (ix, iy) in enumerate(self._quad):
            cx, cy = self._img_to_canvas(ix, iy)
            if (cx - e.x) ** 2 + (cy - e.y) ** 2 <= CORNER_RADIUS ** 2:
                self._drag_index = i
                break

    def _corner_drag(self, e: tk.Event) -> None:
        if self._drag_index is None:
            return
        h, w   = self.original_cv.shape[:2]
        ix, iy = self._canvas_to_img(e.x, e.y)
        self._quad[self._drag_index] = (max(0.0, min(w - 1.0, ix)),
                                        max(0.0, min(h - 1.0, iy)))
        self._draw_corners()

    def _corner_release(self, e: tk.Event) -> None:
        if self._drag_index is None:
            return
        self._drag_index = None
        self._scan()

    # ── Adjustments ────────────────────────────────────────────────────────────
    def _adjusted(self, image: np.ndarray) -> np.ndarray:
        return core.adjust(
            image,
            brightness=self.sl_brightness.get(),
            contrast=self.sl_contrast.get(),
            saturation=self.sl_saturation.get(),
            sharpness=self.sl_sharpness.get(),
        )

    def _apply_adjustments(self) -> None:
        # shortcut: the sharpness blur is sized in pixels, so it looks a little
        # stronger on this small preview than on the full-size export.
        if self._preview_cv is not None:
            self._render_image(self._adjusted(self._preview_cv), self.canvas_result)

    def _reset_sliders(self) -> None:
        self.sl_brightness.set(0)
        self.sl_contrast.set(0)
        self.sl_saturation.set(0)
        self.sl_sharpness.set(0)
        self._apply_adjustments()

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

        img: Optional[np.ndarray] = core.load_image(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return

        self.original_cv = img
        self.scanned_cv  = None
        self._preview_cv = None
        self._quarter_turns = 0

        contour: Optional[np.ndarray] = core.find_document(img)
        self._quad = self._default_quad() if contour is None else core.order_corners(contour)

        self._show_original()
        self._reset_sliders()
        self._scan()

        name: str = os.path.basename(path)
        if contour is None:
            self.status_var.set(f"{name}: no page found. Drag the corners onto the page.")
        else:
            self.status_var.set(f"{name}: page found. Drag the corners to adjust.")

    def _scan(self) -> None:
        """Flatten the area inside the corners and show the result."""
        if not core.is_valid_quad(self._quad, self.original_cv):
            self.status_var.set("The corners do not form a page. Move them apart.")
            return

        self._quad = core.order_corners(self._quad)
        self._draw_corners()
        self.scanned_cv = core.rotate(core.warp(self.original_cv, self._quad),
                                      self._quarter_turns)
        self._update_preview()
        self.status_var.set("Scan updated.")

    def _rotate(self) -> None:
        if self.scanned_cv is None:
            return
        self._quarter_turns = (self._quarter_turns + 1) % 4
        self._scan()

    def _export(self) -> None:
        if self.scanned_cv is None:
            messagebox.showwarning("Nothing to Export", "Import an image first.")
            return

        path: str = filedialog.asksaveasfilename(
            title="Save Scanned Image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"),
                       ("All files", "*.*")],
        )
        if not path:
            return

        ok: bool = core.save_image(path, self._adjusted(self.scanned_cv))
        if ok:
            self.status_var.set(f"Exported: {os.path.basename(path)}")
            messagebox.showinfo("Exported", f"Saved to:\n{path}")
        else:
            messagebox.showerror(
                "Error",
                "Could not save file.\n"
                "Use a .png, .jpg or .pdf name in a folder you can write to.",
            )
