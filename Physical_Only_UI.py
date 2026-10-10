"""Presentation app for physical-only proprioception."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
from queue import Empty, Queue
import statistics
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

from Grasp import grasp
from Physical_Only_Proprioception import (
    collect_one_trial, create_collection, dataset_folder, expand_collection,
    load_model, load_neural_model, open_hand, pending_trials, read_collection,
    recognize_once, release_hand, save_grasp, train_dataset, train_neural_dataset,
)
from utils.Constants import Connection, PhysicalOnly, Proprioception


BG = "#0a0f1c"
SURFACE = "#0f1628"
CARD = "#141c31"
CARD_HI = "#1b2540"
FIELD = "#0f1729"
BORDER = "#24304f"
TEXT = "#eef2fb"
MUTED = "#8e9bb7"
FAINT = "#56627f"
ACCENT = "#3fd8b6"
ACCENT_HI = "#6be6cb"
BLUE = "#6b98ff"
BAR = "#3a5596"
AMBER = "#ffb547"
RED = "#ff6577"

SYSTEM = platform.system()
FAMILY = {"Windows": "Segoe UI", "Darwin": "Helvetica Neue"}.get(SYSTEM, "DejaVu Sans")
MONO = {"Windows": "Consolas", "Darwin": "Menlo"}.get(SYSTEM, "DejaVu Sans Mono")
FINGERS = [("Index", 0), ("Middle", 4), ("Ring", 8), ("Thumb", 12)]
SCALE = 1.0


def px(value):
    return int(round(value * SCALE))


def font(size, weight="normal", mono=False):
    return (MONO if mono else FAMILY, size, weight)


def rounded(canvas, x1, y1, x2, y2, radius, **options):
    radius = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    points = [x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
              x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
              x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1]
    return canvas.create_polygon(points, smooth=True, **options)


def parse_class_counts(text, default_count=None):
    """Read one class or class:count per line from an entry box."""
    classes = {}
    for line in text.replace(",", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 1)
        name = parts[0].strip().lower()
        dataset_folder(name)
        if name in classes:
            raise ValueError(f"Class {name} is listed twice")
        if len(parts) == 2:
            count = int(parts[1].strip())
        elif default_count is not None:
            count = default_count
        else:
            raise ValueError(f"Enter a count for {name}, for example {name}: 2")
        if count < 1:
            raise ValueError("Grasp counts must be at least one")
        classes[name] = count
    return classes


def decision_color(class_name):
    return {"uncertain": AMBER, "unknown": RED}.get(class_name, ACCENT)


class Button(tk.Canvas):
    """Rounded, flat button drawn on a canvas so it looks the same on every OS."""

    STYLES = {
        "primary": (ACCENT, ACCENT_HI, BG, None),
        "secondary": (CARD_HI, "#243051", TEXT, BORDER),
        "ghost": (None, CARD_HI, MUTED, BORDER),
    }

    def __init__(self, parent, text, command, kind="primary", height=42, width=140,
                 size=11):
        super().__init__(parent, height=px(height), width=px(width), bg=parent.cget("bg"),
                         highlightthickness=0, bd=0, cursor="hand2")
        self.text = text
        self.command = command
        self.kind = kind
        self.size = size
        self.enabled = True
        self.hover = False
        self.pressed = False
        self.bind("<Configure>", lambda event: self.draw())
        self.bind("<Enter>", lambda event: self.set_hover(True))
        self.bind("<Leave>", lambda event: self.set_hover(False))
        self.bind("<ButtonPress-1>", self.press)
        self.bind("<ButtonRelease-1>", self.release)

    def set_hover(self, value):
        self.hover = value
        self.pressed = self.pressed and value
        self.draw()

    def press(self, event):
        if self.enabled:
            self.pressed = True
            self.draw()

    def release(self, event):
        fire = self.enabled and self.pressed and self.hover
        self.pressed = False
        self.draw()
        if fire:
            self.command()

    def set_enabled(self, value):
        self.enabled = value
        self.configure(cursor="hand2" if value else "arrow")
        self.draw()

    def set_text(self, text):
        self.text = text
        self.draw()

    def draw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width < 4:
            width, height = int(self.cget("width")), int(self.cget("height"))
        fill, hover, color, outline = self.STYLES[self.kind]
        if not self.enabled:
            fill, color = (CARD_HI if self.kind == "primary" else fill), FAINT
        elif self.hover:
            fill = hover
            color = TEXT if self.kind == "ghost" else color
        offset = 1 if self.pressed else 0
        rounded(self, 1, 1 + offset, width - 1, height - 1, px(10),
                fill=fill or self.cget("bg"), outline=outline or "", width=1)
        self.create_text(width / 2, height / 2 + offset, text=self.text, fill=color,
                         font=font(self.size, "bold"))


class Segmented(tk.Canvas):
    """Pill-shaped segmented control used for navigation and mode selection."""

    def __init__(self, parent, options, command, height=40):
        super().__init__(parent, height=px(height), width=px(200), bg=parent.cget("bg"),
                         highlightthickness=0, bd=0, cursor="hand2")
        self.options = options
        self.command = command
        self.selected = 0
        self.hovered = None
        self.enabled = True
        self.bind("<Configure>", lambda event: self.draw())
        self.bind("<Motion>", self.motion)
        self.bind("<Leave>", lambda event: self.motion(None))
        self.bind("<ButtonRelease-1>", self.click)

    def index_at(self, x):
        width = max(1, self.winfo_width())
        return min(len(self.options) - 1, max(0, int(x / (width / len(self.options)))))

    def motion(self, event):
        self.hovered = None if event is None else self.index_at(event.x)
        self.draw()

    def click(self, event):
        index = self.index_at(event.x)
        if self.enabled and index != self.selected:
            self.select(index)
            self.command(index)

    def select(self, index):
        self.selected = index
        self.draw()

    def set_enabled(self, value):
        self.enabled = value
        self.draw()

    def draw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width < 4:
            return
        rounded(self, 0, 0, width, height, px(12), fill=SURFACE, outline=BORDER)
        step = width / len(self.options)
        for index, option in enumerate(self.options):
            x1, x2 = index * step + px(4), (index + 1) * step - px(4)
            if index == self.selected:
                rounded(self, x1, px(4), x2, height - px(4), px(9), fill=CARD_HI,
                        outline=ACCENT if self.enabled else BORDER)
                color = TEXT if self.enabled else MUTED
            else:
                color = TEXT if index == self.hovered and self.enabled else MUTED
            self.create_text((x1 + x2) / 2, height / 2, text=option, fill=color,
                             font=font(10, "bold"))


class Card(tk.Frame):
    """Bordered panel with a small-caps title row and a body frame."""

    def __init__(self, parent, title=None, subtitle=None, pad=16):
        super().__init__(parent, bg=CARD, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=BORDER)
        self.header = tk.Frame(self, bg=CARD)
        self.title_label = tk.Label(self.header, text=(title or "").upper(), bg=CARD, fg=MUTED,
                                    font=font(9, "bold"))
        if title:
            self.header.pack(fill="x", padx=px(pad), pady=(px(pad - 2), 0))
            self.title_label.pack(side="left")
        if subtitle:
            tk.Label(self, text=subtitle, bg=CARD, fg=FAINT, font=font(10),
                     justify="left", anchor="w").pack(fill="x", padx=px(pad),
                                                     pady=(px(3), 0))
        self.body = tk.Frame(self, bg=CARD)
        self.body.pack(fill="both", expand=True, padx=px(pad), pady=px(pad - 4))


class Chip(tk.Label):
    def __init__(self, parent, text, color=MUTED):
        super().__init__(parent, text=text, bg=CARD_HI, fg=color, font=font(9, "bold"),
                         padx=px(9), pady=px(3))

    def set(self, text, color=MUTED):
        self.configure(text=text, fg=color)


class ScrollFrame(tk.Frame):
    """Vertically scrolling container; the mouse wheel is routed here by the app."""

    def __init__(self, parent, bg=SURFACE):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0,
                                yscrollincrement=px(24))
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                 style="Slim.Vertical.TScrollbar")
        self.inner = tk.Frame(self.canvas, bg=bg)
        window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda event: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(
            window, width=event.width))

    def on_wheel(self, direction):
        if self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(direction, "units")

    def to_top(self):
        self.canvas.yview_moveto(0)


class Progress(tk.Canvas):
    def __init__(self, parent, value=0.0, height=8):
        super().__init__(parent, height=px(height), width=px(60), bg=parent.cget("bg"),
                         highlightthickness=0, bd=0)
        self.value = value
        self.bind("<Configure>", lambda event: self.draw())

    def set(self, value):
        self.value = value
        self.draw()

    def draw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width < 4:
            return
        rounded(self, 0, 0, width, height, height / 2, fill=SURFACE, outline="")
        fill = width * max(0.0, min(1.0, self.value))
        if fill > height:
            rounded(self, 0, 0, fill, height, height / 2,
                    fill=ACCENT if self.value >= 1 else BLUE, outline="")


class Spinner(tk.Canvas):
    def __init__(self, parent, size=16):
        super().__init__(parent, width=px(size), height=px(size), bg=parent.cget("bg"),
                         highlightthickness=0, bd=0)
        self.size = px(size)
        self.angle = 0
        self.active = False
        self.draw()

    def set_active(self, value):
        if value and not self.active:
            self.active = True
            self.spin()
        self.active = value
        self.draw()

    def spin(self):
        if not self.active:
            return
        self.angle = (self.angle - 18) % 360
        self.draw()
        self.after(40, self.spin)

    def draw(self):
        self.delete("all")
        pad = px(2)
        box = (pad, pad, self.size - pad, self.size - pad)
        if self.active:
            self.create_oval(*box, outline=BORDER, width=px(2))
            self.create_arc(*box, start=self.angle, extent=100, style="arc",
                            outline=AMBER, width=px(2))
        else:
            inset = self.size * 0.28
            self.create_oval(inset, inset, self.size - inset, self.size - inset,
                             fill=ACCENT, outline="")


class Gauge(tk.Canvas):
    """Animated confidence ring."""

    def __init__(self, parent, size=132):
        super().__init__(parent, width=px(size), height=px(size), bg=parent.cget("bg"),
                         highlightthickness=0, bd=0)
        self.size = px(size)
        self.value = 0.0
        self.target = None
        self.color = ACCENT
        self.caption = "confidence"
        self.job = None
        self.draw()

    def set(self, value, color=ACCENT, caption="confidence"):
        self.target = value
        self.color = color
        self.caption = caption
        if self.job:
            self.after_cancel(self.job)
        self.animate()

    def animate(self):
        self.job = None
        goal = self.target or 0.0
        self.value += (goal - self.value) * 0.2
        if abs(goal - self.value) < 0.002:
            self.value = goal
        else:
            self.job = self.after(16, self.animate)
        self.draw()

    def draw(self):
        self.delete("all")
        width = px(10)
        box = (width, width, self.size - width, self.size - width)
        self.create_oval(*box, outline=CARD_HI, width=width)
        if self.target is not None and self.value > 0.001:
            self.create_arc(*box, start=90, extent=-359.9 * min(1, self.value),
                            style="arc", outline=self.color, width=width)
        center = self.size / 2
        label = "—" if self.target is None else f"{self.value * 100:.0f}%"
        self.create_text(center, center - px(6), text=label, fill=TEXT,
                         font=font(20, "bold"))
        self.create_text(center, center + px(18), text=self.caption.upper(), fill=MUTED,
                         font=font(8, "bold"))


class ProbabilityChart(tk.Canvas):
    """Ranked per-class bars that animate when a new result arrives."""

    def __init__(self, parent):
        super().__init__(parent, bg=CARD, highlightthickness=0, bd=0,
                         yscrollincrement=px(20))
        self.scores = {}
        self.shown = {}
        self.probability = True
        self.highlight = None
        self.color = ACCENT
        self.job = None
        self.bind("<Configure>", lambda event: self.draw())

    def set(self, scores, probability=True, highlight=None, color=ACCENT):
        self.scores = dict(scores)
        self.probability = probability
        self.highlight = highlight
        self.color = color
        self.shown = {name: self.shown.get(name, 0.0) for name in self.scores}
        if self.job:
            self.after_cancel(self.job)
        self.animate()

    def targets(self):
        if self.probability:
            return {name: max(0.0, min(1.0, value)) for name, value in self.scores.items()}
        best = min(self.scores.values(), default=0)
        return {name: (1.0 if value <= 0 else max(0.04, best / value))
                for name, value in self.scores.items()}

    def animate(self):
        self.job = None
        targets = self.targets()
        moving = False
        for name, goal in targets.items():
            current = self.shown.get(name, 0.0)
            current += (goal - current) * 0.18
            if abs(goal - current) < 0.002:
                current = goal
            else:
                moving = True
            self.shown[name] = current
        self.draw()
        if moving:
            self.job = self.after(16, self.animate)

    def on_wheel(self, direction):
        if self.bbox("all") and self.bbox("all")[3] > self.winfo_height():
            self.yview_scroll(direction, "units")

    def draw(self):
        self.delete("all")
        width = self.winfo_width()
        if width < 20:
            return
        if not self.scores:
            self.create_text(width / 2, px(60), fill=FAINT, font=font(11),
                             text="Select a dataset with a collection plan to see its classes")
            self.configure(scrollregion=(0, 0, width, self.winfo_height()))
            return
        order = sorted(self.scores, key=lambda name: (-self.scores[name] if self.probability
                                                      else self.scores[name]))
        row = px(50)
        name_width = px(150)
        value_width = px(78)
        track_x1 = name_width + px(34)
        track_x2 = width - value_width
        for rank, name in enumerate(order):
            y = rank * row + px(8)
            middle = y + row / 2 - px(4)
            top = name == self.highlight
            if top:
                rounded(self, 0, y, width - 2, y + row - px(8), px(10), fill=CARD_HI,
                        outline="")
            self.create_text(px(14), middle, text=f"{rank + 1:02d}", anchor="w",
                             fill=self.color if top else FAINT, font=font(9, "bold", mono=True))
            self.create_text(px(40), middle, text=name.replace("_", " ").title(), anchor="w",
                             fill=TEXT if top else MUTED,
                             font=font(12, "bold" if top else "normal"), width=name_width - px(10))
            bar_height = px(10)
            rounded(self, track_x1, middle - bar_height / 2, track_x2, middle + bar_height / 2,
                    bar_height / 2, fill=SURFACE, outline="")
            fill_x = track_x1 + (track_x2 - track_x1) * self.shown.get(name, 0.0)
            if fill_x - track_x1 > 2:
                rounded(self, track_x1, middle - bar_height / 2, fill_x,
                        middle + bar_height / 2, bar_height / 2,
                        fill=self.color if top else BAR, outline="")
            score = self.scores[name]
            if self.probability:
                shown = self.shown.get(name, 0.0) if score > 0 else 0.0
                label = f"{shown * 100:.1f}%"
            else:
                label = f"{score:.1f}°"
            self.create_text(width - px(14), middle, text=label, anchor="e",
                             fill=self.color if top else MUTED, font=font(12, "bold", mono=True))
        height = len(order) * row + px(16)
        self.configure(scrollregion=(0, 0, width, max(height, self.winfo_height())))


class JointStrip(tk.Canvas):
    """Sixteen joint closure bars grouped by finger."""

    def __init__(self, parent, height=118):
        super().__init__(parent, height=px(height), bg=CARD, highlightthickness=0, bd=0)
        self.angles = list(Proprioception.grip[0]["start_angles"])
        self.grip_number = 1
        self.bind("<Configure>", lambda event: self.draw())

    def set_grip(self, grip_number):
        self.grip_number = grip_number
        self.draw()

    def set(self, angles):
        if len(angles) == 16:
            self.angles = list(angles)
            self.draw()

    def closure(self, joint):
        grip = Proprioception.grip[self.grip_number - 1]
        start = grip["start_angles"][joint]
        end = grip["max_angles"][joint]
        if end == start:
            return 0.0
        return max(0.0, min(1.0, (self.angles[joint] - start) / (end - start)))

    def draw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width < 40:
            return
        group = width / 4
        bar_top, bar_bottom = px(6), height - px(30)
        for index, (finger, first) in enumerate(FINGERS):
            x0 = index * group
            inner = group - px(18)
            step = inner / 4
            values = [self.closure(first + joint) for joint in range(4)]
            for joint, value in enumerate(values):
                x1 = x0 + px(9) + joint * step + px(3)
                x2 = x1 + step - px(6)
                rounded(self, x1, bar_top, x2, bar_bottom, px(4), fill=SURFACE, outline="")
                fill_top = bar_bottom - (bar_bottom - bar_top) * value
                if bar_bottom - fill_top > 3:
                    color = ACCENT if value < 0.98 else AMBER
                    rounded(self, x1, fill_top, x2, bar_bottom, px(4), fill=color, outline="")
            average = sum(values) / 4
            self.create_text(x0 + group / 2, height - px(13), fill=MUTED, font=font(9, "bold"),
                             text=f"{finger.upper()}  {average * 100:.0f}%")


class ConfusionMatrix(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent, height=px(10), bg=CARD, highlightthickness=0, bd=0)
        self.report = None
        self.bind("<Configure>", lambda event: self.draw())

    def set(self, report):
        self.report = report
        self.draw()

    def draw(self):
        self.delete("all")
        width = self.winfo_width()
        report = self.report
        if not report or "validation_confusion" not in report or width < 40:
            self.configure(height=1)
            return
        classes = report["classes"]
        if len(classes) > 10:
            self.configure(height=px(22))
            self.create_text(0, px(10), anchor="w", fill=FAINT, font=font(9),
                             text="Confusion matrix hidden for more than 10 classes")
            return
        label_width = px(90)
        cell = min(px(34), (width - label_width) / len(classes))
        header = px(20)
        self.configure(height=int(header + cell * len(classes) + px(4)))
        largest = max(max(row.values()) for row in report["validation_confusion"].values()) or 1
        for column, name in enumerate(classes):
            self.create_text(label_width + column * cell + cell / 2, header / 2,
                             text=name[:3].upper(), fill=FAINT, font=font(8, "bold"))
        for row_index, actual in enumerate(classes):
            y = header + row_index * cell
            self.create_text(label_width - px(8), y + cell / 2, text=actual[:12], anchor="e",
                             fill=MUTED, font=font(9))
            for column, predicted in enumerate(classes):
                count = report["validation_confusion"][actual][predicted]
                x = label_width + column * cell
                strength = count / largest
                if count == 0:
                    fill = SURFACE
                elif actual == predicted:
                    fill = blend(SURFACE, ACCENT, 0.35 + 0.65 * strength)
                else:
                    fill = blend(SURFACE, RED, 0.35 + 0.65 * strength)
                self.create_rectangle(x + 1, y + 1, x + cell - 1, y + cell - 1, fill=fill,
                                      outline="")
                if count:
                    self.create_text(x + cell / 2, y + cell / 2, text=str(count),
                                     fill=BG if strength > 0.5 else TEXT, font=font(9, "bold"))


def blend(first, second, amount):
    a = [int(first[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(second[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(x + (y - x) * amount):02x}" for x, y in zip(a, b))


class Dialog(tk.Toplevel):
    """Themed modal for confirmations and errors."""

    def __init__(self, root, title, message, kind="confirm", confirm="Continue"):
        super().__init__(root, bg=CARD)
        self.result = False
        self.title(title)
        self.resizable(False, False)
        self.transient(root)
        color = {"confirm": ACCENT, "error": RED, "info": BLUE}[kind]
        tk.Frame(self, bg=color, height=px(4)).pack(fill="x")
        body = tk.Frame(self, bg=CARD, padx=px(26), pady=px(22))
        body.pack(fill="both", expand=True)
        tk.Label(body, text=title, bg=CARD, fg=TEXT, font=font(16, "bold"),
                 anchor="w").pack(fill="x")
        tk.Label(body, text=message, bg=CARD, fg=MUTED, font=font(11), justify="left",
                 wraplength=px(420), anchor="w").pack(fill="x", pady=(px(8), px(20)))
        buttons = tk.Frame(body, bg=CARD)
        buttons.pack(fill="x")
        if kind == "confirm":
            Button(buttons, confirm, self.accept, width=150).pack(side="right")
            Button(buttons, "Cancel", self.destroy, kind="ghost", width=110).pack(
                side="right", padx=(0, px(10)))
        else:
            Button(buttons, "OK", self.destroy, kind="secondary", width=110).pack(side="right")
        self.bind("<Return>", lambda event: self.accept() if kind == "confirm" else self.destroy())
        self.bind("<Escape>", lambda event: self.destroy())
        self.update_idletasks()
        x = root.winfo_rootx() + (root.winfo_width() - self.winfo_width()) // 2
        y = root.winfo_rooty() + (root.winfo_height() - self.winfo_height()) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        self.grab_set()
        self.focus_set()

    def accept(self):
        self.result = True
        self.destroy()

    @classmethod
    def ask(cls, root, title, message, confirm="Continue"):
        dialog = cls(root, title, message, "confirm", confirm)
        root.wait_window(dialog)
        return dialog.result

    @classmethod
    def show(cls, root, title, message, kind="error"):
        root.wait_window(cls(root, title, message, kind))


class HandView(tk.Canvas):
    """Offscreen MuJoCo render of the LEAP hand that fills its panel."""

    DEFAULT_VIEW = (100.0, -25.0)

    def __init__(self, parent):
        super().__init__(parent, bg="#0c1222", highlightthickness=0, bd=0,
                         width=px(PhysicalOnly.ui_preview_width),
                         height=px(PhysicalOnly.ui_preview_height))
        self.image = None
        self.renderer = None
        self.size = None
        self.last_mouse = None
        self.ready = False
        self.live = False
        self.spinning = False
        self.resize_job = None
        self.error = None
        try:
            import mujoco
            import numpy as np
            from utils.SimHand import SimHand

            self.mujoco = mujoco
            self.np = np
            self.joint_order = SimHand.REAL_TO_SIM
            self.model = mujoco.MjModel.from_xml_path(str(Path(Connection.model_path).resolve()))
            self.model.vis.global_.offwidth = 2400
            self.model.vis.global_.offheight = 1800
            self.model.vis.headlight.ambient[:] = 0.45
            self.model.vis.headlight.diffuse[:] = 0.75
            self.model.vis.headlight.specular[:] = 0.25
            self.data = mujoco.MjData(self.model)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.data.qpos[:] = np.radians(Proprioception.grip[0]["start_angles"])[self.joint_order]
            mujoco.mj_forward(self.model, self.data)
            low, high = self.data.geom_xpos.min(axis=0), self.data.geom_xpos.max(axis=0)
            self.center = (low + high) / 2
            self.span = float(np.linalg.norm(high - low))
            self.backdrop = None
            self.reset_view(draw=False)
            self.ready = True
        except Exception as error:
            self.error = str(error)

        self.bind("<Configure>", self.schedule_resize)
        self.bind("<ButtonPress-1>", self.start_drag)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", lambda event: setattr(self, "last_mouse", None))
        self.bind("<Double-Button-1>", lambda event: self.reset_view())

    def reset_view(self, draw=True):
        if not hasattr(self, "camera"):
            return
        self.camera.lookat[:] = self.center
        self.camera.distance = self.span * 1.3
        self.camera.azimuth, self.camera.elevation = self.DEFAULT_VIEW
        if draw:
            self.draw()

    def schedule_resize(self, event):
        if self.resize_job:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(120, self.rebuild)

    def rebuild(self):
        self.resize_job = None
        width, height = max(160, self.winfo_width()), max(160, self.winfo_height())
        width, height = min(width, 2400), min(height, 1800)
        if self.ready and (width, height) != self.size:
            try:
                renderer = self.mujoco.Renderer(self.model, height=height, width=width)
                old, self.renderer, self.size = self.renderer, renderer, (width, height)
                top, bottom = self.np.array([20, 29, 52]), self.np.array([10, 15, 28])
                shade = self.np.linspace(0, 1, height)[:, None, None]
                self.backdrop = self.np.broadcast_to(top + (bottom - top) * shade,
                                                     (height, width, 3)).astype(self.np.uint8)
                if old is not None:
                    old.close()
            except Exception as error:
                if self.renderer is None:
                    self.ready = False
                    self.error = str(error)
        self.draw()

    def set_pose(self, angles):
        if not self.ready or len(angles) != 16:
            return
        self.data.qpos[:] = self.np.radians(angles)[self.joint_order]
        self.mujoco.mj_forward(self.model, self.data)
        self.draw()

    def set_live(self, value):
        self.live = value
        self.draw_overlay()

    def toggle_spin(self):
        self.spinning = not self.spinning
        if self.spinning:
            self.spin()
        return self.spinning

    def spin(self):
        if not self.spinning:
            return
        if self.ready and self.last_mouse is None:
            self.camera.azimuth = (self.camera.azimuth + 0.6) % 360
            self.draw()
        self.after(33, self.spin)

    def zoom(self, factor):
        if self.ready:
            self.camera.distance = max(self.span * 0.4, min(self.span * 6, self.camera.distance * factor))
            self.draw()

    def on_wheel(self, direction):
        self.zoom(1.1 if direction > 0 else 0.9)

    def start_drag(self, event):
        self.last_mouse = (event.x, event.y)

    def drag(self, event):
        if not self.ready or self.last_mouse is None:
            return
        self.camera.azimuth += (event.x - self.last_mouse[0]) * 0.45
        self.camera.elevation = max(-89, min(89,
            self.camera.elevation + (event.y - self.last_mouse[1]) * 0.45))
        self.last_mouse = (event.x, event.y)
        self.draw()

    def draw(self):
        self.delete("frame")
        width, height = self.winfo_width(), self.winfo_height()
        if not self.ready or self.renderer is None:
            if width > 10:
                message = ("MuJoCo preview unavailable\n" + self.error) if self.error \
                    else "Starting MuJoCo renderer..."
                self.create_text(width / 2, height / 2, text=message, fill=MUTED,
                                 font=font(12), width=width - px(60), justify="center",
                                 tags="frame")
            self.draw_overlay()
            return
        self.renderer.update_scene(self.data, camera=self.camera)
        pixels = self.renderer.render()
        empty = pixels.max(axis=2) < 4
        pixels = self.np.where(empty[:, :, None], self.backdrop, pixels)
        rows, columns = pixels.shape[:2]
        ppm = f"P6\n{columns} {rows}\n255\n".encode() + pixels.tobytes()
        self.image = tk.PhotoImage(data=ppm, format="PPM")
        self.create_image(width / 2, height / 2, image=self.image, tags="frame")
        self.tag_lower("frame")
        self.draw_overlay()

    def draw_overlay(self):
        self.delete("overlay")
        width, height = self.winfo_width(), self.winfo_height()
        if width < 10:
            return
        color = AMBER if self.live else ACCENT
        text = "LIVE  ·  MEASURED ANGLES" if self.live else "IDLE  ·  START POSE"
        rounded(self, px(12), px(12), px(12) + px(196), px(40), px(10), fill="#0b1120",
                outline=BORDER, tags="overlay")
        self.create_oval(px(24), px(22), px(32), px(30), fill=color, outline="",
                         tags="overlay")
        self.create_text(px(40), px(26), text=text, anchor="w", fill=TEXT,
                         font=font(9, "bold"), tags="overlay")
        self.create_text(width - px(14), height - px(14), anchor="se", fill=FAINT,
                         font=font(9), tags="overlay",
                         text="Drag to rotate  ·  Scroll to zoom  ·  Double-click to reset")

    def close(self):
        self.spinning = False
        if self.renderer is not None:
            self.renderer.close()


def dataset_info(name):
    """Summarise a dataset folder for display without touching the hand."""
    folder = dataset_folder(name)
    info = dict(name=name, exists=folder.exists(), classes=[], counts={}, collected={},
                pending=[], legacy=False, plan_error=None, nearest=None, neural=None)
    try:
        plan = read_collection(name)
        info["classes"] = plan["classes"]
        info["counts"] = plan["trial_counts"]
        info["pending"] = pending_trials(name)
        waiting = {}
        for class_name, _ in info["pending"]:
            waiting[class_name] = waiting.get(class_name, 0) + 1
        info["collected"] = {class_name: count - waiting.get(class_name, 0)
                             for class_name, count in info["counts"].items()}
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError) as error:
        info["plan_error"] = str(error)
    if (folder / "model.json").exists():
        try:
            model = json.loads((folder / "model.json").read_text())
            info["nearest"] = model
            if not info["classes"] and not info["plan_error"]:
                info["legacy"] = True
                info["classes"] = model["classes"]
                counts = {class_name: 0 for class_name in model["classes"]}
                for sample in model["samples"]:
                    counts[sample["class_name"]] += 1
                info["counts"] = counts
                info["collected"] = dict(counts)
        except (OSError, ValueError, KeyError):
            pass
    report = folder / PhysicalOnly.network_report_file
    if (folder / PhysicalOnly.network_model_file).exists() and report.exists():
        try:
            info["neural"] = json.loads(report.read_text())
        except (OSError, ValueError):
            pass
    return info


class PhysicalOnlyUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ex-ARM  |  Physical Proprioception Studio")
        self.root.geometry(f"{px(PhysicalOnly.ui_width)}x{px(PhysicalOnly.ui_height)}")
        self.root.minsize(px(1280), px(780))
        self.root.configure(bg=BG)
        self.events = Queue()
        self.busy = False
        self.neural_session = None
        self.score_dataset = None
        self.info = None
        self.mode = "neural"
        self.controls = []
        self.dataset = tk.StringVar(value=PhysicalOnly.default_dataset_name)
        self.status = tk.StringVar(value="Ready")
        self.style()

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.build_header()
        body = tk.Frame(root, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=px(20), pady=(0, px(6)))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, minsize=px(390), weight=0)
        body.grid_columnconfigure(1, minsize=px(430), weight=5)
        body.grid_columnconfigure(2, minsize=px(440), weight=4)
        self.build_sidebar(body)
        self.build_results(body)
        self.build_hand(body)
        self.build_footer()

        self.root.bind_all("<MouseWheel>", self.wheel)
        self.root.bind_all("<Button-4>", self.wheel)
        self.root.bind_all("<Button-5>", self.wheel)
        self.show_page(0)
        self.refresh_dataset()
        self.log("Studio ready. Choose a dataset, then collect, train, or recognize.")
        self.root.after(PhysicalOnly.ui_poll_ms, self.read_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ---------- styling helpers ----------

    def style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dataset.TCombobox", fieldbackground=FIELD, background=CARD_HI,
                        foreground=TEXT, arrowcolor=ACCENT, bordercolor=BORDER,
                        lightcolor=FIELD, darkcolor=FIELD, padding=px(6))
        style.map("Dataset.TCombobox", fieldbackground=[("readonly", FIELD), ("disabled", CARD)],
                  foreground=[("disabled", FAINT)], selectbackground=[("!focus", FIELD)],
                  selectforeground=[("!focus", TEXT)])
        style.configure("Slim.Vertical.TScrollbar", troughcolor=SURFACE, background=CARD_HI,
                        bordercolor=SURFACE, arrowcolor=SURFACE, lightcolor=CARD_HI,
                        darkcolor=CARD_HI, gripcount=0, arrowsize=px(4), width=px(8))
        style.map("Slim.Vertical.TScrollbar", background=[("active", BORDER)])
        self.root.option_add("*TCombobox*Listbox.background", CARD)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", CARD_HI)
        self.root.option_add("*TCombobox*Listbox.selectForeground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.font", font(11))

    def text(self, parent, value, size=11, color=MUTED, weight="normal", **options):
        return tk.Label(parent, text=value, bg=parent.cget("bg"), fg=color,
                        font=font(size, weight), justify="left", anchor="w", **options)

    def field_label(self, parent, value, hint=None):
        row = tk.Frame(parent, bg=parent.cget("bg"))
        row.pack(fill="x", pady=(px(10), px(5)))
        self.text(row, value.upper(), 9, MUTED, "bold").pack(side="left")
        if hint:
            self.text(row, hint, 9, FAINT).pack(side="right")

    def entry(self, parent, value="", width=None):
        field = tk.Entry(parent, bg=FIELD, fg=TEXT, insertbackground=ACCENT, relief="flat",
                         font=font(12), highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT, disabledbackground=CARD, width=width or 10)
        field.insert(0, value)
        return field

    def lines(self, parent, value="", height=3):
        field = tk.Text(parent, height=height, bg=FIELD, fg=TEXT, insertbackground=ACCENT,
                        relief="flat", font=font(11), wrap="word", highlightthickness=1,
                        highlightbackground=BORDER, highlightcolor=ACCENT, padx=px(8),
                        pady=px(6))
        field.insert("1.0", value)
        return field

    def action(self, parent, text, command, kind="primary", height=44):
        button = Button(parent, text, command, kind=kind, height=height)
        self.controls.append(button)
        return button

    def stat(self, parent, title, value="—"):
        cell = tk.Frame(parent, bg=parent.cget("bg"))
        heading = self.text(cell, title.upper(), 8, FAINT, "bold")
        heading.pack(anchor="w")
        label = self.text(cell, value, 15, TEXT, "bold")
        label.pack(anchor="w", pady=(px(2), 0))
        return cell, heading, label

    def wheel(self, event):
        if getattr(event, "num", 0) in (4, 5):
            direction = -1 if event.num == 4 else 1
        else:
            direction = -1 if event.delta > 0 else 1
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if hasattr(widget, "on_wheel"):
                widget.on_wheel(direction)
                return
            widget = getattr(widget, "master", None)

    # ---------- layout ----------

    def build_header(self):
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=px(20), pady=(px(16), px(14)))
        logo = tk.Canvas(header, width=px(46), height=px(46), bg=BG, highlightthickness=0)
        rounded(logo, 0, 0, px(46), px(46), px(13), fill=ACCENT, outline="")
        logo.create_text(px(23), px(23), text="EX", fill=BG, font=font(15, "bold"))
        logo.pack(side="left")
        titles = tk.Frame(header, bg=BG)
        titles.pack(side="left", padx=(px(14), 0))
        self.text(titles, "Physical Proprioception Studio", 19, TEXT, "bold").pack(anchor="w")
        self.text(titles, "Ex-ARM  ·  LEAP Hand object recognition from joint angles alone",
                  10, MUTED).pack(anchor="w")

        pill = tk.Frame(header, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        pill.pack(side="right")
        self.spinner = Spinner(pill)
        self.spinner.configure(bg=CARD)
        self.spinner.pack(side="left", padx=(px(12), px(6)), pady=px(10))
        self.status_pill = tk.Label(pill, text="Idle", bg=CARD, fg=TEXT, font=font(10, "bold"),
                                    width=22, anchor="w")
        self.status_pill.pack(side="left", padx=(0, px(12)))

        Chip(header, f"REAL HAND  ·  {Connection.Port}  ·  {Connection.baudrate // 1000000} Mbps",
             MUTED).pack(side="right", padx=px(14))

        group = tk.Frame(header, bg=BG)
        group.pack(side="right")
        self.text(group, "DATASET", 9, MUTED, "bold").pack(side="left", padx=(0, px(10)))
        self.dataset_box = ttk.Combobox(group, textvariable=self.dataset, width=22,
                                        style="Dataset.TCombobox", font=font(12))
        self.dataset_box.pack(side="left", ipady=px(2))
        self.dataset_box.bind("<<ComboboxSelected>>", lambda event: self.refresh_dataset())
        self.dataset_box.bind("<Return>", lambda event: self.refresh_dataset())
        Button(group, "Refresh", self.refresh_dataset, kind="ghost", height=34,
               width=86, size=10).pack(side="left", padx=(px(8), 0))
        Button(group, "Open folder", self.open_folder, kind="ghost", height=34,
               width=106, size=10).pack(side="left", padx=(px(6), 0))

    def build_sidebar(self, body):
        side = tk.Frame(body, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        side.grid(row=0, column=0, sticky="nsew", padx=(0, px(14)))
        top = tk.Frame(side, bg=SURFACE)
        top.pack(fill="x", padx=px(16), pady=(px(16), px(4)))
        self.nav = Segmented(top, ["Dataset", "Collect", "Train", "Recognize"],
                             self.show_page, height=42)
        self.nav.pack(fill="x")
        self.page_step = self.text(top, "", 9, ACCENT, "bold")
        self.page_step.pack(anchor="w", pady=(px(18), 0))
        self.page_title = self.text(top, "", 17, TEXT, "bold")
        self.page_title.pack(anchor="w")
        self.page_about = self.text(top, "", 10, MUTED, wraplength=px(340))
        self.page_about.pack(anchor="w", pady=(px(2), 0))
        self.scroll = ScrollFrame(side)
        self.scroll.pack(fill="both", expand=True, padx=(px(16), px(6)), pady=(px(12), px(14)))
        self.pages = [self.build_dataset_page(), self.build_collect_page(),
                      self.build_train_page(), self.build_recognize_page()]

    PAGES = [
        ("Dataset", "Pick an existing object set or create a new one to teach the hand."),
        ("Collect", "Record real grasps of each object, one trial at a time."),
        ("Train", "Fit the recognition models on the saved physical trials."),
        ("Recognize", "Grasp an unknown object and let the model identify it."),
    ]

    def show_page(self, index):
        for page in self.pages:
            page.pack_forget()
        self.pages[index].pack(fill="both", expand=True, padx=(0, px(8)))
        self.nav.select(index)
        title, about = self.PAGES[index]
        self.page_step.configure(text=f"STEP {index + 1} OF 4")
        self.page_title.configure(text=title)
        self.page_about.configure(text=about)
        self.scroll.to_top()

    def page(self):
        return tk.Frame(self.scroll.inner, bg=SURFACE)

    def build_dataset_page(self):
        page = self.page()
        overview = Card(page, "Overview")
        overview.pack(fill="x", pady=(0, px(12)))
        self.overview_name = self.text(overview.body, "—", 18, TEXT, "bold")
        self.overview_name.pack(anchor="w")
        self.overview_note = self.text(overview.body, "", 10, MUTED, wraplength=px(320))
        self.overview_note.pack(anchor="w", pady=(px(2), 0))
        stats = tk.Frame(overview.body, bg=CARD)
        stats.pack(fill="x", pady=(px(14), px(6)))
        self.stat_classes = self.stat(stats, "Classes")
        self.stat_grasps = self.stat(stats, "Grasps saved")
        self.stat_models = self.stat(stats, "Models")
        for cell in (self.stat_classes, self.stat_grasps, self.stat_models):
            cell[0].pack(side="left", fill="x", expand=True)
        self.class_list = tk.Frame(overview.body, bg=CARD)
        self.class_list.pack(fill="x", pady=(px(8), 0))

        create = Card(page, "New dataset",
                      "One class per line. Write  name: count  to override the default.")
        create.pack(fill="x")
        self.field_label(create.body, "Dataset name")
        self.new_name = self.entry(create.body)
        self.new_name.pack(fill="x", ipady=px(6))
        self.field_label(create.body, "Classes", "e.g.  mug: 5")
        self.new_classes = self.lines(create.body, "cube\nsphere\ncylinder", 4)
        self.new_classes.pack(fill="x")
        self.field_label(create.body, "Grasps per class")
        self.new_count = self.entry(create.body, str(PhysicalOnly.default_trials_per_class))
        self.new_count.pack(fill="x", ipady=px(6))
        self.action(create.body, "Create dataset", self.create_plan).pack(
            fill="x", pady=(px(16), px(4)))
        return page

    def build_collect_page(self):
        page = self.page()
        card = Card(page, "Next grasp")
        card.pack(fill="x", pady=(0, px(12)))
        self.next_class = self.text(card.body, "—", 28, TEXT, "bold")
        self.next_class.pack(anchor="w")
        self.next_detail = self.text(card.body, "", 11, MUTED, wraplength=px(320))
        self.next_detail.pack(anchor="w")
        self.collect_bar = Progress(card.body)
        self.collect_bar.pack(fill="x", pady=(px(14), px(4)))
        self.collect_count = self.text(card.body, "", 9, FAINT, "bold")
        self.collect_count.pack(anchor="w")
        self.collect_button = self.action(card.body, "Start grasp", self.collect_next, height=50)
        self.collect_button.pack(fill="x", pady=(px(16), px(8)))
        self.text(card.body, "Place the object at the marked position. The hand closes, "
                  "records the final joint angles, saves the trial, then opens.", 10, FAINT,
                  wraplength=px(320)).pack(anchor="w")

        expand = Card(page, "Expand dataset", "Available once every planned grasp is saved.")
        expand.pack(fill="x")
        self.field_label(expand.body, "Extra grasps for every class")
        self.extra_all = self.entry(expand.body, "0")
        self.extra_all.pack(fill="x", ipady=px(6))
        self.field_label(expand.body, "Extra grasps for some classes", "class: count")
        self.extra_classes = self.lines(expand.body, "", 2)
        self.extra_classes.pack(fill="x")
        self.field_label(expand.body, "New classes", "class: count")
        self.add_classes = self.lines(expand.body, "", 2)
        self.add_classes.pack(fill="x")
        self.action(expand.body, "Expand plan", self.expand_plan, kind="secondary").pack(
            fill="x", pady=(px(16), px(4)))
        return page

    def build_train_page(self):
        page = self.page()
        nearest = Card(page, "Nearest-trial model",
                       "Matches a new grasp to the closest saved trial by joint-angle distance.")
        nearest.pack(fill="x", pady=(0, px(12)))
        row = tk.Frame(nearest.body, bg=CARD)
        row.pack(fill="x", pady=(px(6), px(4)))
        self.nearest_state = self.stat(row, "Status")
        self.nearest_score = self.stat(row, "Leave-one-out")
        for cell in (self.nearest_state, self.nearest_score):
            cell[0].pack(side="left", fill="x", expand=True)
        self.action(nearest.body, "Train nearest-trial model", self.train_nearest,
                    kind="secondary").pack(fill="x", pady=(px(14), px(4)))

        neural = Card(page, "Neural classifier",
                      f"{16 * len(Proprioception.grip)} joint readings → {PhysicalOnly.network_hidden_size} tanh units → "
                      f"softmax, with {PhysicalOnly.network_noisy_copies} noisy copies per grasp.")
        neural.pack(fill="x")
        row = tk.Frame(neural.body, bg=CARD)
        row.pack(fill="x", pady=(px(6), px(4)))
        self.neural_state = self.stat(row, "Status")
        self.neural_score = self.stat(row, "Held-out")
        self.neural_temp = self.stat(row, "Temperature")
        for cell in (self.neural_state, self.neural_score, self.neural_temp):
            cell[0].pack(side="left", fill="x", expand=True)
        self.confusion_title = self.text(neural.body, "HELD-OUT CONFUSION  ·  ROWS = ACTUAL",
                                         8, FAINT, "bold")
        self.confusion = ConfusionMatrix(neural.body)
        self.confusion_title.pack(anchor="w", pady=(px(12), px(4)))
        self.confusion.pack(fill="x")
        self.action(neural.body, "Train neural model", self.train_neural).pack(
            fill="x", pady=(px(16), px(6)))
        self.text(neural.body, "Training runs in the background, so the window stays "
                  "responsive.", 10, FAINT, wraplength=px(320)).pack(anchor="w")
        return page

    def build_recognize_page(self):
        page = self.page()
        self.mode_toggle = Segmented(page, ["Neural network", "Nearest trial"],
                                     self.set_mode, height=40)
        self.mode_toggle.pack(fill="x", pady=(0, px(12)))
        card = Card(page, "Identify object")
        card.pack(fill="x", pady=(0, px(12)))
        self.run_hint = self.text(card.body, "", 10, MUTED, wraplength=px(320))
        self.run_hint.pack(anchor="w")
        self.pips = tk.Canvas(card.body, height=px(30), bg=CARD, highlightthickness=0)
        self.pips.pack(fill="x", pady=(px(12), 0))
        self.pips.bind("<Configure>", lambda event: self.draw_pips())
        self.run_button = self.action(card.body, "Run neural grasp", self.run_grasp, height=54)
        self.run_button.pack(fill="x", pady=(px(14), px(8)))
        self.reset_button = self.action(card.body, "Start a new reading", self.reset_neural,
                                        kind="ghost", height=40)
        self.reset_button.pack(fill="x")

        rules = Card(page, "Decision rules")
        rules.pack(fill="x")
        for title, value in (
                ("Accept", f"top probability ≥ {PhysicalOnly.network_min_probability:.0%}"),
                ("Lead", f"over runner-up ≥ {PhysicalOnly.network_min_margin:.0%}"),
                ("Unknown", "when far from every training grasp"),
                ("Repeat", f"up to {PhysicalOnly.network_max_grasps} grasps averaged")):
            row = tk.Frame(rules.body, bg=CARD)
            row.pack(fill="x", pady=px(3))
            self.text(row, title.upper(), 8, ACCENT, "bold", width=9).pack(side="left")
            self.text(row, value, 10, MUTED).pack(side="left")
        return page

    def build_results(self, body):
        column = tk.Frame(body, bg=BG)
        column.grid(row=0, column=1, sticky="nsew", padx=(0, px(14)))
        column.grid_columnconfigure(0, weight=1)
        column.grid_rowconfigure(1, weight=1)

        hero = Card(column, "Final output", pad=20)
        hero.grid(row=0, column=0, sticky="ew")
        self.output_chip = Chip(hero.header, "NEURAL NETWORK", ACCENT)
        self.output_chip.configure(bg=CARD_HI)
        self.output_chip.pack(side="right")
        top = tk.Frame(hero.body, bg=CARD)
        top.pack(fill="x")
        self.gauge = Gauge(top)
        self.gauge.pack(side="right", padx=(px(12), 0))
        words = tk.Frame(top, bg=CARD)
        words.pack(side="left", fill="both", expand=True)
        self.output_label = self.text(words, "AWAITING GRASP", 38, MUTED, "bold")
        self.output_label.pack(anchor="w", pady=(px(10), 0))
        self.output_detail = self.text(words, "", 11, MUTED, wraplength=px(380))
        self.output_detail.pack(anchor="w", pady=(px(2), 0))
        tk.Frame(hero.body, bg=BORDER, height=1).pack(fill="x", pady=(px(14), px(12)))
        stats = tk.Frame(hero.body, bg=CARD)
        stats.pack(fill="x")
        self.output_stats = [self.stat(stats, "—") for _ in range(4)]
        for cell in self.output_stats:
            cell[0].pack(side="left", fill="x", expand=True)

        scores = Card(column, "Class probabilities", pad=16)
        scores.grid(row=1, column=0, sticky="nsew", pady=(px(14), 0))
        self.scores_card = scores
        self.chart_note = self.text(scores.header, "", 9, FAINT)
        self.chart_note.pack(side="right")
        self.chart = ProbabilityChart(scores.body)
        self.chart.pack(fill="both", expand=True)

        activity = Card(column, "Activity", pad=14)
        activity.grid(row=2, column=0, sticky="ew", pady=(px(14), 0))
        self.log_box = tk.Text(activity.body, height=5, bg=CARD, fg=MUTED, relief="flat",
                               font=font(9, mono=True), wrap="word", highlightthickness=0,
                               bd=0, state="disabled", cursor="arrow")
        self.log_box.pack(fill="both", expand=True)
        for tag, color in (("time", FAINT), ("info", MUTED), ("ok", ACCENT),
                           ("warn", AMBER), ("error", RED)):
            self.log_box.tag_configure(tag, foreground=color)

    def build_hand(self, body):
        column = tk.Frame(body, bg=BG)
        column.grid(row=0, column=2, sticky="nsew")
        column.grid_columnconfigure(0, weight=1)
        column.grid_rowconfigure(0, weight=1)
        hand = Card(column, "MuJoCo hand", pad=14)
        hand.grid(row=0, column=0, sticky="nsew")
        Chip(hand.header, "LEAP HAND  ·  16 DOF").pack(side="right")
        self.preview = HandView(hand.body)
        self.preview.pack(fill="both", expand=True)
        tools = tk.Frame(hand.body, bg=CARD)
        tools.pack(fill="x", pady=(px(10), 0))
        Button(tools, "Reset view", self.preview.reset_view, kind="ghost", height=34,
               width=100, size=10).pack(side="left")
        Button(tools, "Zoom in", lambda: self.preview.zoom(0.85), kind="ghost", height=34,
               width=86, size=10).pack(side="left", padx=(px(6), 0))
        Button(tools, "Zoom out", lambda: self.preview.zoom(1.18), kind="ghost", height=34,
               width=86, size=10).pack(side="left", padx=(px(6), 0))
        self.spin_button = Button(tools, "Auto-rotate  off", self.toggle_spin, kind="ghost",
                                  height=34, width=130, size=10)
        self.spin_button.pack(side="right")

        joints = Card(column, "Joint closure  ·  share of grasp range", pad=14)
        joints.grid(row=1, column=0, sticky="ew", pady=(px(14), 0))
        self.joints = JointStrip(joints.body)
        self.joints.pack(fill="x")

    def build_footer(self):
        footer = tk.Frame(self.root, bg=BG)
        footer.grid(row=2, column=0, sticky="ew", padx=px(22), pady=(px(4), px(10)))
        self.footer_dot = tk.Label(footer, text="●", bg=BG, fg=ACCENT, font=font(10))
        self.footer_dot.pack(side="left")
        tk.Label(footer, textvariable=self.status, bg=BG, fg=MUTED, font=font(10)).pack(
            side="left", padx=(px(8), 0))
        tk.Label(footer, text=f"Data  ·  {PhysicalOnly.data_folder}", bg=BG, fg=FAINT,
                 font=font(9)).pack(side="right")

    # ---------- display ----------

    def log(self, message, kind="info"):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", datetime.now().strftime("%H:%M:%S  "), "time")
        self.log_box.insert("end", message + "\n", kind)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, message, kind="info"):
        self.status.set(message)
        self.footer_dot.configure(fg={"warn": AMBER, "error": RED}.get(kind, ACCENT))
        self.log(message, kind)

    def set_stats(self, values):
        for (_, title, value), (name, shown) in zip(self.output_stats, values):
            title.configure(text=name.upper())
            value.configure(text=shown)

    def set_output(self, title, detail, color=TEXT):
        self.output_label.configure(text=title, fg=color)
        self.output_detail.configure(text=detail)

    def clear_output(self):
        classes = self.info["classes"] if self.info else []
        self.set_output("AWAITING GRASP", "Run a grasp from the Recognize step to identify "
                        "the object in the hand.", MUTED)
        self.gauge.set(None)
        self.show_scores({name: 0.0 for name in classes}, probability=True)
        if self.mode == "neural":
            self.set_stats([("Confidence", "—"), ("Margin", "—"), ("Distance", "—"),
                            ("Grasps", f"0 / {PhysicalOnly.network_max_grasps}")])
        else:
            self.set_stats([("Nearest trial", "—"), ("Distance", "—"),
                            ("Runner-up gap", "—"), ("Grasps", "0")])

    def show_scores(self, scores, probability, highlight=None, color=ACCENT):
        self.scores_card.title_label.configure(
            text="CLASS PROBABILITIES" if probability else "DISTANCE TO NEAREST TRIAL")
        self.chart_note.configure(text=f"{len(scores)} classes  ·  "
                                  + ("softmax output" if probability else "lower is better"))
        self.chart.set(scores, probability, highlight, color)

    def render_info(self):
        info = self.info
        if info is None:
            self.overview_name.configure(text="Invalid name")
            self.overview_note.configure(text="Use letters, numbers, spaces, underscores, "
                                         "or hyphens.")
            for cell in (self.stat_classes, self.stat_grasps, self.stat_models):
                cell[2].configure(text="—")
            self.render_classes()
            self.render_collect()
            self.render_models()
            self.update_controls()
            return
        total = sum(info["counts"].values())
        saved = sum(info["collected"].values())
        models = (info["nearest"] is not None) + (info["neural"] is not None)
        self.overview_name.configure(text=info["name"])
        if info["plan_error"]:
            note = info["plan_error"]
        elif info["legacy"]:
            note = "Older dataset without a collection plan. Use Expand plan to add one."
        elif not info["classes"]:
            note = ("New dataset. Create a collection plan below." if not info["exists"]
                    else "No collection plan yet. Create one below.")
        elif info["pending"]:
            note = f"{len(info['pending'])} grasps still to collect."
        else:
            note = "Collection complete. Ready to train and recognize."
        self.overview_note.configure(text=note)
        self.stat_classes[2].configure(text=str(len(info["classes"])) if info["classes"] else "—")
        self.stat_grasps[2].configure(text=f"{saved} / {total}" if total else "—")
        self.stat_models[2].configure(text=f"{models} / 2")
        self.render_classes()
        self.render_collect()
        self.render_models()
        self.update_controls()

    def render_classes(self):
        for child in self.class_list.winfo_children():
            child.destroy()
        if not self.info:
            return
        for class_name in self.info["classes"]:
            count = self.info["counts"][class_name]
            done = self.info["collected"].get(class_name, 0)
            row = tk.Frame(self.class_list, bg=CARD)
            row.pack(fill="x", pady=px(4))
            self.text(row, class_name.replace("_", " ").title(), 11, TEXT, width=13).pack(
                side="left")
            self.text(row, f"{done}/{count}", 10, ACCENT if done == count else AMBER,
                      "bold").pack(side="right")
            bar = Progress(row, done / count if count else 0, height=6)
            bar.pack(side="left", fill="x", expand=True, padx=(px(6), px(10)))

    def render_collect(self):
        info = self.info
        total = sum(info["counts"].values()) if info else 0
        saved = sum(info["collected"].values()) if info else 0
        self.collect_bar.set(saved / total if total else 0)
        self.collect_count.configure(text=f"{saved} OF {total} GRASPS SAVED" if total else "")
        if info and info["pending"]:
            class_name, trial = info["pending"][0]
            self.next_class.configure(text=class_name.replace("_", " ").upper(), fg=TEXT)
            self.next_detail.configure(
                text=f"Grasp {trial} of {info['counts'][class_name]} for this class")
        elif info and info["classes"] and not info["legacy"]:
            self.next_class.configure(text="ALL SAVED", fg=ACCENT)
            self.next_detail.configure(text="Every planned grasp is recorded. Train the models, "
                                       "or expand the plan below.")
        else:
            self.next_class.configure(text="NO PLAN", fg=MUTED)
            self.next_detail.configure(text="Create a dataset in the Dataset step, or expand "
                                       "an older dataset below.")

    def render_models(self):
        nearest = self.info["nearest"] if self.info else None
        neural = self.info["neural"] if self.info else None
        if nearest:
            self.nearest_state[2].configure(text="Trained", fg=ACCENT)
            total = nearest.get("leave_one_out_total") or 0
            score = (f"{nearest['leave_one_out_correct']} / {total}" if total
                     else "needs 2+ per class")
            self.nearest_score[2].configure(text=score)
        else:
            self.nearest_state[2].configure(text="Not trained", fg=MUTED)
            self.nearest_score[2].configure(text="—")
        if neural:
            self.neural_state[2].configure(text="Trained", fg=ACCENT)
            self.neural_score[2].configure(
                text=f"{neural['validation_correct']} / {neural['validation_total']}")
            self.neural_temp[2].configure(text=f"{neural.get('temperature', 0):.2f}")
        else:
            self.neural_state[2].configure(text="Not trained", fg=MUTED)
            self.neural_score[2].configure(text="—")
            self.neural_temp[2].configure(text="—")
        self.confusion.set(neural)

    def session_results(self):
        session = self.neural_session
        if session and self.info and session["name"] == self.info["name"]:
            return session["results"]
        return []

    def draw_pips(self):
        canvas = self.pips
        canvas.delete("all")
        if self.mode != "neural":
            canvas.create_text(0, px(15), anchor="w", fill=FAINT, font=font(9, "bold"),
                               text="SINGLE GRASP  ·  DISTANCE TO EVERY SAVED TRIAL")
            return
        results = self.session_results()
        size, gap = px(16), px(10)
        for index in range(PhysicalOnly.network_max_grasps):
            x = index * (size + gap)
            if index < len(results):
                color = decision_color(results[index]["prediction"]["class_name"])
                canvas.create_oval(x, px(7), x + size, px(7) + size, fill=color, outline="")
            else:
                canvas.create_oval(x, px(7), x + size, px(7) + size, outline=BORDER, width=2)
        canvas.create_text(PhysicalOnly.network_max_grasps * (size + gap) + px(4), px(15),
                           anchor="w", fill=MUTED, font=font(9, "bold"),
                           text=f"{len(results)} OF {PhysicalOnly.network_max_grasps} GRASPS "
                                "IN THIS READING")

    def update_controls(self):
        if self.busy:
            return
        info = self.info
        self.collect_button.set_enabled(bool(info and info["pending"]))
        results = self.session_results()
        if self.mode == "neural":
            number = len(results) + 1
            if number > PhysicalOnly.network_max_grasps:
                self.run_button.set_text("Reading complete")
                self.run_button.set_enabled(False)
            else:
                self.run_button.set_text("Run neural grasp" if number == 1
                                         else f"Grasp again  ({number} of "
                                              f"{PhysicalOnly.network_max_grasps})")
                self.run_button.set_enabled(True)
            self.reset_button.set_enabled(bool(results))
            self.run_hint.configure(text="Place the object, then run a grasp. If the result is "
                                    "uncertain, grasp again: probabilities are averaged across "
                                    "the reading.")
        else:
            self.run_button.set_text("Run nearest-trial grasp")
            self.run_button.set_enabled(True)
            self.reset_button.set_enabled(False)
            self.run_hint.configure(text="One grasp, compared with every saved physical trial. "
                                    "Scores are angle distances, not probabilities.")
        self.draw_pips()

    def set_busy(self, value, label="Idle", live=False):
        self.busy = value
        for control in self.controls:
            control.set_enabled(not value)
        self.mode_toggle.set_enabled(not value)
        self.dataset_box.configure(state="disabled" if value else "normal")
        self.spinner.set_active(value)
        self.status_pill.configure(text=label if value else "Idle",
                                   fg=AMBER if value else TEXT)
        self.preview.set_live(value and live)
        if not value:
            self.update_controls()

    def toggle_spin(self):
        on = self.preview.toggle_spin()
        self.spin_button.set_text("Auto-rotate  on" if on else "Auto-rotate  off")

    # ---------- actions ----------

    def error(self, title, error):
        self.set_status(str(error), "error")
        Dialog.show(self.root, title, str(error))

    def selected_name(self):
        name = self.dataset.get().strip()
        dataset_folder(name)
        return name

    def refresh_dataset(self):
        folder = Path(PhysicalOnly.data_folder)
        names = sorted(path.name for path in folder.iterdir() if path.is_dir()) \
            if folder.exists() else []
        self.dataset_box["values"] = names
        try:
            name = self.selected_name()
            self.info = dataset_info(name)
        except ValueError:
            name = None
            self.info = None
        if name != self.score_dataset:
            self.neural_session = None
            self.score_dataset = name
            self.clear_output()
        self.render_info()

    def open_folder(self):
        try:
            folder = dataset_folder(self.dataset.get().strip())
        except ValueError:
            folder = Path(PhysicalOnly.data_folder)
        if not folder.exists():
            folder = Path(PhysicalOnly.data_folder)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if SYSTEM == "Windows":
                os.startfile(str(folder.resolve()))
            else:
                subprocess.Popen(["open" if SYSTEM == "Darwin" else "xdg-open",
                                  str(folder.resolve())])
        except OSError as error:
            self.error("Open folder", error)

    def set_mode(self, index):
        self.mode = "neural" if index == 0 else "nearest"
        self.neural_session = None
        self.output_chip.set("NEURAL NETWORK" if index == 0 else "NEAREST TRIAL", ACCENT)
        self.clear_output()
        self.update_controls()

    def create_plan(self):
        if self.busy:
            return
        try:
            name = self.new_name.get().strip()
            if not name:
                raise ValueError("Enter a name for the new dataset")
            dataset_folder(name)
            count = int(self.new_count.get().strip())
            classes = parse_class_counts(self.new_classes.get("1.0", "end"), count)
            create_collection(name, list(classes), classes)
            self.dataset.set(name)
            self.neural_session = None
            self.score_dataset = None
            self.refresh_dataset()
            self.new_name.delete(0, "end")
            self.set_status(f"Collection plan saved for {name}: {len(classes)} classes, "
                            f"{sum(classes.values())} grasps", "ok")
            self.show_page(1)
        except (OSError, ValueError, FileExistsError) as error:
            self.error("New dataset", error)

    def expand_plan(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            extra = int(self.extra_all.get().strip() or "0")
            selected = parse_class_counts(self.extra_classes.get("1.0", "end"))
            added = parse_class_counts(self.add_classes.get("1.0", "end"))
            result = expand_collection(name, extra, selected, added)
            self.neural_session = None
            self.score_dataset = None
            self.refresh_dataset()
            if result:
                for field in (self.extra_classes, self.add_classes):
                    field.delete("1.0", "end")
                self.extra_all.delete(0, "end")
                self.extra_all.insert(0, "0")
                self.set_status("Collection expanded. Use Start grasp to record the new "
                                "trials.", "ok")
            else:
                self.set_status("No new grasps requested", "warn")
        except (OSError, ValueError, FileNotFoundError) as error:
            self.error("Expand dataset", error)

    def collect_next(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            pending = pending_trials(name)
            if not pending:
                raise ValueError("No pending trials. Create or expand a collection first")
            class_name, trial = pending[0]
            if not Dialog.ask(self.root, "Ready for grasp",
                              f"Place the {class_name} at the marked position for grasp "
                              f"{trial}. The hand will close, save the trial, then open.",
                              "Start grasp"):
                return

            def work():
                result = collect_one_trial(name, on_angles=self.queue_angles,
                                           on_grip=self.queue_grip)
                if result["remaining"] == 0:
                    train_dataset(name)
                return result
            self.run_job(f"Collecting {class_name} #{trial}", work, self.trial_collected,
                         live=True)
        except (OSError, ValueError, FileNotFoundError) as error:
            self.error("Collect", error)

    def trial_collected(self, result):
        self.refresh_dataset()
        if result["remaining"] == 0:
            self.set_status(f"Saved {result['class_name']} grasp {result['trial']}. Collection "
                            "complete; nearest-trial model trained.", "ok")
        else:
            self.set_status(f"Saved {result['class_name']} grasp {result['trial']}; hand opened. "
                            f"{result['remaining']} grasps remain", "ok")

    def train_nearest(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            self.run_job("Training nearest-trial", lambda: train_dataset(name),
                         self.nearest_trained)
        except ValueError as error:
            self.error("Train nearest trial", error)

    def train_neural(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            self.run_job("Training neural model", lambda: train_neural_dataset(name),
                         self.neural_trained)
        except ValueError as error:
            self.error("Train neural model", error)

    def nearest_trained(self, model):
        self.refresh_dataset()
        total = model["leave_one_out_total"]
        score = f"; leave-one-out {model['leave_one_out_correct']}/{total}" if total else ""
        self.set_status(f"Nearest-trial model saved: {len(model['samples'])} grasps{score}", "ok")

    def neural_trained(self, result):
        _, report = result
        self.neural_session = None
        self.refresh_dataset()
        self.set_status(f"Neural model saved: {report['validation_correct']}/"
                        f"{report['validation_total']} held-out trials correct", "ok")

    def run_grasp(self):
        if self.mode == "neural":
            self.neural_grasp()
        else:
            self.nearest_grasp()

    def nearest_grasp(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            load_model(name)
            if not Dialog.ask(self.root, "Ready for grasp",
                              "Place the object at the training position, then start the hand.",
                              "Start grasp"):
                return
            self.run_job("Nearest-trial grasp",
                         lambda: recognize_once(name, on_angles=self.queue_angles,
                                                on_grip=self.queue_grip),
                         self.nearest_result, live=True)
        except (OSError, ValueError, FileNotFoundError) as error:
            self.error("Recognize", error)

    def nearest_result(self, output):
        model = load_model(output["dataset"])
        angles = output["final_angles_deg"]
        distances = {class_name: min(math.dist(angles, item["angles_deg"])
                                     for item in model["samples"]
                                     if item["class_name"] == class_name)
                     for class_name in model["classes"]}
        prediction = output["prediction"]
        ranked = sorted(distances.values())
        gap = ranked[1] - ranked[0] if len(ranked) > 1 else 0.0
        separation = 1 - ranked[0] / ranked[1] if len(ranked) > 1 and ranked[1] > 0 else 1.0
        class_name = prediction["class_name"]
        self.set_output(class_name.replace("_", " ").upper(),
                        f"Closest saved trial: {class_name} #{prediction['nearest_trial']}, "
                        f"{prediction['distance_deg']:.1f}° away in joint space.", ACCENT)
        self.gauge.set(separation, ACCENT, "separation")
        self.set_stats([("Nearest trial", f"{class_name.title()} #{prediction['nearest_trial']}"),
                        ("Distance", f"{prediction['distance_deg']:.1f}°"),
                        ("Runner-up gap", f"{gap:.1f}°"), ("Grasps", "1")])
        self.show_scores(distances, probability=False, highlight=class_name)
        self.set_status(f"Nearest-trial result: {class_name}. Saved to runs/", "ok")

    def reset_neural(self):
        if self.busy:
            return
        self.neural_session = None
        self.clear_output()
        self.update_controls()
        self.set_status("New neural reading ready")

    def neural_grasp(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            if self.neural_session is None or self.neural_session["name"] != name:
                model = load_neural_model(name)
                timestamp = datetime.now(timezone.utc)
                folder = dataset_folder(name) / "neural_runs" / timestamp.strftime(
                    "%Y%m%d_%H%M%S_%f")
                self.neural_session = dict(name=name, model=model, folder=folder,
                                           time=timestamp, results=[])
            session = self.neural_session
            number = len(session["results"]) + 1
            if number > PhysicalOnly.network_max_grasps:
                raise ValueError("Maximum repeat grasps reached. Start a new neural reading")
            prompt = ("Place the object at the training position, then start the hand."
                      if number == 1 else
                      f"Reposition the object for grasp {number}. Probabilities will be "
                      "averaged with the previous grasps.")
            if not Dialog.ask(self.root, f"Neural grasp {number}", prompt, "Start grasp"):
                return
            self.run_job(f"Neural grasp {number}", lambda: self.neural_step(session, number),
                         self.neural_result, live=True)
        except (OSError, ValueError, FileNotFoundError) as error:
            self.error("Neural recognition", error)

    def neural_step(self, session, number):
        hand = open_hand()
        try:
            try:
                angles = grasp(hand, on_angles=self.queue_angles, on_grip=self.queue_grip)
                grasp_folder = session["folder"] / f"grasp_{number}"
                save_grasp(grasp_folder)
            finally:
                release_hand(hand)
                self.queue_angles(Proprioception.grip[0]["start_angles"])
                self.queue_grip(1)
        finally:
            hand.close()
        model = session["model"]
        prediction = model.predict(angles)
        results = session["results"] + [dict(angles_deg=angles, prediction=prediction)]
        probabilities = [sum(item["prediction"]["probabilities"][class_name]
                             for item in results) / len(results)
                         for class_name in model.class_names]
        distance = statistics.median(item["prediction"]["distance"] for item in results)
        combined = model.evaluate(probabilities, distance)
        output = dict(time_utc=session["time"].isoformat(), dataset=session["name"],
                      grasp_count=len(results), prediction=combined, grasps=results)
        (session["folder"] / "recognition_result.json").write_text(
            json.dumps(output, indent=2) + "\n")
        return output

    def neural_result(self, output):
        session = self.neural_session
        session["results"] = output["grasps"]
        prediction = output["prediction"]
        class_name = prediction["class_name"]
        color = decision_color(class_name)
        count = output["grasp_count"]
        if prediction["reason"]:
            detail = (f"{prediction['reason'].capitalize()}. Best guess: "
                      f"{prediction['best_class']}. Grasp again to add evidence.")
        else:
            detail = f"Accepted after {count} grasp{'s' if count > 1 else ''}."
        self.set_output(class_name.replace("_", " ").upper(), detail, color)
        self.gauge.set(prediction["confidence"], color)
        self.set_stats([("Confidence", f"{prediction['confidence'] * 100:.1f}%"),
                        ("Margin", f"{prediction['margin'] * 100:.1f}%"),
                        ("Distance", f"{prediction['distance']:.3f} / "
                                     f"{session['model'].max_distance:.3f}"),
                        ("Grasps", f"{count} / {PhysicalOnly.network_max_grasps}")])
        self.show_scores(prediction["probabilities"], probability=True,
                         highlight=prediction["best_class"], color=color)
        self.update_controls()
        reason = f" ({prediction['reason']})" if prediction["reason"] else ""
        self.set_status(f"Neural result after {count} grasp(s): {class_name}{reason}",
                        "warn" if prediction["reason"] else "ok")

    # ---------- background work ----------

    def queue_angles(self, angles):
        self.events.put(("angles", list(angles)))

    def queue_grip(self, grip_number):
        self.events.put(("grip", grip_number))

    def run_job(self, label, work, finished, live=False):
        if self.busy:
            self.set_status("Wait for the current operation to finish", "warn")
            return
        self.set_busy(True, label, live)
        self.log(label + "...")

        def worker():
            try:
                value = work()
                self.events.put(("done", (finished, value)))
            except Exception as error:
                self.events.put(("error", error))
        threading.Thread(target=worker, daemon=True).start()

    def read_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "angles":
                    self.preview.set_pose(value)
                    self.joints.set(value)
                elif kind == "grip":
                    self.joints.set_grip(value)
                elif kind == "done":
                    self.set_busy(False)
                    callback, result = value
                    try:
                        callback(result)
                    except Exception as error:
                        self.error("Display failed", error)
                elif kind == "error":
                    self.set_busy(False)
                    self.refresh_dataset()
                    self.error("Operation failed", value)
        except Empty:
            pass
        self.root.after(PhysicalOnly.ui_poll_ms, self.read_events)

    def close(self):
        if self.busy:
            Dialog.show(self.root, "Operation in progress",
                        "Wait for the grasp or training to finish before closing.", "info")
            return
        self.preview.close()
        self.root.destroy()


def main():
    global SCALE
    if SYSTEM == "Windows":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    if SYSTEM == "Windows":
        SCALE = max(1.0, root.winfo_fpixels("1i") / 96)
    PhysicalOnlyUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
