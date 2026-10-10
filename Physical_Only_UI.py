"""Presentation window for physical-only proprioception."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from queue import Empty, Queue
import statistics
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from Grasp import grasp
from Physical_Only_Proprioception import (
    collect_one_trial, create_collection, dataset_folder, expand_collection,
    load_model, load_neural_model, open_hand, pending_trials, read_collection,
    recognize_once, release_hand, save_grasp, train_dataset, train_neural_dataset,
)
from utils.Constants import Connection, PhysicalOnly, Proprioception


BG = "#0c1423"
PANEL = "#172238"
PANEL_LIGHT = "#202e48"
TEXT = "#f1f5fb"
MUTED = "#a9b8ce"
ACCENT = "#65d7c1"
BLUE = "#7aaaf7"
ORANGE = "#ffca82"


def parse_class_counts(text, default_count=None):
    """Read one class or class:count per line from an entry box."""
    classes = {}
    for line in text.splitlines():
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


class HandPreview:
    """Render the URDF inside a Tk image instead of opening another viewer."""

    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, width=PhysicalOnly.ui_preview_width,
                                height=PhysicalOnly.ui_preview_height,
                                bg="#111a2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image = None
        self.last_mouse = None
        self.ready = False
        try:
            import mujoco
            import numpy as np
            from utils.SimHand import SimHand

            self.mujoco = mujoco
            self.np = np
            self.joint_order = SimHand.REAL_TO_SIM
            self.model = mujoco.MjModel.from_xml_path(str(Path(Connection.model_path).resolve()))
            self.model.vis.global_.offwidth = PhysicalOnly.ui_preview_width
            self.model.vis.global_.offheight = PhysicalOnly.ui_preview_height
            self.data = mujoco.MjData(self.model)
            self.renderer = mujoco.Renderer(self.model,
                                            height=PhysicalOnly.ui_preview_height,
                                            width=PhysicalOnly.ui_preview_width)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.lookat[:] = self.model.stat.center
            self.camera.distance = max(0.28, self.model.stat.extent * 2.2)
            self.camera.azimuth = 100
            self.camera.elevation = -25
            self.ready = True
            self.set_pose(Proprioception.grip["start_angles"])
        except Exception as error:
            self.canvas.create_text(PhysicalOnly.ui_preview_width / 2,
                                    PhysicalOnly.ui_preview_height / 2,
                                    text=f"MuJoCo preview unavailable\n{error}",
                                    fill=MUTED, font=("Helvetica", 15),
                                    width=PhysicalOnly.ui_preview_width - 60)

        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-4>", self.zoom)
        self.canvas.bind("<Button-5>", self.zoom)

    def set_pose(self, angles):
        if not self.ready or len(angles) != 16:
            return
        self.data.qpos[:] = self.np.radians(angles)[self.joint_order]
        self.mujoco.mj_forward(self.model, self.data)
        self.draw()

    def draw(self):
        if not self.ready:
            return
        self.renderer.update_scene(self.data, camera=self.camera)
        pixels = self.renderer.render()
        height, width = pixels.shape[:2]
        ppm = f"P6\n{width} {height}\n255\n".encode() + pixels.tobytes()
        self.image = tk.PhotoImage(data=ppm, format="PPM")
        self.canvas.delete("preview")
        self.canvas.create_image(0, 0, anchor="nw", image=self.image, tags="preview")

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

    def zoom(self, event):
        if not self.ready:
            return
        direction = 1 if (getattr(event, "delta", 0) > 0 or
                          getattr(event, "num", 0) == 4) else -1
        self.camera.distance *= 0.88 if direction > 0 else 1.12
        self.draw()

    def close(self):
        if self.ready:
            self.renderer.close()


class PhysicalOnlyUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ex-ARM  |  Physical Proprioception")
        self.root.geometry(f"{PhysicalOnly.ui_width}x{PhysicalOnly.ui_height}")
        self.root.minsize(1240, 760)
        self.root.configure(bg=BG)
        self.events = Queue()
        self.busy = False
        self.neural_session = None
        self.score_dataset = None
        self.dataset = tk.StringVar(value=PhysicalOnly.default_dataset_name)
        self.status = tk.StringVar(value="Ready to collect, train, or recognize")
        self.result = tk.StringVar(value="Awaiting a grasp")
        self.score_title = tk.StringVar(value="CLASS PROBABILITIES")
        self.next_trial = tk.StringVar(value="Choose a dataset to see the next trial")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED,
                        padding=(18, 11), font=("Helvetica", 11, "bold"))
        style.map("TNotebook.Tab", background=[("selected", PANEL_LIGHT)],
                  foreground=[("selected", TEXT)])
        style.configure("Dataset.TCombobox", fieldbackground=PANEL_LIGHT,
                        background=PANEL_LIGHT, foreground=TEXT, padding=6)

        header = tk.Frame(root, bg=BG, padx=28, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="EX-ARM", bg=BG, fg=ACCENT,
                 font=("Helvetica", 13, "bold")).pack(anchor="w")
        tk.Label(header, text="Physical proprioception studio", bg=BG, fg=TEXT,
                 font=("Helvetica", 27, "bold")).pack(anchor="w")
        tk.Label(header, text="Collect real grasps  •  Train class models  •  Identify objects",
                 bg=BG, fg=MUTED, font=("Helvetica", 12)).pack(anchor="w", pady=(3, 0))

        body = tk.Frame(root, bg=BG, padx=28)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=PANEL, width=500)
        left.pack(side="left", fill="both", padx=(0, 18))
        left.pack_propagate(False)
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        top = tk.Frame(left, bg=PANEL, padx=20, pady=18)
        top.pack(fill="x")
        self.label(top, "ACTIVE DATASET", small=True).pack(anchor="w")
        selector = tk.Frame(top, bg=PANEL)
        selector.pack(fill="x", pady=(8, 0))
        self.dataset_box = ttk.Combobox(selector, textvariable=self.dataset,
                                          style="Dataset.TCombobox")
        self.dataset_box.pack(side="left", fill="x", expand=True)
        self.dataset_box.bind("<<ComboboxSelected>>", lambda event: self.refresh_dataset())
        self.button(selector, "Refresh", self.refresh_dataset, quiet=True).pack(side="left", padx=(8, 0))

        tabs = ttk.Notebook(left)
        tabs.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        collect_tab = tk.Frame(tabs, bg=PANEL_LIGHT)
        train_tab = tk.Frame(tabs, bg=PANEL_LIGHT, padx=18, pady=16)
        recognize_tab = tk.Frame(tabs, bg=PANEL_LIGHT, padx=18, pady=16)
        tabs.add(collect_tab, text="Collect")
        tabs.add(train_tab, text="Train")
        tabs.add(recognize_tab, text="Recognize")
        self.build_collect(self.scroll_area(collect_tab))
        self.build_train(train_tab)
        self.build_recognize(recognize_tab)

        preview_card = tk.Frame(right, bg=PANEL, padx=14, pady=14)
        preview_card.pack(fill="x")
        tk.Label(preview_card, text="MUJOCO HAND  ·  LIVE ANGLES", bg=PANEL,
                 fg=MUTED, font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0, 10))
        self.preview = HandPreview(preview_card)
        tk.Label(preview_card, text="Drag to rotate  •  Scroll to zoom  •  Mirrors measured joint angles",
                 bg=PANEL, fg=MUTED, font=("Helvetica", 10)).pack(anchor="w", pady=(9, 0))

        result_card = tk.Frame(right, bg=PANEL, padx=18, pady=14)
        result_card.pack(fill="both", expand=True, pady=(16, 0))
        tk.Label(result_card, text="FINAL OUTPUT", bg=PANEL, fg=MUTED,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(result_card, textvariable=self.result, bg=PANEL, fg=ACCENT,
                 font=("Helvetica", 23, "bold")).pack(anchor="w", pady=(4, 12))
        tk.Label(result_card, textvariable=self.score_title, bg=PANEL, fg=MUTED,
                 font=("Helvetica", 10, "bold")).pack(anchor="w")
        scores = tk.Frame(result_card, bg=PANEL)
        scores.pack(fill="both", expand=True, pady=(8, 0))
        score_canvas = tk.Canvas(scores, bg=PANEL, highlightthickness=0)
        score_scrollbar = tk.Scrollbar(scores, command=score_canvas.yview)
        score_scrollbar.pack(side="right", fill="y")
        score_canvas.pack(side="left", fill="both", expand=True)
        score_canvas.configure(yscrollcommand=score_scrollbar.set)
        self.score_frame = tk.Frame(score_canvas, bg=PANEL)
        score_window = score_canvas.create_window((0, 0), window=self.score_frame,
                                                   anchor="nw")
        self.score_frame.bind("<Configure>", lambda event:
                              score_canvas.configure(scrollregion=score_canvas.bbox("all")))
        score_canvas.bind("<Configure>", lambda event:
                           score_canvas.itemconfigure(score_window, width=event.width))

        footer = tk.Frame(root, bg=BG, padx=28, pady=14)
        footer.pack(fill="x")
        tk.Label(footer, text="●", bg=BG, fg=ACCENT,
                 font=("Helvetica", 14)).pack(side="left")
        tk.Label(footer, textvariable=self.status, bg=BG, fg=MUTED,
                 font=("Helvetica", 11)).pack(side="left", padx=(8, 0))
        self.refresh_dataset()
        self.root.after(PhysicalOnly.ui_poll_ms, self.read_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def label(self, parent, text, small=False):
        return tk.Label(parent, text=text, bg=parent.cget("bg"),
                        fg=MUTED if small else TEXT,
                        font=("Helvetica", 10 if small else 12,
                              "bold" if small else "normal"))

    def entry(self, parent, value=""):
        field = tk.Entry(parent, bg=PANEL, fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=("Helvetica", 12), bd=8)
        field.insert(0, value)
        field.pack(fill="x", pady=(5, 12))
        return field

    def lines(self, parent, value="", height=3):
        field = tk.Text(parent, height=height, bg=PANEL, fg=TEXT,
                        insertbackground=TEXT, relief="flat", bd=8,
                        font=("Helvetica", 11), wrap="word")
        field.insert("1.0", value)
        field.pack(fill="x", pady=(5, 12))
        return field

    def button(self, parent, text, command, quiet=False):
        return tk.Button(parent, text=text, command=command, relief="flat",
                         bg=PANEL if quiet else ACCENT,
                         fg=TEXT if quiet else BG,
                         activebackground=BLUE, activeforeground=BG,
                         font=("Helvetica", 11, "bold"), padx=14, pady=9,
                         cursor="hand2")

    def section(self, parent, title):
        self.label(parent, title, small=True).pack(anchor="w", pady=(12, 4))

    def scroll_area(self, parent):
        canvas = tk.Canvas(parent, bg=PANEL_LIGHT, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        inner = tk.Frame(canvas, bg=PANEL_LIGHT, padx=18, pady=16)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda event:
                   canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event:
                    canvas.itemconfigure(window, width=event.width))
        return inner

    def build_collect(self, parent):
        self.label(parent, "One trial at a time. The hand opens after every saved grasp.").pack(anchor="w")
        self.section(parent, "NEW COLLECTION")
        self.label(parent, "Classes, one per line").pack(anchor="w")
        self.new_classes = self.lines(parent, "cube\nsphere\ncylinder", height=3)
        self.label(parent, "Grasps per class").pack(anchor="w")
        self.new_count = self.entry(parent, str(PhysicalOnly.default_trials_per_class))
        self.button(parent, "Create collection plan", self.create_plan).pack(fill="x")

        self.section(parent, "NEXT PHYSICAL TRIAL")
        tk.Label(parent, textvariable=self.next_trial, bg=PANEL_LIGHT, fg=ORANGE,
                 font=("Helvetica", 12, "bold"), wraplength=440,
                 justify="left").pack(anchor="w", pady=(2, 8))
        self.button(parent, "Start next grasp", self.collect_next).pack(fill="x")
        self.label(parent, "Position the object first. The next trial waits for this button.",
                   small=True).pack(anchor="w", pady=(8, 0))

        self.section(parent, "EXPAND A FINISHED COLLECTION")
        self.label(parent, "Extra grasps for every current class").pack(anchor="w")
        self.extra_all = self.entry(parent, "0")
        self.label(parent, "Extra for selected classes, name: count per line").pack(anchor="w")
        self.extra_classes = self.lines(parent, height=2)
        self.label(parent, "New classes, name: count per line").pack(anchor="w")
        self.add_classes = self.lines(parent, height=2)
        self.button(parent, "Expand collection plan", self.expand_plan, quiet=True).pack(fill="x")

    def build_train(self, parent):
        self.label(parent, "Train from all complete physical trials.").pack(anchor="w")
        self.section(parent, "NEAREST TRIAL")
        self.label(parent, "Compares final joint angles with saved grasps.").pack(anchor="w")
        self.button(parent, "Train nearest-trial model", self.train_nearest).pack(
            fill="x", pady=(12, 12))

        self.section(parent, "NEURAL CLASSIFIER")
        self.label(parent, "Trains with measured trials and small angle variations.").pack(anchor="w")
        self.button(parent, "Train neural model", self.train_neural).pack(
            fill="x", pady=(12, 12))
        self.label(parent, "Training can take a while. The window stays responsive.",
                   small=True).pack(anchor="w")

    def build_recognize(self, parent):
        self.label(parent, "Place an object at the same position used for training.").pack(anchor="w")
        self.section(parent, "NEURAL IDENTIFICATION")
        self.button(parent, "Run neural grasp", self.neural_grasp).pack(fill="x")
        self.label(parent, "Press again for another grasp when uncertain. Each result is saved.",
                   small=True).pack(anchor="w", pady=(9, 12))
        self.button(parent, "Start a new neural reading", self.reset_neural,
                    quiet=True).pack(fill="x")
        self.section(parent, "NEAREST TRIAL IDENTIFICATION")
        self.button(parent, "Run nearest-trial grasp", self.nearest_grasp,
                    quiet=True).pack(fill="x")
        self.label(parent, "Nearest-trial scores are distances, not probabilities.",
                   small=True).pack(anchor="w", pady=(10, 0))

    def selected_name(self):
        name = self.dataset.get().strip()
        dataset_folder(name)
        return name

    def refresh_dataset(self):
        folder = Path(PhysicalOnly.data_folder)
        names = sorted(path.name for path in folder.iterdir() if path.is_dir()) if folder.exists() else []
        self.dataset_box["values"] = names
        try:
            name = self.selected_name()
            plan = read_collection(name)
            if name != self.score_dataset:
                self.neural_session = None
                self.result.set("Awaiting a grasp")
                self.show_scores({class_name: 0.0 for class_name in plan["classes"]},
                                 probability=True)
                self.score_dataset = name
            pending = pending_trials(name)
            total = sum(plan["trial_counts"].values())
            if pending:
                class_name, trial = pending[0]
                self.next_trial.set(f"{class_name.upper()}  ·  grasp {trial}  |  {total - len(pending)}/{total} saved")
            else:
                self.next_trial.set(f"All {total} grasps saved. Ready to train or expand.")
        except (OSError, ValueError, KeyError):
            self.score_dataset = None
            try:
                older_model = (dataset_folder(self.dataset.get().strip()) / "model.json").exists()
            except ValueError:
                older_model = False
            if older_model:
                self.next_trial.set("Older dataset: use Expand collection plan to add a saved plan.")
            else:
                self.next_trial.set("Create a new collection plan to begin.")

    def train_nearest(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            self.run_job("Training nearest-trial model",
                         lambda: train_dataset(name), self.nearest_trained)
        except ValueError as error:
            messagebox.showerror("Train nearest trial", str(error))

    def train_neural(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            self.run_job("Training neural model",
                         lambda: train_neural_dataset(name), self.neural_trained)
        except ValueError as error:
            messagebox.showerror("Train neural model", str(error))

    def create_plan(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            count = int(self.new_count.get().strip())
            classes = parse_class_counts(self.new_classes.get("1.0", "end"), count)
            create_collection(name, list(classes), classes)
            self.neural_session = None
            self.score_dataset = None
            self.refresh_dataset()
            self.status.set(f"Collection plan saved for {name}")
        except (OSError, ValueError, FileExistsError) as error:
            messagebox.showerror("Collection", str(error))

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
            self.status.set("Collection expanded. Use Start next grasp." if result
                            else "No new grasps requested")
        except (OSError, ValueError, FileNotFoundError) as error:
            messagebox.showerror("Expand collection", str(error))

    def collect_next(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            pending = pending_trials(name)
            if not pending:
                raise ValueError("No pending trials. Create or expand a collection first")
            class_name, trial = pending[0]
            if not messagebox.askokcancel("Ready for grasp",
                    f"Place {class_name} for grasp {trial}, then start the hand."):
                return
            def work():
                result = collect_one_trial(name, on_angles=self.queue_angles)
                if result["remaining"] == 0:
                    train_dataset(name)
                return result
            self.run_job(f"Collecting {class_name}, grasp {trial}", work,
                         self.trial_collected)
        except (OSError, ValueError, FileNotFoundError) as error:
            messagebox.showerror("Collect", str(error))

    def trial_collected(self, result):
        self.refresh_dataset()
        self.status.set(f"Saved {result['class_name']} grasp {result['trial']}; hand opened. "
                        f"{result['remaining']} grasps remain")
        if result["remaining"] == 0:
            self.result.set("Collection complete")

    def nearest_trained(self, model):
        self.status.set(f"Nearest-trial model saved: {len(model['samples'])} grasps")

    def neural_trained(self, result):
        _, report = result
        self.neural_session = None
        self.status.set(f"Neural model saved: {report['validation_correct']}/"
                        f"{report['validation_total']} held-out trials correct")

    def nearest_grasp(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            load_model(name)
            if not messagebox.askokcancel("Ready for grasp", "Place the object, then start the hand."):
                return
            self.run_job("Running nearest-trial grasp",
                         lambda: recognize_once(name, on_angles=self.queue_angles),
                         self.nearest_result)
        except (OSError, ValueError, FileNotFoundError) as error:
            messagebox.showerror("Recognize", str(error))

    def nearest_result(self, output):
        name = output["dataset"]
        model = load_model(name)
        angles = output["final_angles_deg"]
        distances = {class_name: min(math.dist(angles, item["angles_deg"])
                         for item in model["samples"] if item["class_name"] == class_name)
                     for class_name in model["classes"]}
        self.result.set(output["prediction"]["class_name"].upper())
        self.show_scores(distances, probability=False)
        self.status.set("Nearest-trial recognition saved")

    def reset_neural(self):
        if self.busy:
            return
        self.neural_session = None
        self.result.set("Awaiting a grasp")
        self.show_scores({}, probability=True)
        self.status.set("New neural reading ready")

    def neural_grasp(self):
        if self.busy:
            return
        try:
            name = self.selected_name()
            if self.neural_session is None or self.neural_session["name"] != name:
                model = load_neural_model(name)
                timestamp = datetime.now(timezone.utc)
                folder = dataset_folder(name) / "neural_runs" / timestamp.strftime("%Y%m%d_%H%M%S_%f")
                self.neural_session = dict(name=name, model=model, folder=folder,
                                           time=timestamp, results=[])
            session = self.neural_session
            number = len(session["results"]) + 1
            if number > PhysicalOnly.network_max_grasps:
                raise ValueError("Maximum repeat grasps reached. Start a new neural reading")
            if not messagebox.askokcancel("Ready for grasp",
                    f"Place the object for neural grasp {number}, then start the hand."):
                return
            self.run_job(f"Running neural grasp {number}",
                         lambda: self.neural_step(session, number), self.neural_result)
        except (OSError, ValueError, FileNotFoundError) as error:
            messagebox.showerror("Neural recognition", str(error))

    def neural_step(self, session, number):
        hand = open_hand()
        try:
            angles = grasp(hand, on_angles=self.queue_angles)
            grasp_folder = session["folder"] / f"grasp_{number}"
            try:
                save_grasp(grasp_folder)
            finally:
                release_hand(hand)
                self.queue_angles(Proprioception.grip["start_angles"])
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
        self.neural_session["results"] = output["grasps"]
        prediction = output["prediction"]
        self.result.set(prediction["class_name"].upper())
        self.show_scores(prediction["probabilities"], probability=True)
        reason = f"  ·  {prediction['reason']}" if prediction["reason"] else ""
        self.status.set(f"{output['grasp_count']} neural grasp(s) saved{reason}")

    def show_scores(self, scores, probability):
        for child in self.score_frame.winfo_children():
            child.destroy()
        self.score_title.set("CLASS PROBABILITIES" if probability
                             else "NEAREST TRIAL DISTANCE  ·  LOWER IS BETTER")
        for class_name, score in sorted(scores.items(),
                                         key=lambda item: -item[1] if probability else item[1]):
            row = tk.Frame(self.score_frame, bg=PANEL)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=class_name.title(), bg=PANEL, fg=TEXT,
                     font=("Helvetica", 11), width=16, anchor="w").pack(side="left")
            bar = tk.Canvas(row, height=12, bg=PANEL_LIGHT, highlightthickness=0)
            bar.pack(side="left", fill="x", expand=True, padx=8)
            value = max(0, min(1, score)) if probability else 1 / (1 + score)
            bar.bind("<Configure>", lambda event, canvas=bar, width=value:
                     (canvas.delete("fill"), canvas.create_rectangle(
                         0, 0, event.width * width, 12, fill=ACCENT, outline="", tags="fill")))
            label = f"{score * 100:.1f}%" if probability else f"{score:.1f}°"
            tk.Label(row, text=label, bg=PANEL, fg=ACCENT,
                     font=("Helvetica", 11, "bold"), width=8,
                     anchor="e").pack(side="right")

    def queue_angles(self, angles):
        self.events.put(("angles", angles))

    def run_job(self, label, work, finished):
        if self.busy:
            self.status.set("Wait for the current operation to finish")
            return
        self.busy = True
        self.dataset_box.configure(state="disabled")
        self.status.set(label + "…")
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
                elif kind == "done":
                    self.busy = False
                    self.dataset_box.configure(state="normal")
                    callback, result = value
                    try:
                        callback(result)
                    except Exception as error:
                        self.status.set(str(error))
                        messagebox.showerror("Display failed", str(error))
                elif kind == "error":
                    self.busy = False
                    self.dataset_box.configure(state="normal")
                    self.status.set(str(value))
                    messagebox.showerror("Operation failed", str(value))
        except Empty:
            pass
        self.root.after(PhysicalOnly.ui_poll_ms, self.read_events)

    def close(self):
        if self.busy:
            messagebox.showinfo("Hand operation in progress",
                                "Wait for the grasp or training to finish before closing.")
            return
        self.preview.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    PhysicalOnlyUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
