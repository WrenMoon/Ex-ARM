"""One window for the Ex-ARM physical, simulation, pose, and vision workflows."""

import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
from queue import Empty, Queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import Physical_Only_UI as physical_ui
from utils.Constants import Connection, PhysicalOnly, Proprioception, Studio


PROJECT_FOLDER = Path(__file__).resolve().parent
BG = physical_ui.BG
CARD = physical_ui.CARD
FIELD = physical_ui.FIELD
BORDER = physical_ui.BORDER
TEXT = physical_ui.TEXT
MUTED = physical_ui.MUTED
FAINT = physical_ui.FAINT
ACCENT = physical_ui.ACCENT
AMBER = physical_ui.AMBER
RED = physical_ui.RED
Button = physical_ui.Button
Dialog = physical_ui.Dialog
ScrollFrame = physical_ui.ScrollFrame
Segmented = physical_ui.Segmented
font = physical_ui.font
px = physical_ui.px


class Card(physical_ui.Card):
    def __init__(self, parent, title=None, subtitle=None, pad=16):
        super().__init__(parent, title, subtitle, pad)
        if subtitle:
            for child in self.winfo_children():
                if isinstance(child, tk.Label) and child is not self.title_label:
                    child.configure(wraplength=px(510))


class SharedJointStrip(physical_ui.JointStrip):
    def set(self, angles):
        super().set(angles)
        if len(angles) == 16 and hasattr(self, "value_label"):
            rows = []
            for finger, start in (("Index", 0), ("Middle", 4),
                                  ("Ring", 8), ("Thumb", 12)):
                rows.append(f"{finger:<6}  " + "  ".join(
                    f"{float(angle):6.1f}°" for angle in angles[start:start + 4]))
            self.value_label.configure(text="\n".join(rows))


def program_command(program, module=False, arguments=None):
    command = [sys.executable]
    command.extend(["-m", program] if module else [program])
    command.extend(arguments or [])
    return command


def scenario_names():
    path = PROJECT_FOLDER / Proprioception.training_steps_path
    if not path.exists():
        return {}
    scenarios = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not {"class", "scale_percent", "grip"}.issubset(reader.fieldnames or []):
            return {}
        for row in reader:
            name = row["class"]
            scale = int(float(row["scale_percent"]))
            scenarios.setdefault(name, set()).add(scale)
    return {name: sorted(scales, reverse=True) for name, scales in scenarios.items()}


def motion_names():
    folder = PROJECT_FOLDER / "Data" / "Motions"
    if not folder.exists():
        return []
    names = set()
    for path in folder.iterdir():
        if path.name.startswith("."):
            continue
        if path.name.endswith("_landmarks.pkl"):
            names.add(path.name[:-14])
        elif path.name.endswith("_trajectory.npy"):
            names.add(path.name[:-15])
        elif path.suffix == ".avi":
            names.add(path.stem)
    return sorted(names)


class ExARMUI:
    PAGES = ("Physical", "Simulation", "Poses", "Vision", "Diagnostics", "Tools")

    def __init__(self, root):
        self.root = root
        root.title("Ex-ARM  |  Project Studio")
        root.geometry(f"{px(max(1560, PhysicalOnly.ui_width))}x"
                      f"{px(max(980, PhysicalOnly.ui_height))}")
        root.minsize(px(1280), px(800))
        root.configure(bg=BG)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)

        self.events = Queue()
        self.process = None
        self.process_label = None
        self.process_done = None
        self.closing = False
        self.tool_buttons = []
        self.pose_control_buttons = []
        self.tool_available = True
        self.last_monitor_positions = None
        self.camera_stop = threading.Event()
        self.camera_cancelled = False
        self.camera_thread = None
        self.camera_frames = Queue(maxsize=1)
        self.pose_frames = Queue(maxsize=1)
        self.manual_positions = Queue(maxsize=1)
        self.operation_stop = threading.Event()
        self.operation_pause = threading.Event()
        self.operation_thread = None
        self.operation_label = None
        self.scenario_frames = None
        self.scenario_loading = False
        self.scenario_index = 0
        self.scenario_playing = False
        self.scenario_job = None
        self.status = tk.StringVar(value="Ready")
        self.pages = []
        self.build_header()

        self.main = tk.PanedWindow(root, orient="horizontal", bg=BG,
                                   sashwidth=px(5), bd=0)
        self.main.grid(row=1, column=0, sticky="nsew")
        self.host = tk.Frame(self.main, bg=BG)
        self.host.grid_columnconfigure(0, weight=1)
        self.host.grid_rowconfigure(0, weight=1)
        self.main.add(self.host, minsize=px(480), width=px(850))
        self.workspace = tk.Frame(self.main, bg=BG)
        self.workspace.grid_rowconfigure(0, weight=1)
        self.workspace.grid_columnconfigure(0, weight=1)
        self.main.add(self.workspace, minsize=px(390))
        self.build_workspace()

        physical_frame = tk.Frame(self.host, bg=BG)
        self.physical = physical_ui.PhysicalOnlyUI(
            physical_frame, embedded=True,
            shared_preview=self.preview, shared_joints=self.joints)
        self.physical.spin_button = self.spin_button
        self.pages.append(physical_frame)
        self.pages.append(self.build_simulation_page())
        self.pages.append(self.build_pose_page())
        self.pages.append(self.build_vision_page())
        self.pages.append(self.build_diagnostics_page())
        self.pages.append(self.build_tools_page())
        for page in self.pages:
            page.grid(row=0, column=0, sticky="nsew")

        self.build_activity()
        self.build_footer()
        if getattr(self, "pose_autosave_error", None):
            self.log("Pose autosave: " + self.pose_autosave_error, "error")
        self.show_page(0)
        root.bind_all("<MouseWheel>", self.wheel, add="+")
        root.bind_all("<Button-4>", self.wheel, add="+")
        root.bind_all("<Button-5>", self.wheel, add="+")
        root.bind_all("<KeyPress>", self.scenario_key, add="+")
        self.refresh_simulation()
        self.refresh_sequences()
        self.refresh_motions()
        root.after(100, self.read_events)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def label(self, parent, text, size=11, color=MUTED, weight="normal", **options):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                        font=font(size, weight), anchor="w", justify="left", **options)

    def button(self, parent, text, command, kind="primary", width=220):
        button = Button(parent, text, command, kind=kind, width=width)
        self.tool_buttons.append(button)
        return button

    def field(self, parent, text, value="", width=18):
        row = tk.Frame(parent, bg=parent.cget("bg"))
        row.pack(fill="x", pady=px(6))
        self.label(row, text, 10, MUTED, "bold").pack(side="left")
        entry = tk.Entry(row, bg=FIELD, fg=TEXT, insertbackground=ACCENT,
                         relief="flat", highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT, font=font(11), width=width)
        entry.insert(0, value)
        entry.pack(side="right", ipady=px(6))
        return entry

    def build_header(self):
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=px(20),
                    pady=(px(14), px(12)))
        mark = tk.Canvas(header, width=px(44), height=px(44), bg=BG,
                         highlightthickness=0)
        physical_ui.rounded(mark, 0, 0, px(44), px(44), px(12),
                            fill=ACCENT, outline="")
        mark.create_text(px(22), px(22), text="EX", fill=BG, font=font(14, "bold"))
        mark.pack(side="left")
        title = tk.Frame(header, bg=BG)
        title.pack(side="left", padx=(px(12), px(26)))
        self.label(title, "Ex-ARM Project Studio", 19, TEXT, "bold").pack(anchor="w")
        self.label(title, "LEAP Hand control, perception, and experiments",
                   10, MUTED).pack(anchor="w")
        self.navigation = Segmented(header, list(self.PAGES), self.show_page)
        self.navigation.pack(side="left", fill="x", expand=True)

    def new_page(self, title, description):
        page = tk.Frame(self.host, bg=BG)
        heading = tk.Frame(page, bg=BG)
        heading.pack(fill="x", padx=px(24), pady=(px(12), px(14)))
        self.label(heading, title, 24, TEXT, "bold").pack(anchor="w")
        self.label(heading, description, 11, MUTED,
                   wraplength=px(540)).pack(anchor="w", pady=(px(3), 0))
        scroll = ScrollFrame(page, bg=BG)
        scroll.pack(fill="both", expand=True, padx=px(24), pady=(0, px(12)))
        return page, scroll.inner

    def build_workspace(self):
        self.viewer_card = Card(self.workspace, "Shared MuJoCo hand",
                                "Live poses and recorded grasps use this view.")
        self.viewer_card.grid(row=0, column=0, sticky="nsew", padx=(0, px(16)),
                              pady=(px(12), px(12)))
        self.preview = physical_ui.HandView(self.viewer_card.body)
        self.preview.pack(fill="both", expand=True)
        tools = tk.Frame(self.viewer_card.body, bg=CARD)
        tools.pack(fill="x", pady=(px(8), 0))
        Button(tools, "Reset", self.preview.reset_view, "ghost", height=32,
               width=80).pack(side="left")
        Button(tools, "+", lambda: self.preview.zoom(0.85), "ghost", height=32,
               width=45).pack(side="left", padx=px(5))
        Button(tools, "−", lambda: self.preview.zoom(1.18), "ghost", height=32,
               width=45).pack(side="left")
        self.spin_button = Button(tools, "Rotate", self.toggle_spin,
                                  "ghost", height=32, width=95)
        self.spin_button.pack(side="right")
        joint_card = Card(self.workspace, "Joint state",
                          "Bars show closure in the selected grip; values are degrees.")
        joint_card.grid(row=1, column=0, sticky="ew", padx=(0, px(16)),
                        pady=(0, px(12)))
        self.joints = SharedJointStrip(joint_card.body)
        self.joints.pack(fill="x")
        self.joints.value_label = tk.Label(joint_card.body, text="", bg=CARD,
                                           fg=MUTED, font=font(9, mono=True),
                                           anchor="w", justify="left")
        self.joints.value_label.pack(fill="x", pady=(px(5), 0))
        self.joints.set(Proprioception.grip[0]["start_angles"])
        self.camera_card = Card(self.workspace, "Camera", "Live hand landmarks")
        self.camera_view = tk.Canvas(self.camera_card.body, bg="#0c1222",
                                     highlightthickness=0, width=px(Studio.camera_width),
                                     height=px(Studio.camera_height))
        self.camera_view.pack(fill="both", expand=True)
        self.camera_image = None
        self.camera_status = self.label(self.camera_card.body, "Camera idle", 10, MUTED)
        self.camera_status.pack(fill="x", pady=(px(8), 0))
        Button(self.camera_card.body, "Stop camera", self.stop_camera, "ghost",
               height=34, width=120).pack(anchor="e", pady=(px(8), 0))

    def toggle_spin(self):
        on = self.preview.toggle_spin()
        self.spin_button.set_text("Rotate  on" if on else "Rotate")

    def show_camera_panel(self, visible):
        if visible:
            self.workspace.grid_columnconfigure(0, weight=1)
            self.workspace.grid_columnconfigure(1, weight=1)
            self.camera_card.grid(row=0, column=1, rowspan=2, sticky="nsew",
                                  padx=(0, px(16)), pady=(px(12), px(12)))
        else:
            self.camera_card.grid_remove()
            self.workspace.grid_columnconfigure(1, weight=0)

    def show_page(self, index):
        self.current_page = index
        self.pages[index].tkraise()
        self.navigation.select(index)
        self.root.after_idle(lambda: self.main.sash_place(
            0, px(850 if index == 0 else 600), 0))
        if index != 1 and self.scenario_frames is not None:
            self.stop_scenario()
            self.preview.show_hand()
        if index == 0:
            self.activity.grid()
            if self.last_monitor_positions is not None:
                self.preview.set_pose(self.last_monitor_positions)
        else:
            self.activity.grid()

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

    def scenario_key(self, event):
        if self.current_page != 1 or self.scenario_frames is None:
            return
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Combobox)):
            return
        if event.keysym == "space":
            self.toggle_scenario()
        elif event.keysym == "Left":
            self.step_scenario(-1)
        elif event.keysym == "Right":
            self.step_scenario(1)
        elif event.keysym.lower() == "r":
            self.restart_scenario()

    def build_simulation_page(self):
        page, body = self.new_page(
            "Simulation-trained recognition",
            "Generate object and size examples in MuJoCo, inspect the grasps, "
            "then run the trained model on the physical hand.")
        training = Card(body, "Generate and train",
                        "Uses the configured STEP objects, size sweep, and grip sequence.")
        training.pack(fill="x", pady=(0, px(12)))
        self.sim_summary = self.label(training.body, "No training report loaded",
                                      11, TEXT, wraplength=px(510))
        self.sim_summary.pack(fill="x", pady=(0, px(12)))
        row = tk.Frame(training.body, bg=CARD)
        row.pack(fill="x")
        self.button(row, "Generate grasps and train",
                    lambda: self.launch("Train_Proprioception.py",
                                        "Simulation training", self.refresh_simulation),
                    width=250).pack(side="left")
        self.button(row, "Refresh report", self.refresh_simulation,
                    "secondary", 140).pack(side="left", padx=px(8))
        self.button(training.body, "Open simulation data",
                    lambda: self.open_folder(PROJECT_FOLDER / Proprioception.data_folder),
                    "ghost", 190).pack(anchor="w", pady=(px(8), 0))

        viewer = Card(body, "Inspect recorded grasps",
                      "Choose an object and size. Use playback controls below or "
                      "Space and the arrow keys while this section is selected.")
        viewer.pack(fill="x", pady=(0, px(12)))
        row = tk.Frame(viewer.body, bg=CARD)
        row.pack(fill="x", pady=(0, px(12)))
        self.label(row, "OBJECT", 10, MUTED, "bold").pack(side="left")
        self.scenario_object = ttk.Combobox(row, state="readonly", width=24,
                                            style="Dataset.TCombobox", font=font(11))
        self.scenario_object.pack(side="left", padx=(px(10), px(24)))
        self.scenario_object.bind("<<ComboboxSelected>>", self.update_scales)
        self.label(row, "SCALE", 10, MUTED, "bold").pack(side="left")
        self.scenario_scale = ttk.Combobox(row, state="readonly", width=12,
                                           style="Dataset.TCombobox", font=font(11))
        self.scenario_scale.pack(side="left", padx=(px(10), 0))
        row = tk.Frame(viewer.body, bg=CARD)
        row.pack(fill="x")
        self.button(row, "Load grasp", self.view_scenario,
                    "secondary", 135).pack(side="left")
        self.button(row, "◀", lambda: self.step_scenario(-1),
                    "ghost", 50).pack(side="left", padx=(px(8), 0))
        self.button(row, "Play / pause", self.toggle_scenario,
                    "ghost", 130).pack(side="left", padx=px(8))
        self.button(row, "▶", lambda: self.step_scenario(1),
                    "ghost", 50).pack(side="left")
        self.button(row, "Restart", self.restart_scenario,
                    "ghost", 95).pack(side="left", padx=(px(8), 0))
        self.scenario_status = self.label(viewer.body, "No grasp loaded", 10, MUTED)
        self.scenario_status.pack(fill="x", pady=(px(8), 0))

        recognize = Card(body, "Physical test with simulation model",
                         "Runs the saved class-and-size model using the real hand. "
                         "The two recognition experiments use separate models.")
        recognize.pack(fill="x")
        self.sim_result = self.label(recognize.body, "No result from this UI session",
                                     12, TEXT, wraplength=px(510))
        self.sim_result.pack(fill="x", pady=(0, px(12)))
        self.button(recognize.body, "Recognize object and size",
                    self.run_sim_recognition, width=250).pack(anchor="w")
        return page

    def build_pose_page(self):
        page, body = self.new_page(
            "Pose tools", "Edit joint angles and named poses, or play a saved JSON sequence.")
        editor = Card(body, "Pose editor",
                      "Edit all 16 joints here. The shared hand shows each change.")
        editor.pack(fill="x", pady=(0, px(12)))
        self.manual_live = tk.BooleanVar(value=False)
        self.pose_values = [tk.DoubleVar(value=0) for _ in range(16)]
        self.pose_entries = []
        self.pose_sliders = []
        self.pose_loading = False
        self.pose_series = []
        self.pose_autosave_error = None
        for finger, finger_name in enumerate(("Index", "Middle", "Ring", "Thumb")):
            group = Card(editor.body, finger_name, pad=10)
            group.pack(fill="x", pady=(0, px(8)))
            for joint in range(4):
                index = finger * 4 + joint
                row = tk.Frame(group.body, bg=CARD)
                row.pack(fill="x", pady=px(2))
                self.label(row, f"J{index}", 10, TEXT, "bold", width=4).pack(side="left")
                slider = tk.Scale(
                    row, from_=Studio.pose_min_deg, to=Studio.pose_max_deg,
                    resolution=1, orient="horizontal", variable=self.pose_values[index],
                    showvalue=False, bg=CARD, fg=TEXT, troughcolor=FIELD,
                    highlightthickness=0, bd=0, length=px(430),
                    command=lambda value, i=index: self.pose_changed(i))
                slider.pack(side="left", fill="x", expand=True)
                self.pose_sliders.append(slider)
                entry = tk.Entry(row, textvariable=self.pose_values[index], width=7,
                                 bg=FIELD, fg=TEXT, insertbackground=ACCENT,
                                 relief="flat", font=font(10))
                entry.pack(side="right", padx=(px(8), 0))
                entry.bind("<Return>", lambda event: self.pose_changed())
                entry.bind("<FocusOut>", lambda event: self.pose_changed())
                self.pose_entries.append(entry)
        row = tk.Frame(editor.body, bg=CARD)
        row.pack(fill="x", pady=(px(6), 0))
        button = Button(row, "Zero all", lambda: self.zero_pose(), "ghost", width=95)
        button.pack(side="left")
        self.pose_control_buttons.append(button)
        for finger, label in enumerate(("Index", "Middle", "Ring", "Thumb")):
            button = Button(row, "Zero " + label,
                            lambda i=finger: self.zero_pose(i), "ghost", width=115)
            button.pack(side="left", padx=(px(5), 0))
            self.pose_control_buttons.append(button)
        self.button(editor.body, "Send current pose to hand", self.send_pose,
                    "secondary", 235).pack(anchor="w", pady=(px(10), 0))
        tk.Checkbutton(editor.body, text="Live hand control", variable=self.manual_live,
                       command=self.toggle_manual_control, bg=CARD, fg=TEXT,
                       selectcolor=FIELD, activebackground=CARD,
                       activeforeground=TEXT, font=font(10)).pack(
                           anchor="w", pady=(px(8), 0))
        row = tk.Frame(editor.body, bg=CARD)
        row.pack(fill="x", pady=(px(8), 0))
        for number, (label, source, target) in enumerate((
            ("Index → Middle", 0, 1), ("Index → Ring", 0, 2),
            ("Middle → Index", 1, 0), ("Middle → Ring", 1, 2),
            ("Ring → Index", 2, 0), ("Ring → Middle", 2, 1),
        )):
            if number == 3:
                row = tk.Frame(editor.body, bg=CARD)
                row.pack(fill="x", pady=(px(4), 0))
            button = Button(row, label,
                            lambda a=source, b=target: self.copy_finger(a, b),
                            "ghost", width=125)
            button.pack(side="left", padx=(0, px(5)))
            self.pose_control_buttons.append(button)

        session = Card(body, "Pose sequence", "Save, edit, reorder, and play named poses.")
        session.pack(fill="x", pady=(0, px(12)))
        self.pose_name = self.field(session.body, "Pose name", "Pose 1")
        self.pose_duration = self.field(session.body, "Duration, seconds", "1.0")
        self.pose_list = tk.Listbox(session.body, height=6, bg=FIELD, fg=TEXT,
                                    selectbackground=ACCENT, selectforeground=BG,
                                    relief="flat", font=font(11))
        self.pose_list.pack(fill="x", pady=px(8))
        self.pose_list.bind("<<ListboxSelect>>", self.select_pose)
        for actions in (
            (("Add", "add"), ("Update", "update"), ("Duplicate", "duplicate"),
             ("Delete", "delete")),
            (("Insert after", "insert"), ("Move up", "up"), ("Move down", "down"),
             ("Clear", "clear")),
            (("Save session", "save"), ("Load session", "load"),
             ("Copy JSON", "copy"), ("Import JSON", "import")),
            (("Copy pose", "copy_pose"),),
        ):
            row = tk.Frame(session.body, bg=CARD)
            row.pack(fill="x", pady=px(3))
            for label, action in actions:
                self.button(row, label, lambda a=action: self.pose_action(a),
                            "ghost", 130).pack(side="left", padx=(0, px(6)))
        row = tk.Frame(session.body, bg=CARD)
        row.pack(fill="x", pady=(px(8), 0))
        self.button(row, "Play edited sequence", self.play_edited_sequence,
                    width=230).pack(side="left")
        self.button(row, "Pause / resume", self.toggle_motion_pause,
                    "secondary", 155).pack(side="left", padx=px(8))
        Button(row, "Stop", self.stop_process, "ghost", width=90).pack(side="left")

        sequence = Card(body, "Play an example sequence",
                        "Select a saved sequence before commanding the configured hand backend.")
        sequence.pack(fill="x", pady=(0, px(12)))
        row = tk.Frame(sequence.body, bg=CARD)
        row.pack(fill="x", pady=(0, px(12)))
        self.sequence_name = ttk.Combobox(row, state="readonly", width=28,
                                          style="Dataset.TCombobox", font=font(11))
        self.sequence_name.pack(side="left")
        self.button(row, "Refresh", self.refresh_sequences,
                    "ghost", 110).pack(side="left", padx=px(8))
        self.button(sequence.body, "Play selected sequence",
                    self.play_sequence, width=230).pack(anchor="w")
        self.button(sequence.body, "Open sequence folder",
                    lambda: self.open_folder(PROJECT_FOLDER / "Data" /
                                             "Example_pose_series"),
                    "ghost", 205).pack(anchor="w", pady=(px(8), 0))
        note = Card(body, "Configured backend")
        note.pack(fill="x")
        self.label(note.body, f"Connection.mode = {Connection.mode}. "
                   "Pose tools use this backend and the port in utils/Constants.py.",
                   11, MUTED).pack(fill="x")
        path = PROJECT_FOLDER / Studio.pose_autosave_path
        if path.exists():
            try:
                self.pose_series = self.validate_pose_series(json.loads(path.read_text()))
                self.refresh_pose_list(0)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.pose_autosave_error = str(error)
        return page

    def build_vision_page(self):
        page, body = self.new_page(
            "Vision and motion teaching",
            "Control the hand from a webcam or record, process, and replay a taught motion.")
        camera = Card(body, "Live webcam control",
                      "The camera appears beside the shared MuJoCo hand. "
                      "Stop it here before starting another motion.")
        camera.pack(fill="x", pady=(0, px(12)))
        row = tk.Frame(camera.body, bg=CARD)
        row.pack(fill="x")
        self.button(row, "Direct angle control",
                    lambda: self.start_camera("direct"),
                    width=215).pack(side="left")
        self.button(row, "Fingertip retargeting",
                    lambda: self.start_camera("retarget"),
                    "secondary", 225).pack(side="left", padx=px(8))
        Button(camera.body, "Stop camera", self.stop_camera,
               "ghost", width=150).pack(anchor="w", pady=(px(8), 0))

        teaching = Card(body, "Kinesthetic teaching",
                        "Record webcam landmarks, solve inverse kinematics, then replay "
                        "the resulting joint trajectory.")
        teaching.pack(fill="x", pady=(0, px(12)))
        self.motion_name = self.field(teaching.body, "Motion name")
        self.motion_duration = self.field(teaching.body, "Recording duration, seconds", "10")
        self.motion_speed = self.field(teaching.body, "Replay speed", "1.0")
        row = tk.Frame(teaching.body, bg=CARD)
        row.pack(fill="x", pady=(px(8), 0))
        self.button(row, "Record", lambda: self.motion_action("record"),
                    width=140).pack(side="left")
        self.button(row, "Process", lambda: self.motion_action("process"),
                    "secondary", 140).pack(side="left", padx=px(8))
        row = tk.Frame(teaching.body, bg=CARD)
        row.pack(fill="x", pady=(px(8), 0))
        self.button(row, "Replay", lambda: self.motion_action("replay"),
                    "secondary", 140).pack(side="left")
        self.button(row, "Full pipeline", lambda: self.motion_action("all"),
                    "secondary", 165).pack(side="left", padx=px(8))

        saved = Card(body, "Saved motions")
        saved.pack(fill="x")
        row = tk.Frame(saved.body, bg=CARD)
        row.pack(fill="x")
        self.saved_motion = ttk.Combobox(row, state="readonly", width=28,
                                         style="Dataset.TCombobox", font=font(11))
        self.saved_motion.pack(side="left")
        self.button(row, "Use name", self.use_saved_motion,
                    "ghost", 105).pack(side="left", padx=px(8))
        self.button(row, "Refresh", self.refresh_motions,
                    "ghost", 105).pack(side="left")
        self.button(row, "Delete", self.delete_motion,
                    "ghost", 105).pack(side="left", padx=px(8))
        self.button(saved.body, "Open motions folder",
                    lambda: self.open_folder(PROJECT_FOLDER / "Data" / "Motions"),
                    "ghost", 205).pack(anchor="w", pady=(px(12), 0))
        return page

    def build_diagnostics_page(self):
        page, body = self.new_page(
            "Hand diagnostics", "Read live joint state or run the configured physical grip.")
        connection = Card(body, "Connection")
        connection.pack(fill="x", pady=(0, px(12)))
        self.label(connection.body,
                   f"Mode: {Connection.mode}     Port: {Connection.Port}     "
                   f"Baudrate: {Connection.baudrate:,}     Motors: {len(Connection.ids)}",
                   11, TEXT).pack(fill="x")
        self.label(connection.body,
                   "Change motor IDs, offsets, connection mode, and grip settings in "
                   "utils/Constants.py before starting a physical run.",
                   10, MUTED, wraplength=px(510)).pack(fill="x", pady=(px(6), 0))

        monitor = Card(body, "Live joint state",
                       "Positions, velocities, and currents for the selected backend.")
        monitor.pack(fill="x", pady=(0, px(12)))
        self.joint_output = tk.Text(monitor.body, height=9, bg=FIELD, fg=TEXT,
                                    font=font(10, mono=True), relief="flat",
                                    state="disabled", padx=px(10), pady=px(8))
        self.joint_output.pack(fill="x", pady=(0, px(12)))
        self.button(monitor.body, "Start monitor",
                    lambda: self.launch("Monitor_Hand.py", "Joint monitor"),
                    width=180).pack(anchor="w")

        grasp = Card(body, "Controlled physical grasp",
                     "Runs the shared grip sequence and saves its final angles and motor log.")
        grasp.pack(fill="x", pady=(0, px(12)))
        self.button(grasp.body, "Run configured grasp",
                    lambda: self.start_grasp(False),
                    width=215).pack(anchor="w")
        self.button(grasp.body, "Open grasp data",
                    lambda: self.open_folder(PROJECT_FOLDER / Proprioception.data_folder),
                    "ghost", 185).pack(anchor="w", pady=(px(8), 0))
        legacy = Card(body, "Legacy current-limit test",
                      "Runs test2.py with its own older grip settings. "
                      "Use the shared grip above for proprioception data.")
        legacy.pack(fill="x", pady=(px(12), 0))
        self.button(legacy.body, "Run legacy test",
                    lambda: self.launch("test2.py", "Legacy current-limit test"),
                    "ghost", 180).pack(anchor="w")
        return page

    def build_tools_page(self):
        page, body = self.new_page(
            "Kinematics and files",
            "Inspect fingertip positions, solve a single-finger target, or update the hand model.")
        forward = Card(body, "Forward kinematics",
                       "Enter 16 joint angles in degrees, in index, middle, ring, thumb order.")
        forward.pack(fill="x", pady=(0, px(12)))
        self.fk_angles = self.field(forward.body, "Joint angles",
                                    ", ".join(["0"] * 16), 48)
        self.fk_result = self.label(forward.body, "Fingertip positions will appear here",
                                    11, TEXT, wraplength=px(510))
        self.fk_result.pack(fill="x", pady=(px(8), px(12)))
        self.button(forward.body, "Calculate fingertips", self.calculate_fk,
                    width=210).pack(anchor="w")

        inverse = Card(body, "Single-finger inverse kinematics",
                       "Target XYZ is in the palm frame, in metres. "
                       "The solution is calculated with the kinematics module's joint limits.")
        inverse.pack(fill="x", pady=(0, px(12)))
        row = tk.Frame(inverse.body, bg=CARD)
        row.pack(fill="x")
        self.label(row, "FINGER", 10, MUTED, "bold").pack(side="left")
        self.ik_finger = ttk.Combobox(row, state="readonly", width=16,
                                      values=("Index", "Middle", "Ring", "Thumb"),
                                      style="Dataset.TCombobox", font=font(11))
        self.ik_finger.set("Index")
        self.ik_finger.pack(side="left", padx=(px(10), 0))
        self.ik_target = self.field(inverse.body, "Target x, y, z", "0.0, 0.0, 0.0", 35)
        self.ik_result = self.label(inverse.body, "Joint angles will appear here",
                                    11, TEXT, wraplength=px(510))
        self.ik_result.pack(fill="x", pady=(px(8), px(12)))
        self.button(inverse.body, "Solve target", self.calculate_ik,
                    "secondary", 180).pack(anchor="w")

        files = Card(body, "Hand model and configuration",
                     "Regenerate the MuJoCo URDF from robot.urdf after editing the source model.")
        files.pack(fill="x")
        row = tk.Frame(files.body, bg=CARD)
        row.pack(fill="x")
        self.button(row, "Convert URDF", self.convert_urdf,
                    "secondary", 170).pack(side="left")
        self.button(row, "Open hand model folder",
                    lambda: self.open_folder(PROJECT_FOLDER / "Data" / "Leap_Model"),
                    "ghost", 215).pack(side="left", padx=px(8))
        self.button(files.body, "Open objects folder",
                    lambda: self.open_folder(PROJECT_FOLDER / Proprioception.object_folder),
                    "ghost", 190).pack(anchor="w", pady=(px(8), 0))
        resources = Card(body, "Project files")
        resources.pack(fill="x", pady=(px(12), 0))
        row = tk.Frame(resources.body, bg=CARD)
        row.pack(fill="x")
        self.button(row, "Open constants",
                    lambda: self.open_file(PROJECT_FOLDER / "utils" / "Constants.py"),
                    "ghost", 175).pack(side="left")
        self.button(row, "Open README",
                    lambda: self.open_file(PROJECT_FOLDER / "README.md"),
                    "ghost", 155).pack(side="left", padx=px(8))
        row = tk.Frame(resources.body, bg=CARD)
        row.pack(fill="x", pady=(px(8), 0))
        self.button(row, "Open manuscript",
                    lambda: self.open_file(PROJECT_FOLDER / "Manuscript" / "Manuscript.tex"),
                    "ghost", 190).pack(side="left")
        self.button(row, "Open IRIS folder",
                    lambda: self.open_folder(PROJECT_FOLDER / "IRIS_Submission"),
                    "ghost", 180).pack(side="left", padx=px(8))
        return page

    def calculate_fk(self):
        try:
            import numpy as np
            from utils.LeapKinematics import LeapKinematics
            angles = [float(part.strip()) for part in self.fk_angles.get().split(",")]
            if len(angles) != 16 or not all(math.isfinite(value) for value in angles):
                raise ValueError("Enter 16 finite angles separated by commas")
            tips = LeapKinematics().fk_degree(np.asarray(angles))
            text = "    ".join(
                f"{name}: ({x:+.3f}, {y:+.3f}, {z:+.3f}) m"
                for name, (x, y, z) in zip(
                    ("Index", "Middle", "Ring", "Thumb"), tips))
            self.fk_result.configure(text=text)
        except (ImportError, ValueError, TypeError) as error:
            Dialog.show(self.root, "Forward kinematics", str(error))

    def calculate_ik(self):
        try:
            from utils.LeapKinematics import LeapKinematics
            target = [float(part.strip()) for part in self.ik_target.get().split(",")]
            if len(target) != 3 or not all(math.isfinite(value) for value in target):
                raise ValueError("Enter three finite XYZ values separated by commas")
            finger = ("Index", "Middle", "Ring", "Thumb").index(self.ik_finger.get())
            angles, info = LeapKinematics().ik_finger_degree(
                finger, target, n_starts=4)
            text = (", ".join(f"{angle:.1f}°" for angle in angles) +
                    f"    error: {info['error_m'] * 1000:.2f} mm")
            if not info["success"]:
                text += "    target not reached within tolerance"
            self.ik_result.configure(text=text)
        except (ImportError, ValueError, TypeError) as error:
            Dialog.show(self.root, "Inverse kinematics", str(error))

    def convert_urdf(self):
        if not Dialog.ask(self.root, "Convert hand model",
                          "Replace mujoco_robot.urdf with a conversion of robot.urdf?",
                          "Convert"):
            return
        self.launch("utils/URDF_Convertor.py", "URDF conversion")

    def build_activity(self):
        self.activity = Card(self.root, "Tool activity")
        self.activity.grid(row=2, column=0, sticky="ew", padx=px(24),
                           pady=(0, px(8)))
        self.activity_text = tk.Text(self.activity.body, height=5, bg=CARD, fg=MUTED,
                                     font=font(9, mono=True), relief="flat",
                                     highlightthickness=0, state="disabled")
        self.activity_text.pack(fill="x")
        self.activity_text.tag_configure("error", foreground=RED)
        self.activity_text.tag_configure("ok", foreground=ACCENT)

    def build_footer(self):
        footer = tk.Frame(self.root, bg=BG)
        footer.grid(row=3, column=0, sticky="ew", padx=px(24),
                    pady=(0, px(10)))
        self.label(footer, "●", 11, ACCENT).pack(side="left")
        tk.Label(footer, textvariable=self.status, bg=BG, fg=MUTED,
                 font=font(10)).pack(side="left", padx=(px(8), 0))
        self.stop_button = Button(footer, "Stop active operation", self.stop_process,
                                  kind="ghost", height=34, width=165, size=10)
        self.stop_button.pack(side="right")
        self.stop_button.set_enabled(False)

    def log(self, message, kind="info"):
        self.status.set(message)
        self.activity_text.configure(state="normal")
        self.activity_text.insert("end", datetime.now().strftime("%H:%M:%S  ") +
                                  message + "\n", kind)
        if int(self.activity_text.index("end-1c").split(".")[0]) > 300:
            self.activity_text.delete("1.0", "60.0")
        self.activity_text.see("end")
        self.activity_text.configure(state="disabled")

    def launch(self, program, label, done=None, module=False,
               arguments=None):
        if (self.process is not None or self.scenario_loading or
                self.camera_thread is not None or
                self.operation_thread is not None or self.physical.busy):
            self.log("Wait for the current tool to finish", "error")
            return
        try:
            command = program_command(program, module, arguments)
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(command, cwd=PROJECT_FOLDER,
                                       stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       text=True, bufsize=1,
                                       creationflags=flags,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1",
                                            "EXARM_UI": "1"})
        except (OSError, ValueError) as error:
            self.log(str(error), "error")
            Dialog.show(self.root, "Cannot start tool", str(error))
            return
        self.process = process
        self.process_label = label
        self.process_done = done
        self.physical.set_busy(True, label)
        if label == "Joint monitor":
            self.preview.set_caption("LIVE  ·  JOINT MONITOR")
        self.update_tool_buttons()
        self.stop_button.set_enabled(True)
        self.log(f"Started {label}")

        def read_output():
            for line in process.stdout:
                self.events.put(("output", line.rstrip()))
            self.events.put(("done", process.wait()))

        threading.Thread(target=read_output, daemon=True).start()

    def stop_process(self):
        process = self.process
        if process is None:
            if self.camera_thread is not None:
                self.stop_camera()
            elif self.operation_thread is not None:
                self.operation_stop.set()
                self.log(f"Stopping {self.operation_label}")
            elif self.physical.busy and self.physical.cancelable:
                self.physical.stop_event.set()
                self.log("Stopping physical grasp; reopening hand")
            return
        if process is None or process.poll() is not None:
            return
        self.log(f"Stopping {self.process_label}")
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        except (OSError, ValueError):
            process.terminate()
        self.root.after(3000, lambda: process.terminate() if process.poll() is None else None)

    def read_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "output":
                    if self.process_label == "Joint monitor":
                        self.render_joint_state(value)
                    elif value:
                        self.log(value)
                elif kind == "done":
                    label, callback = self.process_label, self.process_done
                    self.process = None
                    self.process_label = None
                    self.process_done = None
                    self.physical.set_busy(False)
                    if label == "Joint monitor":
                        self.preview.set_caption(None)
                    self.update_tool_buttons()
                    self.stop_button.set_enabled(False)
                    if value == 0:
                        self.log(f"{label} completed", "ok")
                        if callback and not self.closing:
                            try:
                                callback()
                            except (OSError, ValueError, KeyError) as error:
                                self.log(str(error), "error")
                    else:
                        self.log(f"{label} exited with code {value}", "error")
                elif kind == "scenario_ready":
                    self.scenario_loading = False
                    if self.current_page == 1 and not self.closing:
                        try:
                            self.show_scenario(*value)
                            self.log("Recorded grasp loaded", "ok")
                        except Exception as error:
                            self.stop_scenario()
                            self.preview.show_hand()
                            self.scenario_status.configure(text=str(error))
                            self.log(str(error), "error")
                elif kind == "scenario_error":
                    self.scenario_loading = False
                    self.scenario_status.configure(text=value)
                    self.log(value, "error")
                elif kind == "grasp_result":
                    self.show_sim_result(value)
                elif kind == "camera_done":
                    mode, name, chain, error = value
                    self.camera_thread = None
                    try:
                        self.camera_frames.get_nowait()
                    except Empty:
                        pass
                    self.preview.set_target_markers(None)
                    self.preview.set_caption(None)
                    self.preview.draw()
                    self.physical.set_busy(False)
                    self.show_camera_panel(False)
                    self.camera_status.configure(text="Camera idle")
                    self.stop_button.set_enabled(False)
                    if error:
                        self.log(error, "error")
                    else:
                        self.log("Camera stopped" if mode != "record"
                                 else f"Recorded motion: {name}", "ok")
                        self.refresh_motions()
                        if chain and not self.camera_cancelled and not self.closing:
                            self.launch("Vision.Vision_Kinesthetic_Teaching",
                                        f"Motion process: {name}",
                                        lambda: self.replay_motion(
                                            name, float(self.motion_speed.get())),
                                        module=True,
                                        arguments=["--action", "process", "--name", name])
                elif kind == "operation_done":
                    label = self.operation_label
                    self.operation_thread = None
                    self.operation_label = None
                    self.manual_live.set(False)
                    self.preview.set_caption(None)
                    self.physical.set_busy(False)
                    self.stop_button.set_enabled(False)
                    self.log(f"{label} failed: {value}" if value else
                             f"{label} completed", "error" if value else "ok")
                if self.closing and self.process is None and self.camera_thread is None \
                        and self.operation_thread is None:
                    self.close()
                    return
        except Empty:
            pass
        try:
            ppm, pose, targets = self.camera_frames.get_nowait()
            self.camera_image = tk.PhotoImage(data=ppm, format="PPM")
            self.camera_view.delete("frame")
            self.camera_view.create_image(
                self.camera_view.winfo_width() / 2,
                self.camera_view.winfo_height() / 2,
                image=self.camera_image, tags="frame")
            self.camera_status.configure(text="Live camera")
            had_markers = self.preview.markers is not None
            self.preview.set_target_markers(targets)
            if pose is not None:
                self.preview.set_pose(pose)
                self.joints.set(pose)
            elif had_markers:
                self.preview.draw()
        except Empty:
            pass
        except Exception as error:
            self.log("Camera preview: " + str(error), "error")
        try:
            pose = self.pose_frames.get_nowait()
            self.preview.set_pose(pose)
            self.joints.set(pose)
        except Empty:
            pass
        except Exception as error:
            self.log("Hand preview: " + str(error), "error")
        self.update_tool_buttons()
        if self.closing and not self.physical.busy and self.process is None \
                and self.camera_thread is None and self.operation_thread is None:
            self.close()
            return
        if self.root.winfo_exists():
            self.root.after(Studio.camera_poll_ms, self.read_events)

    def update_tool_buttons(self):
        available = (self.process is None and not self.scenario_loading and
                     self.camera_thread is None and
                     self.operation_thread is None and not self.physical.busy)
        if available != self.tool_available:
            for button in self.tool_buttons:
                button.set_enabled(available)
            self.tool_available = available
        pose_editable = available or self.operation_label == "Manual control"
        for button in self.pose_control_buttons:
            if button.enabled != pose_editable:
                button.set_enabled(pose_editable)
        for widget in self.pose_sliders + self.pose_entries:
            state = "normal" if pose_editable else "disabled"
            if widget.cget("state") != state:
                widget.configure(state=state)
        stoppable = (self.process is not None or self.camera_thread is not None or
                     self.operation_thread is not None or
                     (self.physical.busy and self.physical.cancelable))
        if self.stop_button.enabled != stoppable:
            self.stop_button.set_enabled(stoppable)

    def render_joint_state(self, line):
        try:
            state = json.loads(line)
            positions = state["positions"]
            velocities = state["velocities"]
            currents = state["currents"]
            if any(len(values) != 16 for values in
                   (positions, velocities, currents)):
                raise ValueError("Expected 16 joint values")
        except (ValueError, KeyError, TypeError):
            if line:
                self.log(line)
            return
        rows = ["FINGER       POSITIONS                 VELOCITIES              CURRENTS"]
        for index, name in enumerate(("Index", "Middle", "Ring", "Thumb")):
            start = index * 4
            parts = []
            for values in (positions, velocities, currents):
                parts.append(" ".join(f"{float(number):6.1f}" for number
                                      in values[start:start + 4]))
            rows.append(f"{name:<8}  " + "    ".join(parts))
        self.joint_output.configure(state="normal")
        self.joint_output.delete("1.0", "end")
        self.joint_output.insert("1.0", "\n".join(rows))
        self.joint_output.configure(state="disabled")
        self.last_monitor_positions = positions
        self.preview.set_pose(positions)
        self.joints.set(positions)

    def refresh_simulation(self):
        path = PROJECT_FOLDER / Proprioception.training_report_path
        if path.exists():
            try:
                report = json.loads(path.read_text())
                summary = (f"{report['simulation_count']} simulated grasps  ·  "
                           f"{report['validation_accuracy']:.1%} held-out class accuracy  ·  "
                           f"{report['validation_size_mae_mm']:.2f} mm size error  ·  "
                           f"{report['accepted_count']}/{report['validation_count']} accepted")
                settings = report.get("settings", {})
                if (settings.get("grips") != Proprioception.grip or
                        settings.get("objects") != Proprioception.objects):
                    summary += "\nSaved report uses different grip or object settings; retrain before use."
                self.sim_summary.configure(text=summary)
            except (OSError, ValueError, KeyError, TypeError):
                self.sim_summary.configure(text="Training report could not be read")
        else:
            self.sim_summary.configure(text="No simulation training report yet")
        scenarios = scenario_names()
        self.scenarios = scenarios
        if not scenarios and (PROJECT_FOLDER / Proprioception.training_steps_path).exists():
            self.sim_summary.configure(
                text=self.sim_summary.cget("text") +
                     "\nViewer recordings use an older format; regenerate them by training.")
        names = [item["class_name"] for item in Proprioception.objects
                 if item["class_name"] in scenarios]
        self.scenario_object.configure(values=names)
        if self.scenario_object.get() not in names:
            self.scenario_object.set(names[0] if names else "")
        self.update_scales()

    def update_scales(self, event=None):
        scales = self.scenarios.get(self.scenario_object.get(), [])
        values = [f"{value}%" for value in scales]
        self.scenario_scale.configure(values=values)
        if self.scenario_scale.get() not in values:
            self.scenario_scale.set(values[0] if values else "")

    def view_scenario(self):
        name = self.scenario_object.get()
        scale = self.scenario_scale.get().rstrip("%")
        if not name or not scale:
            Dialog.show(self.root, "No recorded grasp",
                        "Generate simulation grasps before opening the viewer.")
            return
        if (self.process is not None or self.scenario_loading or
                self.camera_thread is not None or self.physical.busy):
            self.log("Wait for the current operation to finish", "error")
            return
        self.stop_scenario()
        self.scenario_loading = True
        self.scenario_status.configure(text="Loading object mesh and grasp frames...")

        def load():
            try:
                from View_Proprioception import load_training_steps
                from utils.ProprioceptionSimulation import GraspSimulation
                key = (name, int(scale))
                frames = load_training_steps()[key]
                object_info = next(item for item in Proprioception.objects
                                   if item["class_name"] == name)
                simulation = GraspSimulation(object_info, int(scale))
                self.events.put(("scenario_ready", (simulation, frames)))
            except Exception as error:
                self.events.put(("scenario_error", str(error)))

        threading.Thread(target=load, daemon=True).start()

    def show_scenario(self, simulation, frames):
        if not self.preview.ready:
            raise RuntimeError("MuJoCo preview unavailable: " +
                               (self.preview.error or "renderer did not start"))
        self.scenario_frames = frames
        self.scenario_index = 0
        simulation.set_angles(frames[0][2])
        self.preview.show_simulation(simulation)
        self.scenario_playing = True
        self.show_scenario_frame()
        self.schedule_scenario()

    def show_scenario_frame(self):
        grip, _, angles, _ = self.scenario_frames[self.scenario_index]
        self.preview.scenario.set_angles(angles)
        self.preview.draw()
        self.joints.set_grip(grip)
        self.joints.set(angles)
        self.update_scenario_label()

    def update_scenario_label(self):
        if self.scenario_frames is None:
            self.scenario_status.configure(text="No grasp loaded")
            return
        grip, step, _, reasons = self.scenario_frames[self.scenario_index]
        state = "Playing" if self.scenario_playing else "Paused"
        stops = ", ".join(f"F{finger}: {reason}" for finger, reason
                           in enumerate(reasons) if reason)
        self.scenario_status.configure(
            text=f"{state} · grip {grip} · step {step} · "
                 f"frame {self.scenario_index + 1}/{len(self.scenario_frames)} · "
                 + (stops or "fingers moving"))

    def step_scenario(self, direction):
        if self.scenario_frames is None:
            return
        self.scenario_playing = False
        self.scenario_index = max(0, min(len(self.scenario_frames) - 1,
                                         self.scenario_index + direction))
        self.show_scenario_frame()

    def toggle_scenario(self):
        if self.scenario_frames is None:
            return
        self.scenario_playing = not self.scenario_playing
        self.update_scenario_label()
        if self.scenario_playing:
            self.schedule_scenario()
        elif self.scenario_job is not None:
            self.root.after_cancel(self.scenario_job)
            self.scenario_job = None

    def schedule_scenario(self):
        if self.scenario_job is not None:
            self.root.after_cancel(self.scenario_job)
        self.scenario_job = self.root.after(
            max(1, int(Proprioception.step_time_s * 1000)), self.tick_scenario)

    def restart_scenario(self):
        if self.scenario_frames is None:
            return
        self.scenario_index = 0
        self.scenario_playing = True
        self.show_scenario_frame()
        self.schedule_scenario()

    def tick_scenario(self):
        self.scenario_job = None
        if not self.scenario_playing or self.scenario_frames is None:
            return
        if self.scenario_index < len(self.scenario_frames) - 1:
            self.scenario_index += 1
            self.show_scenario_frame()
            self.schedule_scenario()
        else:
            self.scenario_playing = False
            self.update_scenario_label()

    def stop_scenario(self):
        was_loaded = self.scenario_frames is not None
        if self.scenario_job is not None:
            self.root.after_cancel(self.scenario_job)
            self.scenario_job = None
        self.scenario_playing = False
        self.scenario_frames = None
        self.scenario_index = 0
        self.joints.set_grip(1)
        if was_loaded:
            self.joints.set(self.preview.last_hand_angles)
        if hasattr(self, "scenario_status"):
            self.update_scenario_label()

    def run_sim_recognition(self):
        if not Dialog.ask(self.root, "Physical recognition",
                          "Place the object at the trained position, then start the grasp.",
                          "Start grasp"):
            return
        self.start_grasp(True)

    def start_grasp(self, recognize):
        if (self.process is not None or self.scenario_loading or
                self.camera_thread is not None or self.operation_thread is not None
                or self.physical.busy):
            self.log("Wait for the current operation to finish", "error")
            return
        if Proprioception.hand_mode != "real":
            Dialog.show(self.root, "Physical grasp",
                        "Set Proprioception.hand_mode to real in Constants.py")
            return
        self.stop_scenario()
        self.preview.show_hand()
        self.operation_stop.clear()
        label = "Simulation-model recognition" if recognize else "Physical grasp"
        self.operation_label = label
        self.physical.set_busy(True, label, live=True)
        self.preview.set_caption("LIVE  ·  MEASURED ANGLES")
        self.operation_thread = threading.Thread(
            target=self.grasp_worker, args=(recognize,), daemon=True)
        self.operation_thread.start()
        self.update_tool_buttons()
        self.log("Started " + label)

    def grasp_worker(self, recognize):
        hand = None
        error = None
        try:
            if recognize:
                from utils.ProprioceptionModel import ProprioceptionModel
                model = ProprioceptionModel.load()
            from utils.LeapHand import LeapHand
            hand = LeapHand(ids=Connection.ids, port=Connection.Port,
                            baudrate=Connection.baudrate, offsets=Connection.offsets)

            def on_angles(angles):
                if self.pose_frames.full():
                    try:
                        self.pose_frames.get_nowait()
                    except Empty:
                        pass
                self.pose_frames.put_nowait(angles)

            if recognize:
                from Run_Proprioception import recognize as run_recognition
                output = run_recognition(hand, model, on_angles=on_angles,
                                         stop_event=self.operation_stop)
                self.events.put(("grasp_result", output))
            else:
                from Grasp import grasp
                grasp(hand, on_angles=on_angles, stop_event=self.operation_stop)
        except Exception as problem:
            error = str(problem)
        finally:
            if hand is not None:
                try:
                    hand.set_torque_enabled(False)
                except Exception as problem:
                    error = error or str(problem)
                try:
                    hand.close_port()
                except Exception as problem:
                    error = error or str(problem)
            self.events.put(("operation_done", error))

    def show_sim_result(self, output=None):
        if output is None:
            path = PROJECT_FOLDER / Proprioception.recognition_result_path
            output = json.loads(path.read_text())
        result = output["prediction"]
        if result["class_name"] == "unknown":
            text = f"UNKNOWN  ·  {result.get('reason', 'no reason saved')}"
        else:
            size = result.get("size_m")
            size_text = f"  ·  {result.get('size_name', 'size')}: {size * 1000:.1f} mm" \
                if size is not None else ""
            text = f"{result['class_name'].upper()}{size_text}"
        text += f"  ·  confidence {result.get('confidence', 0):.1%}"
        self.sim_result.configure(text=text)

    def refresh_sequences(self):
        folder = PROJECT_FOLDER / "Data" / "Example_pose_series"
        names = sorted(path.stem for path in folder.glob("*.json"))
        self.sequence_name.configure(values=names)
        if self.sequence_name.get() not in names:
            self.sequence_name.set(names[0] if names else "")

    def play_sequence(self):
        name = self.sequence_name.get()
        if not name:
            Dialog.show(self.root, "No pose sequence", "Add a JSON pose sequence first.")
            return
        try:
            from Pose_Sequence_Runner import load_pose_sequence, pose_to_array
            poses = load_pose_sequence(name)
            steps = [(pose_to_array(pose), pose["duration"]) for pose in poses]
            self.start_motion_steps(f"Pose sequence {name}", steps)
        except (OSError, ValueError, KeyError) as error:
            Dialog.show(self.root, "Pose sequence", str(error))

    def current_pose_angles(self):
        angles = [value.get() for value in self.pose_values]
        if (len(angles) != 16 or not all(
                math.isfinite(value) and Studio.pose_min_deg <= value <= Studio.pose_max_deg
                for value in angles)):
            raise ValueError("Enter 16 finite angles within the editor limits")
        return angles

    def set_pose_angles(self, angles):
        self.pose_loading = True
        try:
            for value, angle in zip(self.pose_values, angles):
                value.set(angle)
        finally:
            self.pose_loading = False
        self.pose_changed()

    def pose_changed(self, joint=None):
        if self.pose_loading:
            return
        try:
            angles = self.current_pose_angles()
        except (ValueError, tk.TclError):
            return
        self.preview.set_pose(angles)
        self.joints.set(angles)
        if self.manual_live.get() and self.operation_label == "Manual control":
            if self.manual_positions.full():
                try:
                    self.manual_positions.get_nowait()
                except Empty:
                    pass
            self.manual_positions.put_nowait(angles)

    def copy_finger(self, source, target):
        try:
            angles = self.current_pose_angles()
            angles[target * 4:target * 4 + 4] = angles[source * 4:source * 4 + 4]
            self.set_pose_angles(angles)
        except (ValueError, tk.TclError) as error:
            Dialog.show(self.root, "Pose editor", str(error))

    def zero_pose(self, finger=None):
        try:
            angles = self.current_pose_angles()
        except (ValueError, tk.TclError):
            angles = [0] * 16
        start, stop = (0, 16) if finger is None else (finger * 4, finger * 4 + 4)
        angles[start:stop] = [0] * (stop - start)
        self.set_pose_angles(angles)

    def send_pose(self):
        try:
            self.start_motion_steps("Manual pose", [
                (self.current_pose_angles(), Studio.pose_send_seconds)])
        except (ValueError, tk.TclError) as error:
            Dialog.show(self.root, "Pose editor", str(error))

    def selected_pose(self):
        selection = self.pose_list.curselection()
        return selection[0] if selection else None

    def select_pose(self, event=None):
        index = self.selected_pose()
        if index is None or index >= len(self.pose_series):
            return
        pose = self.pose_series[index]
        self.pose_name.delete(0, "end")
        self.pose_name.insert(0, pose["name"])
        self.pose_duration.delete(0, "end")
        self.pose_duration.insert(0, str(pose["duration"]))
        self.set_pose_angles(pose["angles"])

    def refresh_pose_list(self, selected=None):
        self.pose_list.delete(0, "end")
        for index, pose in enumerate(self.pose_series, 1):
            self.pose_list.insert("end", f"{index:02d}  {pose['name']}  ·  "
                                  f"{pose['duration']:.2f}s")
        if selected is not None and 0 <= selected < len(self.pose_series):
            self.pose_list.selection_set(selected)
            self.pose_list.see(selected)

    def pose_from_fields(self):
        name = self.pose_name.get().strip()
        duration = float(self.pose_duration.get())
        if not name or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Pose name and positive duration are required")
        return dict(name=name, duration=duration,
                    angles=self.current_pose_angles())

    def validate_pose_series(self, value):
        if isinstance(value, dict):
            value = value.get("poses", [value])
        if not isinstance(value, list):
            raise ValueError("Pose session must be a list or a poses object")
        poses = []
        for index, pose in enumerate(value):
            if not isinstance(pose, dict):
                raise ValueError(f"Pose {index + 1} is invalid")
            angles = pose.get("angles")
            if angles is None and all(key in pose for key in
                                      ("index", "middle", "ring", "thumb")):
                angles = sum((pose[key] for key in
                              ("index", "middle", "ring", "thumb")), [])
            duration = pose.get("duration")
            if (not isinstance(angles, list) or len(angles) != 16 or
                    any(isinstance(angle, bool) or not isinstance(angle, (int, float))
                        or not math.isfinite(angle) or
                        not Studio.pose_min_deg <= angle <= Studio.pose_max_deg
                        for angle in angles) or
                    isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError(f"Pose {index + 1} has invalid angles or duration")
            poses.append(dict(name=str(pose.get("name") or f"Pose {index + 1}"),
                              duration=float(duration), angles=list(angles)))
        return poses

    def save_pose_autosave(self):
        path = PROJECT_FOLDER / Studio.pose_autosave_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.pose_series, indent=2))

    def pose_action(self, action):
        index = self.selected_pose()
        try:
            if action in ("add", "insert", "update"):
                pose = self.pose_from_fields()
                if action == "add":
                    self.pose_series.append(pose)
                    index = len(self.pose_series) - 1
                elif action == "insert":
                    index = len(self.pose_series) if index is None else index + 1
                    self.pose_series.insert(index, pose)
                elif index is not None:
                    self.pose_series[index] = pose
                else:
                    raise ValueError("Select a pose to update")
            elif action == "duplicate" and index is not None:
                pose = dict(self.pose_series[index])
                pose["name"] += " Copy"
                pose["angles"] = pose["angles"].copy()
                index += 1
                self.pose_series.insert(index, pose)
            elif action == "delete" and index is not None:
                del self.pose_series[index]
                index = min(index, len(self.pose_series) - 1)
            elif action == "up" and index is not None and index > 0:
                self.pose_series[index - 1], self.pose_series[index] = (
                    self.pose_series[index], self.pose_series[index - 1])
                index -= 1
            elif action == "down" and index is not None and index < len(self.pose_series) - 1:
                self.pose_series[index + 1], self.pose_series[index] = (
                    self.pose_series[index], self.pose_series[index + 1])
                index += 1
            elif action == "clear":
                if self.pose_series and not Dialog.ask(self.root, "Clear poses",
                                                       "Remove all edited poses?", "Clear"):
                    return
                self.pose_series.clear()
                index = None
            elif action == "copy":
                self.root.clipboard_clear()
                self.root.clipboard_append(json.dumps(self.pose_series, indent=2))
                self.log("Pose sequence copied as JSON", "ok")
            elif action == "copy_pose":
                self.root.clipboard_clear()
                self.root.clipboard_append(json.dumps(self.pose_from_fields(), indent=2))
                self.log("Current pose copied as JSON", "ok")
            elif action == "import":
                self.pose_series = self.validate_pose_series(
                    json.loads(self.root.clipboard_get()))
                index = 0
            elif action in ("save", "load"):
                folder = PROJECT_FOLDER / "Data" / "Pose_Sessions"
                folder.mkdir(parents=True, exist_ok=True)
                if action == "save":
                    filename = filedialog.asksaveasfilename(
                        parent=self.root, initialdir=folder,
                        defaultextension=".json", filetypes=[("JSON", "*.json")])
                    if filename:
                        Path(filename).write_text(json.dumps(self.pose_series, indent=2))
                        self.log("Saved pose session: " + Path(filename).name, "ok")
                else:
                    filename = filedialog.askopenfilename(
                        parent=self.root, initialdir=folder,
                        filetypes=[("JSON", "*.json")])
                    if filename:
                        self.pose_series = self.validate_pose_series(
                            json.loads(Path(filename).read_text()))
                        index = 0
                        self.log("Loaded pose session: " + Path(filename).name, "ok")
            if action in ("add", "insert", "update", "duplicate", "delete",
                          "up", "down", "clear", "import", "load"):
                self.save_pose_autosave()
            self.refresh_pose_list(index)
        except (OSError, ValueError, KeyError, TypeError, tk.TclError) as error:
            Dialog.show(self.root, "Pose sequence", str(error))

    def play_edited_sequence(self):
        if not self.pose_series:
            Dialog.show(self.root, "Pose sequence", "Add a pose first")
            return
        steps = [(pose["angles"], pose["duration"]) for pose in self.pose_series]
        self.start_motion_steps("Edited pose sequence", steps)

    def toggle_motion_pause(self):
        if self.operation_thread is None:
            return
        if self.operation_pause.is_set():
            self.operation_pause.clear()
            self.log("Resumed " + self.operation_label)
        else:
            self.operation_pause.set()
            self.log("Paused " + self.operation_label)

    def toggle_manual_control(self):
        if not self.manual_live.get():
            if self.operation_label == "Manual control":
                self.operation_stop.set()
            return
        if (self.process is not None or self.scenario_loading or
                self.camera_thread is not None or self.operation_thread is not None
                or self.physical.busy):
            self.manual_live.set(False)
            self.log("Wait for the current operation to finish", "error")
            return
        try:
            angles = self.current_pose_angles()
        except (ValueError, tk.TclError) as error:
            self.manual_live.set(False)
            Dialog.show(self.root, "Pose editor", str(error))
            return
        self.operation_stop.clear()
        self.operation_label = "Manual control"
        self.physical.set_busy(True, "Manual control", live=True)
        self.preview.set_caption("LIVE  ·  POSE EDITOR")
        try:
            self.manual_positions.get_nowait()
        except Empty:
            pass
        self.manual_positions.put_nowait(angles)
        self.operation_thread = threading.Thread(target=self.manual_worker, daemon=True)
        self.operation_thread.start()
        self.update_tool_buttons()
        self.log("Live hand control started")

    def manual_worker(self):
        hand = None
        error = None
        try:
            if Connection.mode not in ("real", "sim", "both"):
                raise ValueError("Connection.mode must be real, sim, or both")
            if Connection.mode in ("real", "both"):
                from utils.LeapHand import LeapHand
                hand = LeapHand(ids=Connection.ids, port=Connection.Port,
                                baudrate=Connection.baudrate, offsets=Connection.offsets)
            while not self.operation_stop.is_set():
                try:
                    angles = self.manual_positions.get(timeout=0.1)
                except Empty:
                    continue
                if hand is not None:
                    hand.set_goal_positions_degree(angles)
                    hand.set_torque_enabled(True)
        except Exception as problem:
            error = str(problem)
        finally:
            if hand is not None:
                try:
                    hand.set_torque_enabled(False)
                except Exception as problem:
                    error = error or str(problem)
                try:
                    hand.close_port()
                except Exception as problem:
                    error = error or str(problem)
            self.events.put(("operation_done", error))

    def refresh_motions(self):
        names = motion_names()
        self.saved_motion.configure(values=names)
        if self.saved_motion.get() not in names:
            self.saved_motion.set(names[0] if names else "")

    def use_saved_motion(self):
        name = self.saved_motion.get()
        if name:
            self.motion_name.delete(0, "end")
            self.motion_name.insert(0, name)

    def start_camera(self, mode, name=None, duration=None, chain=False):
        if (self.process is not None or self.scenario_loading or
                self.camera_thread is not None or
                self.operation_thread is not None or self.physical.busy):
            self.log("Wait for the current operation to finish", "error")
            return
        self.stop_scenario()
        self.preview.show_hand()
        self.camera_stop.clear()
        self.camera_cancelled = False
        self.physical.set_busy(True, "Camera " + mode, live=mode != "record")
        self.preview.set_caption("RECORDING  ·  LANDMARKS" if mode == "record"
                                 else "LIVE  ·  CAMERA TELEOP")
        self.show_camera_panel(True)
        self.camera_status.configure(text="Opening camera...")
        self.log("Starting camera: " + mode)
        self.show_page(3)
        self.camera_thread = threading.Thread(
            target=self.camera_worker, args=(mode, name, duration, chain), daemon=True)
        self.camera_thread.start()
        self.update_tool_buttons()

    def camera_worker(self, mode, name, duration, chain):
        cap = writer = hand = None
        temporary_video = temporary_landmarks = None
        error = None
        frames = []
        try:
            import cv2
            import mediapipe as mp
            import numpy as np
            from Vision import Vision_Teleop as direct
            retarget = None
            if mode == "retarget":
                from Vision import Vision_Retargeting as retarget
                from utils.LeapKinematics import LeapKinematics

            if Connection.mode not in ("real", "sim", "both"):
                raise ValueError("Connection.mode must be real, sim, or both")

            if Connection.mode in ("real", "both") and mode != "record":
                from utils.LeapHand import LeapHand
                hand = LeapHand(ids=Connection.ids, port=Connection.Port,
                                baudrate=Connection.baudrate, offsets=Connection.offsets)
            kin = LeapKinematics() if mode == "retarget" else None
            prior_rad = None
            smooth = np.zeros(16)
            cap = cv2.VideoCapture(Studio.camera_index)
            if not cap.isOpened():
                raise RuntimeError("Could not open the configured camera")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Studio.camera_capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Studio.camera_capture_height)
            if mode == "record":
                path = PROJECT_FOLDER / "Data" / "Motions"
                path.mkdir(parents=True, exist_ok=True)
                temporary_video = path / f".{name}_recording.avi"
                temporary_landmarks = path / f".{name}_recording_landmarks.pkl"
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or fps < 1:
                    fps = Studio.motion_record_hz
                writer = cv2.VideoWriter(str(temporary_video),
                                         cv2.VideoWriter_fourcc(*"XVID"), fps,
                                         (Studio.camera_width, Studio.camera_height))
                if not writer.isOpened():
                    raise RuntimeError("Could not create the motion video")

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(PROJECT_FOLDER / "Data" / "hand_landmarker.task")),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1, min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
            start = time.monotonic()
            last_stamp = -1
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
                while not self.camera_stop.is_set():
                    elapsed = time.monotonic() - start
                    if duration is not None and elapsed >= duration:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera stopped providing frames")
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stamp = max(last_stamp + 1, int(elapsed * 1000))
                    last_stamp = stamp
                    result = detector.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)
                    pose = None
                    targets = None
                    points = None
                    if result.hand_landmarks:
                        landmarks = result.hand_landmarks[0]
                        points = np.asarray([[p.x, p.y, p.z] for p in landmarks],
                                            dtype=np.float64)
                        direct.draw_skeleton(frame, landmarks,
                                             frame.shape[1], frame.shape[0])
                        if mode == "direct":
                            angles = direct.landmarks_to_angles(points)
                            alpha = Studio.direct_smoothing_alpha
                            smooth = alpha * angles + (1 - alpha) * smooth
                            pose = smooth.copy()
                        elif mode == "retarget":
                            targets = retarget.landmarks_to_targets(points)
                            if targets is not None:
                                radians, _ = kin.ik(
                                    targets, q0=prior_rad,
                                    n_starts=retarget.IK_N_STARTS,
                                    tol=retarget.IK_TOL_M,
                                    max_iter=retarget.IK_MAX_ITER)
                                prior_rad = kin.clip_to_limits(radians)
                                smooth = (retarget.ALPHA * np.degrees(prior_rad)
                                          + (1 - retarget.ALPHA) * smooth)
                                pose = smooth.copy()
                    if mode == "record":
                        frames.append(points)
                        cv2.putText(frame, f"REC  {elapsed:.1f}s / {duration:.1f}s",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        writer.write(cv2.resize(frame, (Studio.camera_width,
                                                         Studio.camera_height)))
                    elif pose is not None and hand is not None:
                        hand.set_goal_positions_degree(np.round(pose, 1))
                        hand.set_torque_enabled(True)
                    image = cv2.resize(frame, (Studio.camera_width,
                                               Studio.camera_height))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ppm = (f"P6\n{Studio.camera_width} {Studio.camera_height}\n255\n".encode()
                           + image.tobytes())
                    if self.camera_frames.full():
                        try:
                            self.camera_frames.get_nowait()
                        except Empty:
                            pass
                    self.camera_frames.put_nowait((ppm, pose, targets))
            if mode == "record" and frames:
                import pickle
                writer.release()
                writer = None
                with temporary_landmarks.open("wb") as file:
                    pickle.dump(frames, file)
                temporary_video.replace(path / f"{name}.avi")
                temporary_landmarks.replace(path / f"{name}_landmarks.pkl")
            elif mode == "record":
                raise RuntimeError("No motion frames were recorded")
        except Exception as problem:
            error = str(problem) or type(problem).__name__
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception as problem:
                    error = error or str(problem)
            if writer is not None:
                try:
                    writer.release()
                except Exception as problem:
                    error = error or str(problem)
            if error is not None:
                for leftover in (temporary_video, temporary_landmarks):
                    if leftover is not None:
                        try:
                            leftover.unlink(missing_ok=True)
                        except OSError:
                            pass
            if hand is not None:
                try:
                    hand.set_torque_enabled(False)
                except Exception as problem:
                    error = error or str(problem)
                try:
                    hand.close_port()
                except Exception as problem:
                    error = error or str(problem)
            self.events.put(("camera_done", (mode, name, chain, error)))

    def stop_camera(self):
        if self.camera_thread is not None:
            self.camera_cancelled = True
            self.camera_stop.set()
            self.log("Stopping camera")

    def start_motion_steps(self, label, steps):
        if (self.process is not None or self.scenario_loading or
                self.camera_thread is not None or
                self.operation_thread is not None or self.physical.busy):
            self.log("Wait for the current operation to finish", "error")
            return
        self.stop_scenario()
        self.preview.show_hand()
        self.operation_stop.clear()
        self.operation_pause.clear()
        self.operation_label = label
        self.physical.set_busy(True, label, live=True)
        self.preview.set_caption("LIVE  ·  MOTION PLAYBACK")
        self.operation_thread = threading.Thread(
            target=self.motion_worker, args=(steps,), daemon=True)
        self.operation_thread.start()
        self.update_tool_buttons()
        self.log("Started " + label)

    def motion_worker(self, steps):
        hand = None
        error = None
        try:
            import numpy as np
            if Connection.mode not in ("real", "sim", "both"):
                raise ValueError("Connection.mode must be real, sim, or both")
            if Connection.mode in ("real", "both"):
                from utils.LeapHand import LeapHand
                hand = LeapHand(ids=Connection.ids, port=Connection.Port,
                                baudrate=Connection.baudrate, offsets=Connection.offsets)
            for angles, duration in steps:
                if self.operation_stop.is_set():
                    break
                pose = np.asarray(angles, dtype=float)
                if pose.shape != (16,) or not np.isfinite(pose).all():
                    raise ValueError("Motion contains invalid joint angles")
                end = time.monotonic() + duration
                while time.monotonic() < end and not self.operation_stop.is_set():
                    if self.operation_pause.is_set():
                        pause_start = time.monotonic()
                        while self.operation_pause.is_set() and not self.operation_stop.is_set():
                            time.sleep(0.05)
                        end += time.monotonic() - pause_start
                        continue
                    if hand is not None:
                        hand.set_goal_positions_degree(np.round(pose, 1))
                        hand.set_torque_enabled(True)
                    if self.pose_frames.full():
                        try:
                            self.pose_frames.get_nowait()
                        except Empty:
                            pass
                    self.pose_frames.put_nowait(pose.tolist())
                    time.sleep(min(Studio.motion_update_period_s,
                                   max(0, end - time.monotonic())))
        except Exception as problem:
            error = str(problem)
        finally:
            if hand is not None:
                try:
                    hand.set_torque_enabled(False)
                except Exception as problem:
                    error = error or str(problem)
                try:
                    hand.close_port()
                except Exception as problem:
                    error = error or str(problem)
            self.events.put(("operation_done", error))

    def replay_motion(self, name, speed):
        try:
            import numpy as np
            path = PROJECT_FOLDER / "Data" / "Motions" / f"{name}_trajectory.npy"
            trajectory = np.load(path)
            if trajectory.ndim != 2 or trajectory.shape[1] != 16 or len(trajectory) == 0:
                raise ValueError("Motion trajectory must contain 16 joint angles per frame")
            if not np.isfinite(trajectory).all():
                raise ValueError("Motion trajectory contains invalid angles")
            count = max(1, int(len(trajectory) * Studio.motion_replay_hz /
                               (Studio.motion_record_hz * speed)))
            positions = np.linspace(0, len(trajectory) - 1, count)
            left = np.floor(positions).astype(int)
            right = np.minimum(left + 1, len(trajectory) - 1)
            fraction = (positions - left)[:, None]
            samples = ((1 - fraction) * trajectory[left]
                       + fraction * trajectory[right])
            steps = [(sample, 1 / Studio.motion_replay_hz) for sample in samples]
            self.start_motion_steps(f"Motion replay: {name}", steps)
        except (OSError, ValueError) as error:
            Dialog.show(self.root, "Motion replay", str(error))

    def motion_action(self, action):
        name = self.motion_name.get().strip()
        if not name or name in (".", "..") or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789 _-" for char in name):
            Dialog.show(self.root, "Motion name",
                        "Use a name with letters, numbers, spaces, _ or -.")
            return
        arguments = ["--action", action, "--name", name]
        try:
            if action in ("record", "all"):
                duration = float(self.motion_duration.get())
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Recording duration must be a positive finite number")
                arguments.extend(["--duration", str(duration)])
            if action in ("replay", "all"):
                speed = float(self.motion_speed.get())
                if not math.isfinite(speed) or speed <= 0:
                    raise ValueError("Replay speed must be a positive finite number")
                arguments.extend(["--speed", str(speed)])
        except ValueError as error:
            Dialog.show(self.root, "Motion settings", str(error))
            return
        if action in ("record", "all"):
            if name in motion_names() and not Dialog.ask(
                    self.root, "Replace motion",
                    f"Replace the saved recording for {name}?", "Replace"):
                return
            self.start_camera("record", name, duration, chain=action == "all")
        elif action == "replay":
            self.replay_motion(name, speed)
        else:
            self.launch("Vision.Vision_Kinesthetic_Teaching",
                        f"Motion {action}: {name}", self.refresh_motions,
                        module=True, arguments=arguments)

    def delete_motion(self):
        name = self.saved_motion.get()
        if not name:
            return
        if not Dialog.ask(self.root, "Delete motion",
                          f"Delete the saved files for {name}?", "Delete"):
            return
        self.launch("Vision.Vision_Kinesthetic_Teaching", f"Delete motion {name}",
                    self.refresh_motions, module=True,
                    arguments=["--action", "delete", "--name", name])

    def open_folder(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.open_file(folder)

    def open_file(self, path):
        try:
            if os.name == "nt":
                os.startfile(str(Path(path).resolve()))
            else:
                subprocess.Popen(["open" if platform.system() == "Darwin" else "xdg-open",
                                  str(Path(path).resolve())])
        except OSError as error:
            self.log(str(error), "error")

    def close(self):
        if self.physical.busy and self.physical.cancelable:
            if not self.closing:
                if not Dialog.ask(self.root, "Grasp is running",
                                  "Stop the grasp, reopen the hand, and close Ex-ARM?",
                                  "Stop and close"):
                    return
                self.closing = True
                self.stop_process()
            return
        if (self.process is not None or self.camera_thread is not None or
                self.operation_thread is not None):
            if not self.closing:
                if not Dialog.ask(self.root, "Tool is running",
                                  "Stop the active tool and close Ex-ARM?",
                                  "Stop and close"):
                    return
                self.closing = True
                self.stop_process()
            return
        if self.physical.close():
            self.preview.close()
            self.root.destroy()


def main():
    os.chdir(PROJECT_FOLDER)
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    if platform.system() == "Windows":
        physical_ui.SCALE = max(1.0, root.winfo_fpixels("1i") / 96)
    ExARMUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
