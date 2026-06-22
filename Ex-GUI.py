"""
Ex-GUI.py — LEAP Hand Pose Sequencer GUI

A Tkinter-based graphical tool for creating, editing, and playing back
sequences of LEAP Hand poses. Supports real-time joint control via
sliders, session save/load, clipboard export, and autosave.

Run directly:  python Ex-GUI.py
"""

import json
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import os
from utils.Constants import Connection


try:
    from utils.ExARM import ExArm
except ImportError as e:
    print(f"Import error: {e}")
    # Minimal mock so the GUI can be tested without connected hardware
    class ExArm:
        def __init__(self, **kwargs):
            print("Mock ExArm initialized with:", kwargs)
        def set_goal_positions_degree(self, positions):
            print(f"Mock: Setting positions to {positions}")
        def set_torque_enabled(self, enabled):
            print(f"Mock: Torque {'enabled' if enabled else 'disabled'}")


class Pose:
    """
    Data container for a single hand pose.

    Attributes
    ----------
    name     : human-readable label shown in the pose list
    duration : how long (seconds) the pose is held during playback
    angles   : (16,) list of joint angles in degrees, logical ordering
    """

    def __init__(self, name="", duration=1.0, angles=None):
        self.name     = name
        self.duration = duration
        self.angles   = angles or [0.0] * 16

    def to_dict(self):
        """Serialise to a flat dict (used for JSON session files)."""
        return {
            "name":     self.name,
            "duration": self.duration,
            "angles":   self.angles
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialise from a flat dict (flat 16-element angles list)."""
        return cls(data["name"], data["duration"], data["angles"])

    @classmethod
    def from_finger_dict(cls, data):
        """
        Construct a Pose from a finger-grouped dict.

        Expects keys: 'index', 'middle', 'ring', 'thumb' (each a 4-element list).
        This format is also used by bottle_orient.py and clipboard export.
        """
        angles = (
            data["index"] + data["middle"]
            + data["ring"] + data["thumb"]
        )
        return cls(data.get("name", ""), data["duration"], angles)

    def to_finger_dict(self):
        """
        Serialise to a finger-grouped dict matching the bottle_orient.py POSES format.
        Useful for clipboard export so poses can be pasted directly into scripts.
        """
        return {
            "name":     self.name,
            "duration": self.duration,
            "index":    self.angles[0:4],
            "middle":   self.angles[4:8],
            "ring":     self.angles[8:12],
            "thumb":    self.angles[12:16]
        }


class PoseManager:
    """
    Ordered collection of Pose objects with CRUD and reordering operations.

    Maintains a current_pose_index so the GUI always knows which pose
    is selected, even after inserts, deletes, and moves.
    """

    def __init__(self):
        self.poses               = []
        self.current_pose_index  = -1

    def add_pose(self, pose):
        """Append a pose and select it."""
        self.poses.append(pose)
        self.current_pose_index = len(self.poses) - 1

    def update_pose(self, index, pose):
        """Replace the pose at index with a new Pose object."""
        if 0 <= index < len(self.poses):
            self.poses[index] = pose

    def delete_pose(self, index):
        """Remove the pose at index; adjust current index if necessary."""
        if 0 <= index < len(self.poses):
            del self.poses[index]
            if self.current_pose_index >= len(self.poses):
                self.current_pose_index = len(self.poses) - 1

    def move_pose_up(self, index):
        """Swap pose at index with the one above it."""
        if 1 <= index < len(self.poses):
            self.poses[index], self.poses[index - 1] = (
                self.poses[index - 1], self.poses[index]
            )
            if self.current_pose_index == index:
                self.current_pose_index = index - 1
            elif self.current_pose_index == index - 1:
                self.current_pose_index = index

    def move_pose_down(self, index):
        """Swap pose at index with the one below it."""
        if 0 <= index < len(self.poses) - 1:
            self.poses[index], self.poses[index + 1] = (
                self.poses[index + 1], self.poses[index]
            )
            if self.current_pose_index == index:
                self.current_pose_index = index + 1
            elif self.current_pose_index == index + 1:
                self.current_pose_index = index

    def duplicate_pose(self, index):
        """Insert a copy of the pose at index immediately after it."""
        if 0 <= index < len(self.poses):
            new_pose = Pose(
                name=f"{self.poses[index].name} Copy",
                duration=self.poses[index].duration,
                angles=self.poses[index].angles.copy()
            )
            self.poses.insert(index + 1, new_pose)
            self.current_pose_index = index + 1

    def insert_pose_after(self, index, pose):
        """Insert pose after index, or append if index is out of range."""
        if 0 <= index < len(self.poses):
            self.poses.insert(index + 1, pose)
            self.current_pose_index = index + 1
        else:
            self.poses.append(pose)
            self.current_pose_index = len(self.poses) - 1

    def get_pose(self, index):
        """Return the Pose at index, or None if out of range."""
        return self.poses[index] if 0 <= index < len(self.poses) else None

    def get_current_pose(self):
        """Return the currently selected Pose."""
        return self.get_pose(self.current_pose_index)

    def set_current_pose_index(self, index):
        if -1 <= index < len(self.poses):
            self.current_pose_index = index

    def to_list(self):
        """Serialise all poses to a list of dicts (for JSON save)."""
        return [pose.to_dict() for pose in self.poses]

    def from_list(self, data):
        """Deserialise all poses from a list of dicts (from JSON load)."""
        self.poses               = [Pose.from_dict(item) for item in data]
        self.current_pose_index  = -1

    def save_to_file(self, filename):
        """Write the pose sequence to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_list(), f, indent=2)

    def load_from_file(self, filename):
        """
        Load a pose sequence from a JSON file.

        Returns True on success, False on error.
        """
        try:
            with open(filename, 'r') as f:
                self.from_list(json.load(f))
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False


class RobotController:
    """
    Thread-safe interface between the GUI and the LEAP Hand hardware/sim.

    Runs a background thread that continuously pushes the latest joint
    angles to the robot at ~30 Hz, so the hand tracks GUI slider changes
    in near real-time without blocking the UI event loop.
    """

    def __init__(self):
        self.robot           = None
        self.running         = False
        self.torque_enabled  = False
        self.current_angles  = [0.0] * 16
        self.thread          = None
        self.error           = None

    def connect(self):
        """
        Instantiate ExArm in 'both' mode (real + sim).

        Returns True on success, False if connection fails (e.g. hardware
        not present). The GUI falls back to mock mode on failure.
        """
        try:
            self.robot = ExArm(
                mode="both",
                ids=Connection.ids,
                port=Connection.Port,
                baudrate=Connection.baudrate,
                offsets=Connection.offsets,
                model_path="Data/mujoco_robot.urdf"
            )
            return True
        except Exception as e:
            self.error = str(e)
            return False

    def start_update_thread(self):
        """Start the background 30 Hz robot update thread if not already running."""
        if not self.running:
            self.running = True
            self.thread  = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()

    def stop_update_thread(self):
        """Signal the update thread to stop and wait for it to finish."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _update_loop(self):
        """
        Background loop: sends current_angles to the robot at ~30 Hz.

        Only transmits when torque is enabled, to avoid unnecessary serial
        traffic when the hand is relaxed.
        """
        while self.running:
            try:
                if self.robot and self.torque_enabled:
                    # Normalise to exactly 16 elements, padding/trimming as needed
                    if len(self.current_angles) != 16:
                        if len(self.current_angles) > 16:
                            angles_array = np.array(self.current_angles[:16])
                        else:
                            angles_array = np.array(
                                self.current_angles + [0.0] * (16 - len(self.current_angles))
                            )
                    else:
                        angles_array = np.array(self.current_angles)
                    self.robot.set_goal_positions_degree(angles_array)
                    self.robot.set_torque_enabled(True)
                time.sleep(1 / 30)
            except Exception as e:
                print(f"Error in update loop: {e}")
                self.error = str(e)

    def set_angles(self, angles):
        """
        Update the target joint angles.

        Normalises the input to exactly 16 values (pads with zeros or truncates).
        """
        if len(angles) > 16:
            self.current_angles = angles[:16]
        elif len(angles) < 16:
            self.current_angles = list(angles) + [0.0] * (16 - len(angles))
        else:
            self.current_angles = list(angles)

    def set_torque_enabled(self, enabled):
        """Enable or disable motor torque and forward the command to the robot."""
        self.torque_enabled = enabled
        if self.robot:
            try:
                self.robot.set_torque_enabled(enabled)
            except Exception as e:
                self.error = str(e)

    def get_current_angles(self):
        """Return a normalised copy of the current 16-element angle list."""
        if len(self.current_angles) > 16:
            return self.current_angles[:16]
        elif len(self.current_angles) < 16:
            return self.current_angles + [0.0] * (16 - len(self.current_angles))
        return self.current_angles.copy()


class PlaybackController:
    """
    Sequential playback of a PoseManager's pose list.

    Runs in a dedicated daemon thread. Supports play, pause (toggle),
    and stop. Tracks the currently playing pose index so the GUI can
    highlight it in the pose list.
    """

    def __init__(self, pose_manager, robot_controller):
        self.pose_manager           = pose_manager
        self.robot_controller       = robot_controller
        self.playback_thread        = None
        self.playback_running       = False
        self.paused                 = False
        self.current_playback_index = -1

    def play(self):
        """Start playback from the beginning if not already running."""
        if not self.playback_running:
            self.playback_running = True
            self.paused           = False
            self.playback_thread  = threading.Thread(
                target=self._playback_loop, daemon=True
            )
            self.playback_thread.start()

    def pause(self):
        """Toggle pause state during playback."""
        self.paused = not self.paused

    def stop(self):
        """Stop playback and wait for the playback thread to exit."""
        self.playback_running = False
        self.paused           = False
        if self.playback_thread:
            self.playback_thread.join()

    def _playback_loop(self):
        """
        Internal playback thread: iterates through poses in sequence,
        holds each for its duration, and advances to the next.
        Respects pause and stop signals with 10 ms polling resolution.
        """
        i     = 0
        poses = self.pose_manager.poses

        while self.playback_running and i < len(poses):
            if self.paused:
                time.sleep(0.1)
                continue

            pose                        = poses[i]
            self.current_playback_index = i

            self.robot_controller.set_angles(pose.angles)

            # Wait out the pose duration while watching for stop/pause signals
            start_time = time.time()
            while (
                time.time() - start_time < pose.duration
                and self.playback_running
                and not self.paused
            ):
                time.sleep(0.01)

            if not self.playback_running:
                break

            i += 1

        self.playback_running       = False
        self.current_playback_index = -1


class JointControl:
    """
    Composite widget for a single joint: label, ±1/±5 buttons, slider, and entry.

    The slider spans ±180 °. Direct text entry is also supported.
    Any change triggers on_change_callback(joint_index, new_value).
    """

    def __init__(self, parent, joint_index, label, on_change_callback):
        self.joint_index        = joint_index
        self.on_change_callback = on_change_callback
        self.frame              = ttk.Frame(parent)

        # Widget creation
        self.minus_five_btn = ttk.Button(self.frame, text="-5", width=3, command=self._decrease_five)
        self.minus_one_btn  = ttk.Button(self.frame, text="-1", width=3, command=self._decrease_one)
        self.slider         = ttk.Scale(
            self.frame, from_=-180, to=180,
            orient=tk.HORIZONTAL, command=self._slider_changed, length=120
        )
        self.plus_one_btn   = ttk.Button(self.frame, text="+1", width=3, command=self._increase_one)
        self.plus_five_btn  = ttk.Button(self.frame, text="+5", width=3, command=self._increase_five)
        self.value_var      = tk.StringVar(value="0")
        self.entry          = ttk.Entry(self.frame, textvariable=self.value_var, width=6)
        self.entry.bind('<Return>', self._entry_changed)
        self.label          = ttk.Label(self.frame, text=label)

        # Layout — label above, controls in a row
        self.label.grid(row=0, column=0, columnspan=6, pady=(0, 2))
        self.minus_five_btn.grid(row=1, column=0, padx=(0, 2))
        self.minus_one_btn.grid( row=1, column=1, padx=(0, 2))
        self.slider.grid(        row=1, column=2, padx=(0, 2))
        self.plus_one_btn.grid(  row=1, column=3, padx=(0, 2))
        self.plus_five_btn.grid( row=1, column=4, padx=(0, 2))
        self.entry.grid(         row=1, column=5, padx=(0, 2))

        self.set_value(0)

    def _decrease_five(self):
        new_value = max(-180, float(self.value_var.get()) - 5)
        self.set_value(new_value)
        self.on_change_callback(self.joint_index, new_value)

    def _decrease_one(self):
        new_value = max(-180, float(self.value_var.get()) - 1)
        self.set_value(new_value)
        self.on_change_callback(self.joint_index, new_value)

    def _increase_one(self):
        new_value = min(180, float(self.value_var.get()) + 1)
        self.set_value(new_value)
        self.on_change_callback(self.joint_index, new_value)

    def _increase_five(self):
        new_value = min(180, float(self.value_var.get()) + 5)
        self.set_value(new_value)
        self.on_change_callback(self.joint_index, new_value)

    def _slider_changed(self, value):
        """Called by the ttk.Scale widget on every drag event."""
        new_value = float(value)
        self.value_var.set(f"{new_value:.1f}")
        self.on_change_callback(self.joint_index, new_value)

    def _entry_changed(self, event):
        """Validate and apply a manually typed angle value."""
        try:
            value = float(self.value_var.get())
            if -180 <= value <= 180:
                self.set_value(value)
                self.on_change_callback(self.joint_index, value)
            else:
                messagebox.showerror("Invalid Input", "Value must be between -180 and 180")
                self.set_value(self.slider.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")
            self.set_value(self.slider.get())

    def set_value(self, value):
        """Programmatically set the slider and entry to value (degrees)."""
        self.slider.set(value)
        self.value_var.set(f"{value:.1f}")

    def get_value(self):
        """Return the current joint angle in degrees as a float."""
        return float(self.value_var.get())


class PoseEditorGUI:
    """
    Main application window for the LEAP Hand Pose Sequencer.

    Layout (top → bottom):
      - Title bar
      - Control bar  : torque toggle, finger-copy utilities, zero utilities
      - Joint panel  : four fingers, each with four JointControl widgets
      - Pose display : live read-back of current commanded angles
      - Pose list    : scrollable sequence with CRUD and reorder buttons
      - Playback bar : play / pause / stop
      - Bottom bar   : session save/load, clipboard import/export, clear
    """

    def __init__(self, root):
        self.root         = root
        self.root.title("LEAP Hand Pose Sequencer: Ex-GUI")
        self.root.geometry("1600x900")

        # Core data and control objects
        self.pose_manager       = PoseManager()
        self.robot_controller   = RobotController()
        self.playback_controller = PlaybackController(
            self.pose_manager, self.robot_controller
        )

        # Attempt hardware/sim connection; fall back to mock on failure
        if not self.robot_controller.connect():
            messagebox.showwarning(
                "Connection Warning",
                f"Failed to connect to robot: {self.robot_controller.error}\nUsing mock mode."
            )

        self.robot_controller.start_update_thread()

        self._create_widgets()
        self._check_autosave()

        # Guards against recursive update loops when loading a pose into sliders
        self.changes_pending = False
        self.updating_pose   = False

        self._start_update_loop()

    # ------------------------------------------------------------------ #
    # Widget construction                                                  #
    # ------------------------------------------------------------------ #

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        ttk.Label(
            main_frame,
            text="LEAP Hand Pose Sequencer: Ex-GUI",
            font=("Arial", 16, "bold")
        ).grid(row=0, column=0, pady=(0, 10))

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self._create_control_bar(control_frame)

        joint_frame = ttk.Frame(main_frame)
        joint_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self._create_horizontal_joint_controls(joint_frame)

        # Live angle read-back display
        pose_display_frame = ttk.LabelFrame(main_frame, text="Current Pose", padding="5")
        pose_display_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.current_pose_text = tk.Text(
            pose_display_frame, height=4, width=80, state=tk.DISABLED
        )
        self.current_pose_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        pose_mgmt_frame = ttk.Frame(main_frame)
        pose_mgmt_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        pose_mgmt_frame.columnconfigure(0, weight=1)
        pose_mgmt_frame.rowconfigure(1, weight=1)
        self._create_pose_management(pose_mgmt_frame)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        self._create_bottom_buttons(bottom_frame)

    def _create_control_bar(self, parent):
        """Build the torque toggle, finger-copy, and zero-utility button groups."""
        # Torque toggle
        self.torque_var = tk.StringVar(value="OFF")
        torque_frame    = ttk.LabelFrame(parent, text="Torque Control", padding="5")
        torque_frame.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            torque_frame, textvariable=self.torque_var, command=self._toggle_torque
        ).pack()

        # Finger copy shortcuts (copy one finger's angles to another)
        finger_tools_frame = ttk.LabelFrame(parent, text="Finger Utilities", padding="5")
        finger_tools_frame.pack(side=tk.LEFT, padx=(0, 10))
        finger_tools = [
            ("Index → Ring",    lambda: self._copy_finger(0,  8)),
            ("Index → Middle",  lambda: self._copy_finger(0,  4)),
            ("Middle → Index",  lambda: self._copy_finger(4,  0)),
            ("Middle → Ring",   lambda: self._copy_finger(4,  8)),
            ("Ring → Index",    lambda: self._copy_finger(8,  0)),
            ("Ring → Middle",   lambda: self._copy_finger(8,  4)),
        ]
        for i, (text, cmd) in enumerate(finger_tools):
            ttk.Button(finger_tools_frame, text=text, command=cmd, width=15).grid(
                row=i // 3, column=i % 3, padx=2, pady=2
            )

        # Zero utilities
        zero_frame = ttk.LabelFrame(parent, text="Zero Utilities", padding="5")
        zero_frame.pack(side=tk.LEFT, padx=(0, 10))
        zero_buttons = [
            ("Zero All",    self._zero_all),
            ("Zero Index",  lambda: self._zero_finger(0)),
            ("Zero Middle", lambda: self._zero_finger(4)),
            ("Zero Ring",   lambda: self._zero_finger(8)),
            ("Zero Thumb",  lambda: self._zero_finger(12)),
        ]
        for i, (text, cmd) in enumerate(zero_buttons):
            ttk.Button(zero_frame, text=text, command=cmd, width=12).grid(
                row=i // 3, column=i % 3, padx=2, pady=2
            )

    def _create_horizontal_joint_controls(self, parent):
        """
        Build four labelled columns (INDEX / MIDDLE / RING / THUMB), each
        containing four JointControl widgets for its joints.
        """
        finger_names  = ["INDEX", "MIDDLE", "RING", "THUMB"]
        finger_ranges = [(0, 4), (4, 8), (8, 12), (12, 16)]
        self.joint_controls = []

        for name, (start, end) in zip(finger_names, finger_ranges):
            finger_frame = ttk.LabelFrame(parent, text=name, padding="5")
            finger_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=5)

            for i in range(start, end):
                jc = JointControl(
                    finger_frame, i, f"J{i - start}",
                    self._on_joint_value_changed
                )
                jc.frame.pack(pady=2)
                self.joint_controls.append(jc)

    def _create_pose_management(self, parent):
        """
        Build the pose metadata fields, pose list box, CRUD buttons,
        playback controls, and clipboard import/export buttons.
        """
        # Pose name and duration fields
        meta_frame = ttk.LabelFrame(parent, text="Pose Metadata", padding="5")
        meta_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        meta_frame.columnconfigure(1, weight=1)

        ttk.Label(meta_frame, text="Pose Name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.pose_name_var  = tk.StringVar()
        ttk.Entry(meta_frame, textvariable=self.pose_name_var).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5)
        )
        ttk.Label(meta_frame, text="Duration (s):").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.pose_duration_var = tk.StringVar(value="1.0")
        ttk.Entry(meta_frame, textvariable=self.pose_duration_var, width=10).grid(
            row=0, column=3, sticky=tk.W
        )

        # Scrollable pose list
        list_frame = ttk.LabelFrame(parent, text="Pose Sequence", padding="5")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)

        self.pose_listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE)
        self.pose_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.pose_listbox.bind('<<ListboxSelect>>', self._on_pose_selected)

        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.pose_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.pose_listbox.configure(yscrollcommand=scrollbar.set)

        # CRUD and reorder buttons beside the list
        button_frame = ttk.Frame(list_frame)
        button_frame.grid(row=0, column=1, padx=(10, 0))
        for i, (text, cmd) in enumerate([
            ("Save New Pose",    self._save_new_pose),
            ("Update Selected", self._update_selected_pose),
            ("Duplicate Pose",  self._duplicate_pose),
            ("Delete Pose",     self._delete_pose),
            ("Insert After",    self._insert_pose_after),
            ("Move Up",         self._move_pose_up),
            ("Move Down",       self._move_pose_down),
        ]):
            ttk.Button(button_frame, text=text, command=cmd, width=15).grid(
                row=i, column=0, pady=2, sticky=tk.W
            )

        # Playback controls
        playback_frame = ttk.LabelFrame(parent, text="Playback Controls", padding="5")
        playback_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        self.play_button  = ttk.Button(playback_frame, text="Play",  command=self._play_sequence)
        self.pause_button = ttk.Button(playback_frame, text="Pause", command=self._pause_sequence)
        self.stop_button  = ttk.Button(playback_frame, text="Stop",  command=self._stop_sequence)
        for btn in (self.play_button, self.pause_button, self.stop_button):
            btn.pack(side=tk.LEFT, padx=(0, 5))

        # Clipboard import/export
        export_frame = ttk.Frame(parent)
        export_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        ttk.Button(export_frame, text="Copy Current Pose",    command=self._copy_current_pose).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="Copy Entire Sequence", command=self._copy_entire_sequence).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="Import Sequence",      command=self._import_sequence).pack(side=tk.LEFT, padx=(0, 5))

    def _create_bottom_buttons(self, parent):
        """Save/Load session and clear-all buttons at the bottom of the window."""
        ttk.Button(parent, text="Save Session", command=self._save_session).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(parent, text="Load Session", command=self._load_session).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(parent, text="Clear All",    command=self._clear_all).pack(side=tk.LEFT, padx=(0, 5))

    # ------------------------------------------------------------------ #
    # Joint value helpers                                                  #
    # ------------------------------------------------------------------ #

    def _on_joint_value_changed(self, joint_index, value):
        """Called by any JointControl on change; pushes angles to the robot."""
        self.robot_controller.set_angles(self._get_all_joint_values())
        self.changes_pending = True

    def _get_all_joint_values(self):
        """Return a 16-element list of current slider angles (degrees)."""
        return [ctrl.get_value() for ctrl in self.joint_controls]

    def _set_all_joint_values(self, values):
        """
        Programmatically set all sliders without triggering recursive callbacks.
        Uses the updating_pose flag to suppress re-entrant on_change calls.
        """
        self.updating_pose = True
        for i, value in enumerate(values):
            if 0 <= i < len(self.joint_controls):
                self.joint_controls[i].set_value(value)
        self.updating_pose = False

    # ------------------------------------------------------------------ #
    # Control bar callbacks                                                #
    # ------------------------------------------------------------------ #

    def _toggle_torque(self):
        """Toggle motor torque on/off and update the button label."""
        new_state = self.torque_var.get() != "ON"
        self.robot_controller.set_torque_enabled(new_state)
        self.torque_var.set("ON" if new_state else "OFF")

    def _copy_finger(self, source_start, dest_start):
        """Copy all four joint values from source finger to destination finger."""
        for i in range(4):
            self.joint_controls[dest_start + i].set_value(
                self.joint_controls[source_start + i].get_value()
            )
        self._on_joint_value_changed(0, 0)

    def _zero_all(self):
        """Set all 16 joint sliders to 0°."""
        for ctrl in self.joint_controls:
            ctrl.set_value(0)
        self._on_joint_value_changed(0, 0)

    def _zero_finger(self, start_idx):
        """Set the four joint sliders of one finger to 0°."""
        for i in range(4):
            self.joint_controls[start_idx + i].set_value(0)
        self._on_joint_value_changed(0, 0)

    # ------------------------------------------------------------------ #
    # Pose list callbacks                                                  #
    # ------------------------------------------------------------------ #

    def _save_new_pose(self):
        """Capture current slider state as a new pose and append it to the list."""
        name = self.pose_name_var.get() or f"Pose {len(self.pose_manager.poses)}"
        try:
            duration = float(self.pose_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Duration", "Please enter a valid number for duration")
            return
        self.pose_manager.add_pose(Pose(name, duration, self._get_all_joint_values()))
        self._refresh_pose_list()
        self._autosave()

    def _update_selected_pose(self):
        """Overwrite the selected pose with current slider state and metadata."""
        selection = self.pose_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a pose to update")
            return
        index = selection[0]
        name  = self.pose_name_var.get() or f"Pose {index}"
        try:
            duration = float(self.pose_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Duration", "Please enter a valid number for duration")
            return
        self.pose_manager.update_pose(index, Pose(name, duration, self._get_all_joint_values()))
        self._refresh_pose_list()
        self._autosave()

    def _duplicate_pose(self):
        """Insert a copy of the selected pose immediately after it."""
        selection = self.pose_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a pose to duplicate")
            return
        self.pose_manager.duplicate_pose(selection[0])
        self._refresh_pose_list()
        self._autosave()

    def _delete_pose(self):
        """Delete the selected pose after confirmation."""
        selection = self.pose_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a pose to delete")
            return
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this pose?"):
            self.pose_manager.delete_pose(selection[0])
            self._refresh_pose_list()
            self._autosave()

    def _insert_pose_after(self):
        """Insert the current slider state as a new pose after the selected one."""
        selection = self.pose_listbox.curselection()
        index     = selection[0] if selection else -1
        name      = self.pose_name_var.get() or f"Pose {len(self.pose_manager.poses)}"
        try:
            duration = float(self.pose_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Duration", "Please enter a valid number for duration")
            return
        self.pose_manager.insert_pose_after(index, Pose(name, duration, self._get_all_joint_values()))
        self._refresh_pose_list()
        self._autosave()

    def _move_pose_up(self):
        """Move the selected pose one position earlier in the sequence."""
        selection = self.pose_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a pose to move")
            return
        index = selection[0]
        self.pose_manager.move_pose_up(index)
        self._refresh_pose_list()
        self._select_pose(index - 1)
        self._autosave()

    def _move_pose_down(self):
        """Move the selected pose one position later in the sequence."""
        selection = self.pose_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a pose to move")
            return
        index = selection[0]
        self.pose_manager.move_pose_down(index)
        self._refresh_pose_list()
        self._select_pose(index + 1)
        self._autosave()

    def _on_pose_selected(self, event):
        """
        Load the selected pose into the joint sliders and metadata fields.
        Guarded by updating_pose to prevent recursive callbacks.
        """
        if self.updating_pose:
            return
        selection = self.pose_listbox.curselection()
        if not selection:
            return
        pose = self.pose_manager.get_pose(selection[0])
        if pose:
            self.pose_name_var.set(pose.name)
            self.pose_duration_var.set(str(pose.duration))
            self._set_all_joint_values(pose.angles)
            self.pose_manager.set_current_pose_index(selection[0])

    def _refresh_pose_list(self):
        """
        Rebuild the listbox from the current PoseManager state.
        Marks the currently playing pose with a ▶ indicator.
        Restores the previous selection after rebuilding.
        """
        selection     = self.pose_listbox.curselection()
        current_index = selection[0] if selection else -1

        self.pose_listbox.delete(0, tk.END)
        for i, pose in enumerate(self.pose_manager.poses):
            marker = "▶ " if i == self.playback_controller.current_playback_index else "  "
            self.pose_listbox.insert(tk.END, f"{marker}{i}: {pose.name} ({pose.duration}s)")

        if 0 <= current_index < self.pose_listbox.size():
            self.pose_listbox.selection_set(current_index)

    def _select_pose(self, index):
        """Programmatically select and scroll to a specific pose index."""
        if 0 <= index < self.pose_listbox.size():
            self.pose_listbox.selection_clear(0, tk.END)
            self.pose_listbox.selection_set(index)
            self.pose_listbox.see(index)
            self._on_pose_selected(None)

    # ------------------------------------------------------------------ #
    # Playback callbacks                                                   #
    # ------------------------------------------------------------------ #

    def _play_sequence(self):
        if not self.pose_manager.poses:
            messagebox.showinfo("No Poses", "Please add poses to the sequence first")
            return
        self.playback_controller.play()

    def _pause_sequence(self):
        self.playback_controller.pause()

    def _stop_sequence(self):
        self.playback_controller.stop()

    # ------------------------------------------------------------------ #
    # Clipboard export / import                                            #
    # ------------------------------------------------------------------ #

    def _copy_current_pose(self):
        """
        Copy the current slider state to the clipboard as a Python dict literal
        compatible with the bottle_orient.py POSES format.
        """
        angles   = self._get_all_joint_values()
        pose_str = (
            f"dict(\n"
            f"    duration=1,\n"
            f"    index={angles[0:4]},\n"
            f"    middle={angles[4:8]},\n"
            f"    ring={angles[8:12]},\n"
            f"    thumb={angles[12:16]},\n"
            f"),"
        )
        self.root.clipboard_clear()
        self.root.clipboard_append(pose_str)
        messagebox.showinfo("Copied", "Current pose copied to clipboard!")

    def _copy_entire_sequence(self):
        """
        Copy the entire pose sequence to the clipboard as a Python POSES list
        ready to paste into a scripted sequence file.
        """
        if not self.pose_manager.poses:
            messagebox.showinfo("No Poses", "No poses to export")
            return

        poses_str = "POSES = [\n\n"
        for pose in self.pose_manager.poses:
            fd = pose.to_finger_dict()
            poses_str += (
                f"    dict(\n"
                f"        duration={fd['duration']},\n"
                f"        index={fd['index']},\n"
                f"        middle={fd['middle']},\n"
                f"        ring={fd['ring']},\n"
                f"        thumb={fd['thumb']},\n"
                f"    ),\n\n"
            )
        poses_str += "]\n"

        self.root.clipboard_clear()
        self.root.clipboard_append(poses_str)
        messagebox.showinfo("Copied", "Entire sequence copied to clipboard!")

    def _import_sequence(self):
        """
        Open a dialog where the user can paste a POSES list and import it
        into the current session, replacing all existing poses.
        """
        import_window = tk.Toplevel(self.root)
        import_window.title("Import Sequence")
        import_window.geometry("600x400")

        text_area    = scrolledtext.ScrolledText(import_window, wrap=tk.WORD)
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        button_frame = ttk.Frame(import_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        def do_import():
            content = text_area.get("1.0", tk.END)
            try:
                if "POSES = [" in content:
                    start = content.find("[")
                    end   = content.rfind("]") + 1
                    if start != -1 and end > start:
                        array_str = content[start:end]
                        array_str = array_str.replace("dict(", "{'__type__': 'dict', ").replace(")", "}")
                        array_str = array_str.replace("{'__type__': 'dict', ", "dict(").replace("}", ")")
                        array_str = array_str.replace("np.array(", "").replace(")", "")
                        array_str = (
                            array_str
                            .replace("duration=",  "'duration':")
                            .replace("index=",     "'index':")
                            .replace("middle=",    "'middle':")
                            .replace("ring=",      "'ring':")
                            .replace("thumb=",     "'thumb':")
                        )

                        poses_data = []
                        parts = array_str.split("dict(")
                        for i, part in enumerate(parts):
                            if i == 0:
                                continue
                            paren_count = 1
                            content_end = -1
                            for j, char in enumerate(part):
                                if char == '(':
                                    paren_count += 1
                                elif char == ')':
                                    paren_count -= 1
                                    if paren_count == 0:
                                        content_end = j
                                        break
                            if content_end > 0:
                                dict_content = part[:content_end]
                                dict_content = (
                                    dict_content
                                    .replace("'duration':", "duration=")
                                    .replace("'index':",    "index=")
                                    .replace("'middle':",   "middle=")
                                    .replace("'ring':",     "ring=")
                                    .replace("'thumb':",    "thumb=")
                                )
                                dict_content = "dict(" + dict_content + ")"
                                try:
                                    pose_dict = eval(dict_content, {"__builtins__": {}}, {})
                                    poses_data.append(pose_dict)
                                except Exception:
                                    # Manual key-value fallback parser
                                    pose_dict = {}
                                    for item in dict_content.split(","):
                                        if "duration=" in item:
                                            pose_dict["duration"] = float(item.split("=")[1])
                                        elif "index=" in item:
                                            pose_dict["index"] = [float(v.strip()) for v in item.split("=")[1].strip()[1:-1].split(",") if v.strip()]
                                        elif "middle=" in item:
                                            pose_dict["middle"] = [float(v.strip()) for v in item.split("=")[1].strip()[1:-1].split(",") if v.strip()]
                                        elif "ring=" in item:
                                            pose_dict["ring"] = [float(v.strip()) for v in item.split("=")[1].strip()[1:-1].split(",") if v.strip()]
                                        elif "thumb=" in item:
                                            pose_dict["thumb"] = [float(v.strip()) for v in item.split("=")[1].strip()[1:-1].split(",") if v.strip()]
                                    poses_data.append(pose_dict)

                        self.pose_manager.poses = []
                        for item in poses_data:
                            pose      = Pose.from_finger_dict(item)
                            pose.name = item.get("name", f"Imported Pose {len(self.pose_manager.poses)}")
                            self.pose_manager.add_pose(pose)

                        self._refresh_pose_list()
                        self._autosave()
                        messagebox.showinfo("Success", f"Imported {len(self.pose_manager.poses)} poses")
                        import_window.destroy()
                    else:
                        raise ValueError("Could not parse POSES array")
                else:
                    raise ValueError("No POSES array found in input")
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import sequence:\n{str(e)}")

        ttk.Button(button_frame, text="Import", command=do_import).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=import_window.destroy).pack(side=tk.LEFT)

    # ------------------------------------------------------------------ #
    # Session persistence                                                  #
    # ------------------------------------------------------------------ #

    def _save_session(self):
        """Save the pose sequence to pose_session.json."""
        try:
            self.pose_manager.save_to_file("pose_session.json")
            messagebox.showinfo("Saved", "Session saved to pose_session.json")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save session:\n{str(e)}")

    def _load_session(self):
        """Load a previously saved session from pose_session.json."""
        if os.path.exists("pose_session.json"):
            try:
                if self.pose_manager.load_from_file("pose_session.json"):
                    self._refresh_pose_list()
                    messagebox.showinfo("Loaded", "Session loaded from pose_session.json")
                else:
                    raise Exception("Failed to load file")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load session:\n{str(e)}")
        else:
            messagebox.showinfo("Not Found", "No session file pose_session.json found")

    def _clear_all(self):
        """Remove all poses from the sequence after confirmation."""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all poses?"):
            self.pose_manager.poses              = []
            self.pose_manager.current_pose_index = -1
            self._refresh_pose_list()
            self._autosave()

    # ------------------------------------------------------------------ #
    # Autosave                                                             #
    # ------------------------------------------------------------------ #

    def _check_autosave(self):
        """
        On startup, offer to restore the last autosaved session if the
        autosave file exists.
        """
        autosave_file = "pose_editor_autosave.json"
        if os.path.exists(autosave_file):
            if messagebox.askyesno("Restore Session", "Found autosave file. Restore previous session?"):
                if self.pose_manager.load_from_file(autosave_file):
                    self._refresh_pose_list()
                    messagebox.showinfo("Restored", "Previous session restored")
                else:
                    messagebox.showerror("Restore Error", "Failed to restore session")

    def _autosave(self):
        """Write the current session to the autosave file silently."""
        try:
            self.pose_manager.save_to_file("pose_editor_autosave.json")
            self.changes_pending = False
        except Exception as e:
            print(f"Autosave failed: {e}")

    # ------------------------------------------------------------------ #
    # Live display update loop                                             #
    # ------------------------------------------------------------------ #

    def _start_update_loop(self):
        """
        Start a background thread that refreshes the 'Current Pose' text box
        and the pose list (for playback marker) at 10 Hz.
        """
        def update_loop():
            while True:
                angles      = self.robot_controller.get_current_angles()
                display_text = (
                    f"index  = {angles[0:4]}\n"
                    f"middle = {angles[4:8]}\n"
                    f"ring   = {angles[8:12]}\n"
                    f"thumb  = {angles[12:16]}\n"
                )
                self.current_pose_text.config(state=tk.NORMAL)
                self.current_pose_text.delete(1.0, tk.END)
                self.current_pose_text.insert(tk.END, display_text)
                self.current_pose_text.config(state=tk.DISABLED)

                self._refresh_pose_list()

                if self.changes_pending:
                    self._autosave()

                time.sleep(0.1)

        threading.Thread(target=update_loop, daemon=True).start()

    # ------------------------------------------------------------------ #
    # Window close                                                         #
    # ------------------------------------------------------------------ #

    def on_closing(self):
        """Clean up threads and autosave before destroying the window."""
        self.robot_controller.stop_update_thread()
        self.playback_controller.stop()
        self._autosave()
        self.root.destroy()


def main():
    root = tk.Tk()
    app  = PoseEditorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()