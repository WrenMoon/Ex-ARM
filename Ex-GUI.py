# leap_hand_pose_sequencer.py
import json
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import os

try:
    from utils.ExARM import ExArm
except ImportError as e:
    print(f"Import error: {e}")
    # Create a minimal mock for testing
    class ExArm:
        def __init__(self, **kwargs):
            print("Mock ExArm initialized with:", kwargs)
            
        def set_goal_positions_degree(self, positions):
            print(f"Mock: Setting positions to {positions}")
            
        def set_torque_enabled(self, enabled):
            print(f"Mock: Torque {'enabled' if enabled else 'disabled'}")

class Pose:
    """Represents a single hand pose with name, duration, and joint angles"""
    def __init__(self, name="", duration=1.0, angles=None):
        self.name = name
        self.duration = duration
        self.angles = angles or [0.0] * 16
        
    def to_dict(self):
        return {
            "name": self.name,
            "duration": self.duration,
            "angles": self.angles
        }
        
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["duration"], data["angles"])

    @classmethod
    def from_finger_dict(cls, data):
        """Create a pose from finger-grouped dictionary"""
        angles = []
        angles.extend(data["index"])
        angles.extend(data["middle"])
        angles.extend(data["ring"])
        angles.extend(data["thumb"])
        return cls(data.get("name", ""), data["duration"], angles)

    def to_finger_dict(self):
        """Convert to finger-grouped dictionary"""
        return {
            "name": self.name,
            "duration": self.duration,
            "index": self.angles[0:4],
            "middle": self.angles[4:8],
            "ring": self.angles[8:12],
            "thumb": self.angles[12:16]
        }

class PoseManager:
    """Manages the collection of poses and provides serialization methods"""
    def __init__(self):
        self.poses = []
        self.current_pose_index = -1
        
    def add_pose(self, pose):
        self.poses.append(pose)
        self.current_pose_index = len(self.poses) - 1
        
    def update_pose(self, index, pose):
        if 0 <= index < len(self.poses):
            self.poses[index] = pose
            
    def delete_pose(self, index):
        if 0 <= index < len(self.poses):
            del self.poses[index]
            if self.current_pose_index >= len(self.poses):
                self.current_pose_index = len(self.poses) - 1
                
    def move_pose_up(self, index):
        if 1 <= index < len(self.poses):
            self.poses[index], self.poses[index-1] = self.poses[index-1], self.poses[index]
            if self.current_pose_index == index:
                self.current_pose_index = index - 1
            elif self.current_pose_index == index - 1:
                self.current_pose_index = index
                
    def move_pose_down(self, index):
        if 0 <= index < len(self.poses) - 1:
            self.poses[index], self.poses[index+1] = self.poses[index+1], self.poses[index]
            if self.current_pose_index == index:
                self.current_pose_index = index + 1
            elif self.current_pose_index == index + 1:
                self.current_pose_index = index
                
    def duplicate_pose(self, index):
        if 0 <= index < len(self.poses):
            new_pose = Pose(
                name=f"{self.poses[index].name} Copy",
                duration=self.poses[index].duration,
                angles=self.poses[index].angles.copy()
            )
            self.poses.insert(index + 1, new_pose)
            self.current_pose_index = index + 1
            
    def insert_pose_after(self, index, pose):
        if 0 <= index < len(self.poses):
            self.poses.insert(index + 1, pose)
            self.current_pose_index = index + 1
        else:
            self.poses.append(pose)
            self.current_pose_index = len(self.poses) - 1
            
    def get_pose(self, index):
        if 0 <= index < len(self.poses):
            return self.poses[index]
        return None
        
    def get_current_pose(self):
        return self.get_pose(self.current_pose_index)
        
    def set_current_pose_index(self, index):
        if -1 <= index < len(self.poses):
            self.current_pose_index = index
            
    def to_list(self):
        return [pose.to_dict() for pose in self.poses]
        
    def from_list(self, data):
        self.poses = [Pose.from_dict(item) for item in data]
        self.current_pose_index = -1
        
    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_list(), f, indent=2)
            
    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.from_list(data)
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False

class RobotController:
    """Manages communication with the LEAP hand robot"""
    def __init__(self):
        self.robot = None
        self.running = False
        self.torque_enabled = False
        self.current_angles = [0.0] * 16
        self.thread = None
        self.error = None
        
    def connect(self):
        try:
            self.robot = ExArm(
                mode="both",
                ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                port="COM5",
                baudrate=4000000,
                offsets=[0, -90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                model_path="Data/mujoco_robot.urdf"
            )
            return True
        except Exception as e:
            self.error = str(e)
            return False
            
    def start_update_thread(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            
    def stop_update_thread(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _update_loop(self):
        while self.running:
            try:
                if self.robot and self.torque_enabled:
                    # Ensure we have exactly 16 angles
                    if len(self.current_angles) != 16:
                        # Pad or trim to exactly 16 values
                        if len(self.current_angles) > 16:
                            angles_array = np.array(self.current_angles[:16])
                        else:
                            angles_array = np.array(self.current_angles + [0.0] * (16 - len(self.current_angles)))
                    else:
                        angles_array = np.array(self.current_angles)
                    self.robot.set_goal_positions_degree(angles_array)
                time.sleep(1/30)  # ~30 Hz
            except Exception as e:
                print(f"Error in update loop: {e}")
                self.error = str(e)
                
    def set_angles(self, angles):
        # Ensure we have exactly 16 angles
        if len(angles) > 16:
            self.current_angles = angles[:16]
        elif len(angles) < 16:
            self.current_angles = list(angles) + [0.0] * (16 - len(angles))
        else:
            self.current_angles = list(angles)
        
    def set_torque_enabled(self, enabled):
        self.torque_enabled = enabled
        if self.robot:
            try:
                self.robot.set_torque_enabled(enabled)
            except Exception as e:
                self.error = str(e)
                
    def get_current_angles(self):
        # Ensure we return exactly 16 angles
        if len(self.current_angles) > 16:
            return self.current_angles[:16]
        elif len(self.current_angles) < 16:
            return self.current_angles + [0.0] * (16 - len(self.current_angles))
        else:
            return self.current_angles.copy()

class PlaybackController:
    """Controls playback of pose sequences"""
    def __init__(self, pose_manager, robot_controller):
        self.pose_manager = pose_manager
        self.robot_controller = robot_controller
        self.playback_thread = None
        self.playback_running = False
        self.paused = False
        self.current_playback_index = -1
        
    def play(self):
        if not self.playback_running:
            self.playback_running = True
            self.paused = False
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
            
    def pause(self):
        self.paused = not self.paused
        
    def stop(self):
        self.playback_running = False
        self.paused = False
        if self.playback_thread:
            self.playback_thread.join()
            
    def _playback_loop(self):
        i = 0
        poses = self.pose_manager.poses
        while self.playback_running and i < len(poses):
            if self.paused:
                time.sleep(0.1)
                continue
                
            pose = poses[i]
            self.current_playback_index = i
            
            # Update robot
            self.robot_controller.set_angles(pose.angles)
            
            # Wait for duration
            start_time = time.time()
            while time.time() - start_time < pose.duration and self.playback_running and not self.paused:
                time.sleep(0.01)
                
            if not self.playback_running:
                break
                
            i += 1
            
        self.playback_running = False
        self.current_playback_index = -1

class JointControl:
    """Represents a single joint control widget"""
    def __init__(self, parent, joint_index, label, on_change_callback):
        self.joint_index = joint_index
        self.on_change_callback = on_change_callback
        self.frame = ttk.Frame(parent)
        
        # Create widgets
        self.minus_five_btn = ttk.Button(self.frame, text="-5", width=3, command=self._decrease_five)
        self.minus_one_btn = ttk.Button(self.frame, text="-1", width=3, command=self._decrease_one)
        self.slider = ttk.Scale(self.frame, from_=-180, to=180, orient=tk.HORIZONTAL, 
                               command=self._slider_changed, length=120)
        self.plus_one_btn = ttk.Button(self.frame, text="+1", width=3, command=self._increase_one)
        self.plus_five_btn = ttk.Button(self.frame, text="+5", width=3, command=self._increase_five)
        self.value_var = tk.StringVar(value="0")
        self.entry = ttk.Entry(self.frame, textvariable=self.value_var, width=6)
        self.entry.bind('<Return>', self._entry_changed)
        self.label = ttk.Label(self.frame, text=label)
        
        # Layout
        self.label.grid(row=0, column=0, columnspan=6, pady=(0, 2))
        self.minus_five_btn.grid(row=1, column=0, padx=(0, 2))
        self.minus_one_btn.grid(row=1, column=1, padx=(0, 2))
        self.slider.grid(row=1, column=2, padx=(0, 2))
        self.plus_one_btn.grid(row=1, column=3, padx=(0, 2))
        self.plus_five_btn.grid(row=1, column=4, padx=(0, 2))
        self.entry.grid(row=1, column=5, padx=(0, 2))
        
        # Initialize
        self.set_value(0)
        
    def _decrease_five(self):
        current = float(self.value_var.get())
        new_value = max(-180, current - 5)
        self.set_value(new_value)
        self.on_change_callback(self.joint_index, new_value)
        
    def _decrease_one(self):
        current = float(self.value_var.get())
        new_value = max(-180, current - 1)
        self.set_value(new_value)
        self.on_change_callback(self.joint_index, new_value)
        
    def _increase_one(self):
        current = float(self.value_var.get())
        new_value = min(180, current + 1)
        self.set_value(new_value)
        self.on_change_callback(self.joint_index, new_value)
        
    def _increase_five(self):
        current = float(self.value_var.get())
        new_value = min(180, current + 5)
        self.set_value(new_value)
        self.on_change_callback(self.joint_index, new_value)
        
    def _slider_changed(self, value):
        new_value = float(value)
        self.value_var.set(f"{new_value:.1f}")
        self.on_change_callback(self.joint_index, new_value)
        
    def _entry_changed(self, event):
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
        self.slider.set(value)
        self.value_var.set(f"{value:.1f}")
        
    def get_value(self):
        return float(self.value_var.get())

class PoseEditorGUI:
    """Main GUI application for the LEAP Hand Pose Sequencer"""
    def __init__(self, root):
        self.root = root
        self.root.title("LEAP Hand Pose Sequencer: Ex-GUI")
        self.root.geometry("1600x900")  # Wider window for horizontal layout
        
        # Initialize core components
        self.pose_manager = PoseManager()
        self.robot_controller = RobotController()
        self.playback_controller = PlaybackController(self.pose_manager, self.robot_controller)
        
        # Connect to robot
        if not self.robot_controller.connect():
            messagebox.showwarning("Connection Warning", 
                                 f"Failed to connect to robot: {self.robot_controller.error}\nUsing mock mode.")
        
        # Start robot update thread
        self.robot_controller.start_update_thread()
        
        # Create GUI
        self._create_widgets()
        
        # Load autosave if exists
        self._check_autosave()
        
        # Track changes for autosave
        self.changes_pending = False
        
        # Flag to prevent recursive calls
        self.updating_pose = False
        
        # Start update loop for current pose display
        self._start_update_loop()
        
    def _create_widgets(self):
        # Main layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)  # Pose management row gets weight
        
        # Title
        title_label = ttk.Label(main_frame, text="LEAP Hand Pose Sequencer: Ex-GUI", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Top control bar
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self._create_control_bar(control_frame)
        
        # Joint controls - arranged horizontally
        joint_frame = ttk.Frame(main_frame)
        joint_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self._create_horizontal_joint_controls(joint_frame)
        
        # Current pose display
        pose_display_frame = ttk.LabelFrame(main_frame, text="Current Pose", padding="5")
        pose_display_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.current_pose_text = tk.Text(pose_display_frame, height=4, width=80, state=tk.DISABLED)
        self.current_pose_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Pose management
        pose_mgmt_frame = ttk.Frame(main_frame)
        pose_mgmt_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        pose_mgmt_frame.columnconfigure(0, weight=1)
        pose_mgmt_frame.rowconfigure(1, weight=1)
        self._create_pose_management(pose_mgmt_frame)
        
        # Bottom buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        self._create_bottom_buttons(bottom_frame)
        
    def _create_control_bar(self, parent):
        # Torque control
        self.torque_var = tk.StringVar(value="OFF")
        torque_frame = ttk.LabelFrame(parent, text="Torque Control", padding="5")
        torque_frame.pack(side=tk.LEFT, padx=(0, 10))
        self.torque_button = ttk.Button(torque_frame, textvariable=self.torque_var, 
                                       command=self._toggle_torque)
        self.torque_button.pack()
        
        # Finger utility tools
        finger_tools_frame = ttk.LabelFrame(parent, text="Finger Utilities", padding="5")
        finger_tools_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        finger_tools = [
            ("Index → Ring", lambda: self._copy_finger(0, 8)),
            ("Index → Middle", lambda: self._copy_finger(0, 4)),
            ("Middle → Index", lambda: self._copy_finger(4, 0)),
            ("Middle → Ring", lambda: self._copy_finger(4, 8)),
            ("Ring → Index", lambda: self._copy_finger(8, 0)),
            ("Ring → Middle", lambda: self._copy_finger(8, 4))
        ]
        
        for i, (text, command) in enumerate(finger_tools):
            btn = ttk.Button(finger_tools_frame, text=text, command=command, width=15)
            btn.grid(row=i//3, column=i%3, padx=2, pady=2)
            
        # Zero utilities
        zero_frame = ttk.LabelFrame(parent, text="Zero Utilities", padding="5")
        zero_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        zero_buttons = [
            ("Zero All", self._zero_all),
            ("Zero Index", lambda: self._zero_finger(0)),
            ("Zero Middle", lambda: self._zero_finger(4)),
            ("Zero Ring", lambda: self._zero_finger(8)),
            ("Zero Thumb", lambda: self._zero_finger(12))
        ]
        
        for i, (text, command) in enumerate(zero_buttons):
            btn = ttk.Button(zero_frame, text=text, command=command, width=12)
            btn.grid(row=i//3, column=i%3, padx=2, pady=2)
            
    def _create_horizontal_joint_controls(self, parent):
        # Create a frame for each finger
        finger_names = ["INDEX", "MIDDLE", "RING", "THUMB"]
        finger_ranges = [(0, 4), (4, 8), (8, 12), (12, 16)]
        
        self.joint_controls = []
        
        for finger_idx, (name, (start, end)) in enumerate(zip(finger_names, finger_ranges)):
            finger_frame = ttk.LabelFrame(parent, text=name, padding="5")
            finger_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=5)
            
            # Create joint controls for this finger
            joint_frame = ttk.Frame(finger_frame)
            joint_frame.pack()
            
            for i in range(start, end):
                joint_label = f"J{i-start}"
                joint_control = JointControl(
                    joint_frame, 
                    i, 
                    joint_label, 
                    self._on_joint_value_changed
                )
                joint_control.frame.pack(pady=2)
                self.joint_controls.append(joint_control)
                
    def _create_pose_management(self, parent):
        # Pose metadata
        meta_frame = ttk.LabelFrame(parent, text="Pose Metadata", padding="5")
        meta_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        meta_frame.columnconfigure(1, weight=1)
        
        ttk.Label(meta_frame, text="Pose Name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.pose_name_var = tk.StringVar()
        self.pose_name_entry = ttk.Entry(meta_frame, textvariable=self.pose_name_var)
        self.pose_name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Label(meta_frame, text="Duration (s):").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.pose_duration_var = tk.StringVar(value="1.0")
        self.pose_duration_entry = ttk.Entry(meta_frame, textvariable=self.pose_duration_var, width=10)
        self.pose_duration_entry.grid(row=0, column=3, sticky=tk.W)
        
        # Pose list
        list_frame = ttk.LabelFrame(parent, text="Pose Sequence", padding="5")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Listbox with scrollbar
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
        
        # Pose editing buttons
        button_frame = ttk.Frame(list_frame)
        button_frame.grid(row=0, column=1, padx=(10, 0))
        
        pose_buttons = [
            ("Save New Pose", self._save_new_pose),
            ("Update Selected", self._update_selected_pose),
            ("Duplicate Pose", self._duplicate_pose),
            ("Delete Pose", self._delete_pose),
            ("Insert After", self._insert_pose_after),
            ("Move Up", self._move_pose_up),
            ("Move Down", self._move_pose_down)
        ]
        
        for i, (text, command) in enumerate(pose_buttons):
            btn = ttk.Button(button_frame, text=text, command=command, width=15)
            btn.grid(row=i, column=0, pady=2, sticky=tk.W)
            
        # Playback controls
        playback_frame = ttk.LabelFrame(parent, text="Playback Controls", padding="5")
        playback_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.play_button = ttk.Button(playback_frame, text="Play", command=self._play_sequence)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pause_button = ttk.Button(playback_frame, text="Pause", command=self._pause_sequence)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(playback_frame, text="Stop", command=self._stop_sequence)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Export/Import buttons
        export_frame = ttk.Frame(parent)
        export_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(export_frame, text="Copy Current Pose", command=self._copy_current_pose).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="Copy Entire Sequence", command=self._copy_entire_sequence).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_frame, text="Import Sequence", command=self._import_sequence).pack(side=tk.LEFT, padx=(0, 5))
        
    def _create_bottom_buttons(self, parent):
        ttk.Button(parent, text="Save Session", command=self._save_session).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(parent, text="Load Session", command=self._load_session).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(parent, text="Clear All", command=self._clear_all).pack(side=tk.LEFT, padx=(0, 5))
        
    def _on_joint_value_changed(self, joint_index, value):
        # Update robot controller
        angles = self._get_all_joint_values()
        self.robot_controller.set_angles(angles)
        self.changes_pending = True
        
    def _get_all_joint_values(self):
        return [control.get_value() for control in self.joint_controls]
        
    def _set_all_joint_values(self, values):
        # Prevent recursive calls when updating pose
        self.updating_pose = True
        for i, value in enumerate(values):
            if 0 <= i < len(self.joint_controls):
                self.joint_controls[i].set_value(value)
        self.updating_pose = False
                
    def _toggle_torque(self):
        current_state = self.torque_var.get() == "ON"
        new_state = not current_state
        self.robot_controller.set_torque_enabled(new_state)
        self.torque_var.set("ON" if new_state else "OFF")
        
    def _copy_finger(self, source_start, dest_start):
        for i in range(4):
            source_value = self.joint_controls[source_start + i].get_value()
            self.joint_controls[dest_start + i].set_value(source_value)
        self._on_joint_value_changed(0, 0)  # Trigger update
        
    def _zero_all(self):
        for control in self.joint_controls:
            control.set_value(0)
        self._on_joint_value_changed(0, 0)  # Trigger update
        
    def _zero_finger(self, start_idx):
        for i in range(4):
            self.joint_controls[start_idx + i].set_value(0)
        self._on_joint_value_changed(0, 0)  # Trigger update
        
    def _save_new_pose(self):
        name = self.pose_name_var.get() or f"Pose {len(self.pose_manager.poses)}"
        try:
            duration = float(self.pose_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Duration", "Please enter a valid number for duration")
            return
            
        angles = self._get_all_joint_values()
        pose = Pose(name, duration, angles)
        self.pose_manager.add_pose(pose)
        self._refresh_pose_list()
        self._autosave()
        
    def _update_selected_pose(self):
        selection = self.pose_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a pose to update")
            return
            
        index = selection[0]
        name = self.pose_name_var.get() or f"Pose {index}"
        try:
            duration = float(self.pose_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Duration", "Please enter a valid number for duration")
            return
            
        angles = self._get_all_joint_values()
        pose = Pose(name, duration, angles)
        self.pose_manager.update_pose(index, pose)
        self._refresh_pose_list()
        self._autosave()
        
    def _duplicate_pose(self):
        selection = self.pose_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a pose to duplicate")
            return
            
        index = selection[0]
        self.pose_manager.duplicate_pose(index)
        self._refresh_pose_list()
        self._autosave()
        
    def _delete_pose(self):
        selection = self.pose_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a pose to delete")
            return
            
        index = selection[0]
        result = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this pose?")
        if result:
            self.pose_manager.delete_pose(index)
            self._refresh_pose_list()
            self._autosave()
            
    def _insert_pose_after(self):
        selection = self.pose_listbox.curselection()
        index = selection[0] if selection else -1
        
        name = self.pose_name_var.get() or f"Pose {len(self.pose_manager.poses)}"
        try:
            duration = float(self.pose_duration_var.get())
        except ValueError:
            messagebox.showerror("Invalid Duration", "Please enter a valid number for duration")
            return
            
        angles = self._get_all_joint_values()
        pose = Pose(name, duration, angles)
        self.pose_manager.insert_pose_after(index, pose)
        self._refresh_pose_list()
        self._autosave()
        
    def _move_pose_up(self):
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
        # Prevent recursive calls when updating pose
        if self.updating_pose:
            return
            
        selection = self.pose_listbox.curselection()
        if not selection:
            return
            
        index = selection[0]
        pose = self.pose_manager.get_pose(index)
        if pose:
            self.pose_name_var.set(pose.name)
            self.pose_duration_var.set(str(pose.duration))
            self._set_all_joint_values(pose.angles)
            self.pose_manager.set_current_pose_index(index)
            
    def _refresh_pose_list(self):
        # Save current selection
        selection = self.pose_listbox.curselection()
        current_index = selection[0] if selection else -1
        
        self.pose_listbox.delete(0, tk.END)
        for i, pose in enumerate(self.pose_manager.poses):
            marker = "▶ " if i == self.playback_controller.current_playback_index else "  "
            self.pose_listbox.insert(tk.END, f"{marker}{i}: {pose.name} ({pose.duration}s)")
            
        # Restore selection if possible
        if 0 <= current_index < self.pose_listbox.size():
            self.pose_listbox.selection_set(current_index)
            
    def _select_pose(self, index):
        if 0 <= index < self.pose_listbox.size():
            self.pose_listbox.selection_clear(0, tk.END)
            self.pose_listbox.selection_set(index)
            self.pose_listbox.see(index)
            # Manually trigger the selection event since we're not clicking
            self._on_pose_selected(None)
            
    def _play_sequence(self):
        if not self.pose_manager.poses:
            messagebox.showinfo("No Poses", "Please add poses to the sequence first")
            return
            
        self.playback_controller.play()
        
    def _pause_sequence(self):
        self.playback_controller.pause()
        
    def _stop_sequence(self):
        self.playback_controller.stop()
        
    def _copy_current_pose(self):
        angles = self._get_all_joint_values()
        # Format as dictionary with finger groupings
        pose_dict = {
            "duration": 1,
            "index": angles[0:4],
            "middle": angles[4:8],
            "ring": angles[8:12],
            "thumb": angles[12:16]
        }
        
        # Convert to string representation
        pose_str = f"""dict(
    duration={pose_dict['duration']},
    index={pose_dict['index']},
    middle={pose_dict['middle']},
    ring={pose_dict['ring']},
    thumb={pose_dict['thumb']},
),"""
        
        self.root.clipboard_clear()
        self.root.clipboard_append(pose_str)
        messagebox.showinfo("Copied", "Current pose copied to clipboard!")
        
    def _copy_entire_sequence(self):
        if not self.pose_manager.poses:
            messagebox.showinfo("No Poses", "No poses to export")
            return
            
        # Format all poses
        poses_str = "POSES = [\n\n"
        for pose in self.pose_manager.poses:
            finger_dict = pose.to_finger_dict()
            poses_str += f"""    dict(
        duration={finger_dict['duration']},
        index={finger_dict['index']},
        middle={finger_dict['middle']},
        ring={finger_dict['ring']},
        thumb={finger_dict['thumb']},
    ),

"""
        poses_str += "]\n"
        
        self.root.clipboard_clear()
        self.root.clipboard_append(poses_str)
        messagebox.showinfo("Copied", "Entire sequence copied to clipboard!")
        
    def _import_sequence(self):
        # Create import dialog
        import_window = tk.Toplevel(self.root)
        import_window.title("Import Sequence")
        import_window.geometry("600x400")
        
        # Text area for pasting
        text_area = scrolledtext.ScrolledText(import_window, wrap=tk.WORD)
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(import_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        def do_import():
            content = text_area.get("1.0", tk.END)
            try:
                # Extract the POSES array
                if "POSES = [" in content:
                    # Extract the array portion
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start != -1 and end > start:
                        array_str = content[start:end]
                        # Replace dict with proper dictionary constructor
                        array_str = array_str.replace("dict(", "{'__type__': 'dict', ").replace(")", "}")
                        # Fix the string to be valid Python
                        array_str = array_str.replace("{'__type__': 'dict', ", "dict(").replace("}", ")")
                        # Handle numpy arrays if any
                        array_str = array_str.replace("np.array(", "").replace(")", "")
                        # Add quotes around keys
                        array_str = array_str.replace("duration=", "'duration':").replace("index=", "'index':").replace("middle=", "'middle':").replace("ring=", "'ring':").replace("thumb=", "'thumb':")
                        
                        # Parse the content manually to handle the dict format
                        poses_data = []
                        # Find all dict sections
                        parts = array_str.split("dict(")
                        for i, part in enumerate(parts):
                            if i == 0:  # Skip first part as it's before first dict
                                continue
                            # Extract content until closing parenthesis
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
                                # Convert to valid Python dict
                                dict_content = dict_content.replace("'duration':", "duration=").replace("'index':", "index=").replace("'middle':", "middle=").replace("'ring':", "ring=").replace("'thumb':", "thumb=")
                                dict_content = "dict(" + dict_content + ")"
                                # Evaluate carefully
                                try:
                                    # Replace list representations
                                    exec_globals = {"__builtins__": {}}
                                    pose_dict = eval(dict_content, exec_globals, {})
                                    poses_data.append(pose_dict)
                                except:
                                    # Manual parsing as fallback
                                    pose_dict = {}
                                    items = dict_content.split(",")
                                    for item in items:
                                        if "duration=" in item:
                                            pose_dict["duration"] = float(item.split("=")[1])
                                        elif "index=" in item:
                                            vals = item.split("=")[1].strip()[1:-1].split(",")
                                            pose_dict["index"] = [float(v.strip()) for v in vals if v.strip()]
                                        elif "middle=" in item:
                                            vals = item.split("=")[1].strip()[1:-1].split(",")
                                            pose_dict["middle"] = [float(v.strip()) for v in vals if v.strip()]
                                        elif "ring=" in item:
                                            vals = item.split("=")[1].strip()[1:-1].split(",")
                                            pose_dict["ring"] = [float(v.strip()) for v in vals if v.strip()]
                                        elif "thumb=" in item:
                                            vals = item.split("=")[1].strip()[1:-1].split(",")
                                            pose_dict["thumb"] = [float(v.strip()) for v in vals if v.strip()]
                                    poses_data.append(pose_dict)
                        
                        # Convert to our pose format
                        self.pose_manager.poses = []
                        for item in poses_data:
                            pose = Pose.from_finger_dict(item)
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
                
        def cancel_import():
            import_window.destroy()
            
        ttk.Button(button_frame, text="Import", command=do_import).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=cancel_import).pack(side=tk.LEFT)
        
    def _save_session(self):
        filename = "pose_session.json"
        try:
            self.pose_manager.save_to_file(filename)
            messagebox.showinfo("Saved", f"Session saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save session:\n{str(e)}")
            
    def _load_session(self):
        filename = "pose_session.json"
        if os.path.exists(filename):
            try:
                if self.pose_manager.load_from_file(filename):
                    self._refresh_pose_list()
                    messagebox.showinfo("Loaded", f"Session loaded from {filename}")
                else:
                    raise Exception("Failed to load file")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load session:\n{str(e)}")
        else:
            messagebox.showinfo("Not Found", f"No session file {filename} found")
            
    def _clear_all(self):
        result = messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all poses?")
        if result:
            self.pose_manager.poses = []
            self.pose_manager.current_pose_index = -1
            self._refresh_pose_list()
            self._autosave()
            
    def _check_autosave(self):
        autosave_file = "pose_editor_autosave.json"
        if os.path.exists(autosave_file):
            result = messagebox.askyesno("Restore Session", 
                                       "Found autosave file. Restore previous session?")
            if result:
                if self.pose_manager.load_from_file(autosave_file):
                    self._refresh_pose_list()
                    messagebox.showinfo("Restored", "Previous session restored")
                else:
                    messagebox.showerror("Restore Error", "Failed to restore session")
                    
    def _autosave(self):
        try:
            self.pose_manager.save_to_file("pose_editor_autosave.json")
            self.changes_pending = False
        except Exception as e:
            print(f"Autosave failed: {e}")
            
    def _start_update_loop(self):
        def update_loop():
            while True:
                # Update current pose display
                angles = self.robot_controller.get_current_angles()
                index_vals = angles[0:4]
                middle_vals = angles[4:8]
                ring_vals = angles[8:12]
                thumb_vals = angles[12:16]
                
                display_text = f"index  = {index_vals}\n"
                display_text += f"middle = {middle_vals}\n"
                display_text += f"ring   = {ring_vals}\n"
                display_text += f"thumb  = {thumb_vals}\n"
                
                self.current_pose_text.config(state=tk.NORMAL)
                self.current_pose_text.delete(1.0, tk.END)
                self.current_pose_text.insert(tk.END, display_text)
                self.current_pose_text.config(state=tk.DISABLED)
                
                # Refresh pose list to show playback position
                self._refresh_pose_list()
                
                # Handle autosave
                if self.changes_pending:
                    self._autosave()
                    
                time.sleep(0.1)  # Update 10 times per second
                
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        
    def on_closing(self):
        self.robot_controller.stop_update_thread()
        self.playback_controller.stop()
        self._autosave()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = PoseEditorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()