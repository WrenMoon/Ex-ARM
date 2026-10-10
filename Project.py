"""Open the existing Ex-ARM programs from one terminal menu."""

from pathlib import Path
import platform
import shutil
import subprocess
import sys

from utils.Constants import Connection


PROJECT_FOLDER = Path(__file__).resolve().parent


def choose(title, options):
    print(f"\n{title}")
    for number, label in options:
        print(f"{number}. {label}")
    return input("Choose: ").strip()


def run_program(program, module=False, viewer=False, arguments=None):
    python = sys.executable
    if viewer and platform.system() == "Darwin":
        local_mjpython = Path(sys.executable).with_name("mjpython")
        python = str(local_mjpython) if local_mjpython.exists() else shutil.which("mjpython")
        if python is None:
            print("MuJoCo viewer needs mjpython. Install MuJoCo in this environment first.")
            return

    command = [python]
    command.extend(["-m", program] if module else [program])
    command.extend(arguments or [])
    print(f"\nRunning {program}...\n")
    try:
        result = subprocess.run(command, cwd=PROJECT_FOLDER, check=False)
        if result.returncode:
            print(f"{program} exited with code {result.returncode}")
    except KeyboardInterrupt:
        print("Program stopped")


def connection_viewer():
    return Connection.mode in ("sim", "both")


def physical_menu():
    while True:
        choice = choose("Physical-only object recognition", [
            (1, "Collect, resume, train, or identify (terminal menu)"),
            (2, "Open presentation UI"),
            (3, "Back"),
        ])
        if choice == "1":
            run_program("Physical_Only_Proprioception.py")
        elif choice == "2":
            run_program("Physical_Only_UI.py")
        elif choice == "3":
            return
        else:
            print("Choose 1 through 3")


def simulation_menu():
    while True:
        choice = choose("Simulation-trained object recognition", [
            (1, "Generate simulated grasps and train"),
            (2, "View recorded MuJoCo grasps"),
            (3, "Recognize with the real hand"),
            (4, "Back"),
        ])
        if choice == "1":
            run_program("Train_Proprioception.py")
        elif choice == "2":
            run_program("View_Proprioception.py", viewer=True)
        elif choice == "3":
            run_program("Run_Proprioception.py")
        elif choice == "4":
            return
        else:
            print("Choose 1 through 4")


def pose_menu():
    while True:
        choice = choose("Pose tools", [
            (1, "Open pose editor"),
            (2, "Play a saved pose sequence"),
            (3, "Back"),
        ])
        if choice == "1":
            run_program("Ex-GUI.py", viewer=connection_viewer())
        elif choice == "2":
            folder = PROJECT_FOLDER / "Data" / "Example_pose_series"
            names = sorted(path.stem for path in folder.glob("*.json"))
            if not names:
                print("No saved pose sequences found")
                continue
            print(f"Available sequences: {', '.join(names)}")
            name = input("Sequence name: ").strip()
            if name in names:
                run_program("Pose_Sequence_Runner.py", viewer=connection_viewer(),
                            arguments=[name])
            else:
                print("Choose a listed sequence name")
        elif choice == "3":
            return
        else:
            print("Choose 1 through 3")


def vision_menu():
    while True:
        choice = choose("Vision and motion teaching", [
            (1, "Direct-angle teleoperation"),
            (2, "Fingertip retargeting"),
            (3, "Record, process, or replay a motion"),
            (4, "Back"),
        ])
        if choice == "1":
            run_program("Vision.Vision_Teleop", module=True,
                        viewer=connection_viewer())
        elif choice == "2":
            run_program("Vision.Vision_Retargeting", module=True,
                        viewer=connection_viewer())
        elif choice == "3":
            run_program("Vision.Vision_Kinesthetic_Teaching", module=True,
                        viewer=connection_viewer())
        elif choice == "4":
            return
        else:
            print("Choose 1 through 4")


def diagnostics_menu():
    while True:
        choice = choose("Hand diagnostics", [
            (1, "Display live joint positions, velocities, and currents"),
            (2, "Run the configured physical grasp only"),
            (3, "Back"),
        ])
        if choice == "1":
            run_program("test1.py", viewer=connection_viewer())
        elif choice == "2":
            run_program("Grasp.py")
        elif choice == "3":
            return
        else:
            print("Choose 1 through 3")


def main():
    while True:
        choice = choose("Ex-ARM", [
            (1, "Physical-only object recognition"),
            (2, "Simulation-trained object recognition"),
            (3, "Pose tools"),
            (4, "Vision and motion teaching"),
            (5, "Hand diagnostics"),
            (6, "Quit"),
        ])
        if choice == "1":
            physical_menu()
        elif choice == "2":
            simulation_menu()
        elif choice == "3":
            pose_menu()
        elif choice == "4":
            vision_menu()
        elif choice == "5":
            diagnostics_menu()
        elif choice == "6":
            return
        else:
            print("Choose 1 through 6")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nLeaving Ex-ARM")
