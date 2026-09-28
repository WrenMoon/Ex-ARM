from pathlib import Path


INPUT = Path("Data/robot.urdf")
OUTPUT = Path("Data/mujoco_robot.urdf")


def convert_urdf():
    urdf = INPUT.read_text(encoding="utf-8")

    # MuJoCo version uses mesh paths without the ROS package:// prefix.
    urdf = urdf.replace('filename="package:///', 'filename="')

    OUTPUT.write_text(urdf, encoding="utf-8")

    print(f"Converted {INPUT} -> {OUTPUT}")


if __name__ == "__main__":
    convert_urdf()