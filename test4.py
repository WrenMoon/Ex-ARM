from pathlib import Path

urdf_path = Path("Data/mujoco_robot.urdf")

text = urdf_path.read_text(encoding="utf-8")

# Replace:
# package:///mcp_joint.stl
# package://something/mcp_joint.stl
#
# with:
# mcp_joint.stl
#
import re

text = re.sub(
    r'package:///(?:[^/]+/)?',
    '',
    text
)

urdf_path.write_text(text, encoding="utf-8")

print("URDF paths fixed.")