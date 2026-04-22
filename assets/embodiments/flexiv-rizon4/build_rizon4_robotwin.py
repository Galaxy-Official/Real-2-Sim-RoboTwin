#!/usr/bin/env python3
"""Build a RoboTwin-compatible Flexiv Rizon 4 URDF.

This script downloads the official `flexiv_description` package, vendors it under
the current embodiment directory, rewrites ROS package lookup expressions, and
renders a final `rizon4_robotwin.urdf` that RoboTwin can load directly.
"""

from __future__ import annotations

import argparse
import io
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET


TARBALL_URL = (
    "https://codeload.github.com/flexivrobotics/flexiv_description/"
    "tar.gz/refs/heads/humble"
)
EMBODIMENT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = EMBODIMENT_DIR / "flexiv_description"
URDF_OUTPUT = EMBODIMENT_DIR / "rizon4_robotwin.urdf"
CUROBO_URDF_OUTPUT = EMBODIMENT_DIR / "rizon4_curobo.urdf"
MESH_SCALE_SENTINEL = VENDOR_DIR / ".robotwin_meshes_scaled"
TEXT_SUFFIXES = {
    ".xacro",
    ".xml",
    ".urdf",
    ".srdf",
    ".yaml",
    ".yml",
    ".launch",
    ".py",
    ".txt",
}
MESH_SCALE_FACTOR = 0.001
RAW_MESH_SCALE = "0.001 0.001 0.001"
UNIT_MESH_SCALE = "1 1 1"
RESCALED_MESH_ROOTS = {"Grav"}
WRIST_CAMERA_INSERTION = """
  <link name="camera">
  </link>

  <joint name="camera_joint" type="fixed">
    <origin xyz="0 0 0.10" rpy="0 -1.5707963267948966 0" />
    <parent link="grav_base_link" />
    <child link="camera" />
  </joint>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Delete any existing vendored flexiv_description before downloading.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[flexiv-rizon4] {message}")


def download_tarball(url: str) -> bytes:
    log(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        return response.read()


def prepare_vendor_dir(force_download: bool) -> None:
    if VENDOR_DIR.exists() and force_download:
        log(f"Removing existing vendor directory: {VENDOR_DIR}")
        shutil.rmtree(VENDOR_DIR)

    if VENDOR_DIR.exists():
        log(f"Reusing existing vendor directory: {VENDOR_DIR}")
        return

    tarball = download_tarball(TARBALL_URL)
    with tempfile.TemporaryDirectory(prefix="flexiv_rizon4_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as archive:
            archive.extractall(temp_dir)
        extracted_roots = [path for path in temp_dir.iterdir() if path.is_dir()]
        if not extracted_roots:
            raise RuntimeError("No extracted repository root was found in tarball")
        extracted_root = extracted_roots[0]
        shutil.copytree(extracted_root, VENDOR_DIR)
        log(f"Vendored official package to {VENDOR_DIR}")


def patch_vendor_paths() -> None:
    vendor_abs = VENDOR_DIR.resolve().as_posix()
    replaced_files = 0
    for path in VENDOR_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        patched = original.replace("$(find flexiv_description)", vendor_abs)
        if patched != original:
            path.write_text(patched, encoding="utf-8")
            replaced_files += 1
    log(f"Patched package lookup paths in {replaced_files} vendored text files")


def format_float(value: float) -> str:
    return f"{value:.12g}"


def scale_ascii_stl(path: Path, factor: float) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    scaled_lines = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("vertex "):
            parts = stripped.split()
            coords = [format_float(float(parts[i]) * factor) for i in range(1, 4)]
            scaled_lines.append(f"{indent}vertex {' '.join(coords)}")
        else:
            scaled_lines.append(line)
    path.write_text("\n".join(scaled_lines) + "\n", encoding="utf-8")


def scale_binary_stl(path: Path, factor: float) -> None:
    data = bytearray(path.read_bytes())
    if len(data) < 84:
        raise RuntimeError(f"Binary STL is unexpectedly short: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(data) != expected_size:
        raise RuntimeError(f"Unexpected binary STL size for {path}")
    for index in range(triangle_count):
        offset = 84 + index * 50
        values = list(struct.unpack_from("<12f", data, offset))
        for coord_index in range(3, 12):
            values[coord_index] *= factor
        struct.pack_into("<12f", data, offset, *values)
    path.write_bytes(data)


def scale_stl_mesh(path: Path, factor: float) -> None:
    data = path.read_bytes()
    is_binary = False
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        is_binary = len(data) == 84 + triangle_count * 50
    if is_binary:
        scale_binary_stl(path, factor)
        return
    scale_ascii_stl(path, factor)


def scale_obj_mesh(path: Path, factor: float) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    scaled_lines = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("v "):
            parts = stripped.split()
            coords = [format_float(float(parts[i]) * factor) for i in range(1, 4)]
            extras = parts[4:]
            scaled_line = " ".join(["v", *coords, *extras]).rstrip()
            scaled_lines.append(f"{indent}{scaled_line}")
        else:
            scaled_lines.append(line)
    path.write_text("\n".join(scaled_lines) + "\n", encoding="utf-8")


def normalize_mesh_scale_attributes() -> None:
    replaced_files = 0
    for path in VENDOR_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        patched = original.replace(
            f'scale="{RAW_MESH_SCALE}"',
            f'scale="{UNIT_MESH_SCALE}"',
        ).replace(
            f"scale='{RAW_MESH_SCALE}'",
            f"scale='{UNIT_MESH_SCALE}'",
        )
        if patched != original:
            path.write_text(patched, encoding="utf-8")
            replaced_files += 1
    log(f"Normalized mesh scale attributes in {replaced_files} vendored text files")


def strip_world_base_wrapper(urdf_text: str) -> str:
    """Remove Flexiv's world -> base_link fixed wrapper from rendered URDF.

    RoboTwin already sets the articulation root pose externally. Keeping the
    upstream `world` link and `base_joint` causes the planner and simulation to
    disagree about the robot base frame.
    """

    root = ET.fromstring(urdf_text)

    children = list(root)
    for child in children:
        if child.tag == "link" and child.attrib.get("name") == "world":
            root.remove(child)
            continue

        if child.tag != "joint":
            continue

        parent_link = None
        child_link = None
        for sub in child:
            if sub.tag == "parent":
                parent_link = sub.attrib.get("link")
            elif sub.tag == "child":
                child_link = sub.attrib.get("link")

        if parent_link == "world" and child_link == "base_link":
            root.remove(child)

    return ET.tostring(root, encoding="unicode")


def rescale_vendor_meshes() -> None:
    if MESH_SCALE_SENTINEL.exists():
        log("Vendored meshes are already rescaled to meter units")
        return

    mesh_root = VENDOR_DIR / "meshes"
    if not mesh_root.exists():
        raise FileNotFoundError(f"Missing vendor mesh directory: {mesh_root}")

    scaled_meshes = 0
    for path in mesh_root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(mesh_root).parts
        if not relative_parts or relative_parts[0] not in RESCALED_MESH_ROOTS:
            continue
        suffix = path.suffix.lower()
        if suffix == ".stl":
            scale_stl_mesh(path, MESH_SCALE_FACTOR)
            scaled_meshes += 1
        elif suffix == ".obj":
            scale_obj_mesh(path, MESH_SCALE_FACTOR)
            scaled_meshes += 1

    normalize_mesh_scale_attributes()
    MESH_SCALE_SENTINEL.write_text(
        "Only GN01/Grav mesh vertices were rescaled from millimeters to meters for RoboTwin.\n",
        encoding="utf-8",
    )
    log(
        "Rescaled "
        f"{scaled_meshes} mesh files from millimeters to meters under: "
        + ", ".join(sorted(RESCALED_MESH_ROOTS))
    )


def resolve_xacro_runner() -> list[str]:
    candidates = [
        ["xacro"],
        [sys.executable, "-m", "xacro"],
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                candidate + ["--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            continue
        if result.returncode == 0:
            return candidate
    raise RuntimeError(
        "Could not find a usable xacro executable. Install ROS xacro first, for "
        "example: `sudo apt install ros-humble-xacro` or `pip install xacro`."
    )


def render_urdf() -> None:
    xacro_runner = resolve_xacro_runner()
    xacro_file = VENDOR_DIR / "urdf" / "rizon.urdf.xacro"
    if not xacro_file.exists():
        raise FileNotFoundError(f"Missing xacro file: {xacro_file}")

    log(f"Rendering URDF from {xacro_file}")
    command = xacro_runner + [
        str(xacro_file),
        "rizon_type:=Rizon4",
        "load_gripper:=true",
        "gripper_name:=Flexiv-GN01",
        "load_mounted_ft_sensor:=false",
        "add_world_link:=false",
        "create_base_joint:=false",
        "ros2_control:=false",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "xacro failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    urdf_text = result.stdout
    vendor_abs = VENDOR_DIR.resolve().as_posix()
    urdf_text = urdf_text.replace("package://flexiv_description/", "./flexiv_description/")
    urdf_text = urdf_text.replace(f"{vendor_abs}/", "./flexiv_description/")
    urdf_text = urdf_text.replace(
        f'scale="{RAW_MESH_SCALE}"',
        f'scale="{UNIT_MESH_SCALE}"',
    )
    urdf_text = strip_world_base_wrapper(urdf_text)
    marker = "</robot>"
    if marker not in urdf_text:
        raise RuntimeError("Could not find closing </robot> tag in rendered URDF")
    urdf_text = urdf_text.replace(marker, WRIST_CAMERA_INSERTION + marker, 1)
    URDF_OUTPUT.write_text(urdf_text, encoding="utf-8")
    log(f"Wrote {URDF_OUTPUT}")


def render_curobo_urdf() -> None:
    """Render a cuRobo-safe URDF without articulated gripper joints.

    cuRobo has known issues with robot descriptions that include extra
    non-fixed joints outside the planned arm chain. The Flexiv GN01 gripper
    contains several mimic-driven revolute joints, so we render a second URDF
    that removes the gripper and adds only one fixed TCP link at the gripper's
    closed-center pose.
    """

    xacro_runner = resolve_xacro_runner()
    xacro_file = VENDOR_DIR / "urdf" / "rizon.urdf.xacro"
    command = xacro_runner + [
        str(xacro_file),
        "rizon_type:=Rizon4",
        "load_gripper:=false",
        "load_mounted_ft_sensor:=false",
        "add_world_link:=false",
        "create_base_joint:=false",
        "ros2_control:=false",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "xacro failed when rendering cuRobo URDF.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    urdf_text = result.stdout
    vendor_abs = VENDOR_DIR.resolve().as_posix()
    urdf_text = urdf_text.replace("package://flexiv_description/", "./flexiv_description/")
    urdf_text = urdf_text.replace(f"{vendor_abs}/", "./flexiv_description/")
    urdf_text = urdf_text.replace(
        f'scale="{RAW_MESH_SCALE}"',
        f'scale="{UNIT_MESH_SCALE}"',
    )
    urdf_text = strip_world_base_wrapper(urdf_text)

    insertion = """
  <link name="closed_fingers_tcp">
  </link>

  <joint name="closed_fingers_tcp_joint" type="fixed">
    <origin xyz="0 0 0.20" rpy="0 0 0" />
    <parent link="flange" />
    <child link="closed_fingers_tcp" />
  </joint>
"""
    marker = "</robot>"
    if marker not in urdf_text:
        raise RuntimeError("Could not find closing </robot> tag in rendered cuRobo URDF")
    urdf_text = urdf_text.replace(marker, insertion + marker, 1)
    CUROBO_URDF_OUTPUT.write_text(urdf_text, encoding="utf-8")
    log(f"Wrote {CUROBO_URDF_OUTPUT}")


def validate_urdf() -> None:
    urdf_text = URDF_OUTPUT.read_text(encoding="utf-8")
    required_tokens = [
        'name="joint1"',
        'name="joint7"',
        'name="finger_width_joint"',
        'name="closed_fingers_tcp"',
        'link name="grav_base_link"',
        'name="camera"',
        'name="camera_joint"',
    ]
    missing = [token for token in required_tokens if token not in urdf_text]
    if missing:
        raise RuntimeError(
            "Generated URDF is missing expected Rizon 4 tokens: "
            + ", ".join(missing)
        )
    forbidden_tokens = [
        'link name="world"',
        'joint name="base_joint"',
    ]
    present_forbidden = [token for token in forbidden_tokens if token in urdf_text]
    if present_forbidden:
        raise RuntimeError(
            "Generated URDF still contains a world/base wrapper that RoboTwin "
            "does not expect: " + ", ".join(present_forbidden)
        )
    if RAW_MESH_SCALE in urdf_text:
        raise RuntimeError("Generated URDF still contains non-unit mesh scale attributes")
    log("Validated generated URDF structure")


def validate_curobo_urdf() -> None:
    urdf_text = CUROBO_URDF_OUTPUT.read_text(encoding="utf-8")
    required_tokens = [
        'name="joint1"',
        'name="joint7"',
        'name="closed_fingers_tcp"',
        'name="closed_fingers_tcp_joint"',
    ]
    missing = [token for token in required_tokens if token not in urdf_text]
    if missing:
        raise RuntimeError(
            "Generated cuRobo URDF is missing expected tokens: "
            + ", ".join(missing)
        )
    forbidden_tokens = [
        'name="finger_width_joint"',
        'name="left_outer_knuckle_joint"',
        'name="right_outer_knuckle_joint"',
    ]
    present_forbidden = [token for token in forbidden_tokens if token in urdf_text]
    if present_forbidden:
        raise RuntimeError(
            "cuRobo URDF still contains articulated gripper joints: "
            + ", ".join(present_forbidden)
        )
    forbidden_root_tokens = [
        'link name="world"',
        'joint name="base_joint"',
    ]
    present_forbidden_root = [token for token in forbidden_root_tokens if token in urdf_text]
    if present_forbidden_root:
        raise RuntimeError(
            "Generated cuRobo URDF still contains a world/base wrapper that "
            "RoboTwin does not expect: " + ", ".join(present_forbidden_root)
        )
    if RAW_MESH_SCALE in urdf_text:
        raise RuntimeError(
            "Generated cuRobo URDF still contains non-unit mesh scale attributes"
        )
    log("Validated generated cuRobo URDF structure")


def main() -> int:
    args = parse_args()
    prepare_vendor_dir(force_download=args.force_download)
    patch_vendor_paths()
    rescale_vendor_meshes()
    render_urdf()
    render_curobo_urdf()
    validate_urdf()
    validate_curobo_urdf()
    log("Done")
    log("Next: run `python script/update_embodiment_config_path.py` from the RoboTwin root")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
