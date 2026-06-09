import sys
import platform
import subprocess
import pathlib
import shutil

ROOT = pathlib.Path("..")
WHEEL_ROOT = ROOT / "wheels"

system = platform.system()
machine = platform.machine().lower()

# -------- 1. Check Python Version --------
supported_versions = {(3, 11), (3, 12), (3, 13)}
py_ver = sys.version_info[:2]

if py_ver not in supported_versions:
    raise RuntimeError(
        f"Unsupported Python version: {py_ver[0]}.{py_ver[1]}.\n"
        "This notebook only supports Python 3.11, 3.12, or 3.13."
    )

py_tag = f"cp{py_ver[0]}{py_ver[1]}"

# -------- 2. Find Pattern Beased on Python Version and Platform --------
patterns = []

if system == "Windows":
    if machine in ("amd64", "x86_64"):
        patterns = [
            f"*{py_tag}*win_amd64.whl",
        ]
    elif machine in ("x86", "i386", "i686"):
        patterns = [
            f"*{py_tag}*win32.whl",
        ]
    else:
        raise RuntimeError(
            f"Unsupported Windows architecture: {machine}.\n"
            "Expected amd64/x86_64 or x86/i386/i686."
        )

elif system == "Darwin":
    # Prioritize universal2, then specific architectures
    if machine in ("arm64", "aarch64"):
        patterns = [
            f"*{py_tag}*macosx*universal2.whl",
            f"*{py_tag}*macosx*arm64.whl",
        ]
    elif machine in ("x86_64", "amd64"):
        patterns = [
            f"*{py_tag}*macosx*universal2.whl",
            f"*{py_tag}*macosx*x86_64.whl",
        ]
    else:
        raise RuntimeError(
            f"Unsupported macOS architecture: {machine}.\n"
            "Expected x86_64/amd64 or arm64/aarch64."
        )

elif system == "Linux":
    # Prioritize manylinux, then musllinux, for each architecture
    if machine in ("x86_64", "amd64"):
        patterns = [
            f"*{py_tag}*manylinux*x86_64.whl",
            f"*{py_tag}*musllinux*x86_64.whl",
        ]
    elif machine in ("i686", "x86"):
        patterns = [
            f"*{py_tag}*manylinux*i686.whl",
            f"*{py_tag}*musllinux*i686.whl",
        ]
    elif machine in ("arm64", "aarch64"):
        patterns = [
            f"*{py_tag}*manylinux*aarch64.whl",
            f"*{py_tag}*musllinux*aarch64.whl",
        ]
    else:
        raise RuntimeError(
            f"Unsupported Linux architecture: {machine}."
        )

else:
    raise RuntimeError(
        f"Unsupported operating system: {system}."
    )

# -------- 3. Find wheel --------
wheel = None
for pattern in patterns:
    matches = sorted(WHEEL_ROOT.rglob(pattern))
    if matches:
        wheel = matches[0]
        break

if wheel is None:
    raise RuntimeError(
        f"No compatible wheel found for:\n"
        f"  Python tag: {py_tag}\n"
        f"  System: {system}\n"
        f"  Architecture: {machine}\n\n"
        f"Searched under: {WHEEL_ROOT}\n"
        f"Patterns tried:\n  " + "\n  ".join(patterns)
    )

# -------- 4. Install --------
# This project is managed by uv, whose virtualenvs don't ship pip.
# Prefer `uv pip install` (installs into the active venv) and fall back
# to `python -m pip` only if uv isn't on PATH.
print(f"Installing wheel: {wheel.name}")
if shutil.which("uv"):
    subprocess.check_call([
        "uv", "pip", "install", "--quiet", str(wheel)
    ])
else:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", str(wheel)
    ])
print("Package installed.")
