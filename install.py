import os
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REVISION = "f41394a8930e72905d33928396fcac4574cd70a9"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true", help="Install the optional GeoCalib camera estimator.")
    options = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Install uv, then run: uv run --no-project install.py")
    upstream = ROOT / "upstream"
    if not upstream.exists():
        subprocess.run(["git", "clone", "https://github.com/zju3dv/InfiniSplat.git", str(upstream)], check=True)
    subprocess.run(["git", "checkout", "--detach", REVISION], cwd=upstream, check=True)
    patch = str(ROOT / "upstream.patch")
    applied = subprocess.run(["git", "apply", "--reverse", "--check", patch], cwd=upstream, capture_output=True)
    if applied.returncode:
        subprocess.run(["git", "apply", "--check", patch], cwd=upstream, check=True)
        subprocess.run(["git", "apply", patch], cwd=upstream, check=True)
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        subprocess.run([uv, "venv", "--python", "3.11", str(ROOT / ".venv")], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(python), "torch==2.9.0", "torchvision==0.24.0", "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(python), "-r", str(ROOT / "requirements-runtime.txt")], check=True)
    if options.camera:
        subprocess.run([uv, "pip", "install", "--python", str(python), "-r", str(ROOT / "requirements-runtime.txt"), "opencv-python<4.12", "geocalib @ git+https://github.com/cvg/GeoCalib.git@97b8968e7798a66bf04fcf791fb535624241bda7"], check=True)
    print("InfiniSplat runtime installed. Restart ComfyUI to load the node.")


if __name__ == "__main__":
    main()
