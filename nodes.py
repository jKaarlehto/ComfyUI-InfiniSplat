import json
import logging
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import uuid

import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management as mm
import comfy.utils
from server import PromptServer
from comfy_api.latest import Types
from aiohttp import web

PACK_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(folder_paths.models_dir) / "infinisplat"
folder_paths.add_model_folder_path("infinisplat", str(MODEL_DIR / "checkpoints"))
CHECKPOINT = "infinisplat_rgb.ckpt"


if getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.get("/infinisplat/model_status")
    async def model_status(request):
        missing = []
        checkpoint = folder_paths.get_full_path("infinisplat", CHECKPOINT)
        if not checkpoint or not Path(checkpoint).is_file():
            missing.append("InfiniSplat")
        if request.query.get("camera") == "geocalib" and not (MODEL_DIR / "geocalib/pinhole.tar").is_file():
            missing.append("GeoCalib")
        return web.json_response({"missing": missing})


class InfiniSplatGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "focal_length_mm": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 500.0, "step": 0.1, "tooltip": "35mm-equivalent focal length."}),
                "gaussian_count": ("INT", {"default": 1500000, "min": 10000, "max": 4000000, "step": 10000}),
                "query_chunk_size": ("INT", {"default": 20000, "min": 1000, "max": 160000, "step": 1000, "tooltip": "Lower values reduce peak memory; the upstream default is 80000."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "filter_floaters": ("BOOLEAN", {"default": True}),
                "download_model": ("BOOLEAN", {"default": True, "tooltip": "Download the selected checkpoint on execution if it is missing."}),
                "save": ("BOOLEAN", {"default": False, "tooltip": "Save a permanent PLY in the output directory; otherwise keep a temporary file for preview."}),
            },
            "optional": {
                "file_path": ("STRING", {"default": "infinisplat/InfiniSplat", "tooltip": "Filename prefix relative to ComfyUI's output directory. A counter and .ply extension are added."}),
                "camera_estimation": (["manual", "geocalib"], {"default": "manual", "tooltip": "GeoCalib estimates intrinsics per image. Requires install.py --camera. Connected intrinsics take precedence."}),
                "intrinsics": ("INTRINSICS", {"tooltip": "Normalized camera matrices, shape (B,3,3) or (B,4,4), e.g. MoGe2. One matrix broadcasts to the image batch."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("FILE_3D_PLY", "STRING")
    RETURN_NAMES = ("model_3d", "file_path")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "generate"
    CATEGORY = "3d/InfiniSplat"
    OUTPUT_NODE = True

    def generate(self, image, focal_length_mm, gaussian_count, query_chunk_size, seed, filter_floaters, download_model, save=False, file_path="infinisplat/InfiniSplat", camera_estimation="manual", intrinsics=None, unique_id=None):
        progress = comfy.utils.ProgressBar(1000, node_id=unique_id)
        last_value = 0
        last_stage = None
        def report(value, message):
            nonlocal last_value, last_stage
            last_value = max(last_value, int(value))
            progress.update_absolute(last_value)
            # Chunk counters update the node without filling the console with every chunk.
            stage = message.split("Gaussians")[0]
            if stage != last_stage:
                logging.info("[InfiniSplat] %s", message)
                last_stage = stage
            server = getattr(PromptServer, "instance", None)
            if unique_id is not None and server is not None:
                server.send_sync("infinisplat_progress", {"node_id": str(unique_id), "message": message}, server.client_id)
        report(0, "Preparing input images")
        if camera_estimation not in ("manual", "geocalib"):
            raise ValueError(f"Unknown camera estimator: {camera_estimation}")
        matrices = None
        if intrinsics is not None:
            matrices = intrinsics.detach().cpu().numpy()
            if matrices.ndim == 2:
                matrices = matrices[None]
            if matrices.ndim != 3 or matrices.shape[1:] not in ((3, 3), (4, 4)) or len(matrices) not in (1, len(image)):
                raise ValueError("intrinsics must be normalized 3x3 or 4x4 matrices: one per image, or one shared matrix.")
            matrices = matrices[:, :3, :3].copy()
            if not np.isfinite(matrices).all() or (matrices[:, 0, 0] <= 0).any() or (matrices[:, 1, 1] <= 0).any() or (np.abs(np.linalg.det(matrices)) < 1e-10).any():
                raise ValueError("intrinsics must be finite, non-singular, with positive focal lengths.")
            matrices[:, 0, :] *= image.shape[2]
            matrices[:, 1, :] *= image.shape[1]
        python = PACK_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file() or not (PACK_DIR / "upstream/src/demo/infer_batch_images.py").is_file():
            raise RuntimeError("InfiniSplat runtime is missing. Run: uv run --no-project install.py")
        checkpoint = folder_paths.get_full_path("infinisplat", CHECKPOINT)
        if checkpoint is None:
            checkpoint = str(MODEL_DIR / "checkpoints" / CHECKPOINT)
        if not Path(checkpoint).is_file() and not download_model:
            raise FileNotFoundError(f"Place {CHECKPOINT} in {MODEL_DIR / 'checkpoints'} or enable download_model.")

        output_dir = Path(folder_paths.get_temp_directory()) / "infinisplat" / uuid.uuid4().hex
        save_target = None
        if save:
            prefix = file_path.strip()
            if not prefix:
                raise ValueError("file_path must contain a filename prefix when save is enabled.")
            if prefix.lower().endswith(".ply"):
                prefix = prefix[:-4]
            save_target = folder_paths.get_save_image_path(prefix, folder_paths.get_output_directory())
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("This InfiniSplat runtime requires an NVIDIA CUDA GPU.")
        mm.unload_all_models()
        mm.soft_empty_cache()

        Path(folder_paths.get_temp_directory()).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="infinisplat_", dir=folder_paths.get_temp_directory()) as temporary:
            staging = Path(temporary)
            progress_path = staging / "progress.jsonl"
            for index, rgb in enumerate(image):
                array = (rgb[..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(array).save(staging / f"{index:04d}.png")
            request = {
                "input_dir": str(staging), "output_dir": str(output_dir),
                "checkpoint": checkpoint, "model_dir": str(MODEL_DIR),
                "focal_length_mm": focal_length_mm,
                "gaussian_count": gaussian_count, "query_chunk_size": query_chunk_size,
                "seed": seed, "filter_floaters": filter_floaters,
                "download_model": download_model, "device": str(device),
                "camera_estimation": camera_estimation,
                "intrinsics_px": matrices.tolist() if matrices is not None else None,
                "progress_file": str(progress_path),
            }
            request_path = staging / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            log_path = staging / "worker.log"
            environment = os.environ.copy()
            environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
            environment["PYTHONUNBUFFERED"] = "1"
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            progress_offset = 0
            progress_pending = ""
            def read_progress():
                nonlocal progress_offset, progress_pending
                if not progress_path.exists():
                    return
                with progress_path.open("r", encoding="utf-8") as updates:
                    updates.seek(progress_offset)
                    progress_pending += updates.read()
                    progress_offset = updates.tell()
                lines = progress_pending.split("\n")
                progress_pending = lines.pop()
                for line in lines:
                    event = json.loads(line)
                    report(event["value"], event["message"])
            with log_path.open("w", encoding="utf-8") as log:
                report(5, "Starting runtime")
                process = subprocess.Popen(
                    [str(python), str(PACK_DIR / "worker.py"), str(request_path)],
                    cwd=PACK_DIR / "upstream", env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                try:
                    while True:
                        mm.throw_exception_if_processing_interrupted()
                        read_progress()
                        try:
                            return_code = process.wait(timeout=0.25)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                except BaseException:
                    report(last_value, "Interrupted or failed")
                    raise
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
            read_progress()
            if return_code:
                report(last_value, "Inference failed")
                details = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise RuntimeError(f"InfiniSplat inference failed:\n{details}")
            paths = [str(output_dir / f"{index:04d}" / f"{index:04d}.ply") for index in range(image.shape[0])]
            for path in paths:
                if not Path(path).is_file():
                    raise RuntimeError(f"InfiniSplat did not write its expected output: {path}")
            report(980, "Preparing native 3D output")
            if save_target is not None:
                destination, filename, counter, _, _ = save_target
                saved_paths = []
                for path in paths:
                    while True:
                        target = Path(destination) / f"{filename}_{counter:05d}.ply"
                        try:
                            with target.open("xb") as output, Path(path).open("rb") as source:
                                shutil.copyfileobj(source, output)
                            break
                        except FileExistsError:
                            counter += 1
                    saved_paths.append(str(target))
                    camera_metadata = Path(path).with_name("camera.json")
                    if camera_metadata.is_file():
                        shutil.copy2(camera_metadata, target.with_suffix(".camera.json"))
                    counter += 1
                    report(980 + 19 * len(saved_paths) / len(paths), f"Saving PLY {len(saved_paths)}/{len(paths)}")
                shutil.rmtree(output_dir)
                paths = saved_paths
            files = [Types.File3D(path) for path in paths]
            report(1000, "Complete")
            return {"ui": {"saved_files": paths if save else []}, "result": (files, paths if save else [""] * len(files))}


NODE_CLASS_MAPPINGS = {"InfiniSplatGenerate": InfiniSplatGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"InfiniSplatGenerate": "InfiniSplat (Image to PLY)"}
