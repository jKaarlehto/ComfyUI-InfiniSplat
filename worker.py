import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
UPSTREAM = Path(__file__).resolve().parent / "upstream"
os.chdir(UPSTREAM)
sys.path.insert(0, str(UPSTREAM))

import torch
from huggingface_hub import hf_hub_download
from src.demo import infer_batch_images as inference
from src.demo.infer_single_image import load_demo_config, load_demo_model


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    def report(value, message):
        if request.get("progress_file"):
            with Path(request["progress_file"]).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"value": int(value), "message": message}) + "\n")

    report(10, "Starting runtime")
    checkpoint = Path(request["checkpoint"])
    if not checkpoint.is_file():
        if not request["download_model"]:
            raise FileNotFoundError(checkpoint)
        report(10, "Downloading InfiniSplat weights")
        checkpoint = Path(hf_hub_download(
            repo_id="PLUS-WAVE/InfiniSplat",
            filename="checkpoints/infinisplat_rgb.ckpt",
            local_dir=request["model_dir"],
        ))

    torch.manual_seed(request["seed"])
    device = torch.device(request["device"])
    torch.cuda.reset_peak_memory_stats(device)
    images = inference._collect_images(argparse.Namespace(input_path=Path(request["input_dir"]), limit=0))
    cameras = []
    matrices = request.get("intrinsics_px")
    if matrices is not None:
        cameras = [{"intrinsics_px": matrices[0 if len(matrices) == 1 else index], "source": "connected_intrinsics"} for index in range(len(images))]
    elif request.get("camera_estimation", "manual") == "geocalib":
        try:
            from geocalib import GeoCalib
        except ImportError as exc:
            raise RuntimeError("Install the optional camera addon: uv run --no-project install.py --camera") from exc
        camera_dir = Path(request["model_dir"]) / "geocalib"
        weights = camera_dir / "pinhole.tar"
        if not weights.is_file():
            if not request["download_model"]:
                raise FileNotFoundError(f"GeoCalib weights missing: {weights}. Enable download_model for the first run.")
            camera_dir.mkdir(parents=True, exist_ok=True)
            report(20, "Downloading GeoCalib weights")
            temporary_weights = weights.with_suffix(".download")
            torch.hub.download_url_to_file("https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar", str(temporary_weights))
            temporary_weights.replace(weights)
        calibration_start = time.perf_counter()
        report(20, "Loading camera estimator")
        calibrator = GeoCalib(weights=str(weights)).to(device)
        for index, image_path in enumerate(images):
            report(20 + 80 * index / len(images), f"Estimating camera {index + 1}/{len(images)}")
            calibration = calibrator.calibrate(calibrator.load_image(image_path).to(device), camera_model="pinhole")
            matrix = calibration["camera"].K[0].detach().cpu()
            if not torch.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
                raise RuntimeError(f"GeoCalib returned invalid intrinsics for {image_path.name}. Use manual camera mode.")
            cameras.append({"source": "geocalib", "intrinsics_px": matrix.tolist(),
                            "roll_pitch_radians": calibration["gravity"].rp[0].detach().cpu().tolist(),
                            "focal_uncertainty_px": calibration["focal_uncertainty"].detach().cpu().tolist()})
        print(json.dumps({"camera_seconds": time.perf_counter() - calibration_start, "camera_peak_vram_gb": torch.cuda.max_memory_allocated(device) / 2**30}))
        del calibrator, calibration, matrix
        torch.cuda.empty_cache()
    # Calibration can use random sampling; keep splat sampling reproducible across camera modes.
    torch.manual_seed(request["seed"])
    cfg = load_demo_config(inference.MODE_EXPERIMENTS["rgb"])
    cfg.model.encoder.sample_point_num = request["gaussian_count"]
    cfg.model.encoder.implicit_gs_query_batch_size = request["query_chunk_size"]
    report(100, "Loading InfiniSplat model")
    encoder, decoder = load_demo_model(cfg, checkpoint, device)
    # Observe the existing calls without changing model math or synchronizing CUDA per chunk.
    current_image = 0
    query_done = 0
    query_total = 1

    def image_report(fraction, stage):
        report(200 + 780 * (current_image + fraction) / len(images), f"Image {current_image + 1}/{len(images)}: {stage}")

    encoder.depth_predictor.register_forward_pre_hook(lambda *_: image_report(0.05, "predicting depth"))
    encoder.image_feature_branch.register_forward_pre_hook(lambda *_: image_report(0.3, "encoding image features"))
    original_sample = encoder._sample_sparse_coords
    def sample(*args, **kwargs):
        image_report(0.2, "sampling surface points")
        return original_sample(*args, **kwargs)
    encoder._sample_sparse_coords = sample
    original_decode = encoder._decode_dino_gaussian_delta
    def decode(*args, **kwargs):
        nonlocal query_done, query_total
        coords = kwargs.get("coords_yx", args[4] if len(args) > 4 else None)
        query_done, query_total = 0, coords.shape[1]
        image_report(0.4, f"decoding Gaussians 0/{query_total:,}")
        return original_decode(*args, **kwargs)
    encoder._decode_dino_gaussian_delta = decode
    original_chunk = encoder.implicit_gs_head.decode_dpt
    def chunk(*args, **kwargs):
        nonlocal query_done
        result = original_chunk(*args, **kwargs)
        coords = kwargs.get("coords_yx", args[2] if len(args) > 2 else None)
        query_done += coords.shape[1]
        image_report(0.4 + 0.4 * query_done / query_total, f"decoding Gaussians {query_done:,}/{query_total:,}")
        return result
    encoder.implicit_gs_head.decode_dpt = chunk
    original_filter = inference.filter_final_gaussian_floaters
    def filter_gaussians(*args, **kwargs):
        image_report(0.85, "filtering floaters")
        return original_filter(*args, **kwargs)
    inference.filter_final_gaussian_floaters = filter_gaussians
    original_save = inference.save_ply
    def save_ply(*args, **kwargs):
        image_report(0.9, "writing PLY")
        return original_save(*args, **kwargs)
    inference.save_ply = save_ply
    args = argparse.Namespace(
        mode="rgb", input_path=Path(request["input_dir"]),
        output_dir=Path(request["output_dir"]), limit=0,
        focal_px=None, focal_mm=request["focal_length_mm"], intrinsics_file=None,
        prompt_depth=None, prompt_depth_dir=None,
        disable_floater_filter=not request["filter_floaters"], no_video=True, export_html=False,
    )
    for index, image_path in enumerate(images):
        current_image = index
        image_report(0, "preparing image")
        paths = inference._case_paths(args.output_dir, image_path)
        if cameras:
            paths.scene_ply.parent.mkdir(parents=True, exist_ok=True)
            camera_path = paths.scene_ply.with_name("camera.json")
            camera_path.write_text(json.dumps(cameras[index], indent=2), encoding="utf-8")
            args.intrinsics_file = camera_path
        inference._run_one_image(args, image_path, paths, encoder, decoder, False, device, None)
        image_report(1, "PLY ready")
    print(json.dumps({"peak_vram_gb": torch.cuda.max_memory_allocated(device) / 2**30}))


if __name__ == "__main__":
    main()
