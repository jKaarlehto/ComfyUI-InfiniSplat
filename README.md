# ComfyUI-InfiniSplat

InfiniSplat single-image Gaussian generation with native ComfyUI 3D file output and built-in splat preview. NVIDIA CUDA required.

## Install

Requires ComfyUI 0.39.2 or newer with native 3D file/splat tools. Tested on Windows with an RTX 5090 Laptop (24GB VRAM); Linux paths are supported by the setup script but have not been tested.

Install this package with ComfyUI Manager, or place this folder in `ComfyUI/custom_nodes/`. The inference runtime requires a separate setup: install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then run from this folder:

```powershell
uv run --no-project install.py
```

Restart ComfyUI. Search for **InfiniSplat (Image to PLY)**, under `3d/InfiniSplat`.

The installer creates a separate Python 3.11 / PyTorch 2.9 CUDA 12.8 environment and checks out upstream revision `f41394a8930e72905d33928396fcac4574cd70a9`. It does not install packages into ComfyUI's Python environment. Model construction and inference use upstream model math. Two small patches use memory-mapped checkpoint loading and vectorized PLY export to reduce CPU memory and Python allocation overhead.

## Use

Load Image → InfiniSplat (Image to PLY) → PreviewGaussians. The viewer comes from [ComfyUI-GaussianPack](https://github.com/PozzettiAndrea/ComfyUI-GaussianPack).

- Input: one RGB image, or a batch processed sequentially.
- `focal_length_mm`: 35mm-equivalent focal length; default 30mm. ComfyUI IMAGE tensors do not retain EXIF metadata.
- No calibrated camera is required: upstream constructs centered intrinsics from image dimensions and this focal length, and uses an identity input-camera pose. This is an assumed lens, not automatic focal estimation. Adjust focal length for wide-angle or telephoto images. Native preview accepts no camera metadata.
- `gaussian_count`: default 1,500,000 surface samples; the final floater filter may reduce the exported count.
- `query_chunk_size`: default 20,000, versus upstream's 80,000. Smaller chunks reduce peak decoder memory without reducing the sample count.
- `seed`: controls randomized surface sampling.
- `download_model`: explicitly permits downloading the selected checkpoint when the node executes. Disable for offline use.

The checkpoint is stored under `ComfyUI/models/infinisplat/checkpoints/infinisplat_rgb.ckpt`. Additional `infinisplat` paths can be registered through `extra_model_paths.yaml`.

- `save`: off by default. Preview files live in ComfyUI's temporary directory, with no permanent output export.
- `file_path`: revealed when Save is on. This is a filename prefix relative to the configured output directory, for example `infinisplat/MyScene`. A counter and `.ply` are added, preserving existing files. A trailing `.ply` is accepted. The `file_path` output is revealed when Save is on and returns the permanent saved paths (one per batch image). Disabling Save removes this output and its links.

The native `model_3d` output is a list, one file per input image. Square sockets indicate list processing. Runs unload cached ComfyUI models before launching the isolated worker. It loads once per batch and exits afterward, releasing GPU memory. Cancellation terminates the worker.

Progress appears in ComfyUI's standard node progress bar and a read-only status line. It reports camera estimation, model loading, image/batch number, depth, surface sampling, features, Gaussian query counts, filtering, and PLY export. The overall percentage uses weighted work stages, not an estimate of time remaining. Loading and first-time downloads report their stage without a byte-level percentage. Query updates observe upstream calls without changing model math or adding CUDA synchronization per chunk.

Load `workflows/infinisplat.json` for the example graph; select your input image before running.

The depth-guided checkpoint is not exposed: upstream's xFormers kernels fail on the tested Windows RTX 5090 (SM120), and its non-xFormers attention fallback also fails. RGB inference uses native PyTorch attention and works without xFormers.

## Upstream license

[InfiniSplat](https://github.com/zju3dv/InfiniSplat) uses the [Project Registration License](https://github.com/zju3dv/InfiniSplat/blob/main/LICENSE): personal, research and educational use is free without registration; organizational project use requires registration. Its vendored backbones have their own licenses. Downloadable weights do not imply an MIT license.

## Optional camera estimation

Install `uv run --no-project install.py --camera`, then choose `camera_estimation: geocalib`. This installs the pinned [GeoCalib](https://github.com/cvg/GeoCalib) inference package (Apache 2.0) in the isolated runtime. Its pinhole weights download only if `download_model` is enabled, into `models/infinisplat/geocalib/pinhole.tar`.

Calibration runs once per image before loading InfiniSplat and releases its GPU model first. Batch images are calibrated independently, so mixed-camera batches are supported. Manual focal length remains the default. GeoCalib assumes a centered principal point and estimates intrinsics, roll/pitch, and focal uncertainty; it does not recover translation, yaw, or metric world scale. Estimated roll/pitch are metadata and do not rotate the splat. The inferred intrinsics are passed through upstream's camera override, including its resize/crop corrections. Saving also writes a `.camera.json` sidecar.

Alternatively connect normalized `INTRINSICS` from MoGe2. Accepted tensors are `(B,3,3)` or `(B,4,4)`; one matrix may broadcast to all images. Connected intrinsics override camera estimation and manual focal length. They must match the image dimensions/crop supplied to InfiniSplat.

On the tested RTX 5090 Laptop, one example calibration took 2.68 seconds including model loading and used 1.03 GiB peak PyTorch CUDA allocation. This is an example measurement, not a fixed runtime or accuracy guarantee. Camera parameters inferred from a single image remain estimates, especially for cropped images or scenes with few perspective cues.

## Publish

Registry package: `infinisplat`, publisher: `jkaa`.

Set the repository's GitHub Actions secret `REGISTRY_ACCESS_TOKEN` to a publishing key for this publisher, then run **Publish to Comfy Registry** from Actions. The workflow is manually triggered so ordinary code pushes do not publish a new version. Each registry version is immutable; bump `project.version` before publishing another release.

The registry archive contains this integration, setup script, patches, and example workflows. It excludes the runtime environment, cloned model source, and checkpoints. The original integration is MIT licensed; see `THIRD_PARTY_NOTICES.md` for the separate model licenses.
