# Rank-One Prior Video Dehazing

**English** · [简体中文](README.zh-CN.md)

Single-image dehazing, framewise video dehazing, and motion-refined video dehazing, with a tool for comparing the three video streams. The motion-refined method applies full Rank-One Prior (ROP) to intra-coded frames, propagates the control-grid state using H.264 motion vectors and reference-picture identities, and refreshes unreliable blocks using the current frame. It restricts propagation across detected scene cuts.

| Script | Purpose |
| --- | --- |
| `rank_one_prior_image.py` | Single-image Rank-One Prior (Rank1). |
| `framewise_rop_video.py` | Full Rank1 independently on every video frame. |
| `motion_refined_rop_video.py` | Motion-vector propagation with adaptive block refinement. |
| `compare_three_videos.py` | Synchronized comparison of the original and both restored streams, with a processing information panel. |

## Setup

The full workflow was tested on Windows x64 with Python 3.14. Open Git Bash and run:

```bash
git clone https://github.com/Qiafengfei/rank-one-prior-video-dehazing.git
cd rank-one-prior-video-dehazing
python -m pip install -r requirements.txt
```

This is sufficient for single-image and framewise dehazing. For motion-refined dehazing and the full comparison workflow, first build the bundled H.264 reference exporter. Install a MinGW-w64 x64 toolchain and make sure `gcc` and `mingw32-make` are on `PATH`. Then run from the repository root in Git Bash:

```bash
bash native/build_decoder.sh
```

The script builds `native/native_decoder.exe` from the bundled FFmpeg 7.1.1 source archive. Use a short checkout path if the Windows build tools hit path-length limits.

## Run the examples

| Bundled input | Resolution | Frames | Frame rate |
| --- | --- | ---: | ---: |
| [10s.mp4](examples/10s.mp4) | 960 × 540 | 300 | 30 FPS |
| [20s.mp4](examples/20s.mp4) | 1920 × 1080 | 620 | 30000/1001 FPS |

From the repository root, run these commands in order. The comparison script reads both generated videos and the motion-refined run's `.txt` report. Use `20s` in all three commands for the other example.

```bash
python framewise_rop_video.py 10s
python motion_refined_rop_video.py 10s
python compare_three_videos.py 10s
```

The video scripts resolve the bundled inputs and outputs relative to the repository, so you can clone it anywhere without editing paths. For a single image, supply your own input and output paths:

```bash
python rank_one_prior_image.py input.png outputs/dehazed.png
```

## Output paths

The paths below are relative to the repository root. All video commands default to CPU mode.

| Result | 10-second example | 20-second example |
| --- | --- | --- |
| Framewise ROP | `outputs/framewise/10s-framewise.mp4` | `outputs/framewise/20s-framewise.mp4` |
| Motion-refined ROP | `outputs/cpu/10s-motion-refined.mp4` | `outputs/cpu/20s-motion-refined.mp4` |
| Three-video comparison | `outputs/comparison/10s-cpu-comparison.mp4` | `outputs/comparison/20s-cpu-comparison.mp4` |
| Optional original vs. motion-refined video (`--before-after`) | `outputs/cpu/10s-before-after.mp4` | `outputs/cpu/20s-before-after.mp4` |

The motion-refined script writes a same-named `.txt` report beside its restored MP4. The three-video comparison checks frame alignment and preserves the original audio when present. Its RGB differences include encoding effects.

To enable NVIDIA acceleration, install the CuPy package that matches your CUDA version and pass `--device gpu` to both dehazing scripts and the comparison command. CUDA accelerates dehazing, and NVENC encodes the restored streams. H.264 reference export and comparison encoding still use the CPU. GPU mode writes the motion-refined output to `outputs/gpu/` and uses `-gpu-comparison.mp4` for the comparison. Framewise ROP uses the same output filename for either device, so rerunning it replaces the previous result.

Motion refinement uses a grid at one-eighth resolution. The 960×540 example is edge-padded to 960×544 internally and cropped back at output. Parameter interpolation can use a later intra frame from the same scene.

## License

The project-authored code is available for noncommercial use under the [PolyForm Noncommercial License 1.0.0](LICENSE). See [NOTICE.md](NOTICE.md) for ROP attribution, the FFmpeg license, and the scope of the media license.
