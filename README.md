# Rank-One Prior Video Dehazing

Three dehazing entry points and a synchronized comparison tool. The video method
**Motion-Vector Propagation and Adaptive Block Refinement for Rank-One Prior
Video Dehazing** computes full Rank-One Prior (ROP) on intra-coded frames.
For other frames it warps the control-grid state using H.264 motion vectors and
the decoder's reference-picture identities. It refreshes unreliable blocks
using the current frame and restricts propagation across detected scene cuts.

| Script | Purpose |
| --- | --- |
| `rank_one_prior_image.py` | Single-image Rank-One Prior (Rank1). |
| `framewise_rop_video.py` | Full Rank1 independently on every frame. |
| `motion_refined_rop_video.py` | Motion-vector propagation with adaptive block refinement. |
| `compare_three_videos.py` | Original, framewise ROP, motion-refined ROP, and a processing information panel. |

## Setup

The complete video workflow targets Windows x64. Install Python 3.14 and use a
virtual environment if possible. For the first, second and fourth scripts,
install the Python requirements. For motion-refined ROP, also build the
bundled H.264 reference exporter once.

```bash
python -m pip install -r requirements.txt
bash native/build_decoder.sh
```

Run the build from Git Bash with a MinGW-w64 x64 toolchain (`gcc`,
`mingw32-make`) on `PATH`. The build reads the bundled FFmpeg 7.1.1
source archive and writes `native/native_decoder.exe`. Use a short checkout
path if the build tools encounter Windows path-length limits.

## Examples

| Input | Resolution | Frames | Frame rate |
| --- | --- | ---: | ---: |
| [10s.mp4](examples/10s.mp4) | 960 × 540 | 300 | 30 FPS |
| [20s.mp4](examples/20s.mp4) | 1920 × 1080 | 620 | 30000/1001 FPS |

The commands below run the 10-second example with CPU processing. Use `20s`
in the same commands to process the 20-second example.

```bash
python framewise_rop_video.py 10s
python motion_refined_rop_video.py 10s
python compare_three_videos.py 10s
```

Use an RGB input image for the single-image entry point.

```bash
python rank_one_prior_image.py input.png outputs/dehazed.png
```

The scripts write results under `outputs/`. The three-video comparison
preserves the original audio when present and checks that all three video
timelines match frame by frame. Its RGB differences include encoding effects.
To select the optional NVIDIA path, install the CuPy package that matches your
CUDA version and use `--device gpu` for the video scripts. GPU execution uses
CUDA for dehazing and NVENC for algorithm output encoding. Reference decoding
and export still run on the CPU. The comparison itself uses CPU encoding.

Motion refinement uses a grid at one-eighth resolution. The 960×540 example
is edge-padded internally to 960×544 and cropped back at output. Parameter
interpolation can use a later intra frame from the same scene.

## License

The project-authored code is shared under the
[PolyForm Noncommercial License 1.0.0](LICENSE) for noncommercial use.
See [NOTICE.md](NOTICE.md) for ROP attribution, the FFmpeg license and media
license scope.
