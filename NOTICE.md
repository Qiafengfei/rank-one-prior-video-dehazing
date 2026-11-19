# Attribution and license scope

The project-authored Python code, decoder bridge and build script are available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use of these
contributions requires separate permission from their rights holders.

The Rank-One Prior implementation is based on the work of Jun Liu, Ryan Wen
Liu, Jianing Sun and Tieyong Zeng. The original authors specify noncommercial
use for their code and algorithm and reference Chinese patent ZL 202011281893.3.
Any permissions for their work come from its original rights holders. See the
[upstream ROP repository](https://github.com/junliumath/ROP) and the
[CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Rank-One_Prior_Toward_Real-Time_Scene_Recovery_CVPR_2021_paper.html).

The H.264 reference exporter uses [FFmpeg 7.1.1](https://ffmpeg.org/) under
LGPL 2.1 or later. The archive, reference-export patch, bridge source and build
script are in [`native/`](native/), alongside the [LGPL text](native/COPYING.LGPLv2.1).
The FFmpeg archive and changes to FFmpeg are governed by the applicable FFmpeg
license terms. Python dependencies are distributed separately under their own terms.

The two original example clips are provided as reproducibility inputs.
The software license grants no rights in the example media.
