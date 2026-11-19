# Rank-One Prior 图像与视频去雾

[English](README.md) · **简体中文**

本仓库提供单图去雾、逐帧视频去雾、运动矢量传播与自适应块级修正视频去雾，以及三路视频同步对比。运动传播版在帧内编码帧上执行完整的 Rank-One Prior（ROP），利用 H.264 运动矢量和参考帧关系传播控制网格参数，并在不可靠的块上根据当前帧重新估计参数。方法在检测到的场景切换处限制跨场景传播。

| 脚本 | 功能 |
| --- | --- |
| `rank_one_prior_image.py` | 对单张图像执行 Rank-One Prior（Rank1）去雾。 |
| `framewise_rop_video.py` | 对视频的每一帧独立执行完整的 Rank1 去雾。 |
| `motion_refined_rop_video.py` | 基于运动矢量传播与自适应块级修正的视频去雾。 |
| `compare_three_videos.py` | 同步拼接原片和两种去雾结果，并显示处理信息。 |

## 安装

完整流程已在 Windows x64 和 Python 3.14 上验证。打开 Git Bash 并运行以下命令。

```bash
git clone https://github.com/Qiafengfei/rank-one-prior-video-dehazing.git
cd rank-one-prior-video-dehazing
python -m pip install -r requirements.txt
```

安装 Python 依赖后即可运行单图和逐帧去雾。运行运动传播版及完整对比流程前，需要先构建随仓库提供的 H.264 参考帧导出程序。安装 MinGW-w64 x64 工具链，确认 Git Bash 能找到 `gcc` 和 `mingw32-make`，然后在仓库根目录执行以下命令。

```bash
bash native/build_decoder.sh
```

脚本使用仓库内的 FFmpeg 7.1.1 源码包，生成 `native/native_decoder.exe`。如果 Windows 构建工具报路径过长，可将仓库放到路径较短的目录。

## 运行示例

| 随仓库提供的原片 | 分辨率 | 帧数 | 帧率 |
| --- | --- | ---: | ---: |
| [10s.mp4](examples/10s.mp4) | 960 × 540 | 300 | 30 FPS |
| [20s.mp4](examples/20s.mp4) | 1920 × 1080 | 620 | 30000/1001 FPS |

在仓库根目录依次运行以下命令。拼接脚本会读取前两条命令生成的去雾视频，以及运动传播版的 `.txt` 报告。处理另一个示例时，将三条命令中的 `10s` 都改为 `20s`。

```bash
python framewise_rop_video.py 10s
python motion_refined_rop_video.py 10s
python compare_three_videos.py 10s
```

视频脚本以仓库目录为基准查找示例和保存结果，克隆到其他目录也无需修改代码里的路径。运行单图去雾时，传入自己的输入和输出路径。

```bash
python rank_one_prior_image.py input.png outputs/dehazed.png
```

## 输出路径

以下路径均相对于仓库根目录。视频脚本默认使用 CPU 模式。

| 结果 | 10 秒示例 | 20 秒示例 |
| --- | --- | --- |
| 逐帧 ROP 去雾视频 | `outputs/framewise/10s-framewise.mp4` | `outputs/framewise/20s-framewise.mp4` |
| 运动传播版去雾视频 | `outputs/cpu/10s-motion-refined.mp4` | `outputs/cpu/20s-motion-refined.mp4` |
| 三路视频对比 | `outputs/comparison/10s-cpu-comparison.mp4` | `outputs/comparison/20s-cpu-comparison.mp4` |
| 可选的原片与运动传播版双路对比（`--before-after`） | `outputs/cpu/10s-before-after.mp4` | `outputs/cpu/20s-before-after.mp4` |

运动传播版在去雾视频旁生成同名 `.txt` 报告。三路对比会检查视频的逐帧对齐，并保留原片的音轨（如有）。对比报告中的 RGB 差异包含视频编码造成的差异。

使用 NVIDIA 加速时，安装与 CUDA 版本匹配的 CuPy，并给两条去雾命令和拼接命令加上 `--device gpu`。CUDA 加速去雾，NVENC 编码去雾成片，H.264 参考帧导出和拼接编码仍在 CPU 上执行。GPU 模式的运动传播版输出到 `outputs/gpu/`，拼接视频以 `-gpu-comparison.mp4` 结尾。逐帧 ROP 的两种设备模式共用输出文件名，重新运行会覆盖前一次的结果。

运动传播版使用原图八分之一分辨率的控制网格。960 × 540 示例在计算时沿边缘补齐到 960 × 544，输出时裁回原始尺寸。参数插值可能使用同一场景后续的帧内编码帧。

## 许可

项目原创代码采用 [PolyForm Noncommercial 1.0.0](LICENSE) 非商业许可。ROP 署名、FFmpeg 许可及示例视频的许可范围见 [NOTICE.md](NOTICE.md)。
