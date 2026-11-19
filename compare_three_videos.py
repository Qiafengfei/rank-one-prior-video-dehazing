"""Compose a synchronized comparison of the input, framewise ROP, and motion-refined ROP."""

from argparse import ArgumentParser
from pathlib import Path

import motion_refined_rop_video as pipeline

CELL_WIDTH = 960
CELL_HEIGHT = 540

from fractions import Fraction
from contextlib import ExitStack
from itertools import zip_longest
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import av


def video_info(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = stream.frames
        if not frames:
            frames = sum(1 for _ in container.decode(video=0))
        rate = Fraction(stream.average_rate)
        if frames <= 0 or rate <= 0:
            raise ValueError(f"无法读取有效帧数或帧率：{path}")
        return dict(width=stream.width, height=stream.height, frames=frames, rate=rate)


def read_statistics(path, source, info):
    """读取同一个视频的实测记录，禁止借用别的视频的速度。"""
    fields = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "：" in line:
            key, value = line.split("：", 1)
            fields[key.strip()] = value.strip()
    if Path(fields["输入视频"]).resolve() != source.resolve():
        raise ValueError(f"测速记录对应其他输入视频：{path}")
    size = [int(x) for x in re.findall(r"\d+", fields["视频分辨率"])]
    if size != [info["width"], info["height"]] or int(fields["视频帧数"]) != info["frames"]:
        raise ValueError(f"测速记录的尺寸/帧数与当前原片不同：{path}")
    fps = float(fields["每秒处理帧数"].split()[0])
    if fps <= 0:
        raise ValueError(f"无效的处理速度：{path}")
    return fields, fps


def pixel_differences(reference, experiment):
    """原分辨率逐帧对齐，计算全片RGB最大绝对差和平均绝对差。"""
    maximum = 0
    count = 0
    total_difference = 0
    total_values = 0
    with av.open(str(reference)) as left, av.open(str(experiment)) as right:
        left.streams.video[0].codec_context.thread_count = 1
        right.streams.video[0].codec_context.thread_count = 1
        for a, b in zip_longest(left.decode(video=0), right.decode(video=0)):
            if a is None or b is None:
                raise ValueError("逐帧ROP与实验视频帧数不同，不能计算最大差值")
            if (a.width, a.height) != (b.width, b.height):
                raise ValueError("逐帧ROP与实验视频分辨率不同")
            if a.pts * a.time_base != b.pts * b.time_base:
                raise ValueError("逐帧ROP与实验视频时间戳不同，不能比较错位帧")
            rgb_a = a.reformat(format="rgb24", threads=1).to_ndarray().astype(np.int16)
            rgb_b = b.reformat(format="rgb24", threads=1).to_ndarray().astype(np.int16)
            difference = np.abs(rgb_a - rgb_b)
            maximum = max(maximum, int(difference.max()))
            total_difference += int(difference.sum(dtype=np.uint64))
            total_values += difference.size
            count += 1
    if not count:
        raise ValueError("没有可比较的视频帧")
    return maximum, total_difference / total_values


def overlay_image(name, info, stats, fps, device, max_difference, mean_difference):
    """生成静态标题层与右上信息卡，保留在内存，不另存图片。"""
    w, h = CELL_WIDTH, CELL_HEIGHT
    scale = w / 960
    font = ImageFont.load_default(size=max(14, int(25 * scale)))
    title = ImageFont.load_default(size=max(18, int(31 * scale)))
    image = Image.new("RGB", (w * 2, h * 2), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    header = int(48 * scale)
    for x, y, label in [
        (0, 0, "Original"),
        (0, h, "Framewise ROP"),
        (w, h, "Motion-refined ROP (" + device.upper() + ")"),
    ]:
        draw.rectangle((x, y, x + w - 1, y + header - 1), fill=(20, 24, 30))
        draw.text((x + int(18 * scale), y + int(5 * scale)), label, font=title, fill="white")

    draw.rectangle((w, 0, w * 2 - 1, h - 1), fill=(240, 243, 247))
    x, y = w + int(28 * scale), int(18 * scale)
    draw.text((x, y), "Processing details  /  " + name, font=title, fill=(25, 35, 48))
    y += int(58 * scale)
    rows = [
        f"Resolution: {info['width']} x {info['height']}",
        f"Frames: {info['frames']}",
        f"Playback: {float(info['rate']):.3f} FPS",
        f"{device.upper()} processing: {fps:.2f} FPS",
        "CPU: " + stats["CPU"],
        "Encoder: " + stats["视频编码器"],
        f"Max absolute RGB difference: {max_difference} / 255",
        f"Mean absolute RGB difference: {mean_difference:.4f} / 255",
    ]
    if device == "gpu":
        rows.insert(5, "GPU: " + stats["GPU及驱动"])
    # 长设备名称自动换行，避免文字越过信息格边界。
    max_width = w - int(56 * scale)
    for text in rows:
        line = ""
        for char in text:
            if line and draw.textlength(line + char, font=font) > max_width:
                draw.text((x, y), line, font=font, fill=(30, 40, 52))
                y += int(36 * scale)
                line = ""
            line += char
        draw.text((x, y), line, font=font, fill=(30, 40, 52))
        y += int(39 * scale)
    if y > h - int(12 * scale):
        raise ValueError("设备信息超出右上信息格，请减小信息字体或增加格子尺寸")
    return np.asarray(image).copy()


def main(example, device):
    source = pipeline.ROOT / "examples" / f"{example}.mp4"
    framewise = pipeline.ROOT / "outputs" / "framewise" / f"{example}-framewise.mp4"
    experiment = pipeline.ROOT / "outputs" / device / f"{example}-motion-refined.mp4"
    report = experiment.with_suffix(".txt")
    for path in (source, framewise, experiment, report):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Generate the dehazed videos first.")
    info = video_info(source)
    for path in (framewise, experiment):
        if video_info(path) != info:
            raise ValueError(f"尺寸、帧数或帧率与原片不一致，不能直接同步拼接：{path}")
    stats, fps = read_statistics(report, source, info)
    maximum, mean = pixel_differences(framewise, experiment)
    panel = overlay_image(source.name, info, stats, fps, device, maximum, mean)
    out = pipeline.ROOT / "outputs" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{source.stem}-{device}-comparison.mp4"
    w, h = CELL_WIDTH, CELL_HEIGHT
    header = int(48 * w / 960)
    print(f"正在生成：{target}\n右下实验成片：{experiment}", flush=True)
    with ExitStack() as stack:
        inputs = [stack.enter_context(av.open(str(path))) for path in (source, framewise, experiment)]
        readers = []
        for container in inputs:
            container.streams.video[0].codec_context.thread_count = 1
            readers.append(iter(container.decode(video=0)))
        output = stack.enter_context(av.open(str(target), "w", options={"movflags": "+faststart"}))
        video = output.add_stream("libx264", rate=info["rate"])
        video.width, video.height = w * 2, h * 2
        video.pix_fmt = "yuv420p"
        video.time_base = inputs[0].streams.video[0].time_base
        video.codec_context.thread_count = 8
        video.codec_context.colorspace = 1
        video.codec_context.color_range = 1
        video.options = {"preset": "ultrafast", "crf": "18", "bf": "0"}
        audio_packets = iter(())
        audio_output = None
        if inputs[0].streams.audio:
            audio_source = stack.enter_context(av.open(str(source)))
            audio_stream = audio_source.streams.audio[0]
            audio_output = output.add_stream_from_template(audio_stream)
            audio_packets = (p for p in audio_source.demux(audio_stream) if p.dts is not None)
        audio = next(audio_packets, None)
        fit = min(w / info["width"], (h - header) / info["height"])
        fw, fh = int(info["width"] * fit), int(info["height"] * fit)
        for index in range(info["frames"]):
            frames = [next(reader, None) for reader in readers]
            if any(frame is None for frame in frames):
                raise ValueError("输入成片提前结束，不能生成完整同步对比")
            timestamp = frames[0].pts * frames[0].time_base
            if any(frame.pts * frame.time_base != timestamp for frame in frames[1:]):
                raise ValueError(f"第{index}帧时间戳不同，停止拼接，避免画面错位")
            canvas = panel.copy()
            for frame, (x, y) in zip(frames, ((0, 0), (0, h), (w, h))):
                tile = frame.reformat(
                    width=fw, height=fh, format="rgb24", interpolation="LANCZOS", threads=1
                ).to_ndarray()
                x += (w - fw) // 2
                y += header + (h - header - fh) // 2
                canvas[y : y + fh, x : x + fw] = tile
            encoded = av.VideoFrame.from_ndarray(canvas, format="rgb24")
            encoded = encoded.reformat(format="yuv420p", dst_colorspace="ITU709", threads=1)
            encoded.colorspace = 1
            encoded.color_range = 1
            encoded.pts, encoded.time_base = frames[0].pts, frames[0].time_base
            for packet in video.encode(encoded):
                output.mux(packet)
            # 只复制原片的一条音轨，不混入其他两个成片的声音。
            while audio is not None and audio.dts * audio.time_base <= timestamp:
                audio.stream = audio_output
                output.mux(audio)
                audio = next(audio_packets, None)
            if (index + 1) % 100 == 0:
                print(f"拼接进度：{index + 1}/{info['frames']}", flush=True)
        if any(next(reader, None) is not None for reader in readers):
            raise ValueError("输入帧数比记录更多，停止输出")
        for packet in video.encode():
            output.mux(packet)
        while audio is not None:
            audio.stream = audio_output
            output.mux(audio)
            audio = next(audio_packets, None)
    expected = dict(info, width=w * 2, height=h * 2)
    if video_info(target) != expected:
        raise RuntimeError("输出尺寸、帧数或帧率不符合预期")
    audio_note = "保留原片音轨" if audio_output is not None else "原片无音轨"
    print(f"完成：{target}\n{w*2} × {h*2}，{info['frames']}帧；{audio_note}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=pipeline.EXAMPLES)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    main(args.example, args.device)
