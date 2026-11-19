"""Motion-vector propagation and adaptive block refinement for Rank-One Prior video dehazing."""

from argparse import ArgumentParser
from pathlib import Path

EXAMPLES = ("10s", "20s")
DEVICE = "cpu"

import os

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from fractions import Fraction
from time import perf_counter
from collections import deque
import platform
import sys
import struct
import subprocess
import tempfile
import rank_one_prior_image as rop
import numpy as np
import cv2
import av
from numba import config, njit, set_num_threads
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / "native" / "native_decoder.exe"
SCALE = 8
WORKERS = 4
PIPELINE = True  # CPU/GPU共用的有界编解码流水线。
CPU_RECOVERY_THREADS = 8
CPU_ENCODER_THREADS = 8
cv2.setNumThreads(1)
set_num_threads(min(CPU_RECOVERY_THREADS, config.NUMBA_NUM_THREADS))


def timer(start=None):
    """不传参数时记录起点，传入起点时返回已用秒数。"""
    if start is None:
        return perf_counter()
    return perf_counter() - start


def decoder(path):
    container = av.open(str(path))
    container.streams.video[0].codec_context.thread_count = 1
    return container


class Writer:
    """编码器创建、逐帧编码、刷新与关闭均属于正式处理耗时。"""

    def __init__(self, path, width, height, rate, time_base):
        self.container = av.open(str(path), "w")
        try:
            codec = "libx264"
            if DEVICE == "gpu":
                codec = "h264_nvenc"
                if width > 4096:
                    codec = "hevc_nvenc"
            self.stream = self.container.add_stream(codec, rate=rate)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.time_base = time_base
            self.stream.codec_context.thread_count = CPU_ENCODER_THREADS if DEVICE == "cpu" else 8
            self.stream.codec_context.colorspace = 1
            self.stream.codec_context.color_range = 1
            if codec.endswith("nvenc"):
                self.stream.options = {
                    "preset": "p1",
                    "tune": "ll",
                    "cq": "18",
                    "bf": "0",
                    "rc": "vbr",
                }
            else:
                self.stream.options = {"preset": "ultrafast", "crf": "18", "bf": "0"}
            self.container.start_encoding()
        except BaseException:
            self.container.close()
            raise

    def write(self, rgb, pts, time_base):
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame = frame.reformat(format="yuv420p", dst_colorspace="ITU709", threads=1)
        frame.colorspace = 1
        frame.color_range = 1
        frame.pts = pts
        frame.time_base = time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


class AsyncWriter(Writer):
    """保留有界流水线，用Future统一管理编码任务、异常与线程回收。"""

    def __init__(self, *args):
        super().__init__(*args)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = deque()
        self.closed = False
        self.error = None

    def write(self, rgb, pts, time_base):
        if self.closed:
            raise RuntimeError("编码器已经关闭")
        if self.error is not None:
            raise self.error
        if len(self.pending) >= 4:
            self.pending[0].result()
            self.pending.popleft()
        self.pending.append(self.pool.submit(self.encode, rgb, pts, time_base))

    def encode(self, *frame):
        if self.error is None:
            try:
                super().write(*frame)
            except BaseException as error:
                self.error = error
                raise

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                for task in self.pending:
                    task.result()
            except BaseException as error:
                self.error = error
            finally:
                self.pool.shutdown(wait=True, cancel_futures=True)
                try:
                    super().close()
                except BaseException as error:
                    self.error = error
        if self.error is not None:
            raise self.error


def prefetch(source):
    """只预取下一帧；标准库负责线程回收和异常传递。"""
    context = nullcontext(source)
    if hasattr(source, "close"):
        context = closing(source)
    end = object()
    with context, ThreadPoolExecutor(max_workers=1) as pool:
        iterator = iter(source)
        pending = pool.submit(next, iterator, end)
        while True:
            frame = pending.result()
            if frame is end:
                break
            pending = pool.submit(next, iterator, end)
            yield frame


def read_exact(pipe, size, allow_eof=False):
    data = bytearray()
    while len(data) < size:
        chunk = pipe.read(size - len(data))
        if not chunk:
            if allow_eof and not data:
                return None
            raise EOFError("真实参考导出中断，不能继续处理")
        data.extend(chunk)
    return bytes(data)


def export_frames(source, width, height, time_base, rate):
    """从原码流导出解码顺序、POC、每个参考列表的真实运动和参考编号。"""
    env = os.environ.copy()
    env.pop("STAGE3_GOP", None)
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(
            [str(NATIVE), str(source), "2"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
            env=env,
        )
        try:
            while True:
                header = read_exact(proc.stdout, 40, True)
                if header is None:
                    break
                event, w, h, poc, kind, number, nal, ref, pts = struct.unpack("<8iq", header)
                if event != 3 or (w, h) != (width, height):
                    raise ValueError("解码器输出尺寸或协议不匹配")
                yuv = np.frombuffer(read_exact(proc.stdout, w * h * 3 // 2), np.uint8).reshape(
                    h * 3 // 2, w
                )
                gh, gw = (h + 7) // 8, (w + 7) // 8
                grid = np.frombuffer(read_exact(proc.stdout, 2 * gh * gw * 16), "<i4").reshape(
                    2, gh, gw, 4
                )
                if h % SCALE or w % SCALE:
                    # 边缘不足8像素的格子用当前帧局部ROP更新，不使用隐藏像素的运动。
                    grid = grid.copy()
                    if h % SCALE:
                        grid[:, -1, :, :] = np.iinfo(np.int32).min
                    if w % SCALE:
                        grid[:, :, -1, :] = np.iinfo(np.int32).min
                index = pts * time_base * rate
                if index.denominator != 1:
                    raise ValueError("当前程序要求从0开始的恒定帧率输入")
                yield int(index), pts, poc, kind, nal, ref, yuv, grid
            if proc.wait() != 0:
                log.seek(0)
                raise RuntimeError(log.read().decode(errors="replace"))
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()


def pad_to_grid(rgb):
    """只向右/下复制边缘像素补齐控制网格，不缩放图像；编码前裁回原尺寸。"""
    height, width = rgb.shape[:2]
    bottom = (-height) % SCALE
    right = (-width) % SCALE
    if bottom == 0 and right == 0:
        return rgb
    return np.pad(rgb, ((0, bottom), (0, right), (0, 0)), mode="edge")


def pack(result, rgb):
    """保存 C、E、透射率和当前灰度，不缓存处理图像作为运动参考。"""
    z = result.arrays
    h, w = rgb.shape[:2]
    fields = [
        z["scatter_coefficient"][..., None],
        z["scatter"] - z["scatter_raw"],
        z["transmission"],
    ]
    fields = [cv2.resize(x, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA) for x in fields]
    small = cv2.resize(rgb, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA)
    gray = small.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255
    return np.concatenate(
        (fields[0][..., None], fields[1], fields[2], gray[..., None]), axis=-1
    ).astype(np.float32)


def prepare(source, rate, time_base, engine=None):
    """共用解码、场景编号和I帧调度；GPU计算始终在当前线程执行。"""

    def seed(index, frame, scene):
        rgb = pad_to_grid(frame.reformat(format="rgb24", threads=1).to_ndarray())
        result = rop.rank1(rgb, diagnostics=False)
        z = result.arrays
        return dict(
            index=index,
            scene=scene,
            state=pack(result, rgb),
            image=result.image,
            s=z["spectrum_l1"].reshape(3),
            a=z["atmosphere"].reshape(3),
            threshold=float(z["atmosphere_threshold"]),
        )

    scenes, jobs = [], []
    previous = None
    scene = 0
    cp = None
    if engine is not None:
        cp = engine.cp
    with decoder(source) as container, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts * time_base != Fraction(index) / rate:
                raise ValueError("当前程序要求从0开始的恒定帧率输入")
            if engine is None:
                small = (
                    frame.reformat(width=64, height=36, format="rgb24", threads=1)
                    .to_ndarray()
                    .astype(np.float32)
                    / 255
                )
            else:
                rgb = pad_to_grid(frame.reformat(format="rgb24", threads=1).to_ndarray())
                small = engine.resize(cp.asarray(rgb), (36, 64)).astype(cp.float32) / 255
            if previous is not None:
                change = abs(small - previous).mean()
                if engine is None:
                    cut = float(change) > 0.22
                else:
                    cut = bool((change > 0.22).get())
                if cut:
                    scene += 1
            scenes.append(scene)
            previous = small
            if int(frame.pict_type) == 1:
                if engine is None:
                    jobs.append(pool.submit(seed, index, frame, scene))
                else:
                    entry = engine.seed(rgb)
                    entry.update(index=index, scene=scene)
                    jobs.append(entry)
        if engine is None:
            seeds = [job.result() for job in jobs]
        else:
            seeds = jobs
    if not seeds or seeds[0]["index"] != 0:
        raise ValueError("输入必须从可完整解码的I帧开始")
    return seeds, scenes


def parameters(index, seeds, scenes):
    same = [s for s in seeds if s["scene"] == scenes[index]]
    fallback = not same
    if fallback:
        same = [max((s for s in seeds if s["index"] <= index), key=lambda s: s["index"])]
    position = max(index, same[0]["index"])
    left = max((s for s in same if s["index"] <= position), key=lambda s: s["index"])
    right = next((s for s in same if s["index"] > position), left)
    alpha = 0.0
    if right is not left:
        alpha = (position - left["index"]) / (right["index"] - left["index"])
    return (
        tuple((1 - alpha) * left[k] + alpha * right[k] for k in ("s", "a", "threshold")),
        fallback,
    )


def gpu_step(engine, rgb, grid, refs, fallback, s, a, threshold, index, scene=0, force_local=False):
    """在GPU上执行运动传播与块级修正传播与局部修正，调用ROP模块的缩放和恢复。"""
    cp = engine.cp
    gpu = cp.asarray(rgb)
    small, gray = engine.small_fields(gpu)
    shape = (2, engine.hs, engine.ws)
    z = cp.empty((*shape, 8), cp.float32)
    valid = cp.empty(shape, cp.uint8)
    outside = cp.empty_like(valid)
    same = cp.empty_like(valid)
    x = cp.empty(shape, cp.float32)
    y = cp.empty_like(x)
    rmap = cp.asarray(np.asarray(refs, dtype=np.int32).reshape(-1, 3))
    gmap = cp.asarray(grid)
    n = 2 * engine.hs * engine.ws
    engine.warp_kernel(
        ((n + 255) // 256,),
        (256,),
        (
            engine.pool,
            gmap,
            rmap,
            len(refs),
            z,
            valid,
            outside,
            same,
            x,
            y,
            engine.hs,
            engine.ws,
            scene,
        ),
    )
    ce, initial = engine.local_fields(
        z, valid, outside, small, gray, fallback, s, threshold, force_local
    )
    tiny = engine.resize(initial, engine.tshape, (engine.full_scale * 8,) * 2)
    control = engine.linear(tiny, (engine.hs, engine.ws)).astype(cp.float32)
    current = cp.concatenate((ce, control, gray[..., None]), axis=-1)
    engine.pool[index] = current
    return engine.render(gpu, tiny, a)


def gpu_predict(rgb, grid, cache, pts, params, force_local, index, scene, engine):
    """CPU只整理参考编号；所有图像、运动采样和运动传播与块级修正计算交给GPU。"""
    # 参考编号来自CPU解码器，直接整理元数据，避免GPU排序后再同步下载。
    refs = []
    for field in grid:
        valid = (field[..., 0] >= 0) & (field[..., 3] != np.iinfo(np.int32).min)
        for poc in np.unique(field[..., 3][valid]):
            record = cache[int(poc)]
            refs.append((int(poc), record["index"], record["scene"]))
    if refs:
        fallback_poc = min(
            (r[0] for r in refs),
            key=lambda k: (cache[k]["pts"] > pts, abs(cache[k]["pts"] - pts), k),
        )
    else:
        fallback_poc = max(
            (k for k in cache if cache[k]["pts"] <= pts), key=lambda k: cache[k]["pts"]
        )
    s, a, threshold = params
    return gpu_step(
        engine,
        rgb,
        grid,
        refs,
        cache[fallback_poc]["index"],
        s,
        a,
        threshold,
        index,
        scene,
        force_local=force_local,
    )


@lru_cache(maxsize=8)
def control_coordinates(h, w):
    """传播网格只依赖视频尺寸；只读坐标可安全跨帧复用。"""
    yy, xx = np.indices((h, w), dtype=np.float32)
    yy.flags.writeable = False
    xx.flags.writeable = False
    return yy, xx


def warp(grid, cache, pts):
    """按真实 POC 分别采样两张参考列表，保留原双向融合规则。"""
    _, h, w, _ = grid.shape
    yy, xx = control_coordinates(h, w)
    predictions = []
    referenced = set()
    for field in grid:
        available = field[..., 0] >= 0
        valid = available & (field[..., 3] != np.iinfo(np.int32).min)
        if not valid.any():
            continue
        x = xx + np.where(available, field[..., 1], 0).astype(np.float32) / (4 * SCALE)
        y = yy + np.where(available, field[..., 2], 0).astype(np.float32) / (4 * SCALE)
        pocs = np.unique(field[..., 3][valid])
        # 把真实参考状态并排放入带复制边界的图集，每张参考列表只采样一次。
        tile_height = h + 2
        atlas = np.empty((len(pocs) * tile_height, w + 2, 8), np.float32)
        offsets = np.zeros((h, w), np.float32)
        for tile, poc in enumerate(pocs):
            referenced.add(int(poc))
            selected = valid & (field[..., 3] == poc)
            cv2.copyMakeBorder(
                cache[int(poc)]["state"],
                1,
                1,
                1,
                1,
                cv2.BORDER_REPLICATE,
                dst=atlas[tile * tile_height : (tile + 1) * tile_height],
            )
            offsets[selected] = tile * tile_height
        # 限制整数地址、保留小数权重，避免越界采样跨到相邻图块。
        ix, iy = np.floor(x), np.floor(y)
        map_x = np.clip(ix, -1, w - 1) + (x - ix) + 1
        map_y = np.clip(iy, -1, h - 1) + (y - iy) + 1 + offsets
        warped = cv2.remap(atlas, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        outside = valid & ((x < 0) | (x > w - 1) | (y < 0) | (y > h - 1))
        predictions.append((warped, valid, outside))
    if referenced:
        fallback_key = min(
            referenced, key=lambda k: (cache[k]["pts"] > pts, abs(cache[k]["pts"] - pts), k * 2)
        )
    else:
        past = [k for k in cache if cache[k]["pts"] <= pts]
        fallback_key = max(past, key=lambda k: cache[k]["pts"])
    fallback = cache[fallback_key]["state"]
    if not predictions:
        predictions.append((fallback, np.zeros((h, w), bool), np.zeros((h, w), bool)))
    return predictions, fallback


@njit(cache=True, nogil=True)
def _control_update(small, gray, predictions, fallback, s, basis, threshold, force_local):
    """融合加权传播与局部ROP；按预测元组长度特化单、双参考内核。"""
    first, valid0, edge0 = predictions[0]
    second, valid1, edge1 = predictions[-1]
    has_second = len(predictions) == 2
    h, w = gray.shape
    c = np.empty((h, w), np.float32)
    e = np.empty((h, w, 3), np.float32)
    for y in range(h):
        for x in range(w):
            weight0 = np.float32(valid0[y, x]) / (
                np.float32(0.01) + abs(gray[y, x] - first[y, x, 7])
            )
            weight1 = np.float32(0)
            if has_second:
                weight1 = np.float32(valid1[y, x]) / (
                    np.float32(0.01) + abs(gray[y, x] - second[y, x, 7])
                )
            total = weight0 + weight1
            prior_gray = fallback[y, x, 7]
            if total > 0:
                value = first[y, x, 7] * weight0
                if has_second:
                    value = value + second[y, x, 7] * weight1
                prior_gray = value / total
            bad = total <= 0 or edge0[y, x] or abs(gray[y, x] - prior_gray) > np.float32(0.06)
            if has_second:
                bad = (
                    bad
                    or edge1[y, x]
                    or (
                        valid0[y, x]
                        and valid1[y, x]
                        and abs(first[y, x, 0] - second[y, x, 0]) > np.float32(0.15)
                    )
                )
            if force_local:
                bad = True
            if bad:
                r = small[y, x, 0] / 255.0
                g = small[y, x, 1] / 255.0
                b = small[y, x, 2] / 255.0
                cosine = ((r * basis[0] + g * basis[1]) + b * basis[2]) / max(
                    np.sqrt((r * r + g * g) + b * b), 0.001
                )
                value = cosine * ((r + g) + b)
                c[y, x] = value
                raw0, raw1, raw2 = value * s[0], value * s[1], value * s[2]
                reflected = (raw0 + raw1) + raw2 > threshold
                for k in range(3):
                    raw = value * s[k]
                    adjusted = raw
                    if reflected:
                        adjusted = 2 / 3 * threshold - raw
                    e[y, x, k] = adjusted - raw
            else:
                for k in range(4):
                    value = fallback[y, x, k]
                    if total > 0:
                        value = first[y, x, k] * weight0
                        if has_second:
                            value = value + second[y, x, k] * weight1
                        value = value / total
                    if k == 0:
                        c[y, x] = value
                    else:
                        e[y, x, k - 1] = value
    return c, e


def control_parameters(small, gray, predictions, fallback, s, threshold, force_local):
    basis = s / max(float(np.linalg.norm(s)), 0.001)
    return _control_update(
        small, gray, tuple(predictions), fallback, s, basis, threshold, force_local
    )


def predict(rgb, grid, cache, pts, params, force_local, recovery):
    s, a, threshold = params
    H, W = rgb.shape[:2]
    h, w = H // SCALE, W // SCALE
    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    gray = small.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255
    predictions, fallback = warp(grid, cache, pts)
    c, e = control_parameters(small, gray, predictions, fallback, s, threshold, force_local)
    scatter = c[..., None] * s + e
    scale = 0.02 if min(H, W) < 800 else 0.01 if min(H, W) < 1500 else 0.005
    shape = tuple(max(1, int(np.ceil(n * scale))) for n in (H, W))
    tiny = rop.resize_matlab(1 - 0.8 * scatter, shape, (scale * SCALE,) * 2)
    if not np.isfinite(tiny).all() or tiny.min() <= 0.001:
        raise ValueError("透射率无效，不能继续编码")
    transmission = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_LINEAR)
    image, _ = rop.render(rgb, tiny, a, recovery=recovery)
    state = np.concatenate((c[..., None], e, transmission, gray[..., None]), axis=-1).astype(
        np.float32
    )
    return state, image


def hardware():
    cpu = platform.processor()
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    gpu = "未使用；不初始化CUDA，不使用GPU编码"
    if DEVICE == "gpu":
        try:
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            gpu = "无法查询GPU型号"
    return cpu, gpu


def runtime_information():
    """记录真正运行的解释器、数值库与调试状态，便于复现性能。"""
    import numba

    tools = []
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is not None:
        tools = [monitoring.get_tool(i) for i in range(6) if monitoring.get_tool(i)]
    return dict(
        executable=sys.executable,
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba.__version__,
        opencv=cv2.__version__,
        pyav=av.__version__,
        numba_layer=numba.threading_layer(),
        numba_threads=numba.get_num_threads(),
        trace=bool(sys.gettrace()),
        profile=bool(sys.getprofile()),
        monitoring=tools,
        rop_file=rop.__file__,
    )


def configure_cpu():
    """沿用本项目设置：仅用性能核，每个物理核选一个硬件线程。"""
    if os.name != "nt":
        return "使用系统调度"
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    query = kernel.GetSystemCpuSetInformation
    query.argtypes = [
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
        wintypes.HANDLE,
        wintypes.ULONG,
    ]
    size = wintypes.ULONG()
    query(None, 0, ctypes.byref(size), None, 0)
    buffer = ctypes.create_string_buffer(size.value)
    if not query(buffer, size.value, ctypes.byref(size), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    rows = []
    offset = 0
    while offset < size.value:
        raw = buffer.raw
        length, kind = struct.unpack_from("<II", raw, offset)
        if length < 32 or offset + length > size.value:
            raise ValueError("CPU拓扑信息不完整")
        if kind == 0:
            group = int.from_bytes(raw[offset + 12 : offset + 14], "little")
            if group != 0:
                raise ValueError("当前简洁版本不支持多个CPU处理器组")
            rows.append((raw[offset + 14], raw[offset + 15], raw[offset + 18]))
        offset += length
    best = max(row[2] for row in rows)
    cores = {}
    for logical, core, efficiency in rows:
        if efficiency == best:
            cores.setdefault(core, logical)
    selected = sorted(cores.values())
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), sum(1 << i for i in selected)):
        raise ctypes.WinError(ctypes.get_last_error())
    set_num_threads(min(CPU_RECOVERY_THREADS, len(selected), config.NUMBA_NUM_THREADS))
    return selected


@contextmanager
def compilation_check(engine):
    """正式处理时若请求新CUDA模块，拒绝将该轮当作有效测速。"""
    requests = []
    if engine is None:
        yield requests
        return
    import cupy.cuda.compiler as compiler

    original = compiler._compile_module_with_cache

    def capture(*args, **kwargs):
        requests.append(str(args[0])[:100])
        return original(*args, **kwargs)

    compiler._compile_module_with_cache = capture
    try:
        yield requests
    finally:
        compiler._compile_module_with_cache = original


def warmup(engine, width, height, output_folder=None):
    """仅使用合成数据完成 JIT、数值库和编解码库预热。"""
    rng = np.random.default_rng(177)
    small = rng.integers(0, 256, (96, 128, 3), np.uint8)
    rgb = np.zeros((height, width, 3), np.uint8)
    if engine is None:
        rop.rank1(small)
        state = np.zeros((12, 16, 8), np.float32)
        valid = np.ones((12, 16), bool)
        outside = np.zeros_like(valid)
        control_parameters(
            small[:12, :16].copy(),
            np.zeros((12, 16), np.float32),
            [(state, valid, outside)],
            state,
            np.full(3, 1 / 3),
            1.0,
            True,
        )
        control_parameters(
            small[:12, :16].copy(),
            np.zeros((12, 16), np.float32),
            [(state, valid, outside)] * 2,
            state,
            np.full(3, 1 / 3),
            1.0,
            True,
        )
        rop.render(rgb, np.full((2, 3, 3), 0.7), np.full(3, 0.7))
        # MATLAB缩放结果布局不同，必须也在计时前编译对应签名。
        control = np.full((height // SCALE, width // SCALE, 3), 0.7)
        tiny = rop.resize_matlab(control, (11, 20))
        rop.render(rgb, tiny, np.full(3, 0.7))
    else:
        cp = engine.cp
        seed = engine.seed(rgb)
        engine.pool[0] = seed["state"]
        engine.pool[1] = seed["state"]
        grid = np.zeros((2, height // 8, width // 8, 4), np.int32)
        grid[0, ..., 3] = 4
        grid[1, ..., 3] = 8
        gpu_step(
            engine, rgb, grid, [(4, 0, 0), (8, 1, 0)], 0, seed["s"], seed["a"], seed["threshold"], 2
        )
        gpu_step(
            engine,
            rgb,
            grid,
            [(4, 0, 0), (8, 1, 0)],
            0,
            seed["s"],
            seed["a"],
            seed["threshold"],
            2,
            force_local=True,
        )
        cache = {4: dict(index=0, pts=0, scene=0), 8: dict(index=1, pts=4, scene=0)}
        gpu_predict(
            rgb, grid, cache, 2, (seed["s"], seed["a"], seed["threshold"]), False, 2, 0, engine
        )
        engine.resize(cp.asarray(rgb), (36, 64)).astype(cp.float32) / 255
        scene = cp.zeros((36, 64, 3), cp.float32)
        _ = (cp.abs(scene - scene).mean() > 0.22).get()
        for key in ("s", "a", "threshold"):
            _ = 0.3 * seed[key] + 0.7 * seed[key]
        cp.cuda.Stream.null.synchronize()
    folder = ROOT / ".warmup" if output_folder is None else Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (DEVICE + ".mp4")
    writer = Writer(path, width, height, Fraction(30), Fraction(1, 15360))
    try:
        writer.write(rgb, 0, Fraction(1, 15360))
    finally:
        writer.close()
    with decoder(path) as container:
        for frame in container.decode(video=0):
            frame.to_ndarray(format="rgb24")


def compare(source, processed, target, rate, time_base, width, height, count):
    """算法计时结束后生成左右同步对比，不改变原始播放帧率。"""
    with decoder(source) as left, decoder(processed) as right:
        original = iter(left.decode(video=0))
        result = iter(right.decode(video=0))
        writer = Writer(target, width * 2, height, rate, time_base)
        try:
            for i in range(count):
                a = next(original)
                b = next(result)
                if a.pts * a.time_base != b.pts * b.time_base:
                    raise ValueError("对比画面时间戳不一致")
                images = [
                    a.reformat(format="rgb24", threads=1).to_ndarray(),
                    b.reformat(format="rgb24", threads=1).to_ndarray(),
                ]
                canvas = np.concatenate(images, axis=1)
                for x, label in [(16, "Original"), (width + 16, "运动传播与块级修正 " + DEVICE.upper())]:
                    cv2.putText(
                        canvas, label, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                    )
                writer.write(canvas, a.pts, a.time_base)
            if next(original, None) is not None or next(result, None) is not None:
                raise ValueError("视频帧数不一致")
        finally:
            writer.close()


def main(example, create_comparison=False):
    setup_started = perf_counter()
    source = ROOT / "examples" / f"{example}.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not NATIVE.is_file():
        raise FileNotFoundError(f"{NATIVE}: build the decoder with bash native/build_decoder.sh")
    out = ROOT / "outputs" / DEVICE
    out.mkdir(parents=True, exist_ok=True)
    processed = out / f"{source.stem}-motion-refined.mp4"
    comparison = out / f"{source.stem}-before-after.mp4"
    cpu, gpu = hardware()
    cpu_affinity = configure_cpu()
    # 计时前只查询尺寸用于准备合成缓冲区；真实内容及参数不提前读取。
    with decoder(source) as container:
        width = container.streams.video[0].width
        height = container.streams.video[0].height
        capacity = container.streams.video[0].frames
    working_width = (width + SCALE - 1) // SCALE * SCALE
    working_height = (height + SCALE - 1) // SCALE * SCALE
    engine = None
    if DEVICE == "gpu":
        engine = rop.create_gpu_engine(working_height, working_width, max(3, capacity))
    warmup(engine, working_width, working_height, out / ".warmup")
    compiled = (tuple(rop.pixel_recover.signatures), tuple(_control_update.signatures))
    setup_seconds = perf_counter() - setup_started

    stages = dict(prepass=0.0, prediction=0.0, rgb_bridge=0.0, submit_wait=0.0, close=0.0)
    # t1之前已完成环境和合成预热；从这里起才读取真实视频及计算参数。
    with compilation_check(engine) as new_modules:
        started = timer()
        with decoder(source) as container:
            stream = container.streams.video[0]
            width, height = stream.width, stream.height
            rate, time_base = Fraction(stream.average_rate), Fraction(stream.time_base)
            if (
                stream.codec_context.name != "h264"
                or stream.codec_context.format.name != "yuv420p"
                or width % 2
                or height % 2
            ):
                raise ValueError(
                    f"需要偶数宽高的H.264 YUV420P输入，当前为"
                    f"{stream.codec_context.name}/{stream.codec_context.format.name}，"
                    f"{width}×{height}"
                )
            matrix = int(stream.codec_context.colorspace)
            value_range = int(stream.codec_context.color_range)
        stage_started = perf_counter()
        seeds, scenes = prepare(source, rate, time_base, engine)
        stages["prepass"] = perf_counter() - stage_started
        seedmap = {s["index"]: s for s in seeds}
        cache = {}
        pending = {}
        seen = set()
        count = 0
        use_pipeline = PIPELINE
        writer_type = AsyncWriter if use_pipeline else Writer
        writer = writer_type(processed, width, height, rate, time_base)
        try:
            frame_source = export_frames(source, width, height, time_base, rate)
            if use_pipeline:
                frame_source = prefetch(frame_source)
            with closing(frame_source) as frames:
                for index, pts, poc, kind, nal, ref, yuv, grid in frames:
                    if index in seen or not 0 <= index < len(scenes):
                        raise ValueError("帧序号重复或超出预处理范围")
                    seen.add(index)
                    if nal == 5:
                        cache.clear()
                    stage_started = perf_counter()
                    frame = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
                    frame.colorspace = 1 if matrix == 1 else 5
                    frame.color_range = 2 if value_range == 2 else 1
                    rgb = pad_to_grid(
                        frame.reformat(
                            format="rgb24",
                            src_colorspace="ITU709" if matrix == 1 else "ITU601",
                            threads=1,
                        ).to_ndarray()
                    )
                    stages["rgb_bridge"] += perf_counter() - stage_started
                    if kind == 1:
                        seed = seedmap[index]
                        state, image = seed["state"], seed["image"]
                        if engine is not None:
                            engine.pool[index] = state
                    else:
                        stage_started = perf_counter()
                        params, fallback = parameters(index, seeds, scenes)
                        cut = index > 0 and scenes[index] != scenes[index - 1]
                        if engine is None:
                            state, image = predict(
                                rgb, grid, cache, pts, params, fallback or cut, rop.pixel_recover
                            )
                        else:
                            image = gpu_predict(
                                rgb,
                                grid,
                                cache,
                                pts,
                                params,
                                fallback or cut,
                                index,
                                scenes[index],
                                engine,
                            )
                            state = None
                        stages["prediction"] += perf_counter() - stage_started
                    if ref:
                        cache[poc] = dict(state=state, pts=pts, index=index, scene=scenes[index])
                    pending[index] = (image, pts)
                    while count in pending:
                        image, display_pts = pending.pop(count)
                        stage_started = perf_counter()
                        writer.write(
                            np.ascontiguousarray(image[:height, :width]), display_pts, time_base
                        )
                        stages["submit_wait"] += perf_counter() - stage_started
                        count += 1
            if pending or count != len(scenes):
                raise ValueError("输出帧数与实际输入不一致")
        finally:
            stage_started = perf_counter()
            writer.close()
            stages["close"] = perf_counter() - stage_started
        elapsed = timer(started)  # t2：MP4已经刷新、封装并成功关闭。
    if (tuple(rop.pixel_recover.signatures), tuple(_control_update.signatures)) != compiled:
        raise RuntimeError("计时内出现新的JIT编译，本次测速无效，需补充合成预热")

    if new_modules:
        raise RuntimeError(
            f"计时内请求了{len(new_modules)}个新CUDA模块，本次测速无效，需补充合成预热"
        )

    compare_started = timer()
    compare_seconds = 0.0
    if create_comparison:
        compare(source, processed, comparison, rate, time_base, width, height, count)
        compare_seconds = timer(compare_started)
    runtime = runtime_information()
    lines = [
        f"输入视频：{source}",
        f"视频分辨率：{width} × {height}",
        f"控制网格内部尺寸：{working_width} × {working_height}（边缘复制补齐，输出裁回原尺寸）",
        f"视频帧数：{count}",
        f"源视频帧率：{float(rate):.6f} FPS",
        f"运行时间：{elapsed:.6f} 秒",
        f"准备及合成预热耗时（不计入处理FPS）：{setup_seconds:.6f} 秒",
        f"每秒处理帧数：{count / elapsed:.6f} FPS",
        "主线程分项秒数：" + "；".join(f"{k}={v:.6f}" for k, v in stages.items()),
        f"运算模式：{DEVICE}",
        f"CPU：{cpu}",
        f"Python解释器：{runtime['executable']}；Python版本：{runtime['python']}",
        f"库版本：NumPy {runtime['numpy']}；Numba {runtime['numba']}；OpenCV {runtime['opencv']}；PyAV {runtime['pyav']}",
        f"Numba实际线程后端：{runtime['numba_layer']}；线程数：{runtime['numba_threads']}",
        f"调试状态：trace={runtime['trace']}；profile={runtime['profile']}；监控工具={runtime['monitoring']}",
        f"CPU逻辑线程：{cpu_affinity}",
        f"CPU恢复/编码线程：{CPU_RECOVERY_THREADS}/{CPU_ENCODER_THREADS}；队列并行：{PIPELINE}",
        f"GPU及驱动：{gpu}",
        f"视频编码器：{writer.stream.codec_context.name}",
        "算法运算范围：所选设备执行完整I帧ROP、参数插值、运动传播、局部ROP、透射率、百分位和恢复",
        "GPU模式限制：原生H264解码/真实参考导出、像素格式桥接和文件调度仍在CPU；并非整程序零CPU",
        "计时范围：真实输入读取至处理MP4关闭；排除环境、JIT、合成预热、对比生成和文本输出",
        f"对比生成耗时（不计入处理FPS）：{compare_seconds:.6f} 秒",
        f"处理视频：{processed}",
        f"对比视频：{comparison if create_comparison else '本次仅测速，未生成对比片'}",
        "对比布局：左侧原片，右侧运动传播与块级修正；无音轨",
    ]
    report = "\n".join(lines) + "\n"
    report_path = processed.with_suffix(".txt")
    report_path.write_text(report, encoding="utf-8-sig")
    print(report, flush=True)
    return dict(
        frames=count,
        seconds=elapsed,
        fps=count / elapsed,
        output=str(processed),
        setup_seconds=setup_seconds,
        recovery_threads=CPU_RECOVERY_THREADS,
        encoder_threads=CPU_ENCODER_THREADS,
        pipeline=PIPELINE,
        runtime=runtime,
        stages=stages,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=EXAMPLES)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--before-after", action="store_true", help="Also create a side-by-side video.")
    args = parser.parse_args()
    DEVICE = args.device
    with threadpool_limits(limits=1):
        main(args.example, create_comparison=args.before_after)
