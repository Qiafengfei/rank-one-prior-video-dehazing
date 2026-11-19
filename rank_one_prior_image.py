"""ROP（Rank1）与加速恢复：纯数值计算，不读写视频。

原作者：Jun Liu、Ryan Wen Liu、Jianing Sun、Tieyong Zeng。
原代码/算法仅限非商业用途；相关专利 ZL 202011281893.3。
保留原 Rank1 公式、MATLAB 缩放、分位数及亮度 gamma 规则。
"""

from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import cv2
from numba import njit, prange


def _uint8(x):
    """按照 MATLAB 的规则四舍五入并限制到 0～255。"""
    return np.floor(np.clip(x, 0, 255) + 0.5).astype(np.uint8)


def _cubic(x):
    a = np.abs(x)
    return (1.5 * a**3 - 2.5 * a**2 + 1) * (a <= 1) + (
        -0.5 * a**3 + 2.5 * a**2 - 4 * a + 2
    ) * ((a > 1) & (a <= 2))


@lru_cache(maxsize=24)
def _resize_weights(n, target, scale):
    """缓存仅由尺寸决定的抗混叠权重；不缓存真实图像或参数。"""
    width = 4 / scale if scale < 1 else 4
    centers = (np.arange(target) + 0.5) / scale - 0.5
    left = np.floor(centers - width / 2).astype(int)
    indices = left[:, None] + np.arange(int(np.ceil(width)) + 2)
    distance = centers[:, None] - indices
    weights = scale * _cubic(scale * distance) if scale < 1 else _cubic(distance)
    weights /= weights.sum(axis=1, keepdims=True)
    keep = np.any(weights != 0, axis=0)
    weights, indices = (weights[:, keep], indices[:, keep])
    indices %= 2 * n
    indices = np.where(indices < n, indices, 2 * n - 1 - indices)
    weights.flags.writeable = False
    indices.flags.writeable = False
    return weights, indices


def resize_matlab(image, shape, scales=None):
    """带抗混叠和对称边界的 MATLAB 风格双三次缩放。"""
    result = np.asarray(image, dtype=np.float64)
    if scales is None:
        scales = (shape[0] / result.shape[0], shape[1] / result.shape[1])
    for axis in np.argsort(scales):
        scale = scales[axis]
        n = result.shape[axis]
        weights, indices = _resize_weights(n, shape[axis], scale)
        moved = np.moveaxis(result, axis, 0)
        result = np.moveaxis(np.einsum("ij,ij...->i...", weights, moved[indices]), 0, axis)
    return result


@dataclass
class ROPResult:
    """ROP 的图像、中间参数和计算耗时；不进行文件读写。"""

    arrays: dict
    timings: dict
    parameters: dict
    notes: list

    @property
    def image(self):
        return self.arrays["image"]

    @property
    def spectrum(self):
        return self.arrays["spectrum"]

    @property
    def scattering(self):
        return self.arrays["scatter"]

    @property
    def atmosphere(self):
        return self.arrays["atmosphere"]

    @property
    def transmission(self):
        return self.arrays["transmission"]


def normalize_rgb(image):
    """将整数 RGB 转为浮点数；浮点输入必须在 0～1 范围内。"""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a nonempty H x W x 3 RGB array.")
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255
    elif image.dtype == np.uint16:
        image = image.astype(np.float64) / 65535
    elif np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float64, copy=False)
    else:
        raise ValueError("Use uint8, uint16, or floating-point RGB input.")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Floating RGB input must be finite and in [0,1].")
    return image


def rgb_to_ycbcr(image):
    matrix = np.array(
        [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]]
    )
    matrix = 1 / 255 * matrix
    rgb = image.astype(np.float64)
    return _uint8(
        rgb[..., 0, None] * matrix[:, 0]
        + rgb[..., 1, None] * matrix[:, 1]
        + rgb[..., 2, None] * matrix[:, 2]
        + [16, 128, 128]
    )


def ycbcr_to_rgb(image):
    matrix = np.array(
        [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]]
    )
    inverse = np.linalg.inv(matrix)
    matrix = 255 * inverse
    offset = matrix @ np.array([16, 128, 128])
    ycbcr = image.astype(np.float64)
    return _uint8(
        ycbcr[..., 0, None] * matrix[:, 0]
        + ycbcr[..., 1, None] * matrix[:, 1]
        + ycbcr[..., 2, None] * matrix[:, 2]
        - offset
    )


def _postprocess(recovered, low, high):
    lo, hi = np.percentile(recovered, [low, high], axis=(0, 1), method="hazen")
    stretched = np.divide(recovered - lo, hi - lo, out=np.zeros_like(recovered), where=hi != lo)
    before = _uint8(stretched * 255)
    ycbcr = rgb_to_ycbcr(before)
    lut = _uint8(((np.arange(256) + 0.5) / 256) ** (5 / 6) * 256 - 0.5)
    after = ycbcr.copy()
    after[..., 0] = lut[after[..., 0]]
    return dict(
        percentile_low=lo.reshape(1, 1, 3),
        percentile_high=hi.reshape(1, 1, 3),
        stretched=stretched,
        pre_gamma=before,
        ycbcr_before=ycbcr,
        ycbcr_after=after,
        image=ycbcr_to_rgb(after),
    )


def _atmosphere(image, scatter, fraction, plus, notes):
    magnitude = scatter.sum(axis=2)
    count = int(np.floor(magnitude.size * fraction))
    if count == 0:
        count = 1
        notes.append("Atmosphere sample count clamped to one for a small image.")
    # 只选择需要的最高0.1%样本；同值边界按原F顺序截取，结果与全图稳定排序一致。
    values = magnitude.ravel(order="F")
    cutoff = np.partition(values, values.size - count)[values.size - count]
    higher = np.flatnonzero(values > cutoff)
    equal = np.flatnonzero(values == cutoff)[: count - higher.size]
    selected = np.concatenate((higher, equal))
    indices = selected[np.argsort(-values[selected], kind="stable")]
    atmosphere = image.reshape(-1, 3, order="F")[indices].mean(axis=0).reshape(1, 1, 3)
    threshold = magnitude.ravel(order="F")[indices[-1]]
    if plus:
        pivot = threshold * scatter[0:1, 0:1, :] / max(magnitude[0, 0], 0.001)
        reflected = 2 * pivot - scatter
    else:
        reflected = 2 / 3 * threshold - scatter
    adjusted = np.where((magnitude <= threshold)[..., None], scatter, reflected)
    return (atmosphere, adjusted, indices, threshold)


def _omega(value):
    if not np.isscalar(value) or not np.isfinite(value) or (not 0 < value <= 1):
        raise ValueError("omega must lie in (0,1].")
    return float(value)


def rank1(image, omega=0.8, diagnostics=True):
    """完整 ROP，输入 RGB 图像，返回图像和传播所需参数。"""
    total = perf_counter()
    omega = _omega(omega)
    image = normalize_rgb(image)
    normalized_at = perf_counter()
    notes = []
    spectrum = image.reshape(-1, 3, order="F").mean(axis=0).reshape(1, 1, 3)
    basis = spectrum / np.maximum(np.sqrt(np.sum(spectrum**2, axis=2, keepdims=True)), 0.001)
    spectrum_at = perf_counter()
    fog_basis = image / np.maximum(np.sqrt(np.sum(image**2, axis=2, keepdims=True)), 0.001)
    cosine = np.sum(basis * fog_basis, axis=2, keepdims=True)
    raw = (
        cosine
        * (
            image.sum(axis=2, keepdims=True)
            / np.maximum(spectrum.sum(axis=2, keepdims=True), 0.001)
        )
        * spectrum
    )
    atmosphere, scatter, indices, threshold = _atmosphere(image, raw, 0.001, False, notes)
    initial = 1 - omega * scatter
    size = min(image.shape[:2])
    scale = 0.02 if size < 800 else 0.01 if size < 1500 else 0.005
    small_shape = tuple((max(1, int(np.ceil(n * scale))) for n in image.shape[:2]))
    small = resize_matlab(initial, small_shape, (scale, scale))
    transmission = resize_matlab(small, image.shape[:2])
    transmission_at = perf_counter()
    recovered = (image - atmosphere) / np.maximum(transmission, 0.001) + atmosphere
    arrays = _postprocess(recovered, 1, 99)
    if not diagnostics:
        arrays = {key: arrays[key] for key in ("image", "percentile_low", "percentile_high")}
    if np.any(arrays["percentile_low"] == arrays["percentile_high"]):
        notes.append("Zero percentile range mapped to zero before gamma.")
    arrays.update(
        spectrum=spectrum,
        spectrum_l1=spectrum / np.maximum(spectrum.sum(), 0.001),
        spectrum_l2=basis,
        cosine=cosine[..., 0],
        scatter_raw=raw,
        scatter=scatter,
        scatter_coefficient=cosine[..., 0] * image.sum(axis=2),
        atmosphere=atmosphere,
        atmosphere_indices=indices,
        atmosphere_threshold=np.array(threshold),
        transmission_initial=initial,
        transmission_small=small,
        transmission=transmission,
        recovered=recovered,
    )
    if diagnostics:
        arrays.update(scattering_initial=1 - initial, scattering_final=1 - transmission)
    else:
        for key in ("cosine", "transmission_initial", "transmission_small", "recovered"):
            arrays.pop(key)
    end = perf_counter()
    return ROPResult(
        arrays,
        dict(
            normalization=normalized_at - total,
            spectrum=spectrum_at - normalized_at,
            scattering_transmission=transmission_at - spectrum_at,
            recovery_postprocess=end - transmission_at,
            total=end - total,
        ),
        dict(algorithm="rank1", omega=omega, resize_scale=scale, mi=1, ma=99),
        notes,
    )


@njit(cache=True)
def rounded(v):
    return int(min(255.0, max(0.0, np.floor(v + 0.5))))


@lru_cache(maxsize=8)
def recovery_columns(width, grid_width):
    """只缓存各行共用的横向插值坐标，不缓存输入图像。"""
    position = np.clip(
        (np.arange(width, dtype=np.float64) + 0.5) * grid_width / width - 0.5, 0.0, grid_width - 1.0
    )
    low = position.astype(np.int64)
    result = (low, np.minimum(low + 1, grid_width - 1), position - low)
    for value in result:
        value.flags.writeable = False
    return result


@njit(parallel=True, cache=True, nogil=True)
def pixel_recover(rgb, t, a, low, high, lut, m, inv, table=None, columns=None):
    h, w = rgb.shape[:2]
    out = np.empty_like(rgb)
    th, tw = t.shape[:2]
    for y in prange(h):
        sy = max(0.0, min(th - 1.0, (y + 0.5) * th / h - 0.5))
        y0 = int(sy)
        y1 = min(y0 + 1, th - 1)
        fy = sy - y0
        for x in range(w):
            if columns is None:
                sx = max(0.0, min(tw - 1.0, (x + 0.5) * tw / w - 0.5))
                x0 = int(sx)
                x1 = min(x0 + 1, tw - 1)
                fx = sx - x0
            else:
                x0, x1, fx = columns[0][x], columns[1][x], columns[2][x]
            p0 = stretched_pixel(rgb[y, x, 0], a[0], low[0], high[0], t, y0, y1, x0, x1, fy, fx, 0)
            p1 = stretched_pixel(rgb[y, x, 1], a[1], low[1], high[1], t, y0, y1, x0, x1, fy, fx, 1)
            p2 = stretched_pixel(rgb[y, x, 2], a[2], low[2], high[2], t, y0, y1, x0, x1, fy, fx, 2)
            if table is not None:
                index = (p0 * 256 + p1) * 256 + p2
                for k in range(3):
                    out[y, x, k] = table[index, k]
                continue
            yy = rounded(m[0, 0] * p0 + m[0, 1] * p1 + m[0, 2] * p2 + 16.0)
            cb = rounded(m[1, 0] * p0 + m[1, 1] * p1 + m[1, 2] * p2 + 128.0)
            cr = rounded(m[2, 0] * p0 + m[2, 1] * p1 + m[2, 2] * p2 + 128.0)
            yy = lut[yy]
            for k in range(3):
                out[y, x, k] = rounded(
                    inv[k, 0] * (yy - 16.0) + inv[k, 1] * (cb - 128.0) + inv[k, 2] * (cr - 128.0)
                )
    return out


@njit(inline="always", cache=True)
def stretched_pixel(pixel, a, low, high, t, y0, y1, x0, x1, fy, fx, k):
    transmission = ((1 - fx) * t[y0, x0, k] + fx * t[y0, x1, k]) * (1 - fy) + (
        (1 - fx) * t[y1, x0, k] + fx * t[y1, x1, k]
    ) * fy
    j = (pixel / 255.0 - a) / max(transmission, 0.001) + a
    return rounded((j - low) / (high - low) * 255.0) if high != low else 0


@lru_cache(maxsize=8)
def sampling_maps(H, W, th, tw, step):
    """复用当前尺寸下的百分位采样坐标，避免每帧重复创建网格。"""
    yy, xx = np.meshgrid(
        (np.arange(0, H, step, dtype=np.float32) + 0.5) * th / H - 0.5,
        (np.arange(0, W, step, dtype=np.float32) + 0.5) * tw / W - 0.5,
        indexing="ij",
    )
    yy.flags.writeable = False
    xx.flags.writeable = False
    return yy, xx


def render(rgb, t, a, exact_percentiles=False, stride=8, recovery=None):
    """按当前帧采样统计拉伸范围，再在全分辨率恢复。"""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    step = 1 if exact_percentiles else stride
    sample = rgb[::step, ::step].astype(np.float64) / 255
    if t.shape[:2] == rgb.shape[:2]:
        sample_t = t[::step, ::step]
    else:
        H, W = rgb.shape[:2]
        th, tw = t.shape[:2]
        yy, xx = sampling_maps(H, W, th, tw, step)
        sample_t = cv2.remap(t, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    recovered = (sample - a) / np.maximum(sample_t, 0.001) + a
    low, high = np.percentile(recovered, [1, 99], axis=(0, 1), method="hazen")
    if recovery is None or recovery is pixel_recover:
        image = pixel_recover(
            rgb,
            t,
            a,
            low,
            high,
            LUT,
            M,
            INV,
            color_table(),
            recovery_columns(rgb.shape[1], t.shape[1]),
        )
    else:
        image = recovery(rgb, t, a, low, high, LUT, M, INV)
    return (
        image,
        dict(low=low.tolist(), high=high.tolist(), percentile_stride=step),
    )


# RGB/YCbCr 变换及亮度 gamma 查表，沿用原 ROP 的量化顺序。
M = (
    np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]])
    / 255
)
INV = np.linalg.inv(M)
LUT = _uint8(((np.arange(256) + 0.5) / 256) ** (5 / 6) * 256 - 0.5)


def create_gpu_engine(height, width, capacity):
    """ROP 与运动矢量传播、块级修正的 GPU 数值计算；CPU模式不导入CuPy或初始化CUDA。"""
    import os
    import math

    os.environ["CUPY_CACHE_IN_MEMORY"] = "1"
    # Conda环境的CUDA库可能与CuPy版本不同，显式使用CUDA_PATH指定的工具包。
    cuda_dll_directory = None
    if os.name == "nt" and os.environ.get("CUDA_PATH"):
        cuda_bin = Path(os.environ["CUDA_PATH"]) / "bin"
        if cuda_bin.is_dir():
            cuda_dll_directory = os.add_dll_directory(str(cuda_bin))
    import cupy as cp

    CUDA = r"""

    extern "C" __global__ void resize_axis(const double* src,const int* idx,const double* wt,double* dst,
     int h,int w,int c,int oh,int ow,int taps,int axis){
     int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=oh*ow*c)return;
     int k=q%c,x=(q/c)%ow,y=q/(c*ow),r=axis==0?y:x;double v=0.;
     for(int j=0;j<taps;j++){int p=idx[r*taps+j];v+=wt[r*taps+j]*src[((axis==0?p:y)*w+(axis==0?x:p))*c+k];}
     dst[q]=v;
    }
    extern "C" __global__ void bilinear(const double* src,const float* xx,const float* yy,double* dst,
     int h,int w,int oh,int ow,int c){
     int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=oh*ow*c)return;
     int k=q%c,p=q/c;double x=xx[p],y=yy[p];int x0=floor(x),y0=floor(y);double fx=x-x0,fy=y-y0;
     int x1=max(0,min(w-1,x0+1)),y1=max(0,min(h-1,y0+1));x0=max(0,min(w-1,x0));y0=max(0,min(h-1,y0));
     dst[q]=((1-fx)*src[(y0*w+x0)*c+k]+fx*src[(y0*w+x1)*c+k])*(1-fy)
           +((1-fx)*src[(y1*w+x0)*c+k]+fx*src[(y1*w+x1)*c+k])*fy;
    }
    extern "C" __global__ void warp_states(const float* pool,const int* grid,const int* refs,int nr,
     float* z,unsigned char* valid,unsigned char* outside,unsigned char* same,float* xx,float* yy,
     int h,int w,int scene){
     int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=2*h*w)return;
     int cell=p%(h*w),y=cell/w,x=cell%w;const int* g=grid+p*4;
     float sx=x+(g[0]>=0?g[1]/32.f:0.f),sy=y+(g[0]>=0?g[2]/32.f:0.f);xx[p]=sx;yy[p]=sy;
     int slot=-1,sc=-1;for(int r=0;r<nr;r++){if(refs[3*r]==g[3]){slot=refs[3*r+1];sc=refs[3*r+2];break;}}
     bool ok=g[0]>=0&&slot>=0;valid[p]=ok;outside[p]=ok&&(sx<0||sx>w-1||sy<0||sy>h-1);same[p]=ok&&sc==scene;
     if(!ok){for(int k=0;k<8;k++)z[p*8+k]=0;return;}
     int x0=floorf(sx),y0=floorf(sy);float fx=sx-x0,fy=sy-y0;
     int x1=max(0,min(w-1,x0+1)),y1=max(0,min(h-1,y0+1));x0=max(0,min(w-1,x0));y0=max(0,min(h-1,y0));
     for(int k=0;k<8;k++){
      float a=pool[((slot*h+y0)*w+x0)*8+k],b=pool[((slot*h+y0)*w+x1)*8+k];
      float c=pool[((slot*h+y1)*w+x0)*8+k],d=pool[((slot*h+y1)*w+x1)*8+k];
      z[p*8+k]=(1-fy)*((1-fx)*a+fx*b)+fy*((1-fx)*c+fx*d);
     }
    }

    // 一次读取8x8原像素，完成均值、偶数舍入及灰度，避免全帧float64临时数组。
    extern "C" __global__ void small_fields(const unsigned char* rgb,
        unsigned char* small, float* gray, int h, int w) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        int ws = w / 8;
        if (cell >= (h / 8) * ws) {
            return;
        }
        int y = cell / ws;
        int x = cell % ws;
        int sums[3] = {0, 0, 0};
        for (int dy = 0; dy < 8; ++dy) {
            for (int dx = 0; dx < 8; ++dx) {
                int p = ((y * 8 + dy) * w + x * 8 + dx) * 3;
                for (int k = 0; k < 3; ++k) {
                    sums[k] += rgb[p + k];
                }
            }
        }
        for (int k = 0; k < 3; ++k) {
            int value = sums[k] / 64;
            int remainder = sums[k] % 64;
            if (remainder > 32 || (remainder == 32 && (value & 1))) {
                value += 1;
            }
            small[cell * 3 + k] = value;
        }
        float r = small[cell * 3];
        float g = small[cell * 3 + 1];
        float b = small[cell * 3 + 2];
        gray[cell] = ((r * .299f + g * .587f) + b * .114f) / 255.f;
    }


    // 融合两路可靠性加权、局部ROP修正和初始透射率，消除逐项GPU临时数组。
    extern "C" __global__ void local_fields(const float* z,
        const unsigned char* valid, const unsigned char* outside,
        const unsigned char* small, const float* gray, const float* fallback,
        const double* s, const double* threshold, float* ce, double* initial,
        int n, int force_local) {
        int q = blockDim.x * blockIdx.x + threadIdx.x;
        if (q >= n) {
            return;
        }
        float w0 = valid[q] / (.01f + fabsf(gray[q] - z[q * 8 + 7]));
        float w1 = valid[n + q] / (.01f + fabsf(gray[q] - z[(n + q) * 8 + 7]));
        float weights = w0 + w1;
        float prop[8];
        for (int k = 0; k < 8; ++k) {
            float weighted = z[q * 8 + k] * w0 + z[(n + q) * 8 + k] * w1;
            prop[k] = weights > 0 ? weighted / fmaxf(weights, 1e-30f) : fallback[q * 8 + k];
        }
        bool bad = force_local || weights <= 0 || outside[q] || outside[n + q]
            || fabsf(gray[q] - prop[7]) > .06f
            || (valid[q] && valid[n + q] && fabsf(z[q * 8] - z[(n + q) * 8]) > .15f);
        float c = prop[0];
        float e[3] = {prop[1], prop[2], prop[3]};
        if (bad) {
            double x0 = small[q * 3] / 255.;
            double x1 = small[q * 3 + 1] / 255.;
            double x2 = small[q * 3 + 2] / 255.;
            double sn = fmax(sqrt((s[0] * s[0] + s[1] * s[1]) + s[2] * s[2]), .001);
            double b0 = s[0] / sn;
            double b1 = s[1] / sn;
            double b2 = s[2] / sn;
            double norm = fmax(sqrt((x0 * x0 + x1 * x1) + x2 * x2), .001);
            double coefficient = (((x0 * b0 + x1 * b1) + x2 * b2) / norm) * ((x0 + x1) + x2);
            double raw[3] = {coefficient * s[0], coefficient * s[1], coefficient * s[2]};
            bool reflected = (raw[0] + raw[1]) + raw[2] > threshold[0];
            c = coefficient;
            for (int k = 0; k < 3; ++k) {
                double adjusted = reflected ? (2. / 3.) * threshold[0] - raw[k] : raw[k];
                e[k] = adjusted - raw[k];
            }
        }
        ce[q * 4] = c;
        for (int k = 0; k < 3; ++k) {
            ce[q * 4 + 1 + k] = e[k];
            initial[q * 3 + k] = 1. - .8 * ((double)c * s[k] + (double)e[k]);
        }
    }

    __device__ int byteval(double x){return min(255,max(0,(int)floor(x+.5)));}
    __device__ int stretch(unsigned char v,int k,const double* t,const double* p,
     int th,int tw,int x0,int x1,int y0,int y1,double fx,double fy){
     double tt=((1-fx)*t[(y0*tw+x0)*3+k]+fx*t[(y0*tw+x1)*3+k])*(1-fy)
              +((1-fx)*t[(y1*tw+x0)*3+k]+fx*t[(y1*tw+x1)*3+k])*fy;
     double j=(v/255.-p[k])/fmax(tt,.001)+p[k];
     return p[6+k]!=p[3+k]?byteval((j-p[3+k])/(p[6+k]-p[3+k])*255.):0;
    }
    extern "C" __global__ void recover(const unsigned char* rgb,const double* t,const double* p,
     const unsigned char* lut,const double* m,const double* inv,unsigned char* out,int h,int w,int th,int tw){
     int q=blockDim.x*blockIdx.x+threadIdx.x;if(q>=h*w)return;int y=q/w,x=q%w;
     double sy=fmax(0.,fmin(th-1.,(y+.5)*th/h-.5)),sx=fmax(0.,fmin(tw-1.,(x+.5)*tw/w-.5));
     int y0=sy,x0=sx,y1=min(y0+1,th-1),x1=min(x0+1,tw-1);double fy=sy-y0,fx=sx-x0;
     int a=stretch(rgb[q*3],0,t,p,th,tw,x0,x1,y0,y1,fx,fy);
     int b=stretch(rgb[q*3+1],1,t,p,th,tw,x0,x1,y0,y1,fx,fy);
     int c=stretch(rgb[q*3+2],2,t,p,th,tw,x0,x1,y0,y1,fx,fy);
     int yy=lut[byteval(m[0]*a+m[1]*b+m[2]*c+16.)];
     int cb=byteval(m[3]*a+m[4]*b+m[5]*c+128.),cr=byteval(m[6]*a+m[7]*b+m[8]*c+128.);
     for(int k=0;k<3;k++)out[q*3+k]=byteval(inv[k*3]*(yy-16.)+inv[k*3+1]*(cb-128.)+inv[k*3+2]*(cr-128.));
    }

    """

    class Engine:

        def __init__(self, height=1080, width=1920, capacity=1024):
            self.h, self.w = (height, width)
            self.hs, self.ws = (height // 8, width // 8)
            self.pool = cp.empty((capacity, self.hs, self.ws, 8), cp.float32)
            self.module = cp.RawModule(code=CUDA, options=("--std=c++11", "--fmad=false"))
            self.axis_kernel = self.module.get_function("resize_axis")
            self.bilinear = self.module.get_function("bilinear")
            self.warp_kernel = self.module.get_function("warp_states")
            self.recover_kernel = self.module.get_function("recover")
            self.small_fields_kernel = self.module.get_function("small_fields")
            self.local_fields_kernel = self.module.get_function("local_fields")
            self.weights = {}
            self.maps = {}
            matrix = np.array(
                [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]]
            )
            self.m = cp.asarray(matrix / 255)
            self.inv = cp.asarray(np.linalg.inv(matrix / 255))
            self.mi = cp.asarray(1 / 255 * matrix)
            self.invi = cp.asarray(255 * np.linalg.inv(matrix))
            self.offset = self.invi @ cp.asarray([16.0, 128.0, 128.0])
            self.lut = cp.asarray(
                np.floor(
                    np.clip(((np.arange(256) + 0.5) / 256) ** (5 / 6) * 256 - 0.5, 0, 255) + 0.5
                ).astype(np.uint8)
            )
            self.yy, self.xx = cp.indices((self.hs, self.ws), dtype=cp.float32)
            self.output = cp.empty((height, width, 3), cp.uint8)
            self.full_scale = (
                0.02 if min(height, width) < 800 else 0.01 if min(height, width) < 1500 else 0.005
            )
            self.tshape = tuple(
                (max(1, int(math.ceil(n * self.full_scale))) for n in (height, width))
            )

        def _axis_weights(self, n, target, scale):
            key = (n, target, float(scale))
            if key not in self.weights:
                width = 4 / scale if scale < 1 else 4
                centers = (cp.arange(target) + 0.5) / scale - 0.5
                indices = cp.floor(centers - width / 2).astype(int)[:, None] + cp.arange(
                    int(cp.ceil(width)) + 2
                )
                d = centers[:, None] - indices

                def cubic(x):
                    a = cp.abs(x)
                    return (1.5 * a**3 - 2.5 * a**2 + 1) * (a <= 1) + (
                        -0.5 * a**3 + 2.5 * a**2 - 4 * a + 2
                    ) * ((a > 1) & (a <= 2))

                weights = scale * cubic(scale * d) if scale < 1 else cubic(d)
                weights /= weights.sum(axis=1, keepdims=True)
                keep = cp.any(weights != 0, axis=0)
                weights = weights[:, keep]
                indices = indices[:, keep] % (2 * n)
                indices = cp.where(indices < n, indices, 2 * n - 1 - indices)
                self.weights[key] = (
                    cp.ascontiguousarray(cp.asarray(indices.astype(cp.int32))),
                    cp.ascontiguousarray(cp.asarray(weights)),
                )
            return self.weights[key]

        def resize(self, image, shape, scales=None):
            result = cp.ascontiguousarray(image, dtype=cp.float64)
            if result.ndim == 2:
                result = result[..., None]
            if scales is None:
                scales = (shape[0] / result.shape[0], shape[1] / result.shape[1])
            for axis in np.argsort(scales):
                h, w, c = result.shape
                oh, ow = (shape[0], w) if axis == 0 else (h, shape[1])
                idx, wt = self._axis_weights(result.shape[axis], shape[axis], scales[axis])
                dest = cp.empty((oh, ow, c), cp.float64)
                n = dest.size
                self.axis_kernel(
                    ((n + 255) // 256,),
                    (256,),
                    (result, idx, wt, dest, h, w, c, oh, ow, wt.shape[1], int(axis)),
                )
                result = dest
            return result

        def linear(self, image, shape, quantized=False, step=1):
            h, w, c = image.shape
            oh, ow = shape
            key = (h, w, oh, ow, quantized, step)
            if key not in self.maps:
                y = (
                    (cp.arange(oh, dtype=cp.float32) * step + 0.5) * h / self.h - 0.5
                    if quantized
                    else (cp.arange(oh, dtype=cp.float32) + 0.5) * h / oh - 0.5
                )
                x = (
                    (cp.arange(ow, dtype=cp.float32) * step + 0.5) * w / self.w - 0.5
                    if quantized
                    else (cp.arange(ow, dtype=cp.float32) + 0.5) * w / ow - 0.5
                )
                if quantized:
                    y = cp.rint(y * 32) / 32
                    x = cp.rint(x * 32) / 32
                yy, xx = cp.meshgrid(y, x, indexing="ij")
                self.maps[key] = (xx.astype(cp.float32), yy.astype(cp.float32))
            xx, yy = self.maps[key]
            dest = cp.empty((oh, ow, c), cp.float64)
            n = dest.size
            self.bilinear(
                ((n + 255) // 256,),
                (256,),
                (cp.ascontiguousarray(image, dtype=cp.float64), xx, yy, dest, h, w, oh, ow, c),
            )
            return dest

        def percentiles(self, image):
            values = cp.sort(image.reshape(-1, 3), axis=0)
            n = values.shape[0]
            outputs = []
            for p in (0.01, 0.99):
                position = max(0, min(n - 1, n * p - 0.5))
                lo = int(math.floor(position))
                hi = min(n - 1, lo + 1)
                outputs.append(values[lo] + (values[hi] - values[lo]) * (position - lo))
            return cp.stack(outputs)

        @staticmethod
        def byte(image):
            return cp.floor(cp.clip(image, 0, 255) + 0.5).astype(cp.uint8)

        def area(self, x):
            if x.ndim == 2:
                x = x[..., None]
            return x.reshape(self.hs, 8, self.ws, 8, x.shape[-1]).mean(axis=(1, 3))

        def local_fields(self, z, valid, outside, small, gray, fallback, s, threshold, force_local):
            """一次GPU计算完成参数传播、可靠性判断、局部修正及散射重建。"""
            ce = cp.empty((self.hs, self.ws, 4), cp.float32)
            initial = cp.empty((self.hs, self.ws, 3), cp.float64)
            count = self.hs * self.ws
            self.local_fields_kernel(
                ((count + 255) // 256,), (256,),
                (z, valid, outside, small, gray, self.pool[fallback], s, threshold,
                 ce, initial, count, int(force_local)),
            )
            return ce, initial

        def small_fields(self, rgb):
            """融合控制网格RGB均值、偶数舍入和灰度，保持原来的逐位结果。"""
            small = cp.empty((self.hs, self.ws, 3), cp.uint8)
            gray = cp.empty((self.hs, self.ws), cp.float32)
            count = self.hs * self.ws
            self.small_fields_kernel(
                ((count + 255) // 256,), (256,), (rgb, small, gray, self.h, self.w)
            )
            return small, gray

        def small_rgb(self, rgb):
            return cp.rint(self.area(rgb.astype(cp.float64))).astype(cp.uint8)

        @staticmethod
        def gray(small):
            x = small.astype(cp.float32)
            return (
                x[..., 0] * cp.float32(0.299)
                + x[..., 1] * cp.float32(0.587)
                + x[..., 2] * cp.float32(0.114)
            ) / cp.float32(255)

        def seed(self, rgb, *, include_state=True):
            gpu = cp.asarray(rgb)
            image = gpu.astype(cp.float64) / 255
            spectrum = image.reshape(-1, 3, order="F").mean(axis=0)
            basis = spectrum / cp.maximum(cp.sqrt(cp.sum(spectrum * spectrum)), 0.001)
            fog = image / cp.maximum(cp.sqrt(cp.sum(image * image, axis=2, keepdims=True)), 0.001)
            cosine = cp.sum(basis * fog, axis=2)
            raw = (
                cosine[..., None]
                * (image.sum(axis=2, keepdims=True) / cp.maximum(spectrum.sum(), 0.001))
                * spectrum
            )
            magnitude = raw.sum(axis=2)
            count = max(1, int(math.floor(self.h * self.w * 0.001)))
            order = cp.argsort(-magnitude.ravel(order="F"), kind="stable")[:count]
            atmosphere = image.reshape(-1, 3, order="F")[order].mean(axis=0)
            threshold = magnitude.ravel(order="F")[order[-1]]
            scatter = cp.where((magnitude <= threshold)[..., None], raw, 2 / 3 * threshold - raw)
            tiny = self.resize(1 - 0.8 * scatter, self.tshape, (self.full_scale,) * 2)
            transmission = self.resize(tiny, (self.h, self.w))
            recovered = (image - atmosphere) / cp.maximum(transmission, 0.001) + atmosphere
            low, high = self.percentiles(recovered)
            stretched = cp.where(
                high != low, (recovered - low) / cp.where(high != low, high - low, 1), 0
            )
            before = self.byte(stretched * 255).astype(cp.float64)
            yuv = self.byte(
                before[..., 0, None] * self.mi[:, 0]
                + before[..., 1, None] * self.mi[:, 1]
                + before[..., 2, None] * self.mi[:, 2]
                + cp.asarray([16, 128, 128])
            )
            yuv[..., 0] = self.lut[yuv[..., 0]]
            yuv = yuv.astype(cp.float64)
            final = self.byte(
                yuv[..., 0, None] * self.invi[:, 0]
                + yuv[..., 1, None] * self.invi[:, 1]
                + yuv[..., 2, None] * self.invi[:, 2]
                - self.offset
            )
            if not include_state:
                return {"image": final.get()}
            c = cosine * image.sum(axis=2)
            state = cp.concatenate(
                (
                    self.area(c),
                    self.area(scatter - raw),
                    self.area(transmission),
                    self.gray(self.small_rgb(gpu))[..., None],
                ),
                axis=-1,
            ).astype(cp.float32)
            return dict(
                state=state,
                image=final.get(),
                s=spectrum / cp.maximum(spectrum.sum(), 0.001),
                a=atmosphere,
                threshold=threshold,
            )

        def render(self, rgb, t, a):
            sample = rgb[::8, ::8].astype(cp.float64) / 255
            sampled_t = self.linear(t, sample.shape[:2], True, 8)
            low, high = self.percentiles((sample - a) / cp.maximum(sampled_t, 0.001) + a)
            p = cp.ascontiguousarray(cp.stack((a, low, high)))
            self.recover_kernel(
                ((self.h * self.w + 255) // 256,),
                (256,),
                (
                    rgb,
                    t,
                    p,
                    self.lut,
                    self.m,
                    self.inv,
                    self.output,
                    self.h,
                    self.w,
                    t.shape[0],
                    t.shape[1],
                ),
            )
            return self.output.get()

    engine = Engine(height, width, capacity)
    engine._cuda_dll_directory = cuda_dll_directory
    engine.cp = cp
    return engine


_COLOR_TABLE = None


@njit(parallel=True, cache=True)
def make_color_table(lut, m, inv):
    """输入无关的精确RGB颜色变换查表；复用已有实现，合成预热时准备。"""
    out = np.empty((1 << 24, 3), np.uint8)
    for red in prange(256):
        for green in range(256):
            for blue in range(256):
                yy = rounded(m[0, 0] * red + m[0, 1] * green + m[0, 2] * blue + 16.0)
                cb = rounded(m[1, 0] * red + m[1, 1] * green + m[1, 2] * blue + 128.0)
                cr = rounded(m[2, 0] * red + m[2, 1] * green + m[2, 2] * blue + 128.0)
                yy = lut[yy]
                index = (red * 256 + green) * 256 + blue
                for k in range(3):
                    out[index, k] = rounded(
                        inv[k, 0] * (yy - 16.0)
                        + inv[k, 1] * (cb - 128.0)
                        + inv[k, 2] * (cr - 128.0)
                    )
    return out


def color_table():
    """输入无关的精确RGB颜色变换查表；复用已有实现，合成预热时准备。"""
    global _COLOR_TABLE
    if _COLOR_TABLE is None:
        table = make_color_table(LUT, M, INV)
        table.flags.writeable = False
        _COLOR_TABLE = table
    return _COLOR_TABLE


if __name__ == "__main__":
    from argparse import ArgumentParser
    from PIL import Image

    parser = ArgumentParser(description="Apply Rank-One Prior dehazing to a single image.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with Image.open(args.input) as source:
        recovered = rank1(np.asarray(source.convert("RGB")), diagnostics=False).image
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(recovered).save(args.output)
