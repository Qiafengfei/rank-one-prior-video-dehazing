"""Apply single-image Rank-One Prior dehazing independently to each video frame."""

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
import heapq

import av

import motion_refined_rop_video as pipeline
from rank_one_prior_image import create_gpu_engine, rank1


def remux(video, source, target):
    with av.open(str(video)) as vc, av.open(str(source)) as sc, av.open(str(target), 'w', options={'movflags': '+faststart'}) as out:
        vs = vc.streams.video[0]
        ov = out.add_stream_from_template(vs)
        oa = out.add_stream_from_template(sc.streams.audio[0]) if sc.streams.audio else None
        def packets(container, stream, dest):
            for packet in container.demux(stream):
                if packet.dts is not None:
                    stamp = packet.dts * packet.time_base
                    packet.stream = dest
                    yield stamp, packet
        video_packets = packets(vc, vs, ov)
        audio_packets = packets(sc, sc.streams.audio[0], oa) if oa is not None else ()
        for _, packet in heapq.merge(video_packets, audio_packets, key=lambda row: row[0]):
            out.mux(packet)


def main(example, device):
    pipeline.DEVICE = device
    pipeline.configure_cpu()
    source = pipeline.ROOT / "examples" / f"{example}.mp4"
    folder = pipeline.ROOT / "outputs" / "framewise"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{example}-framewise.mp4"

    with TemporaryDirectory(dir=folder) as scratch:
        video = Path(scratch) / "video.mp4"
        pending = Path(scratch) / "pending.mp4"
        with pipeline.decoder(source) as container:
            stream = container.streams.video[0]
            engine = None
            if device == "gpu":
                engine = create_gpu_engine(stream.height, stream.width, 1)
            writer = pipeline.AsyncWriter(video, stream.width, stream.height,
                                          stream.average_rate, stream.time_base)
            try:
                for count, frame in enumerate(container.decode(video=0), start=1):
                    rgb = frame.to_ndarray(format="rgb24")
                    if engine is None:
                        recovered = rank1(rgb, diagnostics=False).image
                    else:
                        recovered = engine.seed(rgb, include_state=False)["image"]
                    writer.write(recovered, frame.pts, frame.time_base)
                    if count % 100 == 0:
                        print(f"{source.name}: {count}/{stream.frames} frames", flush=True)
            finally:
                writer.close()
        remux(video, source, pending)
        pending.replace(target)
    print(target)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=pipeline.EXAMPLES)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    with pipeline.threadpool_limits(limits=1):
        main(args.example, args.device)
