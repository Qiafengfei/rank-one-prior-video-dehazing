#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
root=$(pwd)
build="$root/build"
mkdir -p "$build/tmp"
export TMPDIR="$build/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

if [[ -d "$build/ffmpeg-7.1.1" ]]; then
    echo "Reusing the unpacked FFmpeg source."
else
    tar -xf "$root/ffmpeg-7.1.1.tar.xz" -C "$build"
fi
cd "$build/ffmpeg-7.1.1"
if git apply --check "$root/ffmpeg-h264-reference-export.patch"; then
    git apply "$root/ffmpeg-h264-reference-export.patch"
else
    git apply --reverse --check "$root/ffmpeg-h264-reference-export.patch"
fi
./configure --target-os=mingw32 --arch=x86_64 --cc=gcc \
    --disable-everything --disable-autodetect --disable-programs --disable-doc \
    --disable-network --disable-debug --enable-decoder=h264 --enable-demuxer=mov,h264 \
    --enable-parser=h264 --enable-protocol=file --enable-static --disable-shared \
    --enable-asm --disable-x86asm

mingw32-make -j"${JOBS:-8}" "SHELL=$(command -v bash)"
mingw32-make "SHELL=$(command -v bash)" libavcodec/aom_film_grain.o
gcc -O2 -I . "$root/native_decoder.c" libavcodec/aom_film_grain.o \
    -o "$root/native_decoder.exe" -Wl,--start-group libavformat/libavformat.a \
    libavcodec/libavcodec.a libavutil/libavutil.a -Wl,--end-group \
    -lm -latomic -lole32 -luser32 -lbcrypt
