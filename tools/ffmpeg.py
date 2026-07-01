"""
FFmpeg Tool - AI Video Director
==================================
Centralised FFmpeg command builder and executor.
Used by: editing_agent, export_agent
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional


def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def run(cmd: List[str], timeout: int = 180) -> subprocess.CompletedProcess:
    """Run an FFmpeg command and return the result."""
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def trim(
    input_path: str,
    start: float,
    end: float,
    output_path: str,
    width: int = 1080,
    height: int = 1920,
    caption: str = "",
    has_audio: bool = True,
) -> bool:
    """Trim, scale, optionally burn caption. Returns True on success."""
    ff = get_ffmpeg()
    duration = round(end - start, 3)
    if duration <= 0:
        return False

    vf_parts = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "format=yuv420p",
    ]
    if caption:
        safe = caption.replace("'", r"'\''").replace(":", r"\:")
        vf_parts.append(
            f"drawtext=text='{safe}':fontcolor=white:fontsize=52:"
            f"x=(w-text_w)/2:y=(h-text_h)-60:box=1:boxcolor=black@0.5:boxborderw=10"
        )

    if has_audio:
        inputs = ["-ss", str(start), "-i", input_path, "-t", str(duration)]
        audio_args = ["-c:a", "aac", "-ar", "44100", "-ac", "2", "-af", "loudnorm"]
    else:
        inputs = [
            "-ss", str(start), "-i", input_path, "-t", str(duration),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        ]
        audio_args = ["-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest"]

    cmd = [ff, "-y", *inputs, "-vf", ",".join(vf_parts),
           "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           *audio_args, output_path, "-loglevel", "error"]
    r = run(cmd, timeout=120)
    return r.returncode == 0 and Path(output_path).exists()


def concat(clip_list: List[str], output_path: str) -> bool:
    """Concatenate pre-encoded clips via stream copy."""
    list_file = output_path.replace(".mp4", "_list.txt")
    with open(list_file, "w") as f:
        for p in clip_list:
            f.write(f"file '{p}'\n")
    cmd = [get_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", list_file,
           "-c", "copy", output_path, "-loglevel", "error"]
    r = run(cmd, timeout=300)
    return r.returncode == 0


def mix_audio(video_path: str, music_path: str, output_path: str, music_vol: float = 0.15) -> bool:
    """Mix background music under video audio."""
    cmd = [
        get_ffmpeg(), "-y", "-i", video_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={music_vol}[music];[0:a][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path, "-loglevel", "error",
    ]
    r = run(cmd, timeout=300)
    return r.returncode == 0 and Path(output_path).exists()


def extract_thumbnail(video_path: str, output_path: str, at_pct: float = 0.2) -> bool:
    """Extract a JPEG thumbnail at a percentage through the video."""
    import json
    try:
        probe = [get_ffmpeg().replace("ffmpeg", "ffprobe"),
                 "-v", "quiet", "-print_format", "json", "-show_format", video_path]
        r = subprocess.run(probe, capture_output=True, text=True, timeout=10)
        dur = float(json.loads(r.stdout).get("format", {}).get("duration", 5))
        cmd = [get_ffmpeg(), "-y", "-ss", str(dur * at_pct), "-i", video_path,
               "-vframes", "1", "-vf", "scale=720:-1", output_path, "-loglevel", "error"]
        run(cmd, timeout=30)
        return Path(output_path).exists()
    except Exception:
        return False
