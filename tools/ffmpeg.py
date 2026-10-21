"""
FFmpeg Tool - AI Video Director
==================================
Centralised FFmpeg command builder and executor.
Used by: editing_agent, export_agent
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List


def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def get_ffprobe() -> str:
    """
    Return the path to ffprobe. Since imageio_ffmpeg only bundles ffmpeg
    (not ffprobe), we look for ffprobe alongside the ffmpeg binary first,
    then fall back to using ffmpeg itself (which accepts the same probing flags).
    """
    import shutil
    # 1. Try a real ffprobe binary next to the bundled ffmpeg
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        ffp = ff.replace("ffmpeg", "ffprobe")
        if Path(ffp).exists():
            return ffp
    except Exception:
        pass
    # 2. Try system PATH
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        return system_ffprobe
    # 3. Fall back to using ffmpeg itself — it accepts all -show_* flags
    return get_ffmpeg()


def run(cmd: List[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def has_audio(path: str) -> bool:
    """Check if a video file has an audio stream by parsing ffmpeg stderr."""
    try:
        r = subprocess.run([get_ffmpeg(), "-i", path], capture_output=True, timeout=10)
        info = r.stderr.decode("utf-8", errors="replace")
        return "Audio:" in info
    except Exception:
        return False


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
    """Trim, scale, optionally burn caption. Always outputs with audio track."""
    ff = get_ffmpeg()
    duration = round(end - start, 3)
    if duration <= 0:
        return False

    vf_parts = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "setsar=1",
        "format=yuv420p",
    ]
    if caption:
        safe = caption.replace("'", "\\'").replace(":", "\\:")
        vf_parts.append(
            f"drawtext=text='{safe}':fontcolor=white:fontsize=48:"
            f"x=(w-text_w)/2:y=(h-text_h)-60:box=1:boxcolor=black@0.5:boxborderw=8"
        )

    if has_audio:
        # Source has audio — encode normally
        cmd = [
            ff, "-y",
            "-ss", str(start), "-i", input_path, "-t", str(duration),
            "-vf", ",".join(vf_parts),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            output_path, "-loglevel", "error",
        ]
    else:
        # No audio — add silent track so all clips are uniform
        cmd = [
            ff, "-y",
            "-ss", str(start), "-i", input_path, "-t", str(duration),
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf", ",".join(vf_parts),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            "-map", "0:v", "-map", "1:a",
            "-shortest",
            output_path, "-loglevel", "error",
        ]

    r = run(cmd, timeout=120)
    return r.returncode == 0 and Path(output_path).exists()


def concat(clip_list: List[str], output_path: str) -> bool:
    """
    Concatenate pre-encoded clips.
    Re-encodes to ensure all clips have matching streams (handles no-audio clips).
    """
    if not clip_list:
        return False

    if len(clip_list) == 1:
        import shutil
        shutil.copy2(clip_list[0], output_path)
        return Path(output_path).exists()

    ff = get_ffmpeg()

    # Build inputs and filter_complex for concat
    inputs = []
    for p in clip_list:
        inputs.extend(["-i", p])

    n = len(clip_list)
    filter_parts = "".join(f"[{i}:v][{i}:a]" for i in range(n))
    filter_str = f"{filter_parts}concat=n={n}:v=1:a=1[v][a]"

    cmd = [
        ff, "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-pix_fmt", "yuv420p",
        output_path, "-loglevel", "error",
    ]
    r = run(cmd, timeout=300)
    return r.returncode == 0 and Path(output_path).exists()


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
    try:
        r = run([get_ffprobe(), "-v", "quiet", "-print_format", "json",
                 "-show_format", video_path], timeout=10)
        dur = float(json.loads(r.stdout).get("format", {}).get("duration", 5))
        cmd = [get_ffmpeg(), "-y", "-ss", str(dur * at_pct), "-i", video_path,
               "-vframes", "1", "-vf", "scale=720:-1", output_path, "-loglevel", "error"]
        run(cmd, timeout=30)
        return Path(output_path).exists()
    except Exception:
        return False
