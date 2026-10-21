"""
FFmpeg Tool - ClipMind AI
==================================
Centralised FFmpeg command builder and executor.
Used by: editing_agent, export_agent
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def get_ffmpeg() -> str:
    """FFmpeg on PATH first, then the binary bundled with imageio-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def run(cmd: List[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe(path: str) -> Dict[str, Any]:
    """
    Read duration, resolution, fps, codec and audio presence by parsing the
    stderr of `ffmpeg -i`. Works with the imageio-ffmpeg binary, which ships
    without ffprobe. Returns {"path", "error"} on failure.
    """
    try:
        r = subprocess.run([get_ffmpeg(), "-hide_banner", "-i", path], capture_output=True, timeout=30)
        info = r.stderr.decode("utf-8", errors="replace")
        if not info:
            return {"path": path, "error": "No output from ffmpeg"}

        duration = 0.0
        dur_match = re.search(r"Duration:\s+(\d+):(\d+):([\d.]+)", info)
        if dur_match:
            h, m, s = dur_match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)

        width, height, fps = 0, 0, 0.0
        vid_match = re.search(r"Stream.*Video.*?(\d{2,5})x(\d{2,5}).*?([\d.]+)\s+fps", info)
        if vid_match:
            width, height, fps = int(vid_match.group(1)), int(vid_match.group(2)), float(vid_match.group(3))

        codec_match = re.search(r"Video:\s+(\w+)", info)

        return {
            "path": path,
            "duration": round(duration, 3),
            "size_bytes": Path(path).stat().st_size,
            "width": width,
            "height": height,
            "codec": codec_match.group(1) if codec_match else "unknown",
            "fps": fps,
            "has_audio": "Audio:" in info,
        }
    except Exception as exc:
        return {"path": path, "error": str(exc)}


def get_duration(path: str) -> float:
    """Duration of a media file in seconds (0.0 if it cannot be read)."""
    return float(probe(path).get("duration", 0.0))


def has_audio(path: str) -> bool:
    """Check if a video file has an audio stream."""
    return bool(probe(path).get("has_audio", False))


def _escape_drawtext(text: str) -> str:
    """Escape a caption for use inside drawtext=text='...'."""
    return (
        text.replace("\\", "\\\\")
        .replace("'", "’")      # typographic apostrophe avoids filtergraph quoting issues
        .replace(":", "\\:")
        .replace("%", "\\%")
    )


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
    """
    Trim, scale/pad to the platform frame, normalise to 30 fps and optionally burn
    a caption. Always outputs an audio track (silent if the source has none) so
    segments can be concatenated. If the caption cannot be drawn (e.g. no font
    available to drawtext), the segment is rendered again without it.
    """
    ff = get_ffmpeg()
    duration = round(end - start, 3)
    if duration <= 0:
        return False

    vf_parts = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "setsar=1",
        "fps=30",
        "format=yuv420p",
    ]
    if caption:
        font_file = os.getenv("CAPTION_FONT_FILE", "").replace("\\", "/").replace(":", "\\:")
        font_opt = f"fontfile='{font_file}':" if font_file else ""
        vf_parts.append(
            f"drawtext={font_opt}text='{_escape_drawtext(caption)}':fontcolor=white:fontsize=48:"
            f"x=(w-text_w)/2:y=(h-text_h)-60:box=1:boxcolor=black@0.5:boxborderw=8"
        )

    cmd = [ff, "-y", "-ss", str(start), "-i", input_path, "-t", str(duration)]
    if not has_audio:
        # No audio: add a silent track so all segments have matching streams
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-map", "0:v", "-map", "1:a", "-shortest"]
    cmd += [
        "-vf", ",".join(vf_parts),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        output_path, "-loglevel", "error",
    ]

    r = run(cmd, timeout=120)
    if r.returncode == 0 and Path(output_path).exists():
        return True
    if caption:
        return trim(input_path, start, end, output_path, width, height, "", has_audio)
    return False


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
        dur = get_duration(video_path) or 5.0
        cmd = [get_ffmpeg(), "-y", "-ss", str(dur * at_pct), "-i", video_path,
               "-vframes", "1", "-vf", "scale=720:-1", output_path, "-loglevel", "error"]
        run(cmd, timeout=30)
        return Path(output_path).exists()
    except Exception:
        return False
