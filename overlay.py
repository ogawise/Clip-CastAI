"""
ClipCast AI - Component 2: Video Overlay.

Combines a base clip and a POV clip into one vertical (1080x1920)
picture-in-picture MP4 by building a raw ffmpeg command and running it.

Usage:
    python overlay.py --base downloads/pexels_XXXX.mp4 --pov my_pov.mp4 --output outputs/result.mp4
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# --- Configuration ----------------------------------------------------------

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
PIP_PADDING_PX = 30

# In "both" mode the base audio is turned down to this fraction of its
# original volume so the POV (at full volume) sits on top of it.
BASE_VOLUME_IN_MIX = 0.3

VIDEO_PRESET = "medium"  # slower preset = smaller file, same quality
VIDEO_CRF = 23  # quality: lower = better and bigger (18-28 is sensible)

# Overlay x:y expressions. W/H = background size, w/h = PIP size.
POSITIONS = {
    "bottom-right": f"W-w-{PIP_PADDING_PX}:H-h-{PIP_PADDING_PX}",
    "bottom-left": f"{PIP_PADDING_PX}:H-h-{PIP_PADDING_PX}",
    "top-right": f"W-w-{PIP_PADDING_PX}:{PIP_PADDING_PX}",
    "top-left": f"{PIP_PADDING_PX}:{PIP_PADDING_PX}",
}

AUDIO_MODES = ["base", "pov", "both", "none"]


# --- Helpers ----------------------------------------------------------------

def pip_width_px(pip_size_percent):
    """Convert a % of output width into an even pixel width (H.264 needs even sizes)."""
    width = round(OUTPUT_WIDTH * pip_size_percent / 100 / 2) * 2
    return max(2, width)


def has_audio(ffprobe, path):
    """Return True if the file has at least one audio stream."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe exited with code {result.returncode}")
    return bool(result.stdout.strip())


def resolve_audio_mode(requested, base_has_audio, pov_has_audio):
    """Decide which audio mode can actually be used, given which clips have audio.

    Returns (mode, note). `note` explains any fallback, or is None.
    """
    if requested == "base" and not base_has_audio:
        return "none", "base clip has no audio track, so the output will be silent"
    if requested == "pov" and not pov_has_audio:
        return "none", "POV clip has no audio track, so the output will be silent"
    if requested == "both":
        if base_has_audio and pov_has_audio:
            return "both", None
        if pov_has_audio:
            return "pov", "base clip has no audio track, using POV audio only"
        if base_has_audio:
            return "base", "POV clip has no audio track, using base audio only"
        return "none", "neither clip has an audio track, so the output will be silent"
    return requested, None


def build_ffmpeg_command(ffmpeg, base, pov, output, pip_width, position, audio_mode):
    """Build the ffmpeg argument list. Input 0 is the base clip, input 1 is the POV."""
    W, H = OUTPUT_WIDTH, OUTPUT_HEIGHT
    filters = [
        # Fit the base inside 1080x1920 without stretching, then centre it on a
        # black canvas. setsar=1 marks the pixels as square.
        f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1[base]",
        # Scale the POV to the PIP width; -2 keeps the aspect ratio with an even height.
        f"[1:v]scale={pip_width}:-2,setsar=1[pov]",
        # Draw the POV on the base; stop when either video ends. yuv420p is the
        # 8-bit pixel format every phone and Instagram can play.
        f"[base][pov]overlay={POSITIONS[position]}:shortest=1,format=yuv420p[v]",
    ]
    if audio_mode == "both":
        filters.append(
            f"[0:a]volume={BASE_VOLUME_IN_MIX}[base_a];"
            f"[base_a][1:a]amix=inputs=2:duration=longest:normalize=0[a]"
        )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-i", str(base),
        "-i", str(pov),
        "-filter_complex", ";".join(filters),
        "-map", "[v]",
    ]

    if audio_mode == "base":
        cmd += ["-map", "0:a:0"]
    elif audio_mode == "pov":
        cmd += ["-map", "1:a:0"]
    elif audio_mode == "both":
        cmd += ["-map", "[a]"]

    cmd += ["-c:v", "libx264", "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF)]
    if audio_mode == "none":
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac"]

    # -shortest makes the whole file (audio included) end with the shortest
    # stream; overlay's shortest=1 alone only stops the video.
    cmd += ["-shortest", "-movflags", "+faststart", str(output)]
    return cmd


def format_command(cmd):
    """Render the argument list as a command you can paste into PowerShell, cmd or bash.

    Only used for display: subprocess.run() receives the list itself, so no
    shell ever parses the real command. Double quotes are safe here because
    none of our arguments contain $, backticks or quote characters.
    """
    special = set(" ;[]()&|<>")
    return " ".join(f'"{arg}"' if special & set(arg) else arg for arg in cmd)


# --- Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine a base clip and a POV clip into a vertical picture-in-picture video."
    )
    parser.add_argument("--base", required=True, type=Path, help="path to the licensed base clip")
    parser.add_argument("--pov", required=True, type=Path, help="path to the POV/reaction clip")
    parser.add_argument("--output", required=True, type=Path, help="output video path (.mp4)")
    parser.add_argument("--pip-size", type=float, default=30,
                        help="PIP width as %% of the output width (default 30)")
    parser.add_argument("--position", choices=list(POSITIONS), default="bottom-right",
                        help="PIP position (default bottom-right)")
    parser.add_argument("--audio-source", choices=AUDIO_MODES, default="base",
                        help="which audio to keep (default base)")
    args = parser.parse_args()
    if not 0 < args.pip_size <= 100:
        parser.error("--pip-size must be greater than 0 and at most 100")
    return args


def main():
    args = parse_args()

    if not args.base.is_file():
        print(f"Error: Base video file does not exist:\n{args.base}", file=sys.stderr)
        return 1
    if not args.pov.is_file():
        print(f"Error: POV video file does not exist:\n{args.pov}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(
            "Error: FFmpeg was not found on your system PATH.\n"
            "Install FFmpeg and confirm that this command works:\n"
            "ffmpeg -version",
            file=sys.stderr,
        )
        return 1
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        print(
            "Error: ffprobe was not found on your system PATH.\n"
            "It ships with FFmpeg; confirm that this command works:\n"
            "ffprobe -version",
            file=sys.stderr,
        )
        return 1

    try:
        base_has_audio = has_audio(ffprobe, args.base)
        pov_has_audio = has_audio(ffprobe, args.pov)
    except RuntimeError as exc:
        print(f"Error: Could not read one of the input videos:\n{exc}", file=sys.stderr)
        return 1

    audio_mode, audio_note = resolve_audio_mode(args.audio_source, base_has_audio, pov_has_audio)
    pip_width = pip_width_px(args.pip_size)
    cmd = build_ffmpeg_command(ffmpeg, args.base, args.pov, args.output,
                               pip_width, args.position, audio_mode)

    print(f"Base:     {args.base}")
    print(f"POV:      {args.pov}")
    print(f"Output:   {args.output} ({OUTPUT_WIDTH}x{OUTPUT_HEIGHT})")
    print(f"PIP position: {args.position} ({pip_width}px wide, {PIP_PADDING_PX}px padding)")
    if audio_note:
        print(f"Audio source: {args.audio_source} -> {audio_mode} ({audio_note})")
    else:
        print(f"Audio source: {audio_mode}")
    print("Duration behavior: the output will end when the shorter clip ends.")
    print()
    print("Running FFmpeg command:")
    print(format_command(cmd))
    print()

    # ffmpeg's stderr is not captured: it streams straight to the terminal, so
    # you see the live progress line and, on failure, ffmpeg's own error text.
    # Flush first so our messages appear before ffmpeg's when output is piped.
    sys.stdout.flush()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: FFmpeg failed with exit code {result.returncode} (see its output above).",
              file=sys.stderr)
        return result.returncode

    print(f"\nOutput created successfully:\n{args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
