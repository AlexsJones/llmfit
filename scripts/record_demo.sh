#!/usr/bin/env bash
# Re-record assets/demo.gif from assets/demo.tape.
#
# Needs: vhs (brew install vhs), ffmpeg, and the llmfit you want to show on PATH.
# Run from the repository root:  scripts/record_demo.sh
#
# VHS renders PNG frames (a text layer and a cursor layer); this script
# composites them and builds the GIF itself because:
#   - dither=none keeps flat terminal colours clean. Dithering is what makes
#     terminal GIFs grainy, and it also bloats them.
#   - 20 fps is exactly 5 centiseconds, the GIF delay unit, so playback speed
#     is exact (24 fps would round to 25 and run 4% fast).
#   - VHS 0.12's own encode step fails silently against ffmpeg 9.
set -euo pipefail

TAPE="$PWD/assets/demo.tape"
OUT="assets/demo.gif"
# VHS renders into its temp dir and renames the frames into place, which
# fails silently across filesystems (tmpfs /tmp -> the repo). Run it from a
# temp working directory so the rename stays on one filesystem.
WORK="$(mktemp -d)"
# Registered before anything can fail, so an early exit leaves nothing behind.
# The tape keeps its throwaway config dir under $WORK too.
trap 'rm -rf "$WORK"' EXIT
FRAMES="$WORK/demo-frames"
FPS=20
BG=0x1e1e2e   # Catppuccin Mocha background, matches the tape's theme
PAD=12        # matches `Set Padding` in the tape

command -v vhs >/dev/null || { echo "vhs not found: brew install vhs" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
[ -f "$TAPE" ] || { echo "run from the repository root" >&2; exit 1; }

echo "Recording with $(llmfit --version)"
(cd "$WORK" && vhs "$TAPE" >/dev/null 2>&1)
[ -f "$FRAMES/frame-text-00001.png" ] || { echo "vhs produced no frames" >&2; exit 1; }

SIZE="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
  -of csv=p=0:s=x "$FRAMES/frame-text-00001.png")"
W="${SIZE%x*}"
H="${SIZE#*x}"
BASE="[0:v][1:v]overlay=format=auto,pad=$((W + 2 * PAD)):$((H + 2 * PAD)):${PAD}:${PAD}:color=${BG},fps=${FPS}"
IN=(-framerate 24 -i "$FRAMES/frame-text-%05d.png" -framerate 24 -i "$FRAMES/frame-cursor-%05d.png")
PALETTE="$WORK/palette.png"   # not `mktemp --suffix`: BSD mktemp (macOS) lacks it

# One palette across the whole clip (it spans three TUI themes), then
# quantise without dithering.
ffmpeg -v error -y "${IN[@]}" \
  -filter_complex "${BASE},palettegen=stats_mode=full:max_colors=256:reserve_transparent=0[p]" \
  -map "[p]" "$PALETTE"
ffmpeg -v error -y "${IN[@]}" -i "$PALETTE" \
  -filter_complex "${BASE}[v];[v][2:v]paletteuse=dither=none:diff_mode=rectangle" \
  -loop 0 "$OUT"

ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration \
  -of default=nw=1 "$OUT" | tr '\n' ' '
echo; ls -la "$OUT" | awk '{printf "%s  %.2f MB\n", $9, $5/1048576}'
