#!/usr/bin/env bash
# group_videos.sh
set -euo pipefail
shopt -s nullglob

# Adjust these if needed
SRC_DIR="VideoVanzetti04042024"
DEST_BASE="videos"

# Set to 1 for a dry run (prints what it would do)
DRY_RUN=${DRY_RUN:-0}

# Process every Ch*.mp4 in the source directory
for f in "$SRC_DIR"/Ch*.mp4; do
  filename="$(basename "$f")"

  # Expect: ChX_<NEWDIRNAME>.mp4
  ch="${filename%%_*}"                 # e.g., Ch1
  rest="${filename#*_}"                # e.g., 04-04-2024_..._S1.mp4
  newdirname="${rest%.mp4}"            # e.g., 04-04-2024_..._S1

  # Make destination dir like videos/04-04-2024_..._S1
  dest_dir="$DEST_BASE/$newdirname"
  dest_file="$dest_dir/$ch.mp4"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "mkdir -p \"$dest_dir\""
    echo "mv -n \"$f\" \"$dest_file\""
  else
    mkdir -p "$dest_dir"
    # -n avoids overwriting if the file already exists
    mv -n "$f" "$dest_file"
  fi
done

echo "Done."
