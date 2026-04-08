#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

usage() {
  cat <<'EOF'
Usage:
  scripts/group_videos.sh [options]

Reorganize flat camera files named like:
  Ch1_<group>.mp4
  Ch4_<group>.mp4

into grouped directories like:
  videos/<group>/Ch1.mp4
  videos/<group>/Ch4.mp4

Options:
  --src DIR          Source directory containing flat Ch*.mp4 files.
                     Default: VideoVanzetti04042024
  --dest DIR         Destination base directory.
                     Default: videos
  --copy             Copy files instead of moving them.
  --dry-run          Print planned operations without changing files.
  --overwrite        Allow overwriting destination files.
  --help             Show this help and exit.

Examples:
  scripts/group_videos.sh --src raw_drop --dest videos --dry-run
  scripts/group_videos.sh --src raw_drop --dest videos
  scripts/group_videos.sh --src raw_drop --dest videos --copy
EOF
}

SRC_DIR="VideoVanzetti04042024"
DEST_BASE="videos"
MODE="move"
DRY_RUN=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC_DIR="$2"
      shift 2
      ;;
    --dest)
      DEST_BASE="$2"
      shift 2
      ;;
    --copy)
      MODE="copy"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory does not exist: $SRC_DIR" >&2
  exit 1
fi

matched=0
for f in "$SRC_DIR"/Ch*.mp4; do
  matched=1
  filename="$(basename "$f")"

  if [[ "$filename" != *_* ]]; then
    echo "Skipping unexpected filename (missing group separator): $filename" >&2
    continue
  fi

  ch="${filename%%_*}"
  rest="${filename#*_}"
  group_name="${rest%.mp4}"

  if [[ -z "$ch" || -z "$group_name" || "$group_name" == "$filename" ]]; then
    echo "Skipping unexpected filename: $filename" >&2
    continue
  fi

  dest_dir="$DEST_BASE/$group_name"
  dest_file="$dest_dir/$ch.mp4"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "mkdir -p \"$dest_dir\""
    if [[ "$MODE" == "copy" ]]; then
      if [[ "$OVERWRITE" -eq 1 ]]; then
        echo "cp \"$f\" \"$dest_file\""
      else
        echo "cp -n \"$f\" \"$dest_file\""
      fi
    else
      if [[ "$OVERWRITE" -eq 1 ]]; then
        echo "mv \"$f\" \"$dest_file\""
      else
        echo "mv -n \"$f\" \"$dest_file\""
      fi
    fi
    continue
  fi

  mkdir -p "$dest_dir"
  if [[ "$MODE" == "copy" ]]; then
    if [[ "$OVERWRITE" -eq 1 ]]; then
      cp "$f" "$dest_file"
    else
      cp -n "$f" "$dest_file"
    fi
  else
    if [[ "$OVERWRITE" -eq 1 ]]; then
      mv "$f" "$dest_file"
    else
      mv -n "$f" "$dest_file"
    fi
  fi
done

if [[ "$matched" -eq 0 ]]; then
  echo "No matching files found in $SRC_DIR" >&2
  exit 1
fi

echo "Done."
