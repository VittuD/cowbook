# Utilities

The top-level `scripts/` and `tools/` directories contain repository utilities. They are optional helpers, not part of the `cowbook` package runtime surface.

`tools/` contains benchmark harnesses and operational experiments. `scripts/` contains smaller one-off helpers such as video regrouping.

## `group_videos.sh`

`scripts/group_videos.sh` reorganizes flat camera files named like:

```text
Ch1_<group>.mp4
Ch4_<group>.mp4
```

into grouped directories like:

```text
videos/<group>/Ch1.mp4
videos/<group>/Ch4.mp4
```

This is useful when a raw video drop arrives as one flat folder and needs to be rearranged into a grouped layout before writing configs.

Dry-run example:

```bash
scripts/group_videos.sh --src raw_drop --dest videos --dry-run
```

Move files:

```bash
scripts/group_videos.sh --src raw_drop --dest videos
```

Copy files instead:

```bash
scripts/group_videos.sh --src raw_drop --dest videos --copy
```

Overwrite existing targets only when `--overwrite` is passed. Without it, existing destination files are preserved.
