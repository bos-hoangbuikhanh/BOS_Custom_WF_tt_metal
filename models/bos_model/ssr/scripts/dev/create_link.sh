#!/usr/bin/env bash
set -Euo pipefail
IFS=$'\n\t'

# --- Inputs ---
: "${WORKING_DIR:?WORKING_DIR must be set}"
: "${DATA_PATH:?DATA_PATH must be set}"

# Resolve to absolute paths if available (nice-to-have)
if command -v realpath >/dev/null 2>&1; then
  WORKING_DIR="$(realpath -m "$WORKING_DIR")"
  DATA_PATH="$(realpath -m "$DATA_PATH")"
fi

# Sanity checks
[ -d "$WORKING_DIR" ] || { echo "Error: WORKING_DIR does not exist: $WORKING_DIR" >&2; }
[ -d "$DATA_PATH" ]   || { echo "Error: DATA_PATH does not exist: $DATA_PATH" >&2; }

# --- Helpers ---
ensure_real_dir() {
  # Make sure $1 is a REAL directory (not a symlink / file)
  local dir="$1"
  if [ -L "$dir" ]; then
    echo "Found symlink: $dir -> removing so we can create a real directory."
    rm -f "$dir"
  elif [ -e "$dir" ] && [ ! -d "$dir" ]; then
    echo "Found non-directory path at $dir -> removing."
    rm -rf "$dir"
  fi
  mkdir -p "$dir"
}

ensure_parent_dir() {
  mkdir -p "$(dirname "$1")"
}

create_symlink() {
  local source="$1"
  local target="$2"

  if [ ! -e "$source" ]; then
    echo "WARN: source does not exist, skipping: $source" >&2
    return 0
  fi

  ensure_parent_dir "$target"

  # Remove any existing file/dir/symlink at target
  if [ -e "$target" ] || [ -L "$target" ]; then
    rm -rf "$target"
  fi

  ln -s "$source" "$target"
  echo "Linked: $target -> $source"
}

# --- Ensure REAL container dirs ---
ensure_real_dir "$WORKING_DIR/data"
ensure_real_dir "$WORKING_DIR/data/dataset"
ensure_real_dir "$WORKING_DIR/data/dataset/nuscenes"

# --- Symlinks outside nuscenes ---
create_symlink "$DATA_PATH/ckpts/SSR/embeddings" "$WORKING_DIR/data/embeddings"
create_symlink "$DATA_PATH/ckpts/SSR/ckpts"       "$WORKING_DIR/data/ckpts"

# --- Symlink nuscenes contents into a REAL nuscenes dir ---
shopt -s nullglob dotglob
for item in "$DATA_PATH/sets/nuScenes/nuscenes"/*; do
  base="$(basename "$item")"
  create_symlink "$item" "$WORKING_DIR/data/dataset/nuscenes/$base"
done

# --- Place the .pkl files alongside those symlinks (still in a REAL dir) ---
create_symlink "$DATA_PATH/sets/nuScenes/SSR/converted_files/vad_nuscenes_infos_temporal_train.pkl" \
               "$WORKING_DIR/data/dataset/nuscenes/vad_nuscenes_infos_temporal_train.pkl"

create_symlink "$DATA_PATH/sets/nuScenes/SSR/converted_files/vad_nuscenes_infos_temporal_val.pkl" \
               "$WORKING_DIR/data/dataset/nuscenes/vad_nuscenes_infos_temporal_val.pkl"

create_symlink "$DATA_PATH/sets/nuScenes/SSR/samples/patch.pt" \
               "$WORKING_DIR/data/dataset/patch.pt"

# --- can_bus goes under dataset root (not inside nuscenes) ---
create_symlink "$DATA_PATH/sets/nuScenes/can_bus/can_bus" \
               "$WORKING_DIR/data/dataset/can_bus"

create_symlink "$WORKING_DIR/data/dataset" \
               "$WORKING_DIR/tt/data"

create_symlink "$WORKING_DIR/data/dataset" \
               "$WORKING_DIR/reference/data"

echo "Symbolic link creation process completed!"
