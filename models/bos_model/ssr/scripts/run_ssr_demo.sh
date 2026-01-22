#!/usr/bin/env bash
set -e

# Keep terminal open on error so you can read the traceback
trap 'status=$?; echo ""; echo "[ERROR] Script failed with exit code $status"; read -p "Press Enter to close terminal, Ctrl+C to abort..."; exit $status' ERR

MODE="$1"
shift || true

if [[ -z "$MODE" ]]; then
    echo "Usage: $0 <functional|performance|cpu|demo> [--config path] [--tt_checkpoint path] [--pt_checkpoint path] [--launcher none] [--eval type] [--patch path] [--bev-map]"
    exit 1
fi

# ====== Default Variables ======
# WORKING_DIR="/path/to/working_dir"

# Defaults for arguments
TT_CONFIG="$WORKING_DIR/tt/projects/configs/SSR_e2e.py"
PT_CONFIG="$WORKING_DIR/reference/projects/configs/SSR_e2e.py"
DATA_CONFIG="$WORKING_DIR/tt/projects/configs/SSR_e2e.py"
EMBED_PATH="$WORKING_DIR/data/embeddings/tensor_dict.pth"
BEV_MAP_FLAG=""

# Mode-specific default checkpoints
DEFAULT_TT_CKPT="$WORKING_DIR/data/ckpts/ssr_tt.pth"
DEFAULT_PT_CKPT="$WORKING_DIR/data/ckpts/ssr_pt.pth"

# ====== Parse optional overrides ======
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --tt_checkpoint) TT_CHECKPOINT="$2"; shift 2 ;;
        --pt_checkpoint) PT_CHECKPOINT="$2"; shift 2 ;;
        --embeddings) EMBED_PATH="$2"; shift 2 ;;
        --bev-map) BEV_MAP_FLAG="--bev_map"; shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TT_CHECKPOINT="${TT_CHECKPOINT:-$DEFAULT_TT_CKPT}"
PT_CHECKPOINT="${PT_CHECKPOINT:-$DEFAULT_PT_CKPT}"

# ====== Mode execution ======
case "$MODE" in
    functional)
        PATCH="${PATCH:-$DEFAULT_PATCH}"
        cd "$WORKING_DIR/tt"
        python run.py \
            --data_config "$DATA_CONFIG" \
            --tt_config "$TT_CONFIG" \
            --pt_config "$PT_CONFIG" \
            --tt_checkpoint "$TT_CHECKPOINT" \
            --pt_checkpoint "$PT_CHECKPOINT" \
            --embeddings "$EMBED_PATH" \
            --enable_persistent_kernel_cache \
            --validate
        ;;

    performance)
        cd "$WORKING_DIR/tt"
        python run.py \
            --data_config "$DATA_CONFIG" \
            --tt_config "$TT_CONFIG" \
            --tt_checkpoint "$TT_CHECKPOINT" \
            --embeddings "$EMBED_PATH" \
            --enable_persistent_kernel_cache 
        ;;

    demo)
        cd "$WORKING_DIR/tt"
        python run.py \
            --data_config "$DATA_CONFIG" \
            --tt_config "$TT_CONFIG" \
            --tt_checkpoint "$TT_CHECKPOINT" \
            --embeddings "$EMBED_PATH" \
            --enable_persistent_kernel_cache \
            --visualize \
            --repeat \
            --realtime \
            $BEV_MAP_FLAG
        ;;

    cpu)
        cd "$WORKING_DIR/reference"
        python run.py \
            --config "$PT_CONFIG" \
            --checkpoint "$PT_CHECKPOINT" \
            --eval "bbox" \
            --launcher "none"
        ;;

    *)
        echo "Invalid mode: $MODE"
        echo "Valid modes are: functional, performance, cpu, demo"
        exit 1
        ;;
esac
