#!/bin/bash
set -uo pipefail   # removed -e

WORKING_DIR=${WORKING_DIR:-$(pwd)}
TEST_DIR="$WORKING_DIR/test/test_pcc"

TT_CKPT="$WORKING_DIR/data/ckpts/ssr_tt.pth"
TT_CONFIG="projects/configs/SSR_e2e.py"
EMBED_PATH="$WORKING_DIR/data/embeddings/tensor_dict.pth"

cd "$WORKING_DIR/tt"
python -m tracy -r run.py \
    --data_config "$TT_CONFIG" \
    --tt_config "$TT_CONFIG" \
    --tt_checkpoint "$TT_CKPT" \
    --embeddings "$EMBED_PATH" \
    --return_model 1