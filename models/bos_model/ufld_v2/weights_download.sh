#!/bin/bash
set -e

# ============================================================
# Check if gdown is installed
# ============================================================
if ! python -c "import gdown" &> /dev/null; then
    echo "gdown not found, installing..."
    pip install gdown
else
    echo "gdown is already installed."
fi

# ============================================================
# Parse arguments
#   default: download both
#   options:
#     --tusimple34
#     --culane34
# ============================================================
DOWNLOAD_TUSIMPLE34=false
DOWNLOAD_CULANE34=false

if [[ $# -eq 0 ]]; then
    DOWNLOAD_TUSIMPLE34=true
    DOWNLOAD_CULANE34=true
else
    for arg in "$@"; do
        case $arg in
            --tusimple34)
                DOWNLOAD_TUSIMPLE34=true
                ;;
            --culane34)
                DOWNLOAD_CULANE34=true
                ;;
            *)
                echo "Unknown option: $arg"
                echo "Usage: $0 [--tusimple34] [--culane34]"
                exit 1
                ;;
        esac
    done
fi

# ============================================================
# Output directory
# ============================================================
WEIGHT_DIR="models/bos_model/ufld_v2"
mkdir -p "${WEIGHT_DIR}"

# ============================================================
# Download: TuSimple ResNet-34
# ============================================================
if [[ "$DOWNLOAD_TUSIMPLE34" == true ]]; then
    OUTPUT_TUSIMPLE="${WEIGHT_DIR}/tusimple_res34.pth"
    GOOGLE_DRIVE_URL_TUSIMPLE="https://drive.google.com/file/d/1pkz8homK433z39uStGK3ZWkDXrnBAMmX/view"

    if [[ -f "$OUTPUT_TUSIMPLE" ]]; then
        echo "[TuSimple34] File already exists, skipping: ${OUTPUT_TUSIMPLE}"
    else
        echo "[TuSimple34] Downloading tusimple_res34.pth..."
        if gdown --fuzzy "$GOOGLE_DRIVE_URL_TUSIMPLE" -O "$OUTPUT_TUSIMPLE"; then
            echo "[TuSimple34] Download completed: ${OUTPUT_TUSIMPLE}"
        else
            echo "[TuSimple34] Download failed."
            exit 1
        fi
    fi
fi

# ============================================================
# Download: CULane ResNet-34
# ============================================================
if [[ "$DOWNLOAD_CULANE34" == true ]]; then
    OUTPUT_CULANE="${WEIGHT_DIR}/culane_res34.pth"
    GOOGLE_DRIVE_URL_CULANE="https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view"

    if [[ -f "$OUTPUT_CULANE" ]]; then
        echo "[CULane34] File already exists, skipping: ${OUTPUT_CULANE}"
    else
        echo "[CULane34] Downloading culane_res34.pth..."
        if gdown --fuzzy "$GOOGLE_DRIVE_URL_CULANE" -O "$OUTPUT_CULANE"; then
            echo "[CULane34] Download completed: ${OUTPUT_CULANE}"
        else
            echo "[CULane34] Download failed."
            exit 1
        fi
    fi
fi

echo "All requested weights are ready."
