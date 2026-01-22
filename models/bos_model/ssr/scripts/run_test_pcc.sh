#!/bin/bash
set -uo pipefail   # removed -e

WORKING_DIR=${WORKING_DIR:-$(pwd)}
TEST_DIR="$WORKING_DIR/test/test_pcc"

echo "Running pytest on all test files in: $TEST_DIR"

FAILED_TESTS=()

for test_file in "$TEST_DIR"/*.py; do
    if [[ -f "$test_file" ]]; then
        echo "========================================="
        echo " Running: pytest -sv $test_file"
        echo "========================================="
        
        if ! pytest -sv "$test_file"; then
            echo "[ERROR] Test failed: $test_file"
            FAILED_TESTS+=("$test_file")
            # continue with next file
        fi
    fi
done

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo ""
    echo "========================================="
    echo "[SUMMARY] Some tests failed:"
    for f in "${FAILED_TESTS[@]}"; do
        echo "  - $f"
    done
    echo "========================================="
    exit 1
else
    echo ""
    echo "[SUMMARY] All tests passed ✅"
fi
