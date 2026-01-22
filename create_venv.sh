#!/bin/bash
set -eo pipefail

# Allow overriding Python command via environment variable
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
else
    echo "Using user-specified Python: $PYTHON_CMD"
fi

# Verify Python command exists
if ! command -v $PYTHON_CMD &>/dev/null; then
    echo "Python command not found: $PYTHON_CMD"
    exit 1
fi

# Function to create a virtual environment
create_venv() {
    local ENV_DIR=$1
    local MODEL_TYPE=$2

    echo ""
    echo "=========================================="
    echo "Creating virtual environment: $ENV_DIR"
    echo "Model type: $MODEL_TYPE"
    echo "=========================================="

    # Create and activate virtual environment
    $PYTHON_CMD -m venv $ENV_DIR
    source $ENV_DIR/bin/activate

    # Import functions for detecting OS
    . ./install_dependencies.sh --source-only
    detect_os

    if [ "$OS_ID" = "ubuntu" ] && [ "$OS_VERSION" = "22.04" ]; then
        echo "Ubuntu 22.04 detected: force pip/setuptools/wheel versions"
        pip install --force-reinstall pip==25.1.1
        python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
        python3 -m pip install setuptools wheel==0.45.1
    else
        echo "$OS_ID $OS_VERSION detected: updating wheel and setuptools to latest"
        python3 -m pip install --upgrade wheel setuptools
    fi

    echo "Installing dev dependencies"
    python3 -m pip install -r $(pwd)/tt_metal/python_env/requirements-dev.txt
    echo "Installing model specific dependencies"
    python3 -m pip install -r $(pwd)/models/requirements-model.txt

    # Install model-specific requirements based on model type
    if [ "$MODEL_TYPE" = "ssr" ]; then
        echo "Installing SSR model dependencies"
        python3 -m pip install -r $(pwd)/models/bos_model/ssr/requirements.txt
    elif [ "$MODEL_TYPE" = "qwen3" ]; then
        echo "Installing Qwen3 model dependencies"
        python3 -m pip install -r $(pwd)/models/bos_model/qwen3_vl/requirements.txt
    else # install qwen25_vl dependencies as default
        echo "Installing default model dependencies"
        python3 -m pip install -r $(pwd)/models/bos_model/qwen25_vl/requirements.txt
    fi

    echo "Installing tt-metal"
    SETUPTOOLS_SCM_PRETEND_VERSION=1.2.3 pip install -e .

    # Do not install hooks when this is a worktree
    if [ $(git rev-parse --git-dir) = $(git rev-parse --git-common-dir) ]; then
        echo "Generating git hooks"
        pre-commit install
        pre-commit install --hook-type commit-msg
    else
        echo "In worktree: not generating git hooks"
    fi

    deactivate
    echo "Virtual environment created: $ENV_DIR"
}

# Create both environments
echo "This script will create two Python virtual environments:"
echo "  1. python_env - default environment (include Qwen2.5-VL)"
echo "  2. python_env_ssr - for SSR model"
echo "  3. python_env_qwen3 - for Qwen3-VL model"
echo ""

# Create general environment
create_venv "$(pwd)/python_env" "default"

# Create SSR environment
create_venv "$(pwd)/python_env_ssr" "ssr"

# Create Qwen3 environment
create_venv "$(pwd)/python_env_qwen3" "qwen3"

echo ""
echo "=========================================="
echo "Both virtual environments created successfully!"
echo ""
echo "To use the default environment:"
echo "  source env_set.sh"
echo ""
echo "To use the SSR environment:"
echo "  source env_set.sh ssr"
echo ""
echo "To use the qwen3 environment:"
echo "  source env_set.sh qwen3"
echo "=========================================="
echo ""
echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
