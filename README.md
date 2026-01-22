# TT-Metal (BOS) Installation Guide


## Initial Setup

```bash
# Clone tt-metal (bos branch)
git clone git@github.com:bos-semi/tt-metal.git --recurse-submodules
cd tt-metal
```


## Build TT-Metal

```bash
# Install required dependencies (sudo required)
sudo ./install_dependencies.sh

# SET ENV
source env_set.sh

# Build TT-Metal
./build_metal.sh -b Release
```


## Hugepage setting for the first set up

Hugepages need to be set up once per computer. Once it's set, there's no need to set it again for the next build.

```bash
cd tt_metal/third_party/umd
sudo -E env PATH=$PATH python3 ./scripts/setup_hugepages.py enable

# restart computer
cd tt_metal/third_party/umd
sudo -E env PATH=$PATH python3 ./scripts/setup_hugepages.py enable
```


## Python Virtual Environment

```bash
# Create virtual environment
./create_venv.sh
```

## Supported Models
Please refer [model](./models/)

## Quick Links

#### [Tracy Profiler](https://github.com/tenstorrent/tracy)
The Tracy Profiler is a real-time nanosecond resolution, remote telemetry, hybrid frame, and sampling tool. Tracy supports profiling CPU, GPU, memory allocation, locks, context switches, and more.

##### IMPORTANT
Currently, cannot print out 1000+ ops at once. Tracy has limitation to print out all the profiling data. Please refer [TT's: profiling_ttnn_operations](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/ttnn/ttnn/profiling_ttnn_operations.rst) or [Ours: profiling_ttnn_operations](./docs/source/ttnn/ttnn/profiling_ttnn_operations.rst)

#### [TT-NN Visualizer](https://github.com/tenstorrent/ttnn-visualizer)
A comprehensive tool for visualizing and analyzing model execution, offering interactive graphs, memory plots, tensor details, buffer overviews, operation flow graphs, and multi-instance support with file or SSH-based report loading.

#### [Watcher](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tools/watcher.rst)
Watcher monitors firmware and kernels for common programming errors, and overall device status. If an error or hang occurs, Watcher displays log data of that occurrence.
