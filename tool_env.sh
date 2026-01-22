#!/bin/bash

# Function to display help
show_help() {
    echo "Usage: [-h] [-w] [-d] [-t] [-v] VISUALIZER_REPORT_NAME"
    echo "  -h  Show this help message."
    echo "  -w  Enable Watcher mode with environment variables set for logging and monitoring."
    echo "  -d  Enable DRPINT"
    echo "  -t  Enable Tracy profiler (also disable DPRINT)"
    echo "  -v  Enable Visualizer and set the report name."
    echo "  -s  Enable Trace Storage/Ops and set the operation storage name."
}

# Default values for the environment variables
generated_dir="generated"

enable_watcher="OFF"
enable_tracy_profiler="OFF"
enable_dprint="OFF"
enable_visualizer="OFF"
visualizer_report_name="visualizer_report"
enable_allocation_tracking="OFF"
allocation_tracker_name="allocation_tracker_data.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h)
            show_help
            return
            ;;
        -w)
            enable_watcher="ON"
            shift
            ;;
        -d)
            enable_dprint="ON"
            shift
            ;;
        -t)
            enable_tracy_profiler="ON"
            shift
            ;;
        -v)
            enable_visualizer="ON"
            if [[ -n $2 && ! $2 =~ ^- ]]; then
                visualizer_report_name=$2
                shift 2
            else
                echo "Error: Option -v requires a report name."
                show_help
                return
            fi
            ;;
        -s)
            enable_allocation_tracking="ON"
            if [[ -n $2 && ! $2 =~ ^- ]]; then
                allocation_tracker_name=$2
                shift 2
            else
                echo "Error: Option -s requires a trace op report name."
                show_help
                return
            fi
            ;;
        *)
            echo "Illegal option $1"
            show_help
            return
            ;;
    esac
done

# Handle Watcher Mode
if [ "$enable_watcher" = "ON" ]; then
    export TT_METAL_WATCHER=120        # the number of seconds between Watcher updates
    # export TT_METAL_WATCHER_APPEND=1   # append to the end of the existing log file
    # export TT_METAL_WATCHER_DUMP_ALL=1 # dump all state including unsafe state
    # Watcher features can be disabled individually using the following environment variables:
    # export TT_METAL_WATCHER_DISABLE_ASSERT=1
    # export TT_METAL_WATCHER_DISABLE_PAUSE=1
    # export TT_METAL_WATCHER_DISABLE_RING_BUFFER=1
    # export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
    # export TT_METAL_WATCHER_DISABLE_STATUS=1
    # export TT_METAL_WATCHER_DISABLE_STACK_USAGE=1
    # In certain cases enabling watcher can cause the binary to be too large. In this case, disable inlining.
    # export TT_METAL_WATCHER_NOINLINE=1
    echo "--- Watcher mode enabled with TT_METAL_WATCHER settings."
else
    # Unset the Watcher environment variables if they are not needed
    unset TT_METAL_WATCHER
    unset TT_METAL_WATCHER_APPEND
    unset TT_METAL_WATCHER_DUMP_ALL
    unset WATCHER_ENABLED
    unset TT_METAL_WATCHER_DISABLE_ASSERT
    unset TT_METAL_WATCHER_DISABLE_PAUSE
    unset TT_METAL_WATCHER_DISABLE_RING_BUFFER
    unset TT_METAL_WATCHER_DISABLE_NOC_SANITIZE
    unset TT_METAL_WATCHER_DISABLE_STATUS
    unset TT_METAL_WATCHER_DISABLE_STACK_USAGE
    unset TT_METAL_WATCHER_NOINLINE
    echo "--- Watcher mode disabled."
fi

# Handle DPRINT Mode
if [ "$enable_dprint" = "ON" ]; then
    export TTNN_ENABLE_LOGGING=1                 # for DPRINT
    export ENABLE_PROFILER=1                     # for DPRINT
    export TT_METAL_DPRINT_CORES=0,0             # for DPRINT
    export TT_METAL_DPRINT_CHIPS=0         # for DPRINT

    echo "--- DPRINT mode enabled"
else
    unset TTNN_ENABLE_LOGGING
    unset ENABLE_PROFILER
    unset TT_METAL_DPRINT_CORES
    unset TT_METAL_DPRINT_CHIPS

    echo "--- DPRINT mode disable"
fi

# Handle Tracy Mode
if [ "$enable_tracy_profiler" = "ON" ]; then
    unset TTNN_ENABLE_LOGGING
    unset ENABLE_PROFILER
    unset TT_METAL_DPRINT_CORES
    unset TT_METAL_DPRINT_CHIPS

    export ENABLE_TRACY=1
    export TT_METAL_DEVICE_PROFILER=1
    export TT_METAL_PROFILER_SYNC=1
    export TT_METAL_DEVICE_PROFILER_DISPATCH=1

    echo "--- Tracy profiler mode enabled."
else
    unset ENABLE_TRACY
    unset TT_METAL_DEVICE_PROFILER
    unset TT_METAL_PROFILER_SYNC
    unset TT_METAL_DEVICE_PROFILER_DISPATCH

    echo "--- Tracy Profiler mode disabled."
fi

# Handle Visualizer Mode
if [ "$enable_visualizer" = "ON" ]; then
    export TTNN_CONFIG_OVERRIDES=$(printf '{
        "enable_fast_runtime_mode": false,
        "enable_logging": true,
        "enable_graph_report": false,
        "enable_detailed_buffer_report": false,
        "enable_detailed_tensor_report": false,
        "enable_comparison_mode": false,
        "report_name": "%s"
    }' "$visualizer_report_name")

    echo "--- Visualizer mode enabled"
    echo "--- TTNN_CONFIG_OVERRIDES='$TTNN_CONFIG_OVERRIDES'"
else
    unset TTNN_CONFIG_OVERRIDES
    echo "--- Visualizer mode disable"
fi


# Handle Trace Storage Mode
if [ "$enable_allocation_tracking" = "ON" ]; then
    export ALLOCATION_TRACKING=1
    export ALLOCATION_TRACKER_NAME=$allocation_tracker_name

    echo "--- Trace storage mode enabled"
    echo "--- ALLOCATION_TRACKER_NAME='$ALLOCATION_TRACKER_NAME'"
else
    unset ALLOCATION_TRACKING
    unset ALLOCATION_TRACKER_NAME
    echo "--- Trace storage mode disable"
fi
