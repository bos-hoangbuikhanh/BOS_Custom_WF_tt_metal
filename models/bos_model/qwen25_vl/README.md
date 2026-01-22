# Qwen2.5-VL

## Introduction
This codebase includes the Qwen2.5-VL family of models and currently supports the model variants:
- Qwen2.5-VL-3B-Instruct: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Qwen2.5-VL-7B-Instruct: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Qwen2.5-VL-3B-Instruct-AWQ: [Qwen/Qwen2.5-VL-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ)
- Qwen2.5-VL-7B-Instruct-AWQ: [Qwen/Qwen2.5-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ)

## Set environment variables
```
# at $TT_METAL_HOME
source env_set.sh
```

## How to Run
### For a single user example:
```
HF_MODEL=<model_name> pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'accuracy and batch1-trace'
```

**Notes:**
- `<model_name>` is the HuggingFace model repo string, e.g. `Qwen/Qwen2.5-VL-3B-Instruct` or `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen2.5-VL-3B-Instruct-AWQ`, `Qwen2.5-VL-7B-Instruct-AWQ`.
- `-k` is the pytest filter; to run a specific test, use `-k <test_name>`; additional test names are listed in `models/bos_model/qwen25_vl/demo/vision_demo.py`
- `models/bos_model/qwen25_vl/demo/outputs` - path to the directory containing dumped vision outputs.
- `--res` — optional flag to specify the input resolution for vision tests (currently supports `128×128` and `224×224`; default is `224×224`).

### For a batch user example:

```
HF_MODEL=<model_name> pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'accuracy and batch2-trace'
```
- Note: The current implementation supports a batch size of 2.


### To capture Tracy report:
```
HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct python -m tracy -m -r -p -v "pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'accuracy and profiler'"
```

**Notes:**
- The model name `Qwen/Qwen2.5-VL-7B-Instruct` can be changed to `Qwen/Qwen2.5-VL-3B-Instruct` if you want to record Tracy 3B model.
- `-k` is the pytest filter; to run a specific test and `profiler` is a special test case for tracy recording or ttnn-visualizer. We've fixed profiling environment for stable device time analysis, so please use this test case for profiling.
  - res = [224, 224], max_batch_size = 1, warmup_iters=0, include_text_only_prompts = False
- The `accuracy`(BF16) can be changed to `performance`(BFP8-mixed).
- `generated/profiler/reports` - path to the directory containing tracy report. Please refer [tt-perf-report](https://github.com/tenstorrent/tt-perf-report) to read the report.

### For Live chatting demo example:
```
HF_MODEL=<model_name> python models/bos_model/qwen25_vl/demo_qwen25_vl.py -i <image_path>
```

**Notes:**
- Use `-i, --image-path` to pass input image path to the model. e.g. `models/bos_model/qwen25_vl/demo/images/dog.jpg`.
- Use `-c, --context` to enable Qwen to remember context.
- Use `-d, --display` to show the image what you add
- Since image tokens are large, the context will grow with each interaction. Over time, this can exceed memory limits, so for longer chats, it is recommended to run Qwen without context.
- You can exit the chat by using `/bye`

## Details
- On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs.
These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models.


## Run for ttnn-visualizer Profiler
- First, export ENV using script file
  - ```$EXPERIMENT_NAME```: input anythings (for example, ```qwen```)
```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```

- Second, run model
  - If the model has finished running successfully, the result report will be generated in the following path (```generated/ttnn/reports/$EXPERIMENT_NAME_MMDD_hhmm/```)
```
HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'accuracy and profiler'
```

- Third, run ttnn-visualizer
    - ```$REPORT_PATH```: It is the path mentioned in the previous step
    - visit ```http://localhost:8000/``` using your web-browser
```
ttnn-visualizer --profiler-path $REPORT_PATH
```

- If the experiment has finished, please run the following command to clear the environment variables
```
source models/bos_model/unset_l1_vis.sh
```
