# BOS-ViT

## How to run

Dependency installation

```sh
# at $TT_METAL_HOME
./create_venv.sh
source $PYTHON_ENV_DIR/bin/activate

```

For basic usage, please refer the [inference.py](./inference.py)

```sh
python models/bos_model/vit/inference.py
```

## E2E benchmark

For the end-to-end throughput benchmark, run

```sh
# at $TT_METAL_HOME
pytest models/bos_model/vit/tests/test_vit_e2e_perf.py::test_e2e_trace2cq
```

You can also generate the tracy result with the command below

```sh
python -m tracy -r -p -m --sync-host pytest models/bos_model/vit/tests/test_vit_e2e_perf.py::test_e2e_trace2cq
```

For the accuracy test with the validation set of [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k). Please locate dataset with this structure

```
/path/to/imagenet-1k
        ├──val
            ├── n01740131
                ├──ILSVRC2012_val_00000337.JPEG
                ├──...
                ├──ILSVRC2012_val_00011776.JPEG
            ├── ..
            ├── n01773157
```

and run

```sh
IMAGENET-1K_VAL_DIR='path/to/imagenet-1k/val' pytest models/bos_model/vit/tests/test_vit_e2e_perf.py::test_accuracy
```

## Run for ttnn-visualizer Profiler
- First, export ENV using script file
  - ```$EXPERIMENT_NAME```: input anythings (for example, ```vit```)
```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```

- Second, run model
  - If the model has finished running successfully, the result report will be generated in the following path (```generated/ttnn/reports/$EXPERIMENT_NAME_MMDD_hhmm/```)
```
python models/bos_model/vit/inference.py
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
