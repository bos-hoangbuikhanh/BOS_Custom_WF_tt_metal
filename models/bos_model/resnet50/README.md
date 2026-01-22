# Run Images Test (from tt-metal home)

## (Optional) Download ImageNet-1K(validation set)
- ImageNet-1K is uploaded to "https://www.kaggle.com/datasets/titericz/imagenet1k-val" (Log-in required)
```
unzip imagenet-val.zip -d imagenet-val
```

## Set environment variables
```
# at $TT_METAL_HOME
source env_set.sh
```

## Run ResNet50 (Image)
```
python models/bos_model/resnet50/run_resnet50.py [--device_id $DEVICE_ID] [--batch $BATCH_SIZE] [-n $ITER] [--trace] [--cq2] [--data_dir /path/to/imagenet-1k] [--seed $SEED_VALUE] [--no_shuffle] [--demo] [--fullscreen] [--delay $DELAY_TIME] [--benchmark]
```
- ```--device_id```: If you have multi device, you can choose device for run model(```$DEVICE_ID``` = ```0``` is default)
- ```-b``` or ```--batch```: You can choose batch size(```B``` = [```1```, ```2```, ```4```(default)])
- ```-n```: You can run not only 1 time but also ```$ITER``` times(default is 1)
- ```--trace```: If you use this option, you can use trace(makes model execution much more faster)
- ```--cq2```: If you use this option with ```--trace```, you can use 2 command queue(makes model execution more faster)
- ```--data_dir```: You can choose dataset(```XXX``` = [```models/bos_model/demo/dataset/sample/```(default), ```/path/to/imagenet-1k```])
- ```--seed```: If you want to shuffle dataset using another random seed, you can use another integer value(```$SEED_VALUE``` = ```0``` is default)
- ```--no_shuffle```: If you don't want to shuffle dataset, use this option
- ```--demo```: If you use this option, you can see demo window (default: only result show on console)
- ```--fullscreen```: If you use this option, you can see full size demo window
- ```-delay```: You can run with delay(```$DELAY_TIME``` is floating number, default is 0.0)
- ```--benchmark```: If you use this option, you can see model's e2e performance which called FPS (you can see FPS also using ```--demo```)

### Example: Run Demo (with Batch 4, ImageNet-1K, Trace, 2CQ, Visualization+Full Screen, 1 Time)
```
python models/bos_model/resnet50/run_resnet50.py --data_dir models/bos_model/demo/dataset/imagenet-val/ --trace --cq2 --demo --fullscreen
```

### Run with Tracy Profiler
```
python -m tracy -r -p -v -m pytest models/bos_model/resnet50/tests/test_ttnn_functional_resnet50.py
```

### Run for ttnn-visualizer Profiler
- First, export ENV using script file
  - ```$EXPERIMENT_NAME```: input anythings (for example, ```resnet```)
```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```

- Second, run model
  - If the model has finished running successfully, the result report will be generated in the following path (```generated/ttnn/reports/$EXPERIMENT_NAME_MMDD_hhmm/```)
```
pytest models/bos_model/resnet50/tests/test_ttnn_functional_resnet50.py -k batch_4
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
