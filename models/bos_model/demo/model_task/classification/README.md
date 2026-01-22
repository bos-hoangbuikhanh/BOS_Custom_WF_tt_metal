# Image Classification Demo

## How to run

### (Optional) Download full ImageNet-1K(validation set)

Currently, we use 32 images from ImageNet-1K(validation set).

If you want runs with the full validation set of [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Please locate dataset with this structure

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


### Just Inference

```bash
# at $TT_METAL_HOME
source env_set.sh
python models/bos_model/demo/model_task/classification/runner.py --model $MODEL_ID [--batch $BATCH_SIZE] [--trace] [--data_dir /path/to/imagenet-1k] [--seed $SEED_VALUE] [--no_shuffle] [--device_id $DEVICE_ID]
```
- ```--model``` : You can choose model which you want to run

| Model                                           | $MODEL_ID                    |
|-------------------------------------------------|------------------------------|
| [ResNet-50 (224x224)](../../../resnet50)        | ```microsoft/resnet-50```          |
| [ViT-base (224x224)](../../../vit)              | ```google/vit-base-patch16-224```  |

- ```--batch``` : You can choose batch size(```$BATCH_SIZE``` = [```1```, ```2```, ```4```(default)])
- ```--trace```: If you use this option, you can use trace(makes model execution much more faster)
- ```--cq2```: If you use this option with ```--trace```, you can use 2 command queue(makes model execution more faster)
- ```--data_dir```: You can choose dataset(```XXX``` = [```demo/dataset/sample```(default), ```/path/to/imagenet-1k```])
- ```--seed```: If you want to shuffle dataset using another random seed, you can use another integer value(```$SEED_VALUE``` = ```0``` is default)
- ```--no_shuffle```: If you don't want to shuffle dataset, use this option
- ```--device_id```: If you have multi device, you can choose device for run model(```$DEVICE_ID``` = ```0``` is default)


### Demo or Benchmark

```bash
# at $TT_METAL_HOME
source env_set.sh
python models/bos_model/demo/model_task/classification/cls_gui_demo.py --model $MODEL_ID [--batch $BATCH_SIZE] [--trace] [--data_dir /path/to/imagenet-1k] [--seed $SEED_VALUE] [--no_shuffle] [--device_id $DEVICE_ID] [-n $ITER] [--demo] [--fullscreen] [-delay $DELAY_TIME]
```
- ```--batch```, ```--trace```, ```--cq2```, ```--data_dir```, ```--seed```, ```--no_shuffle```, ```--device_id```: Same as runner.py's arguments
- ```-n```: You can run not only 1 time but also ```$ITER``` times(default is 1)
- ```--demo```: If you use this option, you can see demo window (default: only result show on console)
- ```-delay```: You can run with delay(```$DELAY_TIME``` is floating number, default is 0.0)
- ```--fullscreen```: If you use this option, you can see full size demo window
