# Run Images Test (from tt-metal home)

## Set environment variables
```
# at $TT_METAL_HOME
source env_set.sh
```

## Download weights
```
python ./models/bos_model/yolov8s/weights_downloader.py
```

## Run Test (Image)
```
python ./models/bos_model/yolov8s/run_yolov8s.py
```
- input: `reference/images`
- output: `results`


## Run Test with Trace (Image)
```
python ./models/bos_model/yolov8s/run_yolov8s.py --trace -i 320 -n 10
```
- Trace makes model execution much more fast.
- Option i means the size of the image. Maximum I is 320.
- 'n' means Number of iteration.

## Run Video Demo

- Video file is required in `videos` . Please make "videos" folder.
- You can download the video from the below links.
  - Internal link: "https://bos-semi.atlassian.net/wiki/spaces/AIMultimed/pages/102400287/Yolov8s+demo+video".
  - External link: "https://bos-semi-demo-contents.s3.ap-northeast-2.amazonaws.com/public/demo_logo_text_v1.1.mp4" (Right click & Save video as ...)


```
python ./models/bos_model/yolov8s/demo_yolov8s.py
```

## Run for ttnn-visualizer Profiler
- First, export ENV using script file
  - ```$EXPERIMENT_NAME```: input anythings (for example, ```yolo```)
```
source models/bos_model/export_l1_vis.sh $EXPERIMENT_NAME
```

- Second, run model
  - If the model has finished running successfully, the result report will be generated in the following path (```generated/ttnn/reports/$EXPERIMENT_NAME_MMDD_hhmm/```)
```
python ./models/bos_model/yolov8s/run_yolov8s.py
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
