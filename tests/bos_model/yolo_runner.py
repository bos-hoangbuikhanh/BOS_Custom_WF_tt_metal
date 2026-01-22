from models.bos_model.yolov8s.yolov8s import YoloV8sRunner

yolov8_args = {
    "image_shape": [320, 320],
    "trace": True,
    "enable_persistent_cache": True,
}
# ModelRunner = YoloV8sRunner(**runner_args)
ModelRunner = YoloV8sRunner(device_id=0, **yolov8_args)
output = ModelRunner.one_shot(num_images=5)
# for key, value in ModelRunner.model_outputs.items():
#     for i, v in enumerate(value):
#         print(f"{key} - Image {i}: {v.shape}")
#     print(1/ModelRunner.performance[key])

print()

yolov8_args = {
    "device_id": 0,
    "image_shape": 256,
    "input_channels": 3,
    "output_classes": 80,
    "trace": False,
    "dataset": "models/bos_model/yolov8s/reference/images/",
    "weights_dir": "models/bos_model/yolov8s/",
    "enable_persistent_cache": True,
}
ModelRunner = YoloV8sRunner(**yolov8_args)
ttnn_outputs = ModelRunner.run_inference()
# print(ttnn_outputs.keys())
torch_outputs = ModelRunner.run_golden()
# print(torch_outputs.keys())
ModelRunner.check_pcc()
ModelRunner.check_performace()
ModelRunner.deallocate_and_close_device()

print()

yolov8_args = {
    "device_id": 0,
    "image_shape": [320, 320],
    "trace": True,
    "enable_persistent_cache": True,
}
ModelRunner = YoloV8sRunner(**yolov8_args)
ttnn_outputs = ModelRunner.run_inference(num_images=10)
# print(ttnn_outputs.keys())
torch_outputs = ModelRunner.run_golden()
# print(torch_outputs.keys())
ModelRunner.check_pcc(print_performance=True)
ModelRunner.deallocate_and_close_device()
