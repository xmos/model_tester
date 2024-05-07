from xmos_ai_tools import xformer

TFLITE_MODEL_PATH = "models/mobilenetv2.tflite"
OPTIMIZED_MODEL_PATH = "host_app/src/model.tflite"

print("Generating host app cpp files for model...")
xformer.convert(
    TFLITE_MODEL_PATH,
    OPTIMIZED_MODEL_PATH,
    {
        "xcore-thread-count": "5",
        # set conv err threshold
        "xcore-conv-err-threshold": "3",
    },
)
xformer.print_optimization_report()

OPTIMIZED_MODEL_PATH = "device_app/src/model.tflite"
WEIGHT_PARAMS_PATH = "device_app/src/model_weights.params"
FLASH_IMAGE_PATH = "device_app/src/xcore_flash_binary.out"
print("Generating device app cpp files for model...")
xformer.convert(
    TFLITE_MODEL_PATH,
    OPTIMIZED_MODEL_PATH,
    {
        "xcore-thread-count": "5",
        # operation splitting
        "xcore-op-split-tensor-arena": "True",
        "xcore-op-split-top-op": "0",
        "xcore-op-split-bottom-op": "10",
        "xcore-op-split-num-splits": "7",
        # move weights to flash params file
        "xcore-weights-file" : WEIGHT_PARAMS_PATH,
        # set conv err threshold
        "xcore-conv-err-threshold": "3",
    },
)
xformer.print_optimization_report()

# Generate flash binary
xformer.generate_flash(
    output_file=FLASH_IMAGE_PATH,
    model_files=[OPTIMIZED_MODEL_PATH],
    param_files=[WEIGHT_PARAMS_PATH],
)

print("Done!")
