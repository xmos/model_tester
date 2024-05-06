from xmos_ai_tools import xformer

TFLITE_MODEL_PATH = "models/MODEL.tflite"
OPTIMIZED_MODEL_PATH = "host_app/src/model.tflite"

OPT_MODEL_PATH = "src/model.tflite"

print("Generating host app cpp files for model...")
xformer.convert(
    TFLITE_MODEL_PATH,
    OPTIMIZED_MODEL_PATH,
    {
        "xcore-thread-count": "5",
        # "xcore-conv-err-threshold": "3",
    },
)

OPTIMIZED_MODEL_PATH = "device_app/src/model.tflite"
# WEIGHT_PARAMS_PATH = "src/model_weights.params"
# FLASH_IMAGE_PATH = "src/xcore_flash_binary.out"
print("Generating device app cpp files for model...")
xformer.convert(
    TFLITE_MODEL_PATH,
    OPTIMIZED_MODEL_PATH,
    {
        "xcore-thread-count": "5",
        # "xcore-conv-err-threshold": "3",
    },
)

# Generate flash binary
# xformer.generate_flash(
#     output_file=FLASH_IMAGE_PATH,
#     model_files=[OPTIMIZED_MODEL_PATH],
#     param_files=[WEIGHT_PARAMS_PATH],
# )

print("Done!")
