from xmos_ai_tools import xformer

model_path = ""
optimized_model_path = "host_app/src/model.tflite"

print("Generating host app cpp files for model...")
xformer.convert(
    model_path,
    optimized_model_path,
    {
        "xcore-thread-count": "5",
    },
)

optimized_model_path = "device_app/src/model.tflite"

print("Generating device app cpp files for model...")
xformer.convert(
    model_path,
    optimized_model_path,
    {
        "xcore-thread-count": "5",
    },
)

print("Done!")
