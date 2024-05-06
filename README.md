# Simple AI model tester for host and device

## Requirements

XTC Tools 15.2.1

Python 3.10

## Cloning the repository

Clone the repository with the following command:

git clone git@github.com:xmos/model_tester.git

## Setup Python Virtual Environment

Run the following command to create a virtual environment:  

    python3 -m venv .venv

Run the following command to activate:

    source .venv/bin/activate

## Install Python Packages

Install xmos-ai-tools with:

    pip install xmos-ai-tools --pre --upgrade

## Copy in model files

Copy your model files into the `models` folder and edit the `generate_optimized_cpp_for_xcore.py` script accordingly.

## Optimizing Models for XCORE

The Python script `generate_optimized_cpp_for_xcore.py` optimizes the quantized Tensorflow Lite model for xcore and creates optimized `model.tflite`, `model.tflite.cpp` and `model.tflite.h` in `host_app/src` and in `device_app/src`. Run the following command in the root of the repository:

    python generate_optimized_cpp_for_xcore.py

## Set environment variable to the installed XMOS AI Tools runtime

Setup ``XMOS_AITOOLSLIB_PATH`` environment variable. This is used by the build system to identify the installed location of xmos-ai-tools library and headers.

  On Windows, run the following command::

    FOR /F "delims=" %i IN ('python -c "import xmos_ai_tools.runtime as rt; import os; print(os.path.dirname(rt.__file__))"') DO set XMOS_AITOOLSLIB_PATH=%i

  On MacOS and Linux, run the following command::

    export XMOS_AITOOLSLIB_PATH=$(python -c "import xmos_ai_tools.runtime as rt; import os; print(os.path.dirname(rt.__file__))")

## Building the XCORE application for host

The `host_app` folder contains source files for host. We are using CMake and Ninja to build app on host so that it is portable across Linux, Windows, and macOS. Run the following command in the `host_app` folder to build the application:

    mkdir build
    cd build
    cmake .. -G Ninja
    ninja

### Running the XCORE application on host

Run:

    cd host_app/build
    ./app_host


## Building the XCORE application for hardware

The `device_app` folder contains source file for device. Run the following command in the `device_app` folder to build the application:

    xmake

### Running the XCORE application on hardware

Run:

    xrun --xscope bin/app_device
