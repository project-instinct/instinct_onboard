import os
import subprocess
import sys

from setuptools import find_packages, setup


def detect_gpu():
    """Detect if GPU/CUDA is available on the system."""
    # Method 1: Check for CUDA runtime
    try:
        import ctypes

        cuda = ctypes.CDLL("libcudart.so")
        return True
    except (ImportError, OSError):
        pass

    # Method 2: Check nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "NVIDIA" in result.stdout:
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 3: Check for CUDA environment variables
    if "CUDA_HOME" in os.environ or "CUDA_PATH" in os.environ:
        return True

    # Method 4: Check for CUDA libraries
    cuda_libs = ["libcudart.so", "cudart.dll", "cudart64.dll"]
    for lib in cuda_libs:
        try:
            ctypes.CDLL(lib)
            return True
        except OSError:
            continue

    return False


# Detect GPU and set ONNX Runtime dependency
# Override with FORCE_CPU=1 or FORCE_GPU=1 environment variables if needed
if os.environ.get("FORCE_CPU") == "1":
    gpu_available = False
    print("Forced CPU mode via FORCE_CPU=1")
elif os.environ.get("FORCE_GPU") == "1":
    gpu_available = True
    print("Forced GPU mode via FORCE_GPU=1")
else:
    gpu_available = detect_gpu()

onnx_runtime_dep = "onnxruntime-gpu" if gpu_available else "onnxruntime"
print(f"GPU detected: {gpu_available}, using: {onnx_runtime_dep}")

opencv_deps = [
    "opencv-python",
]

setup(
    name="instinct_onboard",
    version="0.1.0",
    packages=find_packages(exclude=["scripts", "tests"]),
    install_requires=[
        "numpy",
        "numpy-quaternion",
        "pyyaml",
        # 'rclpy',
        "transformations",
        "prettytable",
        onnx_runtime_dep,  # Auto-detected GPU or CPU version
        "empy==3.3.2",  # codespell:ignore
        "ros2_numpy",  # Required for ROS2 message handling, install by building a ros2 package from https://github.com/Box-Robotics/ros2_numpy.git
    ],
    extras_require={
        "noopencv": [],  # No opencv dependencies
        "all": opencv_deps,
    },
    zip_safe=True,
)
