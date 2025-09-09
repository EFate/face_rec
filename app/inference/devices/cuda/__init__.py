# app/inference/devices/cuda/__init__.py
"""
CUDA推理引擎模块
基于InsightFace和ONNX Runtime实现
"""

from .engine import CudaInferenceEngine

__all__ = ["CudaInferenceEngine"]
