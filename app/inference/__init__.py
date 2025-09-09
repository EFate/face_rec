# app/inference/__init__.py
"""
异构算力卡推理引擎包

支持多种推理设备：
- CUDA: 基于InsightFace + ONNX Runtime
- Hailo8: 基于Degirum库
- RK3588: 基于Degirum库

提供统一的推理接口和模型管理功能。
"""

from .base import BaseInferenceEngine, InferenceResult, FaceDetection
from .factory import InferenceEngineFactory
from .models import InferenceInput, InferenceOutput

__all__ = [
    "BaseInferenceEngine",
    "InferenceResult", 
    "FaceDetection",
    "InferenceEngineFactory",
    "InferenceInput",
    "InferenceOutput"
]
