# app/inference/devices/hailo8/__init__.py
"""
Hailo8推理引擎模块
基于Degirum库实现
"""

from .engine import Hailo8InferenceEngine

__all__ = ["Hailo8InferenceEngine"]
