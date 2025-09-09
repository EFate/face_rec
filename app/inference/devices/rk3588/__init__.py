# app/inference/devices/rk3588/__init__.py
"""
RK3588推理引擎模块
基于Degirum库实现
"""

from .engine import RK3588InferenceEngine

__all__ = ["RK3588InferenceEngine"]
