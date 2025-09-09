# app/inference/factory.py
"""
推理引擎工厂
根据设备类型创建对应的推理引擎实例
"""

from typing import Dict, Any, Optional, List
from .base import BaseInferenceEngine
from app.cfg.logging import app_logger

# 延迟导入，避免循环依赖
_engine_classes = {}


class InferenceEngineFactory:
    """推理引擎工厂类"""
    
    # 支持的设备类型
    SUPPORTED_DEVICES = {
        "cuda": "app.inference.devices.cuda.engine.CudaInferenceEngine",
        "hailo8": "app.inference.devices.hailo8.engine.Hailo8InferenceEngine",
        "rk3588": "app.inference.devices.rk3588.engine.RK3588InferenceEngine"
    }
    
    @classmethod
    def _get_engine_class(cls, device_type: str):
        """获取推理引擎类"""
        if device_type not in cls.SUPPORTED_DEVICES:
            supported = ", ".join(cls.SUPPORTED_DEVICES.keys())
            raise ValueError(f"不支持的设备类型: {device_type}. 支持的设备: {supported}")
        
        if device_type not in _engine_classes:
            try:
                module_path, class_name = cls.SUPPORTED_DEVICES[device_type].rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                _engine_classes[device_type] = getattr(module, class_name)
            except ImportError as e:
                app_logger.error(f"无法导入 {device_type} 推理引擎: {e}")
                raise ImportError(f"无法导入 {device_type} 推理引擎: {e}")
        
        return _engine_classes[device_type]
    
    @classmethod
    def create_engine(cls, device_type: str, config: Dict[str, Any]) -> BaseInferenceEngine:
        """
        创建推理引擎实例
        
        Args:
            device_type: 设备类型 (cuda, hailo8, rk3588)
            config: 设备配置参数
            
        Returns:
            BaseInferenceEngine: 推理引擎实例
            
        Raises:
            ValueError: 不支持的设备类型
            ImportError: 无法导入推理引擎
        """
        engine_class = cls._get_engine_class(device_type)
        app_logger.info(f"创建 {device_type} 推理引擎")
        
        return engine_class(device_type, config)
    
    @classmethod
    def get_supported_devices(cls) -> List[str]:
        """
        获取支持的设备类型列表
        
        Returns:
            List[str]: 支持的设备类型列表
        """
        return list(cls.SUPPORTED_DEVICES.keys())
    
    @classmethod
    def is_device_supported(cls, device_type: str) -> bool:
        """
        检查设备类型是否支持
        
        Args:
            device_type: 设备类型
            
        Returns:
            bool: 是否支持
        """
        return device_type in cls.SUPPORTED_DEVICES
    
    @classmethod
    def is_device_available(cls, device_type: str) -> bool:
        """
        检查设备是否可用（依赖库是否安装）
        
        Args:
            device_type: 设备类型
            
        Returns:
            bool: 是否可用
        """
        try:
            cls._get_engine_class(device_type)
            return True
        except (ImportError, ValueError):
            return False
    
    @classmethod
    def get_available_devices(cls) -> List[str]:
        """
        获取可用的设备类型列表
        
        Returns:
            List[str]: 可用的设备类型列表
        """
        available_devices = []
        for device_type in cls.SUPPORTED_DEVICES:
            if cls.is_device_available(device_type):
                available_devices.append(device_type)
        return available_devices
    
    @classmethod
    def create_engine_with_fallback(cls, device_type: str, config: Dict[str, Any]) -> BaseInferenceEngine:
        """
        创建推理引擎实例，支持降级处理
        
        Args:
            device_type: 首选设备类型
            config: 设备配置参数
            
        Returns:
            BaseInferenceEngine: 推理引擎实例
        """
        # 尝试创建首选设备
        try:
            if not cls.is_device_available(device_type):
                raise ImportError(f"设备 {device_type} 不可用")
            
            engine = cls.create_engine(device_type, config)
            app_logger.info(f"成功创建 {device_type} 推理引擎")
            return engine
        except Exception as e:
            app_logger.warning(f"创建 {device_type} 推理引擎失败: {e}")
            
            # 降级到CUDA/CPU
            if device_type != "cuda":
                app_logger.info("尝试降级到CUDA/CPU推理引擎")
                try:
                    fallback_config = config.copy()
                    # 确保使用CPU执行提供者
                    fallback_config["providers"] = ["CPUExecutionProvider"]
                    engine = cls.create_engine("cuda", fallback_config)
                    app_logger.info("成功创建CPU推理引擎作为降级方案")
                    return engine
                except Exception as fallback_e:
                    app_logger.error(f"降级到CPU推理引擎也失败: {fallback_e}")
            
            # 如果所有尝试都失败，抛出原始异常
            raise e
    
    @classmethod
    def create_best_available_engine(cls, config: Dict[str, Any]) -> BaseInferenceEngine:
        """
        创建最佳可用推理引擎
        
        Args:
            config: 设备配置参数
            
        Returns:
            BaseInferenceEngine: 推理引擎实例
        """
        available_devices = cls.get_available_devices()
        
        if not available_devices:
            raise RuntimeError("没有可用的推理引擎")
        
        # 优先级：cuda > hailo8 > rk3588
        priority_order = ["cuda", "hailo8", "rk3588"]
        
        for device_type in priority_order:
            if device_type in available_devices:
                try:
                    engine = cls.create_engine(device_type, config)
                    app_logger.info(f"使用最佳可用推理引擎: {device_type}")
                    return engine
                except Exception as e:
                    app_logger.warning(f"创建 {device_type} 推理引擎失败: {e}")
                    continue
        
        # 如果所有设备都失败，抛出异常
        raise RuntimeError("所有推理引擎都不可用")
