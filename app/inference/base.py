# app/inference/base.py
"""
推理引擎抽象基类
定义统一的推理接口，确保不同设备实现的一致性
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
import time
import numpy as np
from .models import InferenceInput, InferenceOutput, InferenceResult, FaceDetection, EngineInfo


class BaseInferenceEngine(ABC):
    """推理引擎抽象基类"""
    
    def __init__(self, device_type: str, config: Dict[str, Any]):
        """
        初始化推理引擎
        
        Args:
            device_type: 设备类型 (cuda, hailo8, rk3588)
            config: 设备配置参数
        """
        self.device_type = device_type
        self.config = config
        self._initialized = False
        self._models_loaded = False
        self._initialization_time = None
        self._model_loading_time = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化推理引擎
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def load_models(self) -> bool:
        """
        加载模型
        
        Returns:
            bool: 模型加载是否成功
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: InferenceInput) -> InferenceOutput:
        """
        执行推理预测
        
        Args:
            input_data: 输入数据
            
        Returns:
            InferenceOutput: 推理结果
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """
        清理资源
        
        Returns:
            bool: 清理是否成功
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    @property
    def are_models_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._models_loaded
    
    def get_engine_info(self) -> EngineInfo:
        """
        获取推理引擎信息
        
        Returns:
            EngineInfo: 引擎信息
        """
        return EngineInfo(
            device_type=self.device_type,
            initialized=self._initialized,
            models_loaded=self._models_loaded,
            config=self.config
        )
    
    def _create_face_detection(
        self, 
        bbox: List[float], 
        confidence: float, 
        landmarks: Optional[List[List[float]]] = None,
        embedding: Optional[List[float]] = None
    ) -> FaceDetection:
        """
        创建人脸检测结果对象
        
        Args:
            bbox: 边界框坐标
            confidence: 置信度
            landmarks: 关键点坐标
            embedding: 特征向量
            
        Returns:
            FaceDetection: 人脸检测结果
        """
        return FaceDetection(
            bbox=bbox,
            confidence=confidence,
            landmarks=landmarks,
            embedding=embedding
        )
    
    def _create_inference_result(
        self, 
        faces: List[FaceDetection], 
        processing_time: float
    ) -> InferenceResult:
        """
        创建推理结果对象
        
        Args:
            faces: 检测到的人脸列表
            processing_time: 处理时间
            
        Returns:
            InferenceResult: 推理结果
        """
        return InferenceResult(
            faces=faces,
            processing_time=processing_time,
            device_type=self.device_type
        )
    
    def _create_inference_output(
        self, 
        result: InferenceResult, 
        success: bool = True, 
        error_message: Optional[str] = None
    ) -> InferenceOutput:
        """
        创建推理输出对象
        
        Args:
            result: 推理结果
            success: 是否成功
            error_message: 错误信息
            
        Returns:
            InferenceOutput: 推理输出
        """
        return InferenceOutput(
            result=result,
            success=success,
            error_message=error_message
        )
    
    def _validate_input(self, input_data: InferenceInput) -> bool:
        """
        验证输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            bool: 验证是否通过
        """
        try:
            # 使用Pydantic验证
            input_data.validate(input_data.dict())
            return True
        except Exception:
            return False
    
    def _preprocess_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        import cv2
        
        # 调整图像尺寸
        resized = cv2.resize(image, target_size)
        
        # 确保数据类型为uint8
        if resized.dtype != np.uint8:
            resized = resized.astype(np.uint8)
        
        return resized
    
    def _postprocess_bbox(self, bbox: List[float], original_size: tuple, target_size: tuple) -> List[float]:
        """
        后处理边界框坐标，将目标尺寸的坐标映射回原始尺寸
        
        Args:
            bbox: 目标尺寸下的边界框
            original_size: 原始图像尺寸 (width, height)
            target_size: 目标图像尺寸 (width, height)
            
        Returns:
            List[float]: 原始尺寸下的边界框
        """
        scale_x = original_size[0] / target_size[0]
        scale_y = original_size[1] / target_size[1]
        
        return [
            bbox[0] * scale_x,  # x1
            bbox[1] * scale_y,  # y1
            bbox[2] * scale_x,  # x2
            bbox[3] * scale_y   # y2
        ]
    
    def _align_face(self, image: np.ndarray, landmarks: Optional[List[List[float]]]) -> np.ndarray:
        """
        人脸对齐（如果需要）
        
        Args:
            image: 输入图像
            landmarks: 人脸关键点
            
        Returns:
            np.ndarray: 对齐后的人脸图像
        """
        # 默认实现：直接返回原图像
        # 子类可以重写此方法实现具体的人脸对齐逻辑
        return image
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        归一化特征向量
        
        Args:
            embedding: 原始特征向量
            
        Returns:
            List[float]: 归一化后的特征向量
        """
        if not embedding:
            return embedding
        
        # L2归一化
        import numpy as np
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        
        if norm > 0:
            normalized = embedding_array / norm
            return normalized.tolist()
        
        return embedding
