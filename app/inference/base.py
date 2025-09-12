# app/inference/base.py
"""
推理引擎抽象基类
定义统一的推理接口，确保不同设备实现的一致性
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Tuple
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
    
    def _align_face(self, image: np.ndarray, landmarks: Optional[List[List[float]]], image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据给定的关键点对齐并裁剪图像中的人脸。
        此函数基于人脸识别系统构建综合指南中的实现，针对侧脸识别进行了优化。

        Args:
            image: 完整的原始图像 (未经裁剪的边界框)
            landmarks: 5个关键点（界标）的列表，格式为 (x, y) 坐标
            image_size: 图像应被调整到的大小，默认为112

        Returns:
            对齐后的人脸图像和变换矩阵
        """
        import cv2
        import numpy as np
        
        if landmarks is None or len(landmarks) < 5:
            # 如果没有足够的关键点，返回原图像
            return image, np.eye(3)[:2]
        
        # ArcFace模型中使用的参考关键点
        _arcface_ref_kps = np.array([
            [38.2946, 51.6963],  # 左眼
            [73.5318, 51.5014],  # 右眼
            [56.0252, 71.7366],  # 鼻子
            [41.5493, 92.3655],  # 左嘴角
            [70.7299, 92.2041],  # 右嘴角
        ], dtype=np.float32)

        # 输入验证
        assert len(landmarks) == 5, f"需要5个关键点进行对齐，但收到了 {len(landmarks)} 个。"
        assert image_size % 112 == 0 or image_size % 128 == 0, "图像尺寸必须是112或128的倍数。"

        # 根据目标尺寸计算缩放因子和偏移
        if image_size % 112 == 0:
            ratio = float(image_size) / 112.0
            diff_x = 0  # 112缩放无水平偏移
        else:
            ratio = float(image_size) / 128.0
            diff_x = 8.0 * ratio  # 128缩放有水平偏移

        # 应用缩放和偏移到参考关键点
        dst = _arcface_ref_kps * ratio
        dst[:, 0] += diff_x

        # 使用RANSAC算法估计相似性变换矩阵
        # 增加RANSAC阈值以提高对侧脸的适应性
        M, inliers = cv2.estimateAffinePartial2D(
            np.array(landmarks, dtype=np.float32), 
            dst, 
            method=cv2.RANSAC,
            ransacReprojThreshold=20.0  # 增加阈值以适应侧脸
        )
        
        # 如果相似变换失败，尝试使用仿射变换
        if M is None:
            M, inliers = cv2.estimateAffine2D(
                np.array(landmarks, dtype=np.float32), 
                dst, 
                method=cv2.RANSAC,
                ransacReprojThreshold=20.0
            )
        
        # 健壮性检查：如果对齐失败，返回空图像
        if M is None or inliers is None or not np.all(inliers):
            return np.zeros((image_size, image_size, 3), dtype=np.uint8), np.zeros((2, 3), dtype=np.float32)

        # 应用仿射变换对齐人脸
        aligned_img = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)

        return aligned_img, M
