# app/inference/devices/cuda/engine.py
"""
CUDA推理引擎实现
基于InsightFace和ONNX Runtime
"""

import time
import asyncio
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np

from ...base import BaseInferenceEngine
from ...models import InferenceInput, InferenceOutput, InferenceResult, FaceDetection
from app.cfg.logging import app_logger

# 导入InsightFace
try:
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    app_logger.warning("InsightFace库未安装，CUDA推理引擎不可用")


class CudaInferenceEngine(BaseInferenceEngine):
    """CUDA推理引擎实现"""
    
    def __init__(self, device_type: str, config: Dict[str, Any]):
        """
        初始化CUDA推理引擎
        
        Args:
            device_type: 设备类型
            config: 配置参数，包含：
                - model_pack_name: InsightFace模型包名称
                - providers: ONNX Runtime执行提供者列表
                - home: 模型存储目录
                - detection_size: 检测模型输入尺寸
                - det_thresh: 检测阈值
        """
        super().__init__(device_type, config)
        
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace库未安装，无法使用CUDA推理引擎")
        
        self.face_analysis: Optional[FaceAnalysis] = None
        self.model_pack_name = config.get("model_pack_name", "buffalo_l")
        self.providers = config.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.home = config.get("home", "./data/.insightface")
        self.detection_size = tuple(config.get("detection_size", [640, 640]))
        self.det_thresh = config.get("det_thresh", 0.4)
    
    def initialize(self) -> bool:
        """
        初始化推理引擎
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            app_logger.info(f"初始化CUDA推理引擎: {self.model_pack_name}")
            
            # 设置InsightFace环境变量
            os.environ["INSIGHTFACE_HOME"] = str(self.home)
            
            # 确保模型目录存在
            Path(self.home).mkdir(parents=True, exist_ok=True)
            
            self._initialized = True
            self._initialization_time = time.time()
            app_logger.info("CUDA推理引擎初始化成功")
            return True
            
        except Exception as e:
            app_logger.error(f"CUDA推理引擎初始化失败: {e}")
            return False
    
    def load_models(self) -> bool:
        """
        加载InsightFace模型
        
        Returns:
            bool: 模型加载是否成功
        """
        try:
            if not self._initialized:
                raise RuntimeError("推理引擎未初始化")
            
            app_logger.info(f"加载InsightFace模型: {self.model_pack_name}")
            
            # 创建FaceAnalysis实例
            self.face_analysis = FaceAnalysis(
                name=self.model_pack_name,
                root=str(self.home),
                providers=self.providers,
                download=True
            )
            
            # 准备模型
            ctx_id = 0 if 'CUDAExecutionProvider' in self.providers else -1
            self.face_analysis.prepare(
                ctx_id=ctx_id,
                det_size=self.detection_size,
                det_thresh=self.det_thresh
            )
            
            self._models_loaded = True
            self._model_loading_time = time.time()
            app_logger.info("InsightFace模型加载成功")
            return True
            
        except Exception as e:
            app_logger.error(f"InsightFace模型加载失败: {e}")
            if "CUDA" in str(e):
                app_logger.error("GPU环境配置错误，请检查CUDA、cuDNN版本或使用CPU模式")
            return False
    
    def predict(self, input_data: InferenceInput) -> InferenceOutput:
        """
        执行人脸检测和识别
        
        Args:
            input_data: 输入数据
            
        Returns:
            InferenceOutput: 推理结果
        """
        start_time = time.time()
        
        try:
            if not self._models_loaded:
                raise RuntimeError("模型未加载")
            
            if not self._validate_input(input_data):
                raise ValueError("输入数据验证失败")
            
            # 执行推理
            faces: List[Face] = self.face_analysis.get(input_data.image)
            
            # 转换结果格式
            face_detections = []
            for face in faces:
                # 检查检测置信度
                if face.det_score < input_data.detection_threshold:
                    continue
                
                # 获取特征向量
                embedding = None
                if input_data.extract_embeddings and hasattr(face, 'normed_embedding'):
                    embedding = face.normed_embedding.tolist()
                
                # 获取关键点
                landmarks = None
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    landmarks = face.landmark_2d_106.tolist()
                
                # 创建人脸检测结果
                face_detection = self._create_face_detection(
                    bbox=face.bbox.tolist(),
                    confidence=float(face.det_score),
                    landmarks=landmarks,
                    embedding=embedding
                )
                face_detections.append(face_detection)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 创建推理结果
            result = self._create_inference_result(face_detections, processing_time)
            
            # 创建输出
            output = self._create_inference_output(result, success=True)
            
            app_logger.debug(f"CUDA推理完成: 检测到{len(face_detections)}张人脸, 耗时{processing_time:.3f}s")
            return output
            
        except Exception as e:
            processing_time = time.time() - start_time
            app_logger.error(f"CUDA推理失败: {e}")
            
            # 创建错误输出
            result = self._create_inference_result([], processing_time)
            output = self._create_inference_output(result, success=False, error_message=str(e))
            return output
    
    def cleanup(self) -> bool:
        """
        清理资源
        
        Returns:
            bool: 清理是否成功
        """
        try:
            app_logger.info("清理CUDA推理引擎资源")
            
            # 清理FaceAnalysis实例
            if self.face_analysis is not None:
                # InsightFace没有显式的清理方法，设置为None即可
                self.face_analysis = None
            
            self._models_loaded = False
            self._initialized = False
            
            app_logger.info("CUDA推理引擎资源清理完成")
            return True
            
        except Exception as e:
            app_logger.error(f"CUDA推理引擎资源清理失败: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "device_type": self.device_type,
            "model_pack_name": self.model_pack_name,
            "providers": self.providers,
            "detection_size": self.detection_size,
            "det_thresh": self.det_thresh,
            "initialized": self._initialized,
            "models_loaded": self._models_loaded,
            "initialization_time": self._initialization_time,
            "model_loading_time": self._model_loading_time
        }
