# app/core/inference_adapter.py
"""
推理适配器
统一传统InsightFace和推理引擎的接口
"""

import numpy as np
from typing import List, Optional
from insightface.app.common import Face

from app.cfg.logging import app_logger
from app.inference.models import InferenceInput, InferenceOutput


class InferenceAdapter:
    """推理适配器，统一不同推理引擎的接口"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._use_new_inference = hasattr(model_manager, '_use_new_inference') and model_manager._use_new_inference
    
    async def get_faces(self, image: np.ndarray, extract_embeddings: bool = True, 
                       detection_threshold: float = 0.4) -> List[Face]:
        """
        获取人脸检测结果，统一返回InsightFace的Face对象格式
        
        Args:
            image: 输入图像
            extract_embeddings: 是否提取特征向量
            detection_threshold: 检测阈值
            
        Returns:
            List[Face]: 人脸检测结果列表
        """
        if self._use_new_inference:
            return await self._get_faces_new_inference(image, extract_embeddings, detection_threshold)
        else:
            return await self._get_faces_insightface(image, extract_embeddings, detection_threshold)
    
    async def _get_faces_new_inference(self, image: np.ndarray, extract_embeddings: bool, 
                                     detection_threshold: float) -> List[Face]:
        """使用推理引擎获取人脸"""
        try:
            # 创建输入数据
            input_data = InferenceInput(
                image=image,
                extract_embeddings=extract_embeddings,
                detection_threshold=detection_threshold
            )
            
            # 从模型池获取推理引擎
            engine = await self.model_manager.acquire_model_async()
            try:
                # 执行推理
                output = engine.predict(input_data)
                
                if not output.success:
                    app_logger.error(f"推理失败: {output.error_message}")
                    return []
                
                # 转换为InsightFace的Face对象格式
                faces = []
                for face_detection in output.result.faces:
                    face = self._convert_to_insightface_face(face_detection, image.shape)
                    faces.append(face)
                
                return faces
            finally:
                await self.model_manager.release_model_async(engine)
            
        except Exception as e:
            app_logger.error(f"推理引擎获取人脸失败: {e}")
            return []
    
    async def _get_faces_insightface(self, image: np.ndarray, extract_embeddings: bool, 
                                   detection_threshold: float) -> List[Face]:
        """使用传统InsightFace获取人脸"""
        try:
            model = await self.model_manager.acquire_model_async()
            try:
                faces = model.get(image)
                # 过滤低置信度的人脸
                filtered_faces = [face for face in faces if face.det_score >= detection_threshold]
                return filtered_faces
            finally:
                await self.model_manager.release_model_async(model)
                
        except Exception as e:
            app_logger.error(f"InsightFace获取人脸失败: {e}")
            return []
    
    def _convert_to_insightface_face(self, face_detection, image_shape) -> Face:
        """将推理引擎的结果转换为InsightFace的Face对象"""
        # 创建Face对象
        face = Face()
        
        # 设置边界框
        face.bbox = np.array(face_detection.bbox, dtype=np.float32)
        
        # 设置检测置信度
        face.det_score = face_detection.confidence
        
        # 设置关键点
        if face_detection.landmarks:
            face.landmark_2d_106 = np.array(face_detection.landmarks, dtype=np.float32)
        
        # 设置特征向量 - 使用embedding属性而不是normed_embedding
        if face_detection.embedding:
            try:
                face.embedding = np.array(face_detection.embedding, dtype=np.float32)
            except AttributeError:
                # 如果embedding属性不可写，尝试使用normed_embedding
                try:
                    face.normed_embedding = np.array(face_detection.embedding, dtype=np.float32)
                except AttributeError:
                    # 如果都不可写，使用setattr
                    setattr(face, 'embedding', np.array(face_detection.embedding, dtype=np.float32))
        
        # 设置其他属性
        face.img_shape = image_shape[:2]  # (height, width)
        
        return face
