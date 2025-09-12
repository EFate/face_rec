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
            
    async def extract_embedding(self, image: np.ndarray, bbox: List[float], 
                               landmarks: Optional[List] = None) -> Optional[List[float]]:
        """
        直接从图像中提取人脸特征向量
        
        Args:
            image: 输入图像
            bbox: 人脸边界框 [x1, y1, x2, y2]
            landmarks: 人脸关键点（可选）
            
        Returns:
            Optional[List[float]]: 特征向量，如果提取失败则返回None
        """
        try:
            if self._use_new_inference:
                # 使用推理引擎提取特征向量
                engine = await self.model_manager.acquire_model_async()
                try:
                    # 创建输入数据
                    input_data = InferenceInput(
                        image=image,
                        extract_embeddings=True,
                        detection_threshold=0.1,  # 使用低阈值，因为我们已经有了bbox
                        bbox=bbox,
                        landmarks=landmarks
                    )
                    
                    # 执行推理
                    output = engine.extract_embedding(image, bbox, landmarks)
                    
                    if output:
                        # 确保是numpy数组并归一化
                        embedding_array = np.array(output, dtype=np.float32)
                        norm = np.linalg.norm(embedding_array)
                        if norm > 0:
                            normalized = embedding_array / norm
                            return normalized.tolist()
                        return output
                    return None
                finally:
                    await self.model_manager.release_model_async(engine)
            else:
                # 使用传统InsightFace提取特征向量
                model = await self.model_manager.acquire_model_async()
                try:
                    # 裁剪人脸区域
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)
                    
                    face_crop = image[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        app_logger.warning("裁剪后的人脸区域为空")
                        return None
                    
                    # 使用InsightFace提取特征向量
                    embedding = model.get_embedding(face_crop)
                    if embedding is not None:
                        return embedding.tolist()
                    return None
                finally:
                    await self.model_manager.release_model_async(model)
        except Exception as e:
            app_logger.error(f"提取特征向量失败: {e}")
            return None
    
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
            # 确保关键点格式正确
            landmarks_array = np.array(face_detection.landmarks, dtype=np.float32)
            # 如果是5点关键点，设置为landmark_2d_106（兼容性处理）
            if landmarks_array.shape[0] == 5:
                face.landmark_2d_106 = landmarks_array
                # 同时设置landmark_5点，确保兼容性
                try:
                    face.landmark_5 = landmarks_array
                except AttributeError:
                    # 如果不可写，使用setattr
                    setattr(face, 'landmark_5', landmarks_array)
            else:
                face.landmark_2d_106 = landmarks_array
        
        # 设置特征向量
        if face_detection.embedding:
            embedding_array = np.array(face_detection.embedding, dtype=np.float32)
            
            # 检查是否已经归一化
            norm = np.linalg.norm(embedding_array)
            is_normalized = abs(norm - 1.0) < 1e-4
            
            try:
                if is_normalized:
                    # 如果已经归一化，设置为normed_embedding
                    face.normed_embedding = embedding_array
                    app_logger.debug("设置归一化特征向量到normed_embedding")
                else:
                    # 如果未归一化，先归一化再设置
                    if norm > 0:
                        normalized = embedding_array / norm
                        face.normed_embedding = normalized
                        app_logger.debug("归一化特征向量并设置到normed_embedding")
                    else:
                        face.embedding = embedding_array
                        app_logger.warning("特征向量范数为0，无法归一化")
            except AttributeError:
                # 如果normed_embedding不可写，尝试使用embedding
                try:
                    face.embedding = embedding_array
                    app_logger.debug("设置特征向量到embedding")
                except AttributeError:
                    # 如果都不可写，使用setattr
                    setattr(face, 'embedding', embedding_array)
                    app_logger.debug("使用setattr设置特征向量")
        
        # 设置其他属性
        face.img_shape = image_shape[:2]  # (height, width)
        
        return face
