# app/service/face_service.py
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import cv2
import uuid
import os
from datetime import datetime
from fastapi import HTTPException, status
from deepface import DeepFace
from scipy.spatial.distance import cosine

from app.cfg.config import AppSettings, StorageType
from app.service.face_dao import FaceDataDAO, CSVFaceDataDAO, SQLiteFaceDataDAO
from app.schema.face_schema import FaceInfo, FaceRecognitionResult, UpdateFaceRequest
from app.cfg.logging import app_logger

class FaceService:
    def __init__(self, settings: AppSettings):
        app_logger.info("正在初始化 FaceService...")
        self.settings = settings
        
        # 根据配置选择DAO实现
        if self.settings.deepface.storage_type == StorageType.CSV:
            features_file = Path(self.settings.deepface.image_db_path).parent / (self.settings.deepface.features_file_name + ".csv")
            self.face_dao: FaceDataDAO = CSVFaceDataDAO(features_path=features_file)
            app_logger.warning("正在使用 CSV 作为数据后端，性能低下且不稳定，强烈建议在生产中切换到 SQLite。")
        elif self.settings.deepface.storage_type == StorageType.SQLITE:
            self.face_dao: FaceDataDAO = SQLiteFaceDataDAO(db_url=self.settings.database.url)
        else:
            raise ValueError(f"不支持的存储类型: {self.settings.deepface.storage_type}")

        self.image_db_path = Path(self.settings.deepface.image_db_path)
        self.image_db_path.mkdir(parents=True, exist_ok=True)
        self.model_name = self.settings.deepface.model_name
        self.detector_backend = self.settings.deepface.detector_backend
        self.threshold = self.settings.deepface.recognition_threshold
        self.anti_spoofing = self.settings.deepface.enable_anti_spoofing
        
        self.known_faces_cache: List[Dict[str, Any]] = []
        self.cache_lock = asyncio.Lock()
        self._load_features_to_cache()

    def _load_features_to_cache(self):
        app_logger.info("正在从数据库加载人脸特征到内存缓存...")
        all_faces_data = self.face_dao.get_all()
        self.known_faces_cache = [
            {
                "sn": face["sn"],
                "name": face["name"],
                "features": np.array(face["features"], dtype=np.float32)
            }
            for face in all_faces_data
        ]
        app_logger.info(f"加载完成！缓存中共有 {len(self.known_faces_cache)} 条人脸特征。")

    async def _add_to_cache(self, face_data: Dict[str, Any]):
        async with self.cache_lock:
            self.known_faces_cache.append({
                "sn": face_data["sn"],
                "name": face_data["name"],
                "features": np.array(face_data["features"], dtype=np.float32)
            })

    async def _remove_from_cache(self, sn: str):
        async with self.cache_lock:
            self.known_faces_cache = [face for face in self.known_faces_cache if face["sn"] != sn]

    async def _update_in_cache(self, sn: str, new_name: str):
        async with self.cache_lock:
            for face in self.known_faces_cache:
                if face["sn"] == sn:
                    face["name"] = new_name

    def _decode_and_validate_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("无法解码图像数据。")
            return img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

    def _extract_and_validate_face(self, img: np.ndarray) -> np.ndarray:
        try:
            # enforce_detection=True 确保只处理检测到人脸的图片
            extracted_faces = DeepFace.extract_faces(
                img, detector_backend=self.detector_backend, align=True,
                anti_spoofing=self.anti_spoofing, enforce_detection=True
            )
            # 确认至少检测到一张脸
            if not extracted_faces:
                raise ValueError("未在图像中检测到人脸。")
        except ValueError as e: # DeepFace内部在未检测到人脸时会抛出ValueError
            raise HTTPException(status_code=400, detail=f"人脸检测失败: {e}")
        
        face_obj = extracted_faces[0]
        if self.anti_spoofing and not face_obj.get("is_real", True):
            raise HTTPException(status_code=400, detail="活体检测失败，疑似伪造人脸。")
        # 返回提取到的人脸图像区域 (numpy array)
        return face_obj['face']

    def _get_embedding(self, face_array: np.ndarray) -> List[float]:
        embedding_objs = DeepFace.represent(
            face_array, model_name=self.model_name,
            detector_backend='skip' # 因为已经提取了人脸，所以跳过检测
        )
        return embedding_objs[0]['embedding']

    def _save_face_image_and_get_path(self, face_array_float: np.ndarray, sn: str) -> Path:
        face_array_uint8 = (face_array_float * 255).astype(np.uint8)
        face_img_bgr = cv2.cvtColor(face_array_uint8, cv2.COLOR_RGB2BGR)
        
        file_uuid = str(uuid.uuid4())
        sn_dir = self.image_db_path / sn
        sn_dir.mkdir(parents=True, exist_ok=True)
        file_path = sn_dir / f"face_{sn}_{file_uuid}.jpg"
        
        cv2.imwrite(str(file_path), face_img_bgr)
        app_logger.info(f"注册图像已保存到: {file_path}")
        return file_path

    async def register_face(self, name: str, sn: str, image_bytes: bytes) -> FaceInfo:
        # 1. 检查SN是否已存在 (可选，根据业务决定是否允许一个SN注册多张脸)
        if self.face_dao.get_features_by_sn(sn):
             app_logger.warning(f"SN '{sn}' 已存在，将为其添加一张新的人脸图像。")
             # raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"SN '{sn}' 已被注册。")

        img = self._decode_and_validate_image(image_bytes)
        face_array = self._extract_and_validate_face(img)
        features = self._get_embedding(face_array)
        saved_image_path = self._save_face_image_and_get_path(face_array, sn)
        
        new_face_record = self.face_dao.create(name, sn, features, saved_image_path)
        await self._add_to_cache(new_face_record)
        
        app_logger.info(f"新的人脸 (SN: {sn}) 已注册。缓存大小: {len(self.known_faces_cache)}")
        return FaceInfo.model_validate(new_face_record)

    async def recognize_face(self, image_bytes: bytes) -> List[FaceRecognitionResult]:
        if not self.known_faces_cache:
            return []

        img = self._decode_and_validate_image(image_bytes)
        
        try:
            # 使用 extract_faces 一次性获取所有检测到的人脸及其位置
            extracted_faces = DeepFace.extract_faces(
                img, detector_backend=self.detector_backend, align=True
            )
            if not extracted_faces:
                return []
        except ValueError: # 未检测到人脸
            return []

        final_results = []
        for face_obj in extracted_faces:
            face_array = face_obj['face']
            facial_area = face_obj['facial_area']
            box = [facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']]

            embedding_objs = DeepFace.represent(
                face_array, model_name=self.model_name, detector_backend='skip'
            )
            query_features = np.array(embedding_objs[0]['embedding'], dtype=np.float32)

            best_match_for_this_face = None
            min_dist = float('inf')

            for known_face in self.known_faces_cache:
                distance = cosine(query_features, known_face["features"])
                if distance < self.threshold and distance < min_dist:
                    min_dist = distance
                    best_match_for_this_face = FaceRecognitionResult(
                        name=known_face["name"],
                        sn=known_face["sn"],
                        distance=distance,
                        box=box
                    )
            
            if best_match_for_this_face:
                final_results.append(best_match_for_this_face)
        
        return final_results

    async def get_all_faces(self) -> List[FaceInfo]:
        all_db_records = self.face_dao.get_all()
        return [FaceInfo.model_validate(record) for record in all_db_records]
    
    async def get_face_by_sn(self, sn: str) -> List[FaceInfo]:
        db_records = self.face_dao.get_features_by_sn(sn)
        if not db_records:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"未找到SN为 '{sn}' 的人脸信息。")
        return [FaceInfo.model_validate(record) for record in db_records]

    async def delete_face_by_sn(self, sn: str) -> int:
        features_to_delete = self.face_dao.get_features_by_sn(sn)
        if not features_to_delete:
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SN '{sn}' 未找到。")

        deleted_count = self.face_dao.delete_by_sn(sn)
        if deleted_count > 0:
            await self._remove_from_cache(sn)
            app_logger.info(f"SN {sn} 已从数据库和缓存中移除。")
            # 删除关联的物理图片文件
            for feature in features_to_delete:
                image_path = Path(feature['image_path'])
                if image_path.exists():
                    try:
                        os.remove(image_path)
                    except OSError as e:
                        app_logger.error(f"无法删除图片文件 {image_path}: {e}")
        return deleted_count

    async def update_face_by_sn(self, sn: str, update_data: UpdateFaceRequest) -> FaceInfo:
        # .model_dump(exclude_unset=True) 只获取用户实际传入的字段
        update_dict = update_data.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求体中没有任何需要更新的字段。")

        # 检查SN是否存在
        existing_faces = self.face_dao.get_features_by_sn(sn)
        if not existing_faces:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SN '{sn}' 未找到。")
        
        updated_count = self.face_dao.update_by_sn(sn, update_dict)
        if updated_count > 0:
            if 'name' in update_dict: # 如果更新了姓名，则需要更新缓存
                await self._update_in_cache(sn, update_dict['name'])
                app_logger.info(f"SN {sn} 的姓名已在缓存中更新。")
            
            # 返回更新后的第一条记录作为代表
            updated_face_info = self.face_dao.get_features_by_sn(sn)[0]
            return FaceInfo.model_validate(updated_face_info)
        
        raise HTTPException(status_code=500, detail="更新失败，数据库未返回更新计数。")

    async def stream_video_recognition(self, video_source: str):
        """
        核心函数：处理视频流，进行人脸识别，并将结果绘制在视频帧上返回。
        返回一个生成器，用于流式传输 `multipart/x-mixed-replace` 响应。
        """
        try:
            source = int(video_source) if video_source.isdigit() else video_source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                app_logger.error(f"无法打开视频源: {video_source}")
                # 在视频流的场景下，直接中断生成器即可
                return
        except Exception as e:
            app_logger.error(f"打开视频源失败: {e}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    app_logger.warning("视频流结束或读取帧失败。")
                    break

                try:
                    # 在视频流中，我们通常不强制检测，以避免在无人脸时程序中断
                    extracted_faces = DeepFace.extract_faces(
                        frame, detector_backend=self.detector_backend, align=True, enforce_detection=False
                    )

                    # 遍历检测到的每个人脸
                    for face_obj in extracted_faces:
                        if face_obj['confidence'] > 0.9: # 只处理高置信度的人脸
                            facial_area = face_obj['facial_area']
                            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                            
                            face_array = face_obj['face']
                            embedding_objs = DeepFace.represent(
                               face_array, model_name=self.model_name, detector_backend='skip'
                            )
                            query_features = np.array(embedding_objs[0]['embedding'], dtype=np.float32)

                            best_match = None
                            min_dist = float('inf')
                            
                            # 在缓存中快速匹配
                            for known_face in self.known_faces_cache:
                                dist = cosine(query_features, known_face["features"])
                                if dist < self.threshold and dist < min_dist:
                                    min_dist = dist
                                    best_match = known_face
                            
                            # 绘制结果
                            label = "Unknown"
                            color = (0, 0, 255) # 红色
                            if best_match:
                                label = f"{best_match['name']} ({best_match['sn']})"
                                color = (0, 255, 0) # 绿色
                            
                            # 绘制边界框
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            # 绘制标签
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                except Exception as e:
                    # 即使单帧处理失败，也继续处理下一帧
                    app_logger.error(f"处理视频帧时发生错误: {e}", exc_info=False)

                # 将处理后的帧编码为JPEG
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue

                # 以字节形式产出帧，用于 multipart 响应
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
                
                await asyncio.sleep(0.01) # 控制帧率，避免CPU满载

        finally:
            cap.release()
            app_logger.info(f"视频源 {video_source} 已关闭。")