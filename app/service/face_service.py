# app/service/face_service.py
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import cv2
import uuid
import os
import time
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from app.cfg.config import AppSettings, StorageType
from app.service.face_dao import FaceDataDAO, CSVFaceDataDAO, SQLiteFaceDataDAO
from app.schema.face_schema import FaceInfo, FaceRecognitionResult, UpdateFaceRequest, ActiveStreamInfo, \
    StreamStartRequest
from app.cfg.logging import app_logger


class FaceService:
    def __init__(self, settings: AppSettings, model: FaceAnalysis):
        app_logger.info("正在初始化 FaceService...")
        self.settings = settings
        self.model = model

        # 根据配置选择DAO实现
        if self.settings.insightface.storage_type == StorageType.CSV:
            features_file = Path(self.settings.insightface.image_db_path).parent / (
                    self.settings.insightface.features_file_name + ".csv")
            self.face_dao: FaceDataDAO = CSVFaceDataDAO(features_path=features_file)
            app_logger.warning("正在使用 CSV 作为数据后端，性能低下且不稳定，强烈建议在生产中切换到 SQLite。")
        elif self.settings.insightface.storage_type == StorageType.SQLITE:
            self.face_dao: FaceDataDAO = SQLiteFaceDataDAO(db_url=self.settings.database.url)
        else:
            raise ValueError(f"不支持的存储类型: {self.settings.insightface.storage_type}")

        self.image_db_path = Path(self.settings.insightface.image_db_path)
        self.image_db_path.mkdir(parents=True, exist_ok=True)
        self.threshold = self.settings.insightface.recognition_threshold

        self.known_faces_cache: List[Dict[str, Any]] = []
        self.cache_lock = asyncio.Lock()

        # --- 【新增】视频流状态管理 ---
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_lock = asyncio.Lock()
        # 注意：初始加载在应用启动时完成，这里不再调用

    async def load_and_cache_features(self):
        """从数据库加载人脸特征到内存缓存。应在服务启动后调用。"""
        app_logger.info("正在从数据库加载人脸特征到内存缓存...")
        async with self.cache_lock:
            all_faces_data = self.face_dao.get_all()
            self.known_faces_cache = [
                {
                    "sn": face["sn"],
                    "name": face["name"],
                    "features": np.array(face["features"], dtype=np.float32)
                }
                for face in all_faces_data
            ]
        app_logger.info(f"✅ 加载完成！缓存中共有 {len(self.known_faces_cache)} 条人脸特征。")

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

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """从字节解码图像"""
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("无法解码图像数据，可能格式不受支持或文件已损坏。")
            return img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

    def _get_faces_from_image(self, img: np.ndarray) -> List[Face]:
        """使用 InsightFace 模型从图像中提取人脸"""
        try:
            faces = self.model.get(img)
            return faces
        except Exception as e:
            app_logger.error(f"使用 InsightFace 提取人脸时出错: {e}")
            raise HTTPException(status_code=500, detail="人脸分析服务内部错误。")

    def _crop_face_image(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """根据边界框裁剪人脸"""
        x1, y1, x2, y2 = bbox.astype(int)
        # 增加一些边距，避免裁剪过紧
        y1 = max(0, y1 - 20)
        y2 = min(img.shape[0], y2 + 20)
        x1 = max(0, x1 - 20)
        x2 = min(img.shape[1], x2 + 20)
        return img[y1:y2, x1:x2]

    def _save_face_image_and_get_path(self, face_img: np.ndarray, sn: str) -> Path:
        """保存裁剪后的人脸图像"""
        file_uuid = str(uuid.uuid4())
        sn_dir = self.image_db_path / sn
        sn_dir.mkdir(parents=True, exist_ok=True)
        file_path = sn_dir / f"face_{sn}_{file_uuid}.jpg"

        cv2.imwrite(str(file_path), face_img)
        app_logger.info(f"注册图像已保存到: {file_path}")
        return file_path

    async def register_face(self, name: str, sn: str, image_bytes: bytes) -> FaceInfo:
        img = self._decode_image(image_bytes)
        faces = self._get_faces_from_image(img)

        if not faces:
            raise HTTPException(status_code=400, detail="未在图像中检测到任何人脸。")
        if len(faces) > 1:
            raise HTTPException(status_code=400, detail=f"图像中检测到 {len(faces)} 张人脸，注册时必须确保只有一张人脸。")

        face = faces[0]

        if face.det_score < 0.8:
            app_logger.warning(f"注册人脸的检测分数较低: {face.det_score:.2f}")

        features = face.normed_embedding.tolist()
        cropped_face_img = self._crop_face_image(img, face.bbox)
        saved_image_path = self._save_face_image_and_get_path(cropped_face_img, sn)

        new_face_record = self.face_dao.create(name, sn, features, saved_image_path)
        await self._add_to_cache(new_face_record)

        app_logger.info(f"新的人脸 (SN: {sn}, Name: {name}) 已成功注册。缓存大小: {len(self.known_faces_cache)}")
        return FaceInfo.model_validate(new_face_record)

    async def recognize_face(self, image_bytes: bytes) -> List[FaceRecognitionResult]:
        if not self.known_faces_cache:
            return []

        img = self._decode_image(image_bytes)
        detected_faces = self._get_faces_from_image(img)

        if not detected_faces:
            return []

        known_features_matrix = np.array([face["features"] for face in self.known_faces_cache])
        known_metadata = [{"name": face["name"], "sn": face["sn"]} for face in self.known_faces_cache]
        detected_features_matrix = np.array([face.normed_embedding for face in detected_faces])
        similarity_matrix = np.dot(detected_features_matrix, known_features_matrix.T)

        final_results = []
        for i, detected_face in enumerate(detected_faces):
            similarities = similarity_matrix[i]
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
            min_dist = 1 - best_similarity

            if min_dist < self.threshold:
                best_match_meta = known_metadata[best_match_index]
                result = FaceRecognitionResult(
                    name=best_match_meta["name"],
                    sn=best_match_meta["sn"],
                    distance=min_dist,
                    box=detected_face.bbox.astype(int).tolist(),
                    detection_confidence=detected_face.det_score,
                    landmark=detected_face.landmark_2d_106
                )
                final_results.append(result)

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
            for feature in features_to_delete:
                image_path = Path(feature['image_path'])
                if image_path.exists():
                    try:
                        os.remove(image_path)
                        parent_dir = image_path.parent
                        if not any(parent_dir.iterdir()):
                            app_logger.info(f"目录 {parent_dir} 为空，正在删除...")
                            os.rmdir(parent_dir)
                    except OSError as e:
                        app_logger.error(f"无法删除图片文件或目录 {image_path}: {e}")
        return deleted_count

    async def update_face_by_sn(self, sn: str, update_data: UpdateFaceRequest) -> FaceInfo:
        update_dict = update_data.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求体中没有任何需要更新的字段。")

        existing_faces = self.face_dao.get_features_by_sn(sn)
        if not existing_faces:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SN '{sn}' 未找到。")

        updated_count = self.face_dao.update_by_sn(sn, update_dict)
        if updated_count > 0:
            if 'name' in update_dict:
                await self._update_in_cache(sn, update_dict['name'])
                app_logger.info(f"SN {sn} 的姓名已在缓存中更新。")

            first_face = existing_faces[0]
            first_face['name'] = update_dict.get('name', first_face['name'])
            return FaceInfo.model_validate(first_face)

        raise HTTPException(status_code=500, detail="更新失败，数据库未返回更新计数。")

    # =======================================================================================
    # === 【全新】视频流管理核心逻辑 (START) =================================================
    # =======================================================================================
    def _blocking_video_processor(
            self,
            video_source: str,
            frame_queue: asyncio.Queue,
            stop_event: asyncio.Event,
            loop: asyncio.AbstractEventLoop
    ):
        cap = None
        try:
            source = int(video_source) if video_source.isdigit() else video_source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                app_logger.error(f"无法打开视频源: {video_source}")
                loop.call_soon_threadsafe(frame_queue.put_nowait, None)
                return

            has_known_faces = len(self.known_faces_cache) > 0
            if has_known_faces:
                known_features_matrix = np.array([face["features"] for face in self.known_faces_cache])
                known_metadata = [{"name": face["name"], "sn": face["sn"]} for face in self.known_faces_cache]

            while not stop_event.is_set():
                ret_grab = cap.grab()
                if stop_event.is_set() or not ret_grab:
                    break

                ret_retrieve, frame = cap.retrieve()
                if not ret_retrieve:
                    continue

                try:
                    detected_faces = self._get_faces_from_image(frame)
                    if detected_faces and has_known_faces:
                        detected_features_matrix = np.array([face.normed_embedding for face in detected_faces])
                        similarity_matrix = np.dot(detected_features_matrix, known_features_matrix.T)
                        for i, face in enumerate(detected_faces):
                            similarities = similarity_matrix[i]
                            best_match_index = np.argmax(similarities)
                            min_dist = 1 - similarities[best_match_index]
                            box = face.bbox.astype(int)
                            color, label = ((0, 0, 255), "Unknown")
                            if min_dist < self.threshold:
                                best_match_meta = known_metadata[best_match_index]
                                label = f"{best_match_meta['name']} ({min_dist:.2f})"
                                color = (0, 255, 0)
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    app_logger.error(f"处理视频帧时发生错误: {e}", exc_info=False)

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if flag:
                    jpeg_bytes = bytearray(encodedImage)
                    loop.call_soon_threadsafe(
                        frame_queue.put_nowait,
                        (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                    )
                time.sleep(0.001)
        except Exception as e:
            app_logger.error(f"视频处理线程中发生致命错误: {e}", exc_info=True)
        finally:
            if cap and cap.isOpened():
                cap.release()
                app_logger.info(f"✅ 视频源 {video_source} 已成功释放。")
            loop.call_soon_threadsafe(frame_queue.put_nowait, None)

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes

        async with self.stream_lock:
            if stream_id in self.active_streams:
                raise HTTPException(status_code=409, detail="Stream ID conflict. Please try again.")

            frame_queue = asyncio.Queue()
            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()

            processing_task = loop.run_in_executor(
                None, self._blocking_video_processor, req.source, frame_queue, stop_event, loop
            )

            started_at = datetime.now()
            expires_at = None
            if lifetime != -1:
                expires_at = started_at + timedelta(minutes=lifetime)

            stream_info = ActiveStreamInfo(
                stream_id=stream_id,
                source=req.source,
                started_at=started_at,
                expires_at=expires_at,
                lifetime_minutes=lifetime
            )

            self.active_streams[stream_id] = {
                "info": stream_info,
                "queue": frame_queue,
                "stop_event": stop_event,
                "task": processing_task
            }
            app_logger.info(f"🚀 视频流已启动: ID={stream_id}, Source={req.source}, Lifetime={lifetime} mins")
            return stream_info

    async def stop_stream(self, stream_id: str) -> bool:
        async with self.stream_lock:
            stream = self.active_streams.get(stream_id)
            if not stream:
                app_logger.warning(f"尝试停止一个不存在或已停止的视频流: ID={stream_id}")
                return False  # Not Found, but we can return False gracefully

            app_logger.info(f"⏹️ 正在请求停止视频流: ID={stream_id}...")
            stream["stop_event"].set()
            await stream["task"]  # Wait for the background task to finish cleanup

            # Clean up the queue to avoid memory leaks
            while not stream["queue"].empty():
                stream["queue"].get_nowait()

            del self.active_streams[stream_id]
            app_logger.info(f"✅ 视频流已成功停止并清理: ID={stream_id}")
            return True

    async def get_stream_feed(self, stream_id: str):
        # This check is done without a lock for performance.
        # It's okay if the stream is stopped between this check and the `await queue.get()`.
        stream = self.active_streams.get(stream_id)
        if not stream:
            raise HTTPException(status_code=404, detail="Stream not found.")

        frame_queue = stream["queue"]
        try:
            while True:
                frame = await frame_queue.get()
                if frame is None:
                    break
                yield frame
        except asyncio.CancelledError:
            app_logger.info(f"客户端断开连接，正在关闭流: ID={stream_id}")
            # The stop logic will be handled by the cleanup tasks or explicit stop call
        finally:
            # This generator exiting does not automatically mean the stream should stop.
            # Only an explicit call to /stop or the cleanup task should stop it.
            app_logger.debug(f"一个客户端已从流 {stream_id} 断开。")

    async def get_all_streams(self) -> List[ActiveStreamInfo]:
        async with self.stream_lock:
            return [stream["info"] for stream in self.active_streams.values()]

    async def cleanup_expired_streams(self):
        """A background task to periodically clean up expired streams."""
        while True:
            await asyncio.sleep(self.settings.app.stream_cleanup_interval_seconds)
            now = datetime.now()
            # Create a copy of keys to avoid runtime errors during dict modification
            stream_ids_to_check = list(self.active_streams.keys())

            for stream_id in stream_ids_to_check:
                # Use a lock to safely access and potentially modify the dictionary
                async with self.stream_lock:
                    stream = self.active_streams.get(stream_id)
                    # Check if stream still exists and is not permanent
                    if stream and stream["info"].expires_at and now >= stream["info"].expires_at:
                        app_logger.info(f"🗑️ 发现过期视频流，正在清理: ID={stream_id}")
                        # Drop the lock before awaiting the stop_stream which has its own lock
                        await self.stop_stream(stream_id)

    async def stop_all_streams(self):
        """Gracefully stops all active streams. Called on application shutdown."""
        app_logger.info("应用程序关闭，正在停止所有活动的视频流...")
        # Create a copy of keys as stop_stream modifies the dictionary
        all_stream_ids = list(self.active_streams.keys())
        stopped_count = 0
        for stream_id in all_stream_ids:
            if await self.stop_stream(stream_id):
                stopped_count += 1
        app_logger.info(f"✅ 所有 {stopped_count} 个活动流已清理完毕。")
    # =======================================================================================
    # === 【全新】视频流管理核心逻辑 (END) =====================================================
    # =======================================================================================