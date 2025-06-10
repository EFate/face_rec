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

        # 初始化空的缓存
        self.known_faces_cache: Dict[str, Any] = {
            "features_matrix": np.empty((0, 512), dtype=np.float32),
            "metadata": []
        }
        self.cache_lock = asyncio.Lock()

        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_lock = asyncio.Lock()

    async def _rebuild_cache_from_db(self):
        """
        [全量更新] 从数据库完全重建人脸特征缓存。仅在服务启动时调用。
        """
        app_logger.info("正在从数据库全量重建人脸特征缓存...")
        all_faces_data = self.face_dao.get_all()
        if not all_faces_data:
            self.known_faces_cache = {
                "features_matrix": np.empty((0, 512), dtype=np.float32),
                "metadata": []
            }
        else:
            features_list = [face["features"] for face in all_faces_data]
            self.known_faces_cache["features_matrix"] = np.array(features_list, dtype=np.float32)
            self.known_faces_cache["metadata"] = [{"sn": face["sn"], "name": face["name"]} for face in all_faces_data]
        app_logger.info(f"✅ 缓存重建完成！缓存中共有 {len(self.known_faces_cache['metadata'])} 条人脸特征。")

    async def load_and_cache_features(self):
        """服务启动时，加载所有特征到缓存中。"""
        async with self.cache_lock:
            await self._rebuild_cache_from_db()

    def get_known_faces_cache_copy(self) -> Dict[str, Any]:
        """获取缓存的一份线程安全的浅拷贝，用于识别任务。"""
        # 对于 numpy 数组，.copy() 是深拷贝
        return {
            "features_matrix": self.known_faces_cache["features_matrix"].copy(),
            "metadata": self.known_faces_cache["metadata"][:]  # 列表浅拷贝
        }

    async def _add_to_cache(self, face_data: Dict[str, Any]):
        """
        [增量更新] 向缓存中添加一条新的人脸数据。
        """
        async with self.cache_lock:
            app_logger.debug(f"正在向缓存中增量添加 SN: {face_data['sn']}")
            current_features = self.known_faces_cache["features_matrix"]
            new_feature = face_data["features"].reshape(1, -1).astype(np.float32)

            if current_features.size == 0:
                self.known_faces_cache["features_matrix"] = new_feature
            else:
                self.known_faces_cache["features_matrix"] = np.vstack([current_features, new_feature])

            self.known_faces_cache["metadata"].append({"sn": face_data["sn"], "name": face_data["name"]})
            app_logger.info(
                f"缓存更新成功，新增 SN: {face_data['sn']}. 当前缓存大小: {len(self.known_faces_cache['metadata'])}")

    async def _remove_from_cache(self, sn: str):
        """
        [增量更新] 从缓存中删除指定SN的所有人脸数据。
        """
        async with self.cache_lock:
            app_logger.debug(f"正在从缓存中增量删除 SN: {sn}")
            metadata = self.known_faces_cache["metadata"]
            indices_to_delete = [i for i, meta in enumerate(metadata) if meta["sn"] == sn]

            if not indices_to_delete:
                app_logger.warning(f"尝试从缓存删除SN {sn}，但在缓存中未找到。")
                return

            self.known_faces_cache["features_matrix"] = np.delete(
                self.known_faces_cache["features_matrix"], indices_to_delete, axis=0
            )

            # 从后往前删除，避免索引变化问题
            for i in sorted(indices_to_delete, reverse=True):
                del self.known_faces_cache["metadata"][i]

            app_logger.info(
                f"缓存更新成功，删除 {len(indices_to_delete)} 条 SN 为 '{sn}' 的记录。当前缓存大小: {len(self.known_faces_cache['metadata'])}")

    async def _update_in_cache(self, sn: str, new_name: str):
        """
        [增量更新] 更新缓存中指定SN的元数据（如姓名）。
        """
        async with self.cache_lock:
            app_logger.debug(f"正在更新缓存中 SN {sn} 的信息...")
            updated_count = 0
            for item in self.known_faces_cache["metadata"]:
                if item["sn"] == sn:
                    item["name"] = new_name
                    updated_count += 1
            app_logger.info(f"缓存更新成功，{updated_count} 条 SN 为 '{sn}' 的记录姓名已更新。")

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("无法解码图像数据，可能格式不受支持或文件已损坏。")
            return img
        except Exception as e:
            app_logger.error(f"图像解码失败: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

    def _get_faces_from_image(self, img: np.ndarray) -> List[Face]:
        try:
            return self.model.get(img)
        except Exception as e:
            app_logger.error(f"使用 InsightFace 提取人脸时出错: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="人脸分析服务内部错误。")

    def _crop_face_image(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype(int)
        # 稍微扩大截图范围
        y1 = max(0, y1 - 20);
        y2 = min(img.shape[0], y2 + 20)
        x1 = max(0, x1 - 20);
        x2 = min(img.shape[1], x2 + 20)
        return img[y1:y2, x1:x2]

    def _save_face_image_and_get_path(self, face_img: np.ndarray, sn: str) -> Path:
        file_uuid = str(uuid.uuid4())
        sn_dir = self.image_db_path / sn
        sn_dir.mkdir(parents=True, exist_ok=True)
        file_path = sn_dir / f"face_{sn}_{file_uuid}.jpg"

        success, encoded_image = cv2.imencode(".jpg", face_img)
        if not success:
            raise HTTPException(status_code=500, detail="无法编码裁剪的人脸图像。")
        with open(file_path, "wb") as f:
            f.write(encoded_image.tobytes())
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
        det_score_threshold = self.settings.insightface.recognition_det_score_threshold
        if face.det_score < det_score_threshold:
            raise HTTPException(status_code=400,
                                detail=f"人脸质量不佳，检测分数({face.det_score:.2f})过低，请上传更清晰的人脸图像。")

        features = face.normed_embedding
        cropped_face_img = self._crop_face_image(img, face.bbox)
        saved_image_path = self._save_face_image_and_get_path(cropped_face_img, sn)

        # 1. 持久化到数据库
        new_face_record = self.face_dao.create(name, sn, features, saved_image_path)

        # 2. 增量更新缓存
        await self._add_to_cache(new_face_record)

        app_logger.info(f"新的人脸 (SN: {sn}, Name: {name}) 已成功注册并加入缓存。")
        return FaceInfo.model_validate(new_face_record)

    async def recognize_face(self, image_bytes: bytes) -> List[FaceRecognitionResult]:
        # 从缓存中获取数据进行识别
        cache_copy = self.get_known_faces_cache_copy()
        known_features_matrix = cache_copy["features_matrix"]
        known_metadata = cache_copy["metadata"]

        if known_features_matrix.size == 0: return []

        img = self._decode_image(image_bytes)
        detected_faces = self._get_faces_from_image(img)
        if not detected_faces: return []

        detected_features_matrix = np.array([face.normed_embedding for face in detected_faces])
        # 使用点积计算余弦相似度 (因为特征向量已归一化)
        similarity_matrix = np.dot(detected_features_matrix, known_features_matrix.T)

        final_results = []
        for i, detected_face in enumerate(detected_faces):
            similarities = similarity_matrix[i]
            best_match_index = np.argmax(similarities)
            # 余弦距离 = 1 - 余弦相似度
            min_dist = 1 - similarities[best_match_index]

            if min_dist < self.threshold:
                best_match_meta = known_metadata[best_match_index]
                final_results.append(FaceRecognitionResult(
                    name=best_match_meta["name"], sn=best_match_meta["sn"], distance=min_dist,
                    box=detected_face.bbox.astype(int).tolist(),
                    detection_confidence=float(detected_face.det_score),
                    landmark=detected_face.landmark_2d_106
                ))
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

        # 1. 从数据库删除
        deleted_count = self.face_dao.delete_by_sn(sn)

        # 2. 如果数据库删除成功，则从缓存删除并清理文件
        if deleted_count > 0:
            await self._remove_from_cache(sn)
            app_logger.info(f"SN {sn} 已从数据库和缓存中移除。")
            # 清理关联的图片文件
            for feature in features_to_delete:
                try:
                    image_path = Path(feature['image_path'])
                    if image_path.exists():
                        os.remove(image_path)
                        # 尝试删除空的父目录
                        parent_dir = image_path.parent
                        if not any(parent_dir.iterdir()):
                            os.rmdir(parent_dir)
                except OSError as e:
                    app_logger.error(f"无法删除图片文件或目录 {feature.get('image_path', 'N/A')}: {e}")
        return deleted_count

    async def update_face_by_sn(self, sn: str, update_data: UpdateFaceRequest) -> FaceInfo:
        update_dict = update_data.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求体中没有任何需要更新的字段。")

        # 检查是否存在
        existing_faces = self.face_dao.get_features_by_sn(sn)
        if not existing_faces:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SN '{sn}' 未找到。")

        # 1. 更新数据库
        updated_count = self.face_dao.update_by_sn(sn, update_dict)

        # 2. 如果更新了姓名，则同步更新缓存
        if updated_count > 0 and 'name' in update_dict:
            await self._update_in_cache(sn, update_dict['name'])

        # 返回更新后的最新信息
        updated_face_info = await self.get_face_by_sn(sn)
        return updated_face_info[0]

    # =======================================================================================
    # === 视频流管理核心逻辑 (精炼后) ==========================================================
    # =======================================================================================
    def _draw_recognition_results_on_frame(self, frame: np.ndarray, last_results: List[Dict]):
        if not last_results: return
        for result in last_results:
            box = result['box'];
            label = result['label'];
            color = result['color']
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (box[0], box[1] - lh - 10), (box[0] + lw, box[1] - 5), color, cv2.FILLED)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _blocking_video_processor(self, stream_id: str, video_source: str, frame_queue: asyncio.Queue,
                                  stop_event: asyncio.Event,
                                  loop: asyncio.AbstractEventLoop):
        """
        [后台线程] 负责视频处理，将纯粹的JPEG字节放入队列。
        """
        cap = None
        try:
            source = int(video_source) if video_source.isdigit() else video_source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                app_logger.error(f"【后台线程 - {stream_id}】无法打开视频源: {video_source}")
                return

            last_rec_time, last_cache_update_time = 0, 0
            last_results, known_faces_cache = [], {}

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    app_logger.warning(f"【后台线程 - {stream_id}】无法从视频源 {video_source} 读取帧，流可能已结束。")
                    break

                current_time = time.time()
                # 定期更新人脸库缓存
                if current_time - last_cache_update_time > self.settings.app.stream_cache_update_interval_seconds:
                    known_faces_cache = self.get_known_faces_cache_copy()
                    last_cache_update_time = current_time

                # 控制识别频率
                if current_time - last_rec_time > self.settings.app.stream_recognition_interval_seconds:
                    last_rec_time = current_time
                    if known_faces_cache.get("metadata"):
                        try:
                            # 人脸识别核心逻辑
                            detected_faces = self._get_faces_from_image(frame)
                            temp_results = []
                            if detected_faces:
                                known_features = known_faces_cache["features_matrix"]
                                known_meta = known_faces_cache["metadata"]
                                detected_features = np.array([f.normed_embedding for f in detected_faces])
                                sim_matrix = np.dot(detected_features, known_features.T)
                                for i, face in enumerate(detected_faces):
                                    sims = sim_matrix[i]
                                    best_idx = np.argmax(sims)
                                    min_dist = 1 - sims[best_idx]
                                    box = face.bbox.astype(int)
                                    color, label = ((0, 0, 255), "Unknown")
                                    if min_dist < self.threshold:
                                        meta = known_meta[best_idx]
                                        label = f"{meta['name']} ({min_dist:.2f})"
                                        color = (0, 255, 0)
                                    temp_results.append({"box": box, "label": label, "color": color})
                            last_results = temp_results
                        except Exception as e:
                            app_logger.error(f"【后台线程 - {stream_id}】处理视频帧时发生错误: {e}", exc_info=False)

                # 绘制结果并编码
                self._draw_recognition_results_on_frame(frame, last_results)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if flag:
                    try:
                        # 只将JPEG字节流放入队列
                        frame_queue.put_nowait(encodedImage.tobytes())
                    except asyncio.QueueFull:
                        app_logger.warning(f"【后台线程 - {stream_id}】视频流队列已满，丢弃一帧。")

                time.sleep(0.01)  # 短暂休眠，避免CPU空转
        except Exception as e:
            app_logger.error(f"【后台线程 - {stream_id}】发生致命错误: {e}", exc_info=True)
        finally:
            if cap and cap.isOpened():
                cap.release()
            # 发送终结信号 (None)
            try:
                loop.call_soon_threadsafe(frame_queue.put_nowait, None)
            except asyncio.QueueFull:
                pass
            app_logger.info(f"✅ 【后台线程 - {stream_id}】处理线程已安全结束。")

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        # 预检视频源
        source_to_check = int(req.source) if req.source.isdigit() else req.source
        cap_check = cv2.VideoCapture(source_to_check)
        if not cap_check.isOpened():
            cap_check.release()
            app_logger.error(f"启动视频流失败：无法打开视频源 '{req.source}'")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"无法打开视频源 '{req.source}'。请检查路径或摄像头ID是否正确。")
        cap_check.release()

        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes

        async with self.stream_lock:
            if stream_id in self.active_streams:
                raise HTTPException(status_code=409, detail="Stream ID conflict. Please try again.")

            frame_queue = asyncio.Queue(maxsize=120)  # 增大队列以应对网络波动
            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()

            processing_task = loop.run_in_executor(
                None, self._blocking_video_processor, stream_id, req.source, frame_queue, stop_event, loop
            )

            started_at = datetime.now()
            expires_at = None if lifetime == -1 else started_at + timedelta(minutes=lifetime)
            stream_info = ActiveStreamInfo(stream_id=stream_id, source=req.source, started_at=started_at,
                                           expires_at=expires_at, lifetime_minutes=lifetime)
            self.active_streams[stream_id] = {"info": stream_info, "queue": frame_queue, "stop_event": stop_event,
                                              "task": processing_task}
            app_logger.info(f"🚀 视频流已启动: ID={stream_id}, Source={req.source}, Lifetime={lifetime} mins")
            return stream_info

    async def stop_stream(self, stream_id: str) -> bool:
        async with self.stream_lock:
            stream = self.active_streams.pop(stream_id, None)
            if not stream:
                app_logger.warning(f"尝试停止一个不存在或已停止的视频流: ID={stream_id}")
                return False

        app_logger.info(f"⏹️ 正在请求停止视频流: ID={stream_id}...")
        stream["stop_event"].set()
        try:
            # 等待后台任务结束
            await asyncio.wait_for(stream["task"], timeout=5.0)
        except asyncio.TimeoutError:
            app_logger.error(f"停止视频流 {stream_id} 的后台任务超时！")

        # 清空队列中可能残留的帧
        while not stream["queue"].empty():
            stream["queue"].get_nowait()

        app_logger.info(f"✅ 视频流已成功停止并清理: ID={stream_id}")
        return True

    async def get_stream_feed(self, stream_id: str):
        """
        [前台协程] 从队列获取JPEG字节，包装成 multipart chunk 并推送给客户端。
        """
        async with self.stream_lock:
            if stream_id not in self.active_streams:
                raise HTTPException(status_code=404, detail="Stream not found.")
            frame_queue = self.active_streams[stream_id]["queue"]

        try:
            while True:
                frame_bytes = await frame_queue.get()
                if frame_bytes is None:  # 检查终结信号
                    app_logger.info(f"接收到流 {stream_id} 的终结信号，关闭连接。")
                    break

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except asyncio.CancelledError:
            app_logger.info(f"客户端断开连接，正在关闭流生成器: ID={stream_id}")
        finally:
            app_logger.debug(f"一个客户端已从流 {stream_id} 断开。")

    async def get_all_active_streams_info(self) -> List[ActiveStreamInfo]:
        async with self.stream_lock:
            return [stream["info"] for stream in self.active_streams.values()]

    async def cleanup_expired_streams(self):
        while True:
            await asyncio.sleep(self.settings.app.stream_cleanup_interval_seconds)
            now = datetime.now()

            # 使用列表推导式创建一个副本，避免在迭代时修改字典
            streams_to_check = list(self.active_streams.items())

            expired_stream_ids = [
                stream_id for stream_id, stream in streams_to_check
                if stream["info"].expires_at and now >= stream["info"].expires_at
            ]
            if expired_stream_ids:
                app_logger.info(f"🗑️ 发现 {len(expired_stream_ids)} 个过期视频流，正在清理: {expired_stream_ids}")
                cleanup_tasks = [self.stop_stream(stream_id) for stream_id in expired_stream_ids]
                await asyncio.gather(*cleanup_tasks)

    async def stop_all_streams(self):
        app_logger.info("应用程序关闭，正在停止所有活动的视频流...")
        all_stream_ids = list(self.active_streams.keys())
        if all_stream_ids:
            stop_tasks = [self.stop_stream(stream_id) for stream_id in all_stream_ids]
            results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            stopped_count = sum(1 for res in results if res is True)
            app_logger.info(f"✅ 所有 {stopped_count}/{len(all_stream_ids)} 个活动流已清理完毕。")