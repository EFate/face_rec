# app/service/face_service.py
import asyncio
import multiprocessing
import queue
import threading
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
from app.service.face_dao import FaceDataDAO, SQLiteFaceDataDAO
from app.schema.face_schema import FaceInfo, FaceRecognitionResult, UpdateFaceRequest, ActiveStreamInfo, \
    StreamStartRequest
from app.cfg.logging import app_logger
# --- 【重要】导入统一的模型创建函数 ---
from app.core.model_manager import create_face_analysis_model


# --- 辅助函数：非极大值抑制 (Non-Max Suppression, NMS) ---
def non_max_suppression(tracks: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    """
    对检测结果列表应用非极大值抑制，消除重叠的框。
    """
    if not tracks:
        return []

    boxes = np.array([t['box'] for t in tracks])
    scores = np.array([t.get('det_score', 0.9) for t in tracks])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]

    keep_indices = []
    while order.size > 0:
        i = order[0]
        keep_indices.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [tracks[i] for i in keep_indices]


def _draw_results_on_frame(frame: np.ndarray, results: List[Dict[str, Any]]):
    """在帧上绘制识别结果（边界框和标签）"""
    for res in results:
        box = res['box'].astype(int)
        label = f"{res['name']}"
        if res['similarity'] is not None:
            label += f" ({res['similarity']:.2f})"
        color = (0, 255, 0) if res['name'] != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (box[0], box[1] - lh - 10), (box[0] + lw, box[1]), color, cv2.FILLED)
        cv2.putText(frame, label, (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ==============================================================================
#  视频处理工作函数 (修改)
# ==============================================================================
def video_stream_process_worker(
        stream_id: str,
        video_source: str,
        settings_dict: Dict[str, Any],
        result_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event
):
    """
    在独立进程中运行的视频处理工作函数。
    【架构优化】: 内部采用多线程实现生产者-消费者模式，分离视频帧的读取和处理。
    - 抓取线程(生产者): 仅负责从视频源高速读取帧，放入内部队列。
    - 处理线程(消费者): 从内部队列获取帧，进行模型推理和图像绘制。
    """

    def frame_grabber_thread(cap, internal_queue, internal_stop_event):
        app_logger.info(f"【子进程 {stream_id} -> 抓取线程】启动。")
        while not internal_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                app_logger.warning(f"【子进程 {stream_id} -> 抓取线程】无法读取到视频帧，可能已结束。将发出停止信号。")
                internal_queue.put(None)
                break
            if not internal_queue.empty():
                try:
                    internal_queue.get_nowait()
                except queue.Empty:
                    pass
            internal_queue.put(frame)
        app_logger.info(f"【子进程 {stream_id} -> 抓取线程】已停止。")

    grabber_thread_handle = None
    cap = None
    internal_thread_stop_event = threading.Event()
    try:
        # --- 【优化】复用配置和统一的模型加载逻辑 ---
        settings = AppSettings.model_validate(settings_dict)
        app_logger.info(f"【子进程 {stream_id}】正在启动，目标处理频率: {settings.app.stream_capture_fps} FPS。")

        # 设置环境变量并使用统一函数加载模型
        os.environ["INSIGHTFACE_HOME"] = str(settings.insightface.home)
        model = create_face_analysis_model(settings)

        app_logger.info(f"✅【子进程 {stream_id}】模型加载成功。")

        # --- 后续逻辑保持不变 ---
        face_dao = SQLiteFaceDataDAO(db_url=settings.database.url)

        def get_latest_known_faces() -> Dict[str, Any]:
            all_faces = face_dao.get_all()
            if not all_faces:
                return {"features_matrix": np.empty((0, 512), dtype=np.float32), "metadata": []}
            features = np.array([f['features'] for f in all_faces], dtype=np.float32)
            metadata = [{"sn": f['sn'], "name": f['name']} for f in all_faces]
            return {"features_matrix": features, "metadata": metadata}

        known_faces_cache = get_latest_known_faces()
        app_logger.info(f"【子进程 {stream_id}】初始人脸缓存加载: {len(known_faces_cache['metadata'])} 条记录。")
        if "rtsp://" in video_source and settings.app.rtsp_use_tcp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        elif "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
            del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        source = int(video_source) if video_source.isdigit() else video_source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {video_source}")
        internal_frame_queue = queue.Queue(maxsize=2)
        grabber_thread_handle = threading.Thread(
            target=frame_grabber_thread,
            args=(cap, internal_frame_queue, internal_thread_stop_event)
        )
        grabber_thread_handle.start()
        target_fps = settings.app.stream_capture_fps
        target_interval = 1.0 / target_fps if target_fps > 0 else 0
        last_process_time = 0
        last_cache_update_time = time.time()
        while not stop_event.is_set():
            current_time = time.time()
            if (current_time - last_process_time) < target_interval:
                time.sleep(0.001)
                continue
            try:
                frame = internal_frame_queue.get(timeout=1.0)
                if frame is None:
                    app_logger.info(f"【子进程 {stream_id} -> 处理线程】接收到流结束信号。")
                    break
            except queue.Empty:
                app_logger.warning(f"【子进程 {stream_id} -> 处理线程】等待帧超时，视频源可能已卡顿或结束。")
                continue
            last_process_time = current_time
            if current_time - last_cache_update_time > settings.app.stream_cache_update_interval_seconds:
                known_faces_cache = get_latest_known_faces()
                last_cache_update_time = time.time()
                app_logger.info(f"【子进程 {stream_id}】人脸缓存已刷新: {len(known_faces_cache['metadata'])} 条记录。")
            final_results = []
            try:
                detected_faces = model.get(frame)
                current_results = []
                if detected_faces and known_faces_cache['metadata']:
                    detected_features = np.array([face.normed_embedding for face in detected_faces])
                    sim_matrix = np.dot(detected_features, known_faces_cache['features_matrix'].T)
                    for i, face in enumerate(detected_faces):
                        sims = sim_matrix[i]
                        best_match_idx = np.argmax(sims)
                        max_sim = sims[best_match_idx]
                        name, sn, similarity = "Unknown", None, None
                        if max_sim > settings.insightface.recognition_similarity_threshold:
                            meta = known_faces_cache['metadata'][best_match_idx]
                            name, sn, similarity = meta['name'], meta['sn'], float(max_sim)
                        current_results.append({
                            'box': face.bbox, 'name': name, 'sn': sn,
                            'similarity': similarity, 'det_score': face.det_score
                        })
                final_results = non_max_suppression(current_results, iou_threshold=0.4)
            except Exception as e:
                app_logger.error(f"【子进程 {stream_id}】帧处理时发生错误: {e}", exc_info=True)
                final_results = []
            if final_results:
                _draw_results_on_frame(frame, final_results)
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if flag:
                try:
                    result_queue.put_nowait(encodedImage.tobytes())
                except queue.Full:
                    pass
    except Exception as e:
        app_logger.error(f"【子进程 {stream_id}】发生致命错误，进程退出: {e}", exc_info=True)
    finally:
        internal_thread_stop_event.set()
        if grabber_thread_handle and grabber_thread_handle.is_alive():
            grabber_thread_handle.join(timeout=2.0)
        if cap and cap.isOpened():
            cap.release()
        try:
            result_queue.put_nowait(None)
        except (queue.Full, ValueError):
            pass
        app_logger.info(f"✅【子进程 {stream_id}】处理工作已结束。")


class FaceService:
    def __init__(self, settings: AppSettings, model: FaceAnalysis):
        app_logger.info("正在初始化 FaceService...")
        self.settings = settings
        self.model = model
        if self.settings.insightface.storage_type == StorageType.SQLITE:
            self.face_dao: FaceDataDAO = SQLiteFaceDataDAO(db_url=self.settings.database.url)
        else:
            raise ValueError(f"不支持的存储类型: {self.settings.insightface.storage_type}")
        self.image_db_path = Path(self.settings.insightface.image_db_path)
        self.image_db_path.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = self.settings.insightface.recognition_similarity_threshold
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_lock = asyncio.Lock()
        self.known_faces_cache: Dict[str, Any] = {}
        self.cache_lock = asyncio.Lock()

    # --- 缓存管理 ---
    async def _rebuild_cache_from_db(self):
        app_logger.info("正在为【主进程】从数据库重建人脸特征缓存...")
        all_faces_data = self.face_dao.get_all()
        if not all_faces_data:
            self.known_faces_cache = {"features_matrix": np.empty((0, 512), dtype=np.float32), "metadata": []}
        else:
            features_list = [face["features"] for face in all_faces_data]
            self.known_faces_cache["features_matrix"] = np.array(features_list, dtype=np.float32)
            self.known_faces_cache["metadata"] = [{"sn": face["sn"], "name": face["name"]} for face in all_faces_data]
        app_logger.info(f"✅ 【主进程】缓存重建完成！缓存中共有 {len(self.known_faces_cache['metadata'])} 条人脸特征。")

    async def load_and_cache_features(self):
        async with self.cache_lock:
            await self._rebuild_cache_from_db()

    async def _add_to_cache(self, face_data: Dict[str, Any]):
        async with self.cache_lock:
            current_features = self.known_faces_cache["features_matrix"]
            new_feature = face_data["features"].reshape(1, -1).astype(np.float32)
            if current_features.size == 0:
                self.known_faces_cache["features_matrix"] = new_feature
            else:
                self.known_faces_cache["features_matrix"] = np.vstack([current_features, new_feature])
            self.known_faces_cache["metadata"].append({"sn": face_data["sn"], "name": face_data["name"]})

    async def _remove_from_cache(self, sn: str):
        async with self.cache_lock:
            metadata = self.known_faces_cache["metadata"]
            indices_to_delete = [i for i, meta in enumerate(metadata) if meta["sn"] == sn]
            if not indices_to_delete: return
            self.known_faces_cache["features_matrix"] = np.delete(self.known_faces_cache["features_matrix"],
                                                                  indices_to_delete, axis=0)
            for i in sorted(indices_to_delete, reverse=True): del self.known_faces_cache["metadata"][i]

    async def _update_in_cache(self, sn: str, new_name: str):
        async with self.cache_lock:
            for item in self.known_faces_cache["metadata"]:
                if item["sn"] == sn: item["name"] = new_name

    # --- 内部辅助函数 ---
    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: raise ValueError("无法解码图像数据")
            return img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

    def _get_faces_from_image(self, img: np.ndarray) -> List[Face]:
        try:
            return self.model.get(img)
        except Exception as e:
            app_logger.error(f"调用 model.get() 时发生异常: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"人脸分析服务内部错误: {e}")

    def _crop_face_image(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype(int)
        y1, y2 = max(0, y1 - 20), min(img.shape[0], y2 + 20)
        x1, x2 = max(0, x1 - 20), min(img.shape[1], x2 + 20)
        return img[y1:y2, x1:x2]

    def _save_face_image_and_get_path(self, face_img: np.ndarray, sn: str) -> Path:
        file_uuid = str(uuid.uuid4())
        sn_dir = self.image_db_path / sn
        sn_dir.mkdir(parents=True, exist_ok=True)
        file_path = sn_dir / f"face_{sn}_{file_uuid}.jpg"
        success, encoded_image = cv2.imencode(".jpg", face_img)
        if not success: raise HTTPException(status_code=500, detail="无法编码裁剪的人脸图像。")
        with open(file_path, "wb") as f: f.write(encoded_image.tobytes())
        return file_path

    # --- 人脸管理核心 API (修改) ---
    async def register_face(self, name: str, sn: str, image_bytes: bytes) -> FaceInfo:
        img = self._decode_image(image_bytes)
        faces = self._get_faces_from_image(img)

        # --- 【优化】改进错误提示 ---
        if not faces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="未在图像中检测到任何人脸。请确保上传的图片清晰、光照良好，且人脸正对前方。"
            )
        if len(faces) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"检测到 {len(faces)} 张人脸，注册时必须确保图像中只有一张人脸。"
            )

        face = faces[0]
        if face.det_score < self.settings.insightface.recognition_det_score_threshold:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"人脸质量不佳，检测置信度({face.det_score:.2f})过低，低于阈值 {self.settings.insightface.recognition_det_score_threshold}。"
            )

        features = face.normed_embedding
        cropped_face_img = self._crop_face_image(img, face.bbox)
        saved_image_path = self._save_face_image_and_get_path(cropped_face_img, sn)
        new_face_record = self.face_dao.create(name, sn, features, saved_image_path)
        await self._add_to_cache(new_face_record)
        return FaceInfo.model_validate(new_face_record)

    async def get_all_faces(self) -> List[FaceInfo]:
        """【新增】获取所有已注册人脸信息"""
        all_faces_data = self.face_dao.get_all()
        return [FaceInfo.model_validate(face_data) for face_data in all_faces_data]

    async def get_face_by_sn(self, sn: str) -> List[FaceInfo]:
        """【新增】根据SN获取人脸信息"""
        faces_data = self.face_dao.get_features_by_sn(sn)
        if not faces_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"未找到SN为 '{sn}' 的人脸记录。")
        return [FaceInfo.model_validate(face_data) for face_data in faces_data]

    async def update_face_by_sn(self, sn: str, update_data: UpdateFaceRequest) -> FaceInfo:
        """【新增】根据SN更新人员信息（如姓名）"""
        update_dict = update_data.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求体中未提供任何更新数据。")

        # 确保该SN存在
        await self.get_face_by_sn(sn)

        updated_count = self.face_dao.update_by_sn(sn, update_dict)
        if updated_count > 0 and 'name' in update_dict:
            await self._update_in_cache(sn, update_dict['name'])
            app_logger.info(f"人员信息已更新: SN={sn}, 新数据={update_dict}")

        # 返回更新后的第一条记录作为代表
        updated_face_info = self.face_dao.get_features_by_sn(sn)[0]
        return FaceInfo.model_validate(updated_face_info)

    async def delete_face_by_sn(self, sn: str) -> int:
        """【新增】根据SN删除人脸记录及关联图片"""
        records_to_delete = self.face_dao.get_features_by_sn(sn)
        if not records_to_delete:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"未找到SN为 '{sn}' 的人脸记录。")

        # 1. 从数据库删除记录
        deleted_count = self.face_dao.delete_by_sn(sn)

        # 2. 如果数据库删除成功，则清理文件和缓存
        if deleted_count > 0:
            app_logger.info(f"准备为SN '{sn}' 删除 {len(records_to_delete)} 个关联图片文件...")
            for record in records_to_delete:
                try:
                    image_path = Path(record["image_path"])
                    if image_path.exists():
                        os.remove(image_path)
                        app_logger.info(f"  - 已删除文件: {image_path}")
                except Exception as e:
                    app_logger.error(f"删除图片文件 {record['image_path']} 失败: {e}", exc_info=True)

            # 尝试删除空的SN目录
            try:
                sn_dir = self.image_db_path / sn
                if sn_dir.exists() and not any(sn_dir.iterdir()):
                    os.rmdir(sn_dir)
                    app_logger.info(f"已删除空的SN目录: {sn_dir}")
            except Exception as e:
                app_logger.error(f"删除SN目录 {sn_dir} 失败: {e}", exc_info=True)

            await self._remove_from_cache(sn)
            app_logger.info(f"SN '{sn}' 的人脸记录已从缓存中移除。")

        return deleted_count

    # --- 识别与视频流 ---
    async def recognize_face(self, image_bytes: bytes) -> List[FaceRecognitionResult]:
        async with self.cache_lock:
            if not self.known_faces_cache or not self.known_faces_cache.get("metadata"):
                await self._rebuild_cache_from_db()
            known_features_matrix = self.known_faces_cache["features_matrix"]
            known_metadata = self.known_faces_cache["metadata"]

        if known_features_matrix.size == 0: return []
        img = self._decode_image(image_bytes)
        detected_faces = self._get_faces_from_image(img)
        if not detected_faces: return []
        detected_features_matrix = np.array([face.normed_embedding for face in detected_faces])
        similarity_matrix = np.dot(detected_features_matrix, known_features_matrix.T)
        final_results = []
        for i, detected_face in enumerate(detected_faces):
            similarities = similarity_matrix[i]
            if not similarities.any(): continue
            best_match_index = np.argmax(similarities)
            max_similarity = similarities[best_match_index]
            if max_similarity > self.similarity_threshold:
                best_match_meta = known_metadata[best_match_index]
                final_results.append(FaceRecognitionResult(
                    name=best_match_meta["name"], sn=best_match_meta["sn"], similarity=float(max_similarity),
                    box=detected_face.bbox.astype(int).tolist(), detection_confidence=float(detected_face.det_score),
                    landmark=detected_face.landmark_2d_106
                ))
        return final_results

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        source_to_check = int(req.source) if req.source.isdigit() else req.source
        cap_check = cv2.VideoCapture(source_to_check)
        if not cap_check.isOpened():
            cap_check.release()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"无法打开视频源 '{req.source}'")
        cap_check.release()

        if "rtsp://" in req.source and not self.settings.app.rtsp_use_tcp:
            app_logger.warning(
                f"正在启动一个RTSP视频流 (source: {req.source})，但配置中 'rtsp_use_tcp' 为 false。如果出现花屏，建议在配置中启用它。")

        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes

        async with self.stream_lock:
            if stream_id in self.active_streams:
                raise HTTPException(status_code=409, detail="Stream ID conflict.")
            result_queue = multiprocessing.Queue(maxsize=120)
            stop_event = multiprocessing.Event()
            settings_dict = self.settings.model_dump()
            process = multiprocessing.Process(
                target=video_stream_process_worker,
                args=(stream_id, req.source, settings_dict, result_queue, stop_event),
                daemon=True
            )
            process.start()
            started_at = datetime.now()
            expires_at = None if lifetime == -1 else started_at + timedelta(minutes=lifetime)
            stream_info = ActiveStreamInfo(stream_id=stream_id, source=req.source, started_at=started_at,
                                           expires_at=expires_at, lifetime_minutes=lifetime)
            self.active_streams[stream_id] = {
                "info": stream_info, "queue": result_queue,
                "stop_event": stop_event, "process": process
            }
            app_logger.info(
                f"🚀 视频流进程已启动: ID={stream_id}, Source={req.source}, Target FPS={self.settings.app.stream_capture_fps}, Lifetime={lifetime} mins")
            return stream_info

    async def stop_stream(self, stream_id: str) -> bool:
        async with self.stream_lock:
            stream = self.active_streams.pop(stream_id, None)
            if not stream: return False
        app_logger.info(f"⏹️ 正在停止视频流: ID={stream_id}...")
        stream["stop_event"].set()
        stream["process"].join(timeout=5.0)
        if stream["process"].is_alive():
            app_logger.warning(f"视频流进程 {stream_id} 未能在5秒内正常退出，将强制终止。")
            stream["process"].terminate()
            stream["process"].join()
        while not stream["queue"].empty():
            try:
                stream["queue"].get_nowait()
            except queue.Empty:
                break
        stream["queue"].close()
        stream["queue"].join_thread()
        app_logger.info(f"✅ 视频流已成功停止: ID={stream_id}")
        return True

    async def get_stream_feed(self, stream_id: str):
        async with self.stream_lock:
            if stream_id not in self.active_streams:
                raise HTTPException(status_code=404, detail="Stream not found.")
            frame_queue = self.active_streams[stream_id]["queue"]
        try:
            while True:
                try:
                    frame_bytes = frame_queue.get(timeout=0.01)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                if frame_bytes is None:
                    app_logger.info(f"接收到流 {stream_id} 的结束信号。")
                    break
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except ValueError as e:
            app_logger.info(f"流 {stream_id} 的队列已被关闭，此为正常停止流程的一部分。({e})")
        except asyncio.CancelledError:
            app_logger.info(f"客户端断开，关闭流生成器: ID={stream_id}")
        finally:
            app_logger.debug(f"一个客户端已从流 {stream_id} 断开。")

    async def get_all_active_streams_info(self) -> List[ActiveStreamInfo]:
        async with self.stream_lock:
            active_infos = []
            dead_stream_ids = []
            for stream_id, stream in self.active_streams.items():
                if stream["process"].is_alive():
                    active_infos.append(stream["info"])
                else:
                    app_logger.warning(f"检测到视频流进程 {stream_id} 已意外终止。将从活动列表移除。")
                    dead_stream_ids.append(stream_id)
            for sid in dead_stream_ids:
                self.active_streams.pop(sid, None)
            return active_infos

    async def cleanup_expired_streams(self):
        while True:
            await asyncio.sleep(self.settings.app.stream_cleanup_interval_seconds)
            now = datetime.now()
            streams_to_check = list(self.active_streams.items())
            expired_stream_ids = [
                stream_id for stream_id, stream in streams_to_check
                if stream["info"].expires_at and now >= stream["info"].expires_at
            ]
            if expired_stream_ids:
                app_logger.info(f"检测到过期的视频流: {expired_stream_ids}，将进行清理。")
                cleanup_tasks = [self.stop_stream(stream_id) for stream_id in expired_stream_ids]
                await asyncio.gather(*cleanup_tasks)

    async def stop_all_streams(self):
        app_logger.info("正在停止所有活动的视频流...")
        async with self.stream_lock:
            all_stream_ids = list(self.active_streams.keys())
        if all_stream_ids:
            stop_tasks = [self.stop_stream(stream_id) for stream_id in all_stream_ids]
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            app_logger.info("所有视频流已停止。")