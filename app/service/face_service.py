# app/service/face_service.py
import asyncio
import multiprocessing as mp
import queue
import threading
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import cv2
import uuid
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from app.cfg.config import AppSettings
from app.service.face_dao import FaceDataDAO, LanceDBFaceDataDAO
from app.core.pipeline import FaceStreamPipeline
from app.schema.face_schema import FaceInfo, FaceRecognitionResult, UpdateFaceRequest, ActiveStreamInfo, \
    StreamStartRequest
from app.cfg.logging import app_logger


# ==============================================================================
#  视频处理工作函数 (子进程入口点)
# ==============================================================================
def video_stream_process_worker(
        stream_id: str,
        video_source: str,
        settings_dict: Dict[str, Any],
        result_queue: mp.Queue,
        stop_event: mp.Event
):
    """
    在独立子进程中运行。
    职责：实例化并运行一个完整的 FaceStreamPipeline，直到 stop_event 被设置。
    """
    # 在子进程中，根据视频源类型设置环境变量，以优化OpenCV的视频捕获行为
    if video_source.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        app_logger.info(f"【子进程 {stream_id}】检测到RTSP源，已设置强制TCP传输以提高稳定性。")

    pipeline = None
    try:
        settings = AppSettings.model_validate(settings_dict)
        os.environ["INSIGHTFACE_HOME"] = str(settings.insightface.home)
        # pipeline的初始化和启动，包含了对视频源是否有效的检查
        pipeline = FaceStreamPipeline(
            settings=settings,
            stream_id=stream_id,
            video_source=video_source,
            output_queue=result_queue
        )

        def on_stop():
            stop_event.wait()
            if pipeline:
                pipeline.stop()

        stop_monitor = threading.Thread(target=on_stop, daemon=True)
        stop_monitor.start()
        pipeline.start()
    except Exception as e:
        app_logger.error(f"【子进程 {stream_id}】发生致命错误，无法启动或运行流水线: {e}", exc_info=True)
    finally:
        try:
            result_queue.put_nowait(None)  # 确保无论如何都通知主进程结束
        except (queue.Full, ValueError):
            pass
        app_logger.info(f"✅【子进程 {stream_id}】处理工作已结束。")


# ==============================================================================
#  FaceService 管理类 (最终修正版)
# ==============================================================================
class FaceService:
    def __init__(self, settings: AppSettings, model: FaceAnalysis):
        app_logger.info("正在初始化 FaceService (v4.4 - 连接逻辑修正)...")
        self.settings = settings
        self.model = model
        self.face_dao: FaceDataDAO = LanceDBFaceDataDAO(
            db_uri=self.settings.insightface.lancedb_uri,
            table_name=self.settings.insightface.lancedb_table_name,
        )
        self.image_db_path = Path(self.settings.insightface.image_db_path)
        self.image_db_path.mkdir(parents=True, exist_ok=True)
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_lock = asyncio.Lock()
        self.mp_context = mp.get_context("spawn")

    async def initialize(self):
        app_logger.info("FaceService 正在初始化...")
        app_logger.info("✅ FaceService 初始化完毕。")

    # --- 省略了所有其他无需修改的方法 (register_face, get_all_faces, update_face_by_sn 等)... ---
    # ... 您可以保留您文件中这些已经正常工作的方法 ...
    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: raise ValueError("无法解码图像数据")
            return img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

    def _get_faces_from_image(self, img: np.ndarray) -> List[Face]:
        return self.model.get(img)

    def _save_face_image(self, face_img: np.ndarray, sn: str) -> Path:
        file_uuid = str(uuid.uuid4())
        sn_dir = self.image_db_path / sn
        sn_dir.mkdir(parents=True, exist_ok=True)
        file_path = sn_dir / f"face_{sn}_{file_uuid}.jpg"
        cv2.imwrite(str(file_path), face_img)
        return file_path

    async def register_face(self, name: str, sn: str, image_bytes: bytes) -> FaceInfo:
        img = self._decode_image(image_bytes)
        faces = self._get_faces_from_image(img)
        if not faces: raise HTTPException(status_code=400, detail="未在图像中检测到任何人脸。")
        if len(faces) > 1: raise HTTPException(status_code=400,
                                               detail=f"检测到 {len(faces)} 张人脸，注册时必须确保只有一张。")
        face = faces[0]
        if face.det_score < self.settings.insightface.recognition_det_score_threshold:
            raise HTTPException(status_code=400, detail=f"人脸质量不佳，检测置信度({face.det_score:.2f})过低。")
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_img = img[y1:y2, x1:x2]
        saved_path = self._save_face_image(face_img, sn)
        new_record = self.face_dao.create(name, sn, face.normed_embedding, saved_path)
        return FaceInfo.model_validate(new_record)

    async def get_all_faces(self) -> List[FaceInfo]:
        all_faces_data = self.face_dao.get_all()
        return [FaceInfo.model_validate(face) for face in all_faces_data]

    async def get_face_by_sn(self, sn: str) -> List[FaceInfo]:
        faces_data = self.face_dao.get_features_by_sn(sn)
        if not faces_data:
            raise HTTPException(status_code=404, detail=f"未找到SN为 '{sn}' 的人脸记录。")
        return [FaceInfo.model_validate(face) for face in faces_data]

    async def update_face_by_sn(self, sn: str, update_data: UpdateFaceRequest) -> Tuple[int, FaceInfo]:
        update_dict = update_data.model_dump(exclude_unset=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="请求体中未提供任何更新数据。")
        await self.get_face_by_sn(sn)
        updated_count = self.face_dao.update_by_sn(sn, update_dict)
        if updated_count == 0:
            app_logger.warning(f"更新操作成功，但SN为'{sn}'的0条记录被更新。")
        updated_face_info_list = self.face_dao.get_features_by_sn(sn)
        if not updated_face_info_list:
            raise HTTPException(status_code=500, detail="更新后无法找回记录，数据可能不一致。")
        app_logger.info(f"人员信息已更新: SN={sn}, 新数据={update_dict}, 影响记录数={updated_count}")
        return updated_count, FaceInfo.model_validate(updated_face_info_list[0])

    async def delete_face_by_sn(self, sn: str) -> int:
        records_to_delete = await self.get_face_by_sn(sn)
        deleted_count = self.face_dao.delete_by_sn(sn)
        if deleted_count > 0:
            for record in records_to_delete:
                try:
                    # 注意：Pydantic v2 .image_path可以直接访问
                    if (p := Path(record.image_path)).exists(): os.remove(p)
                except Exception as e:
                    app_logger.error(f"删除图片文件 {record.image_path} 失败: {e}")
        return deleted_count

    async def recognize_face(self, image_bytes: bytes) -> List[FaceRecognitionResult]:
        img = self._decode_image(image_bytes)
        detected_faces = self._get_faces_from_image(img)
        if not detected_faces: return []
        results = []
        for face in detected_faces:
            search_res = self.face_dao.search(face.normed_embedding,
                                              self.settings.insightface.recognition_similarity_threshold)
            if search_res:
                name, sn, similarity = search_res
                results.append(FaceRecognitionResult(
                    name=name, sn=sn, similarity=similarity, box=face.bbox.astype(int).tolist(),
                    detection_confidence=float(face.det_score), landmark=face.landmark_2d_106
                ))
        return results

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        # 视频源的有效性检查交由子进程处理，这是解决h264问题的关键。

        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes

        async with self.stream_lock:
            result_queue = self.mp_context.Queue(maxsize=120)
            stop_event = self.mp_context.Event()
            process = self.mp_context.Process(
                target=video_stream_process_worker,
                args=(stream_id, req.source, self.settings.model_dump(), result_queue, stop_event),
                daemon=True
            )
            process.start()
            started_at = datetime.now()
            expires_at = None if lifetime == -1 else started_at + timedelta(minutes=lifetime)
            stream_info = ActiveStreamInfo(stream_id=stream_id, source=req.source, started_at=started_at,
                                           expires_at=expires_at, lifetime_minutes=lifetime)
            self.active_streams[stream_id] = {"info": stream_info, "queue": result_queue, "stop_event": stop_event,
                                              "process": process}
            app_logger.info(f"🚀 视频流进程已启动: ID={stream_id}, Source={req.source}")
            return stream_info

    # ... 其余方法 (stop_stream, get_stream_feed等) 保持不变 ...
    async def stop_stream(self, stream_id: str) -> bool:
        async with self.stream_lock:
            stream = self.active_streams.pop(stream_id, None)
            if not stream: return False

        stream["stop_event"].set()
        stream["queue"].close()
        stream["process"].join(timeout=5.0)
        if stream["process"].is_alive():
            stream["process"].terminate()
            stream["process"].join()
        stream["queue"].join_thread()
        app_logger.info(f"✅ 视频流已成功停止: ID={stream_id}")
        return True

    async def get_stream_feed(self, stream_id: str):
        async with self.stream_lock:
            if stream_id not in self.active_streams: raise HTTPException(status_code=404, detail="Stream not found.")
            frame_queue = self.active_streams[stream_id]["queue"]

        try:
            while True:
                try:
                    frame_bytes = frame_queue.get(timeout=0.01)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                if frame_bytes is None: break
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except (ValueError, asyncio.CancelledError):
            app_logger.info(f"客户端从流 {stream_id} 断开。")

    async def get_all_active_streams_info(self) -> List[ActiveStreamInfo]:
        async with self.stream_lock:
            active_infos = []
            dead_stream_ids = []
            for stream_id, stream in self.active_streams.items():
                if stream["process"].is_alive():
                    active_infos.append(stream["info"])
                else:
                    app_logger.warning(f"检测到视频流进程 {stream_id} 已意外终止。")
                    dead_stream_ids.append(stream_id)
            for sid in dead_stream_ids:
                self.active_streams.pop(sid, None)
            return active_infos

    async def cleanup_expired_streams(self):
        while True:
            await asyncio.sleep(self.settings.app.stream_cleanup_interval_seconds)
            now = datetime.now()
            expired_ids = [sid for sid, s in list(self.active_streams.items()) if
                           s["info"].expires_at and now >= s["info"].expires_at]
            if expired_ids:
                app_logger.info(f"正在清理过期视频流: {expired_ids}")
                await asyncio.gather(*[self.stop_stream(sid) for sid in expired_ids])

    async def stop_all_streams(self):
        if not self.active_streams: return
        all_ids = list(self.active_streams.keys())
        app_logger.info(f"正在停止所有活动流: {all_ids}")
        await asyncio.gather(*[self.stop_stream(sid) for sid in all_ids])