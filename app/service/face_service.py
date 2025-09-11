# app/service/face_service.py
import asyncio
import queue
import threading
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import cv2
import uuid
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from insightface.app.common import Face

from app.cfg.config import AppSettings
from app.service.face_dao import FaceDataDAO, LanceDBFaceDataDAO
from app.core.pipeline import FaceStreamPipeline
from app.schema.face_schema import FaceInfo, FaceRecognitionResult, UpdateFaceRequest, ActiveStreamInfo, \
    StreamStartRequest
from app.cfg.logging import app_logger
from app.core.model_manager import ModelManager  # 引入 ModelManager
from app.core.inference_adapter import InferenceAdapter  # 引入推理适配器
from app.cfg.mqtt_manager import MQTTManager




class FaceService:
    # --- 修改 __init__ 方法以接收队列 ---
    def __init__(self, settings: AppSettings, model_manager: ModelManager, result_queue: queue.Queue, mqtt_manager: MQTTManager):
        app_logger.info("正在初始化 FaceService (多线程 + 模型池)...")
        self.settings = settings
        self.model_manager = model_manager  # 注入模型管理器
        self.inference_adapter = InferenceAdapter(model_manager)  # 创建推理适配器
        self.result_persistence_queue = result_queue  # 注入结果持久化队列
        self.mqtt_manager = mqtt_manager  # 注入MQTT管理器
        self.face_dao: FaceDataDAO = LanceDBFaceDataDAO(
            db_uri=self.settings.insightface.lancedb_uri,
            table_name=self.settings.insightface.lancedb_table_name,
        )
        self.image_db_path = Path(self.settings.insightface.image_db_path)
        self.image_db_path.mkdir(parents=True, exist_ok=True)
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_lock = asyncio.Lock()

    async def initialize(self):
        app_logger.info("FaceService 正在初始化...")
        app_logger.info("✅ FaceService 初始化完毕。")

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: raise ValueError("无法解码图像数据")
            return img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的图像文件: {e}")

    def _save_face_image(self, face_img: np.ndarray, sn: str) -> Path:
        file_uuid = str(uuid.uuid4())
        sn_dir = self.image_db_path / sn
        sn_dir.mkdir(parents=True, exist_ok=True)
        file_path = sn_dir / f"face_{sn}_{file_uuid}.jpg"
        cv2.imwrite(str(file_path), face_img)
        return file_path

    async def register_face(self, name: str, sn: str, image_bytes: bytes) -> FaceInfo:
        img = self._decode_image(image_bytes)
        
        # 使用推理适配器获取人脸
        faces = await self.inference_adapter.get_faces(
            img, 
            extract_embeddings=True, 
            detection_threshold=self.settings.insightface.registration_det_score_threshold
        )
        
        if not faces: 
            # 如果没有检测到人脸，尝试使用更宽松的阈值重新检测
            app_logger.warning(f"使用默认阈值未检测到人脸，尝试使用更宽松的阈值重新检测")
            faces = await self.inference_adapter.get_faces(
                img, 
                extract_embeddings=True, 
                detection_threshold=0.1  # 使用更低的阈值
            )
            
            if not faces:
                raise HTTPException(status_code=400, detail="未在图像中检测到任何人脸。请确保图像清晰且包含正面人脸。")
        
        if len(faces) > 1: 
            raise HTTPException(status_code=400, detail=f"检测到 {len(faces)} 张人脸，注册时必须确保只有一张。")
        
        face = faces[0]
        # 使用专门的注册检测阈值，比识别阈值更宽松
        registration_threshold = self.settings.insightface.registration_det_score_threshold
        if face.det_score < registration_threshold:
            raise HTTPException(
                status_code=400, 
                detail=f"人脸质量不佳，检测置信度({face.det_score:.2f})低于注册要求({registration_threshold})。请使用更清晰的正面人脸图片。"
            )
        
        app_logger.debug(f"注册人脸检测成功: 姓名={name}, SN={sn}, 置信度={face.det_score:.3f}")
        
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_img = img[y1:y2, x1:x2]
        saved_path = self._save_face_image(face_img, sn)
        new_record = self.face_dao.create(name, sn, face.normed_embedding, saved_path)
        return FaceInfo.model_validate(new_record)

    async def recognize_face(self, image_bytes: bytes) -> List[FaceRecognitionResult]:
        img = self._decode_image(image_bytes)
        
        # 使用推理适配器获取人脸
        detected_faces = await self.inference_adapter.get_faces(
            img, 
            extract_embeddings=True, 
            detection_threshold=self.settings.insightface.recognition_det_score_threshold
        )
        
        if not detected_faces: 
            app_logger.info("未检测到任何人脸")
            return []
        
        app_logger.debug(f"检测到 {len(detected_faces)} 张人脸，开始识别")
        
        results = []
        for i, face in enumerate(detected_faces):
            app_logger.debug(f"处理第 {i+1} 张人脸，检测置信度: {face.det_score:.3f}")
            
            # 尝试获取embedding
            embedding = getattr(face, 'normed_embedding', None)
            if embedding is None:
                embedding = getattr(face, 'embedding', None)
            
            if embedding is None:
                app_logger.warning(f"第 {i+1} 张人脸无法获取特征向量")
                continue
            
            # 执行搜索
            search_res = self.face_dao.search(embedding,
                                              self.settings.insightface.recognition_similarity_threshold)
            
            if search_res:
                name, sn, similarity = search_res
                app_logger.debug(f"识别成功: {name} (SN: {sn}), 相似度: {similarity:.3f}")
                results.append(FaceRecognitionResult(
                    name=name, sn=sn, similarity=similarity, box=face.bbox.astype(int).tolist(),
                    detection_confidence=float(face.det_score), landmark=face.landmark_2d_106
                ))
            else:
                app_logger.debug(f"第 {i+1} 张人脸未匹配到已知身份，相似度阈值: {self.settings.insightface.recognition_similarity_threshold}")
                
        app_logger.debug(f"识别完成，匹配到 {len(results)} 个身份")
        return results

    async def get_all_faces(self) -> List[FaceInfo]:
        all_faces_data = self.face_dao.get_all()
        faces = []
        for face_data in all_faces_data:
            face_info = FaceInfo.model_validate(face_data)
            # 将image_path转换为可访问的URL
            if face_info.image_path:
                try:
                    # 将本地文件路径转换为相对路径
                    image_path = Path(face_info.image_path)
                    # 构建相对URL路径，相对于data/faces目录
                    if "faces" in str(image_path):
                        # 提取相对于data/faces的路径
                        rel_path = str(image_path).split("faces", 1)[-1].lstrip("/\\")
                        face_info.image_url = f"http://{self.settings.app.host_ip}:{self.settings.server.port}/api/static/faces/{rel_path}"
                    else:
                        # 使用完整的相对路径
                        face_info.image_url = f"http://{self.settings.app.host_ip}:{self.settings.server.port}/api/static/{image_path.name}"
                except Exception as e:
                    app_logger.warning(f"转换图片路径为URL失败: {e}")
                    face_info.image_url = None
            faces.append(face_info)
        return faces

    async def get_face_by_sn(self, sn: str) -> List[FaceInfo]:
        faces_data = self.face_dao.get_features_by_sn(sn)
        if not faces_data:
            raise HTTPException(status_code=404, detail=f"未找到SN为 '{sn}' 的人脸记录。")
        
        faces = []
        for face_data in faces_data:
            face_info = FaceInfo.model_validate(face_data)
            # 将image_path转换为可访问的URL
            if face_info.image_path:
                try:
                    image_path = Path(face_info.image_path)
                    if "faces" in str(image_path):
                        rel_path = str(image_path).split("faces", 1)[-1].lstrip("/\\")
                        face_info.image_url = f"/api/static/faces/{rel_path}"
                    else:
                        face_info.image_url = f"/api/static/{image_path.name}"
                except Exception as e:
                    app_logger.warning(f"转换图片路径为URL失败: {e}")
                    face_info.image_url = None
            faces.append(face_info)
        return faces

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
        app_logger.debug(f"人员信息已更新: SN={sn}, 新数据={update_dict}, 影响记录数={updated_count}")
        return updated_count, FaceInfo.model_validate(updated_face_info_list[0])

    async def delete_face_by_sn(self, sn: str) -> int:
        records_to_delete = await self.get_face_by_sn(sn)
        deleted_count = self.face_dao.delete_by_sn(sn)
        if deleted_count > 0:
            for record in records_to_delete:
                try:
                    if (p := Path(record.image_path)).exists(): os.remove(p)
                except Exception as e:
                    app_logger.error(f"删除图片文件 {record.image_path} 失败: {e}")
        return deleted_count

    def _get_model_sync(self):
        """同步获取模型，用于在线程中调用"""
        try:
            return self.model_manager.acquire_model()
        except Exception as e:
            app_logger.error(f"同步获取模型失败: {e}")
            raise

    def _release_model_sync(self, model):
        """同步释放模型，用于在线程中调用"""
        try:
            self.model_manager.release_model(model)
        except Exception as e:
            app_logger.error(f"同步释放模型失败: {e}")

    def _pipeline_worker_thread(self, stream_id: str, video_source: str, result_queue: queue.Queue,
                                stop_event: threading.Event, task_id: int, app_id: int, app_name: str, domain_name: str):
        """在独立线程中运行，管理单个视频流管道的生命周期。"""
        if video_source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            app_logger.info(f"【线程 {stream_id}】检测到RTSP源，已设置强制TCP传输。")

        model = None
        pipeline = None
        try:
            # 在独立线程中异步获取模型
            # 使用线程安全的方式获取模型，避免创建新的事件循环
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._get_model_sync)
                model = future.result(timeout=30.0)  # 30秒超时
            
            # --- 在创建流水线时注入持久化队列 ---
            pipeline = FaceStreamPipeline(
                settings=self.settings, stream_id=stream_id, video_source=video_source,
                output_queue=result_queue, model=model,
                result_persistence_queue=self.result_persistence_queue,
                task_id=task_id, app_id=app_id, app_name=app_name, domain_name=domain_name,
                mqtt_manager=self.mqtt_manager
            )
            pipeline.start()  # 启动内部读帧、推理等线程

            # 线程主循环，等待停止信号
            while not stop_event.is_set():
                # 检查pipeline内部线程是否意外终止
                if pipeline and not all(t.is_alive() for t in pipeline.threads):
                    app_logger.error(f"【线程 {stream_id}】检测到内部流水线线程异常终止，正在停止...")
                    break
                time.sleep(1)  # 主循环休眠，不消耗CPU

        except Exception as e:
            app_logger.error(f"【线程 {stream_id}】发生致命错误，无法启动或运行流水线: {e}", exc_info=True)
        finally:
            # 安全清理资源
            try:
                if pipeline:
                    pipeline.stop()
            except Exception as e:
                app_logger.error(f"【线程 {stream_id}】停止流水线时出错: {e}")
            
            try:
                if model:
                    # 同步释放模型
                    self._release_model_sync(model)
            except Exception as e:
                app_logger.error(f"【线程 {stream_id}】释放模型时出错: {e}")
            
            # 发送结束信号到结果队列
            try:
                result_queue.put_nowait(None)
            except (queue.Full, ValueError):
                pass
            
            app_logger.debug(f"✅【线程 {stream_id}】处理工作已结束。")

    async def start_stream(self, req: StreamStartRequest) -> ActiveStreamInfo:
        stream_id = str(uuid.uuid4())
        lifetime = req.lifetime_minutes if req.lifetime_minutes is not None else self.settings.app.stream_default_lifetime_minutes

        async with self.stream_lock:
            result_queue = queue.Queue(maxsize=120)
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._pipeline_worker_thread,
                args=(stream_id, req.source, result_queue, stop_event, req.taskId, req.appId, req.appName, req.domainName),
                daemon=True
            )
            thread.start()
            started_at = datetime.now()
            expires_at = None if lifetime == -1 else started_at + timedelta(minutes=lifetime)
            stream_info = ActiveStreamInfo(
                stream_id=stream_id, 
                task_id=req.taskId,
                app_id=req.appId,
                app_name=req.appName,
                domain_name=req.domainName,
                source=req.source, 
                started_at=started_at,
                expires_at=expires_at, 
                lifetime_minutes=lifetime
            )
            self.active_streams[stream_id] = {"info": stream_info, "queue": result_queue, "stop_event": stop_event,
                                              "thread": thread}
            app_logger.debug(f"🚀 视频流线程已启动: ID={stream_id}, TaskID={req.taskId}, Source={req.source}")
            return stream_info

    async def stop_stream(self, stream_id: str) -> bool:
        async with self.stream_lock:
            stream = self.active_streams.pop(stream_id, None)
            if not stream: 
                app_logger.warning(f"尝试停止不存在的视频流: ID={stream_id}")
                return False
        
        try:
            # 设置停止事件
            stream["stop_event"].set()
            
            # 等待线程结束
            if stream["thread"].is_alive():
                stream["thread"].join(timeout=5.0)
                if stream["thread"].is_alive():
                    app_logger.warning(f"视频流线程 {stream_id} 未能及时退出")
                else:
                    app_logger.debug(f"✅ 视频流已成功停止: ID={stream_id}")
            else:
                app_logger.debug(f"✅ 视频流线程已自然结束: ID={stream_id}")
            
            return True
        except Exception as e:
            app_logger.error(f"停止视频流 {stream_id} 时发生错误: {e}", exc_info=True)
            return False

    async def stop_stream_by_task_id(self, task_id: int) -> bool:
        """根据task_id停止视频流"""
        async with self.stream_lock:
            stream_to_stop = None
            for stream_id, stream in self.active_streams.items():
                if stream["info"].task_id == task_id:
                    stream_to_stop = stream_id
                    break
            
            if not stream_to_stop:
                app_logger.warning(f"尝试停止不存在的视频流: TaskID={task_id}")
                return False
                
            stream = self.active_streams.pop(stream_to_stop, None)
            if not stream: 
                app_logger.warning(f"视频流已被移除: TaskID={task_id}, StreamID={stream_to_stop}")
                return False
            
        try:
            # 设置停止事件
            stream["stop_event"].set()
            
            # 等待线程结束
            if stream["thread"].is_alive():
                stream["thread"].join(timeout=5.0)
                if stream["thread"].is_alive():
                    app_logger.warning(f"视频流线程 TaskID={task_id} 未能及时退出")
                else:
                    app_logger.debug(f"✅ 视频流已成功停止: TaskID={task_id}, StreamID={stream_to_stop}")
            else:
                app_logger.debug(f"✅ 视频流线程已自然结束: TaskID={task_id}, StreamID={stream_to_stop}")
            
            return True
        except Exception as e:
            app_logger.error(f"停止视频流 TaskID={task_id} 时发生错误: {e}", exc_info=True)
            return False

    async def get_stream_feed(self, stream_id: str):
        async with self.stream_lock:
            if stream_id not in self.active_streams: 
                app_logger.warning(f"请求不存在的视频流: StreamID={stream_id}")
                raise HTTPException(status_code=404, detail="Stream not found.")
            stream_info = self.active_streams[stream_id]
            frame_queue = stream_info["queue"]
            
            # 检查线程是否还活着
            if not stream_info["thread"].is_alive():
                app_logger.warning(f"视频流线程已死亡: StreamID={stream_id}")
                # 清理死亡的流
                self.active_streams.pop(stream_id, None)
                raise HTTPException(status_code=404, detail="Stream not found.")
        
        try:
            while True:
                try:
                    # 使用非阻塞方式获取帧数据
                    frame_bytes = frame_queue.get_nowait()
                except queue.Empty:
                    # 队列为空时，异步等待一小段时间
                    await asyncio.sleep(0.01)
                    continue
                
                if frame_bytes is None: 
                    app_logger.debug(f"视频流 {stream_id} 已结束")
                    break
                    
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except (ValueError, asyncio.CancelledError) as e:
            app_logger.debug(f"客户端从流 {stream_id} 断开: {e}")
        except Exception as e:
            app_logger.error(f"获取视频流 {stream_id} 时发生错误: {e}", exc_info=True)
            raise

    async def get_stream_feed_by_task_id(self, task_id: int):
        """根据task_id获取视频流"""
        async with self.stream_lock:
            stream_id = None
            stream_info = None
            for sid, stream in self.active_streams.items():
                if stream["info"].task_id == task_id:
                    stream_id = sid
                    stream_info = stream
                    break
            
            if not stream_id or not stream_info:
                app_logger.warning(f"请求不存在的视频流: TaskID={task_id}")
                raise HTTPException(status_code=404, detail="Stream not found.")
            
            # 检查线程是否还活着
            if not stream_info["thread"].is_alive():
                app_logger.warning(f"视频流线程已死亡: TaskID={task_id}, StreamID={stream_id}")
                # 清理死亡的流
                self.active_streams.pop(stream_id, None)
                raise HTTPException(status_code=404, detail="Stream not found.")
                
            frame_queue = stream_info["queue"]
        
        try:
            while True:
                try:
                    # 使用非阻塞方式获取帧数据
                    frame_bytes = frame_queue.get_nowait()
                except queue.Empty:
                    # 队列为空时，异步等待一小段时间
                    await asyncio.sleep(0.01)
                    continue
                
                if frame_bytes is None: 
                    app_logger.debug(f"视频流 TaskID={task_id} 已结束")
                    break
                    
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except (ValueError, asyncio.CancelledError) as e:
            app_logger.debug(f"客户端从流 TaskID={task_id} 断开: {e}")
        except Exception as e:
            app_logger.error(f"获取视频流 TaskID={task_id} 时发生错误: {e}", exc_info=True)
            raise

    async def get_all_active_streams_info(self) -> List[ActiveStreamInfo]:
        async with self.stream_lock:
            active_infos = []
            dead_stream_ids = []
            for stream_id, stream in self.active_streams.items():
                if stream["thread"].is_alive():
                    active_infos.append(stream["info"])
                else:
                    app_logger.warning(f"检测到视频流线程 {stream_id} 已意外终止。")
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
        if not self.active_streams: 
            app_logger.info("没有活动的视频流需要停止")
            return
        
        all_ids = list(self.active_streams.keys())
        app_logger.info(f"正在停止所有活动流: {all_ids}")
        
        try:
            # 并发停止所有流，但捕获异常避免一个失败影响其他
            results = await asyncio.gather(*[self.stop_stream(sid) for sid in all_ids], return_exceptions=True)
            
            # 统计结果
            success_count = sum(1 for r in results if r is True)
            error_count = sum(1 for r in results if isinstance(r, Exception))
            
            app_logger.debug(f"停止视频流完成: 成功={success_count}, 错误={error_count}")
            
            # 记录错误详情
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    app_logger.error(f"停止流 {all_ids[i]} 时出错: {result}")
                    
        except Exception as e:
            app_logger.error(f"停止所有视频流时发生严重错误: {e}", exc_info=True)