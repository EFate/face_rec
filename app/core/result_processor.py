# app/core/result_processor.py
import queue
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time
import cv2
import numpy as np

from app.cfg.config import AppSettings, IMAGE_URL_IP
from app.cfg.logging import app_logger
from app.core.database.database import get_db_session
from app.core.database.results_ops import insert_new_result


class ResultPersistenceService:
    """
    一个后台服务，用于将识别结果异步地、带节流地持久化到磁盘和数据库。
    """

    def __init__(self, settings: AppSettings, result_queue: queue.Queue):
        self.settings = settings
        self.result_queue = result_queue
        self.stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        # 维护一个字典来跟踪每个SN的最后保存时间，用于节流
        self._last_saved_timestamps: Dict[str, datetime] = {}
        # 从配置中获取保存间隔
        self.save_interval = timedelta(seconds=self.settings.app.recognition_save_interval_seconds)

    def start(self):
        """启动后台工作线程。"""
        app_logger.info("启动结果持久化服务...")
        self._thread.start()

    def stop(self):
        """停止后台工作线程。"""
        app_logger.info("正在停止结果持久化服务...")
        self.stop_event.set()
        # 放入一个None来唤醒可能在等待的队列
        try:
            self.result_queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        app_logger.info("✅ 结果持久化服务已停止。")

    def _is_ready_to_save(self, sn: str) -> bool:
        """检查是否应该保存此SN的结果（节流逻辑）。"""
        now = datetime.now()
        last_saved_time = self._last_saved_timestamps.get(sn)
        if last_saved_time is None or (now - last_saved_time) > self.save_interval:
            self._last_saved_timestamps[sn] = now
            return True
        return False

    def _worker(self):
        """工作线程的主循环，支持批处理以减少数据库阻塞。"""
        app_logger.info("结果持久化工作线程已启动。")
        
        batch_data = []
        batch_size = 10  # 批处理大小
        last_batch_time = time.time()
        batch_timeout = 2.0  # 批处理超时时间(秒)
        
        while not self.stop_event.is_set():
            try:
                # 尝试获取数据，使用较短的超时时间以支持批处理
                try:
                    data: Dict[str, Any] = self.result_queue.get(timeout=0.5)
                    if data is None:
                        break
                except queue.Empty:
                    # 队列为空时检查是否需要处理批数据
                    current_time = time.time()
                    if batch_data and (current_time - last_batch_time) > batch_timeout:
                        self._process_batch(batch_data)
                        batch_data.clear()
                        last_batch_time = current_time
                    continue

                sn = data.get("sn")
                # 执行节流检查
                if not self._is_ready_to_save(sn):
                    continue

                # 添加到批处理队列
                batch_data.append(data)
                
                # 检查是否达到批处理条件
                current_time = time.time()
                if (len(batch_data) >= batch_size or 
                    (batch_data and (current_time - last_batch_time) > batch_timeout)):
                    self._process_batch(batch_data)
                    batch_data.clear()
                    last_batch_time = current_time

            except Exception as e:
                app_logger.error(f"结果持久化服务发生错误: {e}", exc_info=True)
        
        # 处理剩余的批数据
        if batch_data:
            self._process_batch(batch_data)
            
        app_logger.info("结果持久化工作线程已停止。")
    
    def _process_batch(self, batch_data: List[Dict[str, Any]]):
        """批处理保存数据到磁盘和数据库"""
        if not batch_data:
            return
            
        try:
            # 批量保存图片
            saved_files = []
            for data in batch_data:
                face_crop: np.ndarray = data.get("face_crop")
                if face_crop is None or face_crop.size == 0:
                    continue
                    
                timestamp_str = data.get("timestamp").strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{data.get('sn')}_{timestamp_str}.jpg"
                save_path = self.settings.app.detected_imgs_path / filename
                
                # 保存图片到磁盘
                cv2.imwrite(str(save_path), face_crop)
                image_url = f"http://{IMAGE_URL_IP}:{self.settings.server.port}/static/detected_imgs/{filename}"
                
                saved_files.append({
                    "sn": data.get("sn"),
                    "name": data.get("name"),
                    "similarity": float(data.get("similarity")),
                    "image_url": image_url
                })
            
            # 批量保存到数据库
            if saved_files:
                with next(get_db_session()) as db_session:
                    for db_data in saved_files:
                        insert_new_result(db_session, db_data)
                    # 一次性提交所有数据
                    db_session.commit()
                
                app_logger.debug(f"批量保存了 {len(saved_files)} 条识别结果")
                
        except Exception as e:
            app_logger.error(f"批处理保存数据时发生错误: {e}", exc_info=True)
