# app/core/result_processor.py
import queue
import threading
from datetime import datetime, timedelta
from typing import Dict, Any

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
        """工作线程的主循环。"""
        app_logger.info("结果持久化工作线程已启动。")
        while not self.stop_event.is_set():
            try:
                data: Dict[str, Any] = self.result_queue.get(timeout=1.0)
                if data is None:
                    break

                sn = data.get("sn")
                # 执行节流检查
                if not self._is_ready_to_save(sn):
                    continue

                # --- 执行耗时的IO操作 ---
                face_crop: np.ndarray = data.get("face_crop")
                timestamp_str = data.get("timestamp").strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{sn}_{timestamp_str}.jpg"
                save_path = self.settings.app.detected_imgs_path / filename

                # 1. 保存图片到磁盘
                if face_crop.size > 0:
                    cv2.imwrite(str(save_path), face_crop)
                    image_url = f"http://{IMAGE_URL_IP}:{self.settings.server.port}/static/detected_imgs/{filename}"

                    # 2. 保存记录到数据库 (使用独立的 session)
                    with next(get_db_session()) as db_session:
                        db_data = {
                            "sn": sn,
                            "name": data.get("name"),
                            "similarity": float(data.get("similarity")),
                            "image_url": image_url
                        }
                        insert_new_result(db_session, db_data)

            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"结果持久化服务发生错误: {e}", exc_info=True)
        app_logger.info("结果持久化工作线程已停止。")