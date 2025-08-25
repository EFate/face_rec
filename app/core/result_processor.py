# app/core/result_processor.py
import queue
import threading
import time
import cv2
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from sqlalchemy.orm import Session

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.database.database import get_db_session
from app.core.database.results_ops import insert_new_result


class ResultPersistenceProcessor:
    """
    结果持久化处理器
    负责异步处理pipeline检测到的结果并保存到数据库
    """

    def __init__(self, settings: AppSettings, result_queue: queue.Queue):
        self.settings = settings
        self.result_queue = result_queue
        self.stop_event = threading.Event()
        self.worker_thread = None

        # 确保检测结果图片保存目录存在 - 保存到static目录方便HTTP访问
        self.detected_imgs_path = Path("app/static/detected_imgs")
        self.detected_imgs_path.mkdir(parents=True, exist_ok=True)

        # 获取宿主机地址配置
        self.host_ip = getattr(self.settings.app, 'host_ip', 'localhost')
        self.server_port = getattr(self.settings.server, 'port', 8000)

    def start(self):
        """启动结果持久化处理线程"""
        if self.worker_thread and self.worker_thread.is_alive():
            app_logger.warning("结果持久化处理器已在运行")
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ResultPersistenceWorker",
            daemon=True
        )
        self.worker_thread.start()
        app_logger.info("✅ 结果持久化处理器已启动")

    def stop(self):
        """停止结果持久化处理线程"""
        if not self.worker_thread or not self.worker_thread.is_alive():
            return

        self.stop_event.set()
        # 放入停止信号
        try:
            self.result_queue.put_nowait(None)
        except queue.Full:
            pass

        self.worker_thread.join(timeout=5.0)
        app_logger.info("✅ 结果持久化处理器已停止")

    def _worker_loop(self):
        """工作线程主循环"""
        app_logger.info("结果持久化处理器工作线程已启动")

        while not self.stop_event.is_set():
            try:
                # 从队列获取检测结果数据
                data = self.result_queue.get(timeout=1.0)

                # 检查停止信号
                if data is None:
                    break

                # 处理检测结果
                self._process_detection_result(data)

            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"处理检测结果时发生错误: {e}", exc_info=True)

        app_logger.info("结果持久化处理器工作线程已停止")

    def _process_detection_result(self, data: Dict[str, Any]):
        """处理单个检测结果"""
        try:
            sn = data.get('sn')
            name = data.get('name')
            similarity = data.get('similarity')
            face_crop = data.get('face_crop')
            timestamp = data.get('timestamp', datetime.now())

            if not all([sn, name, face_crop is not None]):
                app_logger.warning(f"检测结果数据不完整，跳过处理: {data}")
                return

            # 保存人脸裁剪图片
            image_filename = f"detected_{sn}_{int(timestamp.timestamp())}.jpg"
            image_path = self.detected_imgs_path / image_filename

            # 保存图片
            success = cv2.imwrite(str(image_path), face_crop)
            if not success:
                app_logger.error(f"保存检测结果图片失败: {image_path}")
                return

            # 构建可访问的URL
            image_url = f"http://{self.host_ip}:{self.server_port}/static/detected_imgs/{image_filename}"

            # 准备数据库记录
            detection_record = {
                'sn': sn,
                'name': name,
                'similarity': similarity,
                'image_url': image_url,
                'create_time': timestamp,
                'update_time': timestamp
            }

            # 保存到数据库
            db_session = next(get_db_session())
            try:
                insert_new_result(db_session, detection_record)
                app_logger.info(f"成功保存检测结果: SN={sn}, Name={name}, Similarity={similarity:.3f}")
            except Exception as e:
                app_logger.error(f"保存检测结果到数据库失败: {e}")
                db_session.rollback()
                # 删除已保存的图片文件
                try:
                    os.remove(image_path)
                except:
                    pass
            finally:
                db_session.close()

        except Exception as e:
            app_logger.error(f"处理检测结果时发生异常: {e}", exc_info=True)