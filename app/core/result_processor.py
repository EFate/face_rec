import queue
import threading
import time
import cv2
import os
from datetime import datetime
import pytz
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.database.database import get_db_session
from app.core.database.results_ops import insert_batch_results
from app.cfg.mqtt_manager import MQTTManager


class ResultPersistenceProcessor:
    """
    结果持久化处理器
    负责异步处理pipeline检测到的结果并批量保存到数据库
    """

    def __init__(self, settings: AppSettings, result_queue: queue.Queue, mqtt_manager: MQTTManager = None):
        self.settings = settings
        self.result_queue = result_queue
        self.mqtt_manager = mqtt_manager
        self.stop_event = threading.Event()
        self.worker_thread = None
        
        # 保存间隔控制 - 使用配置参数
        self.last_save_time: Dict[str, float] = {}
        self.frame_counters: Dict[str, int] = {}
        self.save_interval_seconds = self.settings.app.recognition_save_interval_seconds
        self.frame_interval = self.settings.app.recognition_frame_interval
        
        # 批量保存缓冲区 - 确保每10帧批量保存
        self.batch_buffer: List[Dict[str, Any]] = []
        self.last_batch_save_time = time.time()
        self.batch_size = self.frame_interval  # 使用帧间隔作为批量大小，确保一致性
        self.batch_timeout = 3.0  # 最长3秒保存一次，避免数据积压

        # 确保检测结果图片保存目录存在 - 保存到static目录方便HTTP访问
        self.detected_imgs_path = Path("app/static/detected_imgs")
        self.detected_imgs_path.mkdir(parents=True, exist_ok=True)

        # 获取宿主机地址配置
        self.host_ip = getattr(self.settings.app, 'host_ip', 'localhost')
        self.server_port = getattr(self.settings.server, 'port', 8000)
        
        app_logger.info(f"结果持久化处理器初始化完成，保存间隔：{self.save_interval_seconds}秒，帧间隔：每{self.frame_interval}帧")

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

        # 确保最后的批次被保存
        self._save_batch_buffer(force=True)
        
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
                
                # 检查是否需要批量保存（基于时间）
                current_time = time.time()
                if current_time - self.last_batch_save_time >= self.batch_timeout:
                    self._save_batch_buffer()

            except queue.Empty:
                # 队列为空时，检查是否需要保存批次（基于时间）
                current_time = time.time()
                if self.batch_buffer and current_time - self.last_batch_save_time >= self.batch_timeout:
                    self._save_batch_buffer()
                continue
            except Exception as e:
                app_logger.error(f"处理检测结果时发生错误: {e}", exc_info=True)

        # 确保最后的批次被保存
        self._save_batch_buffer(force=True)
        app_logger.info("结果持久化处理器工作线程已停止")

    def _process_detection_result(self, data: Dict[str, Any]):
        """处理单个检测结果"""
        try:
            sn = data.get('sn')
            name = data.get('name')
            similarity = data.get('similarity')
            face_crop = data.get('face_crop')
            # 使用中国时区 (Asia/Shanghai)
            china_tz = pytz.timezone('Asia/Shanghai')
            timestamp = data.get('timestamp', datetime.now(china_tz))
            task_id = data.get('task_id')
            app_id = data.get('app_id')
            app_name = data.get('app_name')
            domain_name = data.get('domain_name')

            if not all([sn, name, face_crop is not None, task_id is not None, app_id is not None, app_name, domain_name]):
                app_logger.warning(f"检测结果数据不完整，跳过处理: {data}")
                return
                
            # 检查保存间隔 - 使用配置的时间间隔
            current_time = time.time()
            cache_key = f"{sn}"
            
            # 初始化帧计数器（如果不存在）
            if cache_key not in self.frame_counters:
                self.frame_counters[cache_key] = 0
                self.last_save_time[cache_key] = 0
                
            # 增加帧计数
            self.frame_counters[cache_key] += 1
            
            # 检查是否需要保存（基于时间间隔和帧间隔）
            last_save = self.last_save_time.get(cache_key, 0)
            time_interval_ok = current_time - last_save >= self.settings.app.recognition_save_interval_seconds
            frame_interval_ok = self.frame_counters[cache_key] % self.settings.app.recognition_frame_interval == 0
            
            # 只有同时满足时间间隔和帧间隔条件才保存
            if not (time_interval_ok and frame_interval_ok):
                return
                
            # 更新保存时间
            self.last_save_time[cache_key] = current_time

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
                'update_time': timestamp,
                'task_id': task_id,
                'app_id': app_id,
                'app_name': app_name,
                'domain_name': domain_name
            }

            # 添加到批处理缓冲区
            self.batch_buffer.append(detection_record)
            
            # 注意：MQTT消息将在批量保存后发送，使用真实的record_id
            
            # 检查是否达到批处理大小
            if len(self.batch_buffer) >= self.batch_size:
                self._save_batch_buffer()

        except Exception as e:
            app_logger.error(f"处理检测结果时发生异常: {e}", exc_info=True)
            
    def _save_batch_buffer(self, force=False):
        """批量保存缓冲区中的记录"""
        if not self.batch_buffer and not force:
            return
            
        if not self.batch_buffer:
            self.last_batch_save_time = time.time()
            return
            
        batch_size = len(self.batch_buffer)
        if batch_size == 0:
            self.last_batch_save_time = time.time()
            return
            
        app_logger.debug(f"批量保存 {batch_size} 条检测结果")
        
        # 获取数据库会话
        db_session = next(get_db_session())
        try:
            # 使用单条插入来获取记录ID，而不是批量插入
            inserted_records = []
            for record in self.batch_buffer:
                try:
                    # 使用单条插入获取ID
                    from app.core.database.results_ops import insert_single_result
                    record_id = insert_single_result(db_session, record)
                    
                    # 保存记录信息用于MQTT消息
                    inserted_records.append({
                        'record_id': record_id,
                        'task_id': record.get('task_id'),
                        'app_id': record.get('app_id'),
                        'app_name': record.get('app_name'),
                        'domain_name': record.get('domain_name'),
                        'sn': record.get('sn'),
                        'name': record.get('name'),
                        'similarity': record.get('similarity'),
                        'image_url': record.get('image_url')
                    })
                    
                except Exception as e:
                    app_logger.error(f"插入单条记录失败: {e}")
                    # 删除对应的图片文件
                    try:
                        image_url = record.get('image_url', '')
                        if image_url:
                            filename = image_url.split('/')[-1]
                            image_path = self.detected_imgs_path / filename
                            if image_path.exists():
                                os.remove(image_path)
                    except Exception as img_e:
                        app_logger.error(f"删除图片文件失败: {img_e}")
            
            app_logger.info(f"成功批量保存 {len(inserted_records)} 条检测结果")
            
            # 发送MQTT消息（使用真实的record_id）
            if self.mqtt_manager and inserted_records:
                for record_info in inserted_records:
                    try:
                        self.mqtt_manager.queue_detection_message(
                            task_id=record_info['task_id'],
                            app_id=record_info['app_id'],
                            app_name=record_info['app_name'],
                            domain_name=record_info['domain_name'],
                            record_id=record_info['record_id'],  # 使用真实的数据库记录ID
                            sn=record_info['sn'],
                            name=record_info['name'],
                            similarity=record_info['similarity'],
                            image_url=record_info['image_url']
                        )
                    except Exception as mqtt_e:
                        app_logger.error(f"发送MQTT消息失败: {mqtt_e}")
            
            # 清空缓冲区
            self.batch_buffer = []
            
            # 更新最后批量保存时间
            self.last_batch_save_time = time.time()
            
        except Exception as e:
            app_logger.error(f"批量保存检测结果到数据库失败: {e}")
            db_session.rollback()
            
            # 删除已保存的图片文件（如果保存失败）
            for record in self.batch_buffer:
                try:
                    image_url = record.get('image_url', '')
                    if image_url:
                        filename = image_url.split('/')[-1]
                        image_path = self.detected_imgs_path / filename
                        if image_path.exists():
                            os.remove(image_path)
                except Exception as img_e:
                    app_logger.error(f"删除图片文件失败: {img_e}")
        finally:
            db_session.close()