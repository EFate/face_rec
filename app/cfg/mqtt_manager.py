# app/cfg/mqtt_manager.py
import queue
import threading
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from app.cfg.config import MQTTConfig
from app.cfg.logging import app_logger
from app.cfg.mqtt_client import MQTTClient


class MQTTManager:
    """
    MQTT管理器
    负责异步处理MQTT消息上报，不阻塞推理流程
    """
    
    def __init__(self, config: MQTTConfig):
        self.config = config
        self.mqtt_client: Optional[MQTTClient] = None
        self._message_queue = queue.Queue(maxsize=config.max_queue_size)
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.connected = False
        
    def start(self):
        """启动MQTT管理器"""
        if not self.config.enabled:
            app_logger.info("MQTT功能已禁用，跳过启动")
            return
            
        if self.worker_thread and self.worker_thread.is_alive():
            app_logger.warning("MQTT管理器已在运行")
            return
            
        try:
            # 初始化MQTT客户端
            self.mqtt_client = MQTTClient(
                broker_host=self.config.broker_host,
                broker_port=self.config.broker_port,
                keepalive=self.config.keepalive
            )
            
            # 连接到MQTT服务器
            self.connected = self.mqtt_client.connect(
                username=self.config.username,
                password=self.config.password
            )
            
            if not self.connected:
                app_logger.error("MQTT连接失败，将禁用MQTT功能")
                return
                
            # 启动工作线程
            self.stop_event.clear()
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                name="MQTTWorker",
                daemon=True
            )
            self.worker_thread.start()
            app_logger.info("✅ MQTT管理器已启动")
            
        except Exception as e:
            app_logger.error(f"启动MQTT管理器失败: {e}", exc_info=True)
            self.connected = False
            
    def stop(self):
        """停止MQTT管理器"""
        if not self.worker_thread or not self.worker_thread.is_alive():
            return
            
        self.stop_event.set()
        
        # 放入停止信号
        try:
            self._message_queue.put_nowait(None)
        except queue.Full:
            pass
            
        # 等待工作线程结束
        self.worker_thread.join(timeout=5.0)
        
        # 断开MQTT连接
        if self.mqtt_client:
            self.mqtt_client.disconnect()
            
        app_logger.info("✅ MQTT管理器已停止")
        
    def _worker_loop(self):
        """工作线程主循环"""
        app_logger.info("MQTT工作线程已启动")
        
        while not self.stop_event.is_set():
            try:
                # 从队列获取消息
                message_data = self._message_queue.get(timeout=1.0)
                
                # 检查停止信号
                if message_data is None:
                    break
                    
                # 发布消息
                self._publish_message(message_data)
                
                # 控制发布频率
                time.sleep(self.config.publish_interval)
                
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"MQTT工作线程处理消息时发生错误: {e}", exc_info=True)
                
        app_logger.info("MQTT工作线程已停止")
        
    def _publish_message(self, message_data: Dict[str, Any]):
        """发布MQTT消息"""
        if not self.connected or not self.mqtt_client:
            return
            
        try:
            # 使用新的主题格式：abt/visio/face/ip
            topic = self.config.get_detection_topic()
            success = self.mqtt_client.push_message(topic, message_data)
            
            if success:
                app_logger.debug(f"MQTT消息发布成功到主题 {topic}: {message_data.get('recordId', 'unknown')}")
            else:
                app_logger.warning(f"MQTT消息发布失败到主题 {topic}: {message_data.get('recordId', 'unknown')}")
                
        except Exception as e:
            app_logger.error(f"发布MQTT消息到主题 {topic} 时发生错误: {e}", exc_info=True)
            
    def queue_detection_message(self, task_id: int, app_id: int, app_name: str, 
                               domain_name: str, record_id: int, sn: str, 
                               name: str, similarity: float, image_url: str):
        """将检测结果消息加入队列"""
        if not self.config.enabled or not self.connected:
            return
            
        try:
            message_data = {
                "taskId": task_id,
                "appId": app_id,
                "appName": app_name,
                "domainName": domain_name,
                "deviceAddress": self.config.device_address,
                "appType": self.config.app_type,
                "recordId": record_id,
                "createTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sn": sn,
                "name": name,
                "similarity": similarity,
                "imageUrl": image_url
            }
            
            # 非阻塞放入队列
            self._message_queue.put_nowait(message_data)
            
        except queue.Full:
            app_logger.warning("MQTT消息队列已满，丢弃消息")
        except Exception as e:
            app_logger.error(f"将检测结果加入MQTT队列时发生错误: {e}", exc_info=True)
            
