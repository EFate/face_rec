# app/cfg/mqtt_manager.py
import queue
import threading
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from app.cfg.config import MQTTConfig, ServerConfig
from app.cfg.logging import app_logger
from app.cfg.mqtt_client import MQTTClient
import requests
import asyncio


class MQTTManager:
    """
    MQTT管理器
    负责异步处理MQTT消息上报，不阻塞推理流程
    """
    
    def __init__(self, config: MQTTConfig, server_config: ServerConfig):
        self.config = config
        self.mqtt_client: Optional[MQTTClient] = None
        self._message_queue = queue.Queue(maxsize=config.max_queue_size)
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.connected = False
        self._face_register_topic = "abt/visio/face_register"
        self._api_base_url = f"http://localhost:{server_config.port}/api/face"
        
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
            # 订阅人脸注册主题
            if self.mqtt_client.subscribe_topic(self._face_register_topic):
                self.mqtt_client.set_message_handler(self._handle_face_register_message)
            
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
            
    def _handle_face_register_message(self, topic: str, payload: str):
        """处理人脸注册相关消息"""
        if topic != self._face_register_topic:
            return
            
        try:
            message = json.loads(payload)
            action_type = message.get("actionType")
            items = message.get("items", [])
            
            if not action_type or not items:
                app_logger.warning(f"收到无效的人脸注册消息: {payload}")
                return
                
            app_logger.info(f"处理人脸注册消息, actionType: {action_type}, 共 {len(items)} 条记录")
            
            # 根据actionType调用不同的处理逻辑
            if action_type == "save":
                self._handle_save_action(items)
            elif action_type == "sync":
                self._handle_sync_action(items)
            elif action_type == "delete":
                self._handle_delete_action(items)
            else:
                app_logger.warning(f"未知的actionType: {action_type}")
                
        except json.JSONDecodeError:
            app_logger.error(f"解析人脸注册消息失败, 无效的JSON格式: {payload}")
        except Exception as e:
            app_logger.error(f"处理人脸注册消息时发生错误: {str(e)}", exc_info=True)

    def _handle_save_action(self, items: List[Dict[str, Any]]):
        """处理save操作"""
        app_logger.info(f"开始处理save操作，共 {len(items)} 条记录")
        
        for item in items:
            try:
                sn = item.get("sn")
                name = item.get("name")
                image_urls = item.get("imageUrls", [])
                
                if not sn or not name or not image_urls:
                    app_logger.warning(f"跳过无效的人脸注册记录: {item}")
                    continue
                    
                app_logger.info(f"正在注册人脸: SN={sn}, 姓名={name}, 共 {len(image_urls)} 张图片")
                
                # 下载并注册每张图片
                for image_url in image_urls:
                    try:
                        # 下载图片
                        response = requests.get(image_url, timeout=15)
                        response.raise_for_status()
                        
                        # 检查图片是否有效
                        if not response.content:
                            app_logger.error(f"下载的图片内容为空: {image_url}")
                            continue
                            
                        # 确保图片是RGB格式的JPEG
                        from PIL import Image
                        import io
                        
                        # 转换图片格式
                        try:
                            img = Image.open(io.BytesIO(response.content))
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # 调整图片大小到模型输入尺寸
                            img = img.resize((640, 640))
                            
                            # 转换为JPEG格式
                            output = io.BytesIO()
                            img.save(output, format='JPEG', quality=95)
                            image_bytes = output.getvalue()
                            
                        except Exception as img_e:
                            app_logger.error(f"图片处理失败: {str(img_e)}")
                            # 使用原始图片作为后备方案
                            image_bytes = response.content
                            
                        # 调用router接口注册人脸
                        files = {'image_file': (f'{sn}.jpg', image_bytes)}
                        data = {'name': name, 'sn': sn}
                        api_response = requests.post(
                            f"{self._api_base_url}/faces",
                            files=files,
                            data=data,
                            timeout=30
                        )
                        
                        if api_response.status_code == 201:
                            app_logger.info(f"成功注册人脸: SN={sn}, 图片URL={image_url}")
                        else:
                            error_msg = api_response.json().get('msg', '未知错误')
                            app_logger.error(f"注册人脸失败: SN={sn}, 错误: {error_msg}")
                            # 如果是因为人脸检测失败，尝试调整图片
                            if "未在图像中检测到任何人脸" in error_msg:
                                # 尝试转换图片格式
                                try:
                                    from PIL import Image
                                    import io
                                    img = Image.open(io.BytesIO(response.content))
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    # 调整图片大小
                                    img = img.resize((640, 640))
                                    # 重新保存为jpg
                                    output = io.BytesIO()
                                    img.save(output, format='JPEG', quality=95)
                                    files = {'image_file': (f'{sn}.jpg', output.getvalue())}
                                    api_response = requests.post(
                                        f"{self._api_base_url}/faces",
                                        files=files,
                                        data=data,
                                        timeout=30
                                    )
                                    if api_response.status_code == 201:
                                        app_logger.info(f"调整图片后成功注册人脸: SN={sn}")
                                    else:
                                        app_logger.error(f"调整图片后注册仍然失败: {api_response.json().get('msg', '未知错误')}")
                                except Exception as img_e:
                                    app_logger.error(f"图片处理失败: {str(img_e)}")
                            
                    except requests.exceptions.RequestException as e:
                        app_logger.error(f"下载图片或调用API失败: {image_url}, 错误: {str(e)}")
                        # 重试一次
                        try:
                            response = requests.get(image_url, timeout=15)
                            response.raise_for_status()
                            files = {'image_file': (f'{sn}.jpg', response.content)}
                            data = {'name': name, 'sn': sn}
                            api_response = requests.post(
                                f"{self._api_base_url}/faces",
                                files=files,
                                data=data,
                                timeout=30
                            )
                            if api_response.status_code == 201:
                                app_logger.info(f"重试后成功注册人脸: SN={sn}")
                        except Exception as retry_e:
                            app_logger.error(f"重试仍然失败: {str(retry_e)}")
                    except Exception as e:
                        app_logger.error(f"注册人脸失败: SN={sn}, 错误: {str(e)}", exc_info=True)
                        
            except Exception as e:
                app_logger.error(f"处理人脸注册记录时发生错误: {str(e)}", exc_info=True)
                
    def _handle_sync_action(self, items: List[Dict[str, Any]]):
        """处理sync操作"""
        app_logger.info(f"开始处理sync操作，共 {len(items)} 条记录")
        
        try:
            # 获取当前数据库中的所有sn
            api_response = requests.get(f"{self._api_base_url}/faces")
            if api_response.status_code != 200:
                app_logger.error(f"获取人脸列表失败: {api_response.json().get('msg', '未知错误')}")
                return
                
            all_faces = api_response.json().get("data", {}).get("faces", [])
            existing_sns = {face.get("sn") for face in all_faces}
            received_sns = {item.get("sn") for item in items if item.get("sn")}
            
            # 删除数据库中多余的数据
            sns_to_delete = existing_sns - received_sns
            for sn in sns_to_delete:
                try:
                    delete_response = requests.delete(f"{self._api_base_url}/faces/{sn}")
                    if delete_response.status_code == 200:
                        deleted_count = delete_response.json().get("data", {}).get("deleted_count", 0)
                        app_logger.info(f"同步删除多余数据: SN={sn}, 删除记录数={deleted_count}")
                    else:
                        app_logger.error(f"删除SN={sn}失败: {delete_response.json().get('msg', '未知错误')}")
                except Exception as e:
                    app_logger.error(f"删除SN={sn}时发生错误: {str(e)}")
            
            # 添加缺失的数据
            for item in items:
                sn = item.get("sn")
                name = item.get("name")
                image_urls = item.get("imageUrls", [])
                
                if not sn or not name or not image_urls:
                    continue
                    
                # 如果sn不存在，则添加
                if sn not in existing_sns:
                    self._handle_save_action([item])
                    
        except Exception as e:
            app_logger.error(f"处理sync操作时发生错误: {str(e)}", exc_info=True)
            
    def _handle_delete_action(self, items: List[Dict[str, Any]]):
        """处理delete操作"""
        app_logger.info(f"开始处理delete操作，共 {len(items)} 条记录")
        
        try:
            for item in items:
                sn = item.get("sn")
                if not sn:
                    continue
                    
                try:
                    delete_response = requests.delete(f"{self._api_base_url}/faces/{sn}")
                    if delete_response.status_code == 200:
                        deleted_count = delete_response.json().get("data", {}).get("deleted_count", 0)
                        app_logger.info(f"删除人脸数据: SN={sn}, 删除记录数={deleted_count}")
                    else:
                        app_logger.error(f"删除SN={sn}失败: {delete_response.json().get('msg', '未知错误')}")
                except Exception as e:
                    app_logger.error(f"删除SN={sn}时发生错误: {str(e)}")
                    
        except Exception as e:
            app_logger.error(f"处理delete操作时发生错误: {str(e)}", exc_info=True)
            
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
            
