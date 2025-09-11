import paho.mqtt.client as mqtt
import uuid
import json
import time

from typing import Optional, Dict, Any

class MQTTClient:
    def __init__(self, broker_host: str, broker_port: int = 1883, 
                 client_id: Optional[str] = None, keepalive: int = 60):
        """
        初始化MQTT客户端
        
        :param broker_host: MQTT服务器主机地址
        :param broker_port: MQTT服务器端口，默认1883
        :param client_id: 客户端ID，若为None则自动生成UUID
        :param keepalive: 保持连接的时间间隔(秒)，默认60
        """
        # 生成唯一ID，若未提供则使用UUID
        self.client_id = client_id if client_id else str(uuid.uuid4())
        
        # 创建MQTT客户端实例
        self.client = mqtt.Client(client_id=self.client_id)
        
        # 配置MQTT服务器信息
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.keepalive = keepalive
        
        # 连接状态
        self.connected = False
        
        # 设置回调函数
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_log = self._on_log  # 用于调试日志

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Dict, rc: int) -> None:
        """连接回调函数"""
        if rc == 0:
            self.connected = True
            print(f"客户端 {self.client_id} 连接成功")
        else:
            self.connected = False
            error_messages = {
                1: "连接被拒绝 - 不正确的协议版本",
                2: "连接被拒绝 - 无效的客户端标识符",
                3: "连接被拒绝 - 服务器不可用",
                4: "连接被拒绝 - 不正确的用户名或密码",
                5: "连接被拒绝 - 未授权"
            }
            error_msg = error_messages.get(rc, f"连接失败，错误代码: {rc}")
            print(f"客户端 {self.client_id} {error_msg}")

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """断开连接回调函数"""
        self.connected = False
        if rc != 0:
            print(f"客户端 {self.client_id} 意外断开连接，错误代码: {rc}")
        else:
            print(f"客户端 {self.client_id} 已正常断开连接")

    def _on_publish(self, client: mqtt.Client, userdata: Any, mid: int) -> None:
        """发布消息回调函数"""
        print(f"消息发布成功，消息ID: {mid}")

    def _on_log(self, client: mqtt.Client, userdata: Any, level: int, buf: str) -> None:
        """日志回调函数，用于调试"""
        # 可以根据需要开启，用于调试
        # print(f"日志: {buf}")
        pass

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        """消息接收回调函数"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            print(f"收到来自主题 {topic} 的消息: {payload}")
            
            # 将消息传递给消息处理器
            if hasattr(self, 'message_handler'):
                self.message_handler(topic, payload)
                
        except Exception as e:
            print(f"处理MQTT消息时发生错误: {str(e)}")

    def subscribe_topic(self, topic: str, qos: int = 1) -> bool:
        """
        订阅指定主题
        
        :param topic: 要订阅的主题
        :param qos: 服务质量等级，默认为1
        :return: 订阅是否成功
        """
        if not self.connected:
            print("未连接到MQTT服务器，无法订阅主题")
            return False
            
        try:
            result, mid = self.client.subscribe(topic, qos=qos)
            if result == mqtt.MQTT_ERR_SUCCESS:
                print(f"成功订阅主题: {topic}, QoS: {qos}")
                return True
            else:
                print(f"订阅主题 {topic} 失败，错误代码: {result}")
                return False
        except Exception as e:
            print(f"订阅主题 {topic} 时发生错误: {str(e)}")
            return False

    def set_message_handler(self, handler):
        """
        设置消息处理器
        
        :param handler: 消息处理函数，接收(topic, payload)参数
        """
        self.message_handler = handler
        self.client.on_message = self._on_message

    def connect(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        连接到MQTT服务器
        
        :param username: 用户名，可选
        :param password: 密码，可选
        :return: 连接是否成功
        """
        if self.connected:
            print("已经处于连接状态，无需重复连接")
            return True
            
        try:
            # 如果提供了用户名和密码，则设置
            if username and password:
                self.client.username_pw_set(username, password)
                
            # 连接到MQTT服务器
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            
            # 启动网络循环线程
            self.client.loop_start()
            
            # 等待连接成功
            timeout = 5  # 5秒超时
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if not self.connected:
                print(f"连接超时，未能在{timeout}秒内建立连接")
                self.client.loop_stop()
                
            return self.connected
        except Exception as e:
            print(f"连接过程中发生错误: {str(e)}")
            return False

    def disconnect(self) -> bool:
        """断开与MQTT服务器的连接"""
        if not self.connected:
            print("未处于连接状态，无需断开连接")
            return True
            
        try:
            self.client.loop_stop()
            self.client.disconnect()
            print(f"客户端 {self.client_id} 已主动断开连接")
            return True
        except Exception as e:
            print(f"断开连接过程中发生错误: {str(e)}")
            return False

    def push_message(self, topic: str, message_data: Optional[Dict[str, Any]] = None, qos: int = 1) -> bool:
        """
        推送消息到指定主题
        
        :param topic: 消息主题
        :param message_data: 消息数据字典，若为None则不发布
        :param qos: 服务质量等级，默认为1
        :return: 发布是否成功
        """
        # 没有信息则不发布
        if message_data is None:
            print("没有消息数据，不发布任何内容")
            return False
            
        # 检查连接状态
        if not self.connected:
            print("未连接到MQTT服务器，无法发布消息")
            return False
            
        try:
            # 验证消息数据是否包含必要的字段
            required_fields = ["taskId", "appId", "appName", "recordId"]
            for field in required_fields:
                if field not in message_data:
                    print(f"消息数据缺少必要字段: {field}")
                    return False
            
            # 将消息转换为JSON字符串
            message_json = json.dumps(message_data, ensure_ascii=False)
            
            # 发布消息
            result = self.client.publish(topic, message_json, qos=qos)
            
            # 等待消息发布完成
            result.wait_for_publish()
            
            if result.is_published():
                print(f"消息已成功发布到主题: {topic}")
                return True
            else:
                print(f"消息发布到主题 {topic} 失败")
                return False
                
        except json.JSONDecodeError:
            print("消息数据转换为JSON时发生错误")
            return False
        except Exception as e:
            print(f"发布消息过程中发生错误: {str(e)}")
            return False

# 使用示例
if __name__ == "__main__":
    # 创建MQTT客户端实例，会自动生成UUID作为客户端ID
    mqtt_client = MQTTClient(broker_host="172.16.104.108", broker_port=1883)
    
    # 连接到MQTT服务器
    if mqtt_client.connect(username="abtnet", password="Abt@Rabbit#123"):  # 可以传入username和password参数进行认证
        # 准备消息数据
        message_data = {
            "taskId": 10,
            "appId": 30,
            "appName": "人脸识别",
            "domainName": "xx.com",
            "deviceAddress": "172.16.104.111",
            "appType": "FACE_DETECT",
            "recordId": 11111,
            "createTime": "2025-09-04 15:19:06",
            "updateTime": "2025-09-04 15:19:06",
            "sn": "sn001",
            "name": "xxx人员名称",
            "similarity": 0.86,
            "imageUrl": "xxxxxxxxxxx人脸识别url地址"
        }
        
        # 推送消息到指定主题
        mqtt_client.push_message(topic="face/detection", message_data=message_data)
        
        # 测试没有信息时不发布
        mqtt_client.push_message(topic="face/detection")  # 这不会发布任何消息
        
        # 断开连接
        mqtt_client.disconnect()
    else:
        print("无法连接到MQTT服务器，程序退出")
