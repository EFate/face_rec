#!/usr/bin/env python3
"""
MQTT注册问题诊断和修复脚本
"""
import json
import requests
import os
import sys
from pathlib import Path
import base64
import time

# 配置
API_BASE_URL = "http://localhost:12010/api/face"

class MQTTDebugTester:
    def __init__(self):
        self.api_base_url = API_BASE_URL
        
    def check_api_health(self):
        """检查API健康状态"""
        print("🔍 检查API健康状态...")
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API服务运行正常")
                return True
            else:
                print(f"❌ API服务异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API连接失败: {e}")
            return False
    
    def test_direct_registration(self):
        """测试直接API注册"""
        print("\n🔍 测试直接API注册...")
        
        test_image = Path("test/imgs/女.png")
        if not test_image.exists():
            print(f"❌ 测试图片不存在: {test_image}")
            return None
            
        with open(test_image, 'rb') as f:
            image_data = f.read()
        
        files = {'image_file': ('direct_test.jpg', image_data)}
        data = {'name': '直接API测试', 'sn': 'DIRECT_TEST_001'}
        
        try:
            response = requests.post(f"{self.api_base_url}/faces", files=files, data=data, timeout=30)
            print(f"直接API注册状态码: {response.status_code}")
            
            if response.status_code == 201:
                result = response.json()
                print("✅ 直接API注册成功")
                return result.get('data', {}).get('uuid')
            else:
                print(f"❌ 直接API注册失败: {response.text}")
                return None
        except Exception as e:
            print(f"❌ 直接API注册异常: {e}")
            return None
    
    def simulate_mqtt_processing(self):
        """模拟MQTT图片处理流程"""
        print("\n🔍 模拟MQTT图片处理...")
        
        try:
            from PIL import Image
            import io
            
            test_image = Path("test/imgs/女.png")
            if not test_image.exists():
                print(f"❌ 测试图片不存在: {test_image}")
                return None
                
            # 读取原始图片
            with open(test_image, 'rb') as f:
                original_data = f.read()
            
            # 模拟MQTT处理：调整大小，转换格式
            img = Image.open(io.BytesIO(original_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整为640x640
            img = img.resize((640, 640))
            
            # 转换为JPEG
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95)
            processed_data = output.getvalue()
            
            print(f"原始大小: {len(original_data)} bytes")
            print(f"处理后大小: {len(processed_data)} bytes")
            
            # 测试处理后的图片注册
            files = {'image_file': ('mqtt_processed.jpg', processed_data)}
            data = {'name': 'MQTT处理测试', 'sn': 'MQTT_PROC_TEST_001'}
            
            response = requests.post(f"{self.api_base_url}/faces", files=files, data=data, timeout=30)
            print(f"处理后图片注册状态码: {response.status_code}")
            
            if response.status_code == 201:
                result = response.json()
                print("✅ MQTT处理后注册成功")
                return result.get('data', {}).get('uuid')
            else:
                print(f"❌ MQTT处理后注册失败: {response.text}")
                return None
                
        except ImportError:
            print("❌ PIL库未安装，跳过图片处理测试")
            return None
        except Exception as e:
            print(f"❌ 图片处理异常: {e}")
            return None
    
    def test_recognition(self, sn):
        """测试人脸识别"""
        print(f"\n🔍 测试人脸识别 (SN: {sn})...")
        
        test_image = Path("test/imgs/女.png")
        if not test_image.exists():
            print(f"❌ 测试图片不存在: {test_image}")
            return False
            
        with open(test_image, 'rb') as f:
            image_data = f.read()
        
        files = {'image_file': ('recognition_test.jpg', image_data)}
        
        try:
            response = requests.post(f"{self.api_base_url}/recognize", files=files, timeout=30)
            print(f"识别状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                faces = result.get('data', {}).get('faces', [])
                
                if faces:
                    print("✅ 识别成功，找到匹配人脸:")
                    for face in faces:
                        print(f"  - SN: {face.get('sn')}, 姓名: {face.get('name')}, 相似度: {face.get('similarity')}")
                    return True
                else:
                    print("❌ 识别成功但未找到匹配人脸")
                    return False
            else:
                print(f"❌ 识别失败: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 识别异常: {e}")
            return False
    
    def check_registered_faces(self):
        """检查已注册的人脸"""
        print("\n🔍 检查已注册人脸...")
        try:
            response = requests.get(f"{self.api_base_url}/faces", timeout=10)
            if response.status_code == 200:
                data = response.json()
                faces = data.get('data', {}).get('faces', [])
                
                print(f"当前注册人脸数量: {len(faces)}")
                test_faces = [f for f in faces if 'TEST' in str(f.get('sn', ''))]
                
                if test_faces:
                    print("测试人脸列表:")
                    for face in test_faces:
                        print(f"  - SN: {face.get('sn')}, 姓名: {face.get('name')}, UUID: {face.get('uuid')}")
                
                return faces
            else:
                print(f"❌ 获取人脸列表失败: {response.text}")
                return []
        except Exception as e:
            print(f"❌ 获取人脸列表异常: {e}")
            return []
    
    def clean_test_data(self):
        """清理测试数据"""
        print("\n🧹 清理测试数据...")
        faces = self.check_registered_faces()
        test_faces = [f for f in faces if 'TEST' in str(f.get('sn', ''))]
        
        for face in test_faces:
            uuid = face.get('uuid')
            sn = face.get('sn')
            try:
                response = requests.delete(f"{self.api_base_url}/faces/{uuid}", timeout=10)
                print(f"删除测试人脸 {sn}: {response.status_code}")
            except Exception as e:
                print(f"删除测试人脸 {sn} 失败: {e}")
    
    def test_mqtt_endpoint_config(self):
        """测试MQTT端点配置"""
        print("\n🔍 测试MQTT端点配置...")
        
        # 检查环境变量
        env_vars = ['HOST', 'PORT', 'API_BASE_URL']
        for var in env_vars:
            value = os.environ.get(var, '未设置')
            print(f"环境变量 {var}: {value}")
        
        # 测试不同的主机配置
        test_urls = [
            "http://localhost:12010/api/face",
            "http://127.0.0.1:12010/api/face",
            "http://0.0.0.0:12010/api/face"
        ]
        
        for url in test_urls:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"✅ {url} 可达")
                else:
                    print(f"❌ {url} 不可达 (状态码: {response.status_code})")
            except Exception as e:
                print(f"❌ {url} 连接失败: {e}")

def main():
    """主测试流程"""
    print("🚀 MQTT注册问题诊断工具")
    print("=" * 50)
    
    tester = MQTTDebugTester()
    
    # 步骤1: 检查API健康状态
    if not tester.check_api_health():
        print("❌ API服务未运行，请先启动应用")
        return
    
    # 步骤2: 清理之前的测试数据
    tester.clean_test_data()
    
    # 步骤3: 测试直接API注册
    direct_uuid = tester.test_direct_registration()
    
    # 步骤4: 测试MQTT处理后的注册
    mqtt_uuid = tester.simulate_mqtt_processing()
    
    # 步骤5: 检查注册结果
    faces = tester.check_registered_faces()
    
    # 步骤6: 测试识别
    if direct_uuid:
        print(f"\n测试直接注册的人脸识别...")
        tester.test_recognition('DIRECT_TEST_001')
    
    if mqtt_uuid:
        print(f"\n测试MQTT处理后注册的人脸识别...")
        tester.test_recognition('MQTT_PROC_TEST_001')
    
    # 步骤7: 测试端点配置
    tester.test_mqtt_endpoint_config()
    
    print("\n" + "=" * 50)
    print("📋 诊断总结:")
    print(f"直接API注册UUID: {direct_uuid}")
    print(f"MQTT处理后注册UUID: {mqtt_uuid}")
    
    if direct_uuid and mqtt_uuid:
        print("✅ 两种注册方式都成功，问题可能在MQTT消息格式或网络配置")
    elif direct_uuid and not mqtt_uuid:
        print("❌ 直接API成功但MQTT处理失败，问题在图片处理环节")
    elif not direct_uuid and not mqtt_uuid:
        print("❌ 两种注册都失败，问题在API服务或图片本身")
    else:
        print("⚠️ 需要进一步检查日志和配置")

if __name__ == "__main__":
    main()