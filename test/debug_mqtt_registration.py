#!/usr/bin/env python3
"""
MQTT注册问题调试工具
"""
import sys
import os
sys.path.append('/home/abt/lx/face_rec')

import json
import requests
from PIL import Image
import io
import base64
import numpy as np

def test_api_endpoint():
    """测试API端点"""
    print("🔍 测试API端点")
    
    # 获取当前运行的服务端口
    try:
        with open('/tmp/face_service_port.txt', 'r') as f:
            port = int(f.read().strip())
    except:
        port = 12010
    
    api_base_url = f"http://localhost:{port}/api/face"
    print(f"API端点: {api_base_url}")
    
    # 测试连通性
    try:
        response = requests.get(f"{api_base_url}/status", timeout=5)
        if response.status_code == 200:
            print("✅ API服务正常")
        else:
            print(f"❌ API服务异常: {response.status_code}")
    except Exception as e:
        print(f"❌ API连接失败: {e}")
    
    return api_base_url

def test_direct_registration():
    """测试直接注册"""
    print("\n📋 测试直接注册")
    
    api_base_url = test_api_endpoint()
    
    # 使用本地测试图片
    test_image_path = "/home/abt/lx/face_rec/data/test_images/test_face.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"创建测试图片: {test_image_path}")
        os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
        
        # 创建一个简单的测试图片
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (640, 640), color='white')
        draw = ImageDraw.Draw(img)
        draw.ellipse([(200, 200), (440, 440)], fill='black', outline='black')
        img.save(test_image_path)
    
    # 读取测试图片
    with open(test_image_path, 'rb') as f:
        img_data = f.read()
    
    # 转换为base64
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    
    # 注册数据
    register_data = {
        "name": "test_mqtt_user",
        "image": img_base64
    }
    
    # 直接注册
    try:
        response = requests.post(
            f"{api_base_url}/register",
            json=register_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 直接注册成功: {result}")
            return True
        else:
            print(f"❌ 直接注册失败: {response.status_code}, {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 直接注册异常: {e}")
        return False

def test_mqtt_simulation():
    """模拟MQTT注册"""
    print("\n🔄 模拟MQTT注册")
    
    # 模拟MQTT消息
    mqtt_message = {
        "actionType": "save",
        "data": [
            {
                "id": "mqtt_test_user_001",
                "name": "MQTT测试用户",
                "image_url": "https://via.placeholder.com/640x640/ffffff/000000?text=MQTT+Face"
            }
        ]
    }
    
    print("模拟MQTT消息:")
    print(json.dumps(mqtt_message, indent=2, ensure_ascii=False))
    
    # 模拟MQTT处理流程
    api_base_url = test_api_endpoint()
    
    for item in mqtt_message["data"]:
        print(f"\n处理用户: {item['name']}")
        
        # 1. 下载图片
        image_url = item["image_url"]
        print(f"下载图片: {image_url}")
        
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            
            print(f"✅ 图片下载成功: {len(response.content)} bytes")
            
            # 2. 验证图片
            try:
                img = Image.open(io.BytesIO(response.content))
                print(f"图片格式: {img.format}, 模式: {img.mode}, 尺寸: {img.size}")
                
                # 转换为base64
                img_base64 = base64.b64encode(response.content).decode('utf-8')
                
                # 3. 注册人脸
                register_data = {
                    "name": item["name"],
                    "image": img_base64
                }
                
                response = requests.post(
                    f"{api_base_url}/register",
                    json=register_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ MQTT模拟注册成功: {result}")
                else:
                    print(f"❌ MQTT模拟注册失败: {response.status_code}, {response.text}")
                    
            except Exception as e:
                print(f"❌ 图片处理失败: {e}")
                
        except Exception as e:
            print(f"❌ 图片下载失败: {e}")

def check_registered_faces():
    """检查已注册的人脸"""
    print("\n👥 检查已注册人脸")
    
    api_base_url = test_api_endpoint()
    
    try:
        response = requests.get(f"{api_base_url}/list", timeout=5)
        if response.status_code == 200:
            faces = response.json()
            print(f"当前注册人脸数量: {len(faces)}")
            for face in faces:
                print(f"  - {face.get('name', '未知')} (ID: {face.get('id', '未知')})")
        else:
            print(f"获取人脸列表失败: {response.status_code}")
    except Exception as e:
        print(f"获取人脸列表异常: {e}")

def main():
    """主函数"""
    print("🚀 MQTT注册问题调试工具")
    print("="*50)
    
    # 检查当前注册状态
    check_registered_faces()
    
    # 测试直接注册
    test_direct_registration()
    
    # 模拟MQTT注册
    test_mqtt_simulation()
    
    # 再次检查注册状态
    check_registered_faces()
    
    print("\n" + "="*50)
    print("📋 调试建议")
    print("="*50)
    print("1. 检查MQTT消息中的图片URL是否可访问")
    print("2. 验证网络连接和防火墙设置")
    print("3. 测试图片URL是否返回有效图片格式")
    print("4. 检查图片下载后的处理逻辑")
    print("5. 确认API端点配置正确")

if __name__ == "__main__":
    main()