#!/usr/bin/env python3
"""
调试MQTT注册流程的脚本
测试MQTT注册是否真的执行了注册流程
"""
import json
import requests
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, '/home/abt/lx/face_rec')

# 测试配置
API_BASE_URL = "http://localhost:12010/api/face"
MQTT_API_URL = "http://localhost:12010/api/face"

def test_direct_api_registration():
    """测试直接API注册"""
    print("=== 测试直接API注册 ===")
    
    # 使用测试图片
    test_image_path = "/home/abt/lx/face_rec/app/static/self_test_face.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图片不存在: {test_image_path}")
        return False
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image_file': ('test_face.jpg', f, 'image/jpeg')}
            data = {'name': '测试用户API', 'sn': f'TEST_API_{datetime.now().strftime("%H%M%S")}'}
            
            response = requests.post(f"{API_BASE_URL}/faces", files=files, data=data, timeout=30)
            
        if response.status_code == 201:
            result = response.json()
            print(f"✅ 直接API注册成功: {result}")
            return True
        else:
            print(f"❌ 直接API注册失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 直接API注册异常: {e}")
        return False

def test_mqtt_registration_flow():
    """测试MQTT注册流程"""
    print("\n=== 测试MQTT注册流程 ===")
    
    # 模拟MQTT消息
    mqtt_message = {
        "actionType": "save",
        "items": [
            {
                "sn": f"TEST_MQTT_{datetime.now().strftime('%H%M%S')}",
                "name": "测试用户MQTT",
                "imageUrls": [
                        "http://localhost:12010/static/self_test_face.jpg"
                    ]
            }
        ]
    }
    
    print(f"模拟MQTT消息: {json.dumps(mqtt_message, indent=2, ensure_ascii=False)}")
    
    # 手动执行MQTT注册流程
    try:
        for item in mqtt_message["items"]:
            sn = item.get("sn")
            name = item.get("name")
            image_urls = item.get("imageUrls", [])
            
            print(f"处理注册: SN={sn}, 姓名={name}")
            
            for image_url in image_urls:
                print(f"下载图片: {image_url}")
                
                # 下载图片
                response = requests.get(image_url, timeout=15)
                if response.status_code == 200:
                    print(f"✅ 图片下载成功: {len(response.content)} bytes")
                    
                    # 调用API注册
                    files = {'image_file': (f'{sn}.jpg', response.content)}
                    data = {'name': name, 'sn': sn}
                    
                    api_response = requests.post(
                        f"{MQTT_API_URL}/faces",
                        files=files,
                        data=data,
                        timeout=30
                    )
                    
                    if api_response.status_code == 201:
                        print(f"✅ MQTT注册流程成功: {api_response.json()}")
                        return True
                    else:
                        print(f"❌ MQTT注册API调用失败: {api_response.status_code} - {api_response.text}")
                        return False
                else:
                    print(f"❌ 图片下载失败: {response.status_code}")
                    return False
                    
    except Exception as e:
        print(f"❌ MQTT注册流程异常: {e}")
        return False

def check_registered_faces():
    """检查已注册的人脸"""
    print("\n=== 检查已注册的人脸 ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/faces", timeout=10)
        if response.status_code == 200:
            faces = response.json().get("data", {}).get("faces", [])
            print(f"当前共注册 {len(faces)} 张人脸:")
            
            for face in faces:
                print(f"  - SN: {face.get('sn')}, 姓名: {face.get('name')}, ID: {face.get('id')}")
            
            return faces
        else:
            print(f"❌ 获取人脸列表失败: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ 检查注册人脸异常: {e}")
        return []

def test_api_endpoints():
    """测试API端点"""
    print("\n=== 测试API端点 ===")
    
    endpoints = [
        "/health",
        "/faces"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
            print(f"GET {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"GET {endpoint}: 失败 - {e}")

if __name__ == "__main__":
    print("开始调试MQTT注册流程...")
    
    # 测试API端点
    test_api_endpoints()
    
    # 检查当前注册的人脸
    faces = check_registered_faces()
    
    # 测试直接API注册
    direct_success = test_direct_api_registration()
    
    # 测试MQTT注册流程
    mqtt_success = test_mqtt_registration_flow()
    
    # 再次检查注册的人脸
    faces_after = check_registered_faces()
    
    print(f"\n=== 测试结果总结 ===")
    print(f"直接API注册: {'成功' if direct_success else '失败'}")
    print(f"MQTT注册流程: {'成功' if mqtt_success else '失败'}")
    print(f"注册前人脸数: {len(faces)}")
    print(f"注册后人脸数: {len(faces_after)}")