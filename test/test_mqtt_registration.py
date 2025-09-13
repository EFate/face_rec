#!/usr/bin/env python3
"""
MQTT注册流程测试脚本
"""
import json
import requests
import asyncio
import aiohttp
from pathlib import Path
import base64

# API配置
API_BASE_URL = "http://localhost:12010/api/face"

def test_api_registration():
    """测试直接API注册"""
    print("=== 测试直接API注册 ===")
    
    # 使用本地测试图片
    test_image_path = Path(__file__).parent / "imgs" / "女.png"
    if not test_image_path.exists():
        print(f"❌ 测试图片不存在: {test_image_path}")
        return None
    
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
    
    files = {'image_file': ('test_face.jpg', image_data)}
    data = {'name': 'API测试用户', 'sn': 'API_TEST_001'}
    
    try:
        response = requests.post(f"{API_BASE_URL}/faces", files=files, data=data)
        print(f"API注册响应状态码: {response.status_code}")
        
        if response.status_code == 201:
            result = response.json()
            print(f"✅ 直接API注册成功: {result}")
            return result.get('data', {}).get('uuid')
        else:
            print(f"❌ 直接API注册失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ API注册异常: {e}")
        return None

def test_recognition(image_path):
    """测试人脸识别"""
    print(f"=== 测试人脸识别: {image_path.name} ===")
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    files = {'image_file': ('test_image.jpg', image_data)}
    
    try:
        response = requests.post(f"{API_BASE_URL}/recognize", files=files)
        print(f"识别响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 识别结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return result
        else:
            print(f"❌ 识别失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 识别异常: {e}")
        return None

def simulate_mqtt_registration():
    """模拟MQTT注册流程"""
    print("=== 模拟MQTT注册流程 ===")
    
    # 使用本地图片进行模拟
    test_image_path = Path(__file__).parent / "imgs" / "女.png"
    if not test_image_path.exists():
        print(f"❌ 测试图片不存在: {test_image_path}")
        return
    
    # 将图片转换为base64
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # 构建模拟的MQTT消息数据
    mqtt_data = {
        "actionType": "save",
        "items": [
            {
                "sn": "MQTT_TEST_001",
                "name": "MQTT测试用户",
                "image": base64_image  # 使用base64图片数据
            }
        ]
    }
    
    print(f"模拟MQTT消息数据: {json.dumps(mqtt_data, ensure_ascii=False, indent=2)[:200]}...")
    
    # 直接调用API注册（模拟MQTT的处理方式）
    try:
        # 解码base64图片
        image_bytes = base64.b64decode(base64_image)
        
        files = {'image_file': ('mqtt_face.jpg', image_bytes)}
        data = {'name': 'MQTT测试用户', 'sn': 'MQTT_TEST_001'}
        
        response = requests.post(f"{API_BASE_URL}/faces", files=files, data=data)
        print(f"模拟MQTT注册响应状态码: {response.status_code}")
        
        if response.status_code == 201:
            result = response.json()
            print(f"✅ 模拟MQTT注册成功: {result}")
            return result.get('data', {}).get('uuid')
        else:
            print(f"❌ 模拟MQTT注册失败: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 模拟MQTT注册异常: {e}")
        return None

def check_registered_faces():
    """检查已注册的人脸"""
    print("=== 检查已注册人脸 ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/faces")
        if response.status_code == 200:
            data = response.json()
            faces = data.get('data', {}).get('faces', [])
            print(f"当前注册的人脸数量: {len(faces)}")
            
            if faces:
                print("已注册的人脸:")
                for face in faces[-5:]:  # 显示最后5个
                    print(f"  - {face.get('sn', 'N/A')}: {face.get('name', 'N/A')} (UUID: {face.get('uuid', 'N/A')})")
            return faces
        else:
            print(f"❌ 获取人脸列表失败: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ 获取人脸列表异常: {e}")
        return []

def clear_test_faces():
    """清除测试人脸"""
    print("=== 清除测试人脸 ===")
    
    faces = check_registered_faces()
    test_faces = [f for f in faces if 'TEST' in f.get('sn', '')]
    
    for face in test_faces:
        uuid = face.get('uuid')
        sn = face.get('sn')
        try:
            response = requests.delete(f"{API_BASE_URL}/faces/{uuid}")
            print(f"删除测试人脸 {sn}: {response.status_code}")
        except Exception as e:
            print(f"删除测试人脸 {sn} 失败: {e}")

async def main():
    """主测试函数"""
    print("开始MQTT注册问题调试...")
    
    # 清除之前的测试数据
    clear_test_faces()
    
    # 步骤1: 测试直接API注册
    print("\n" + "="*50)
    api_uuid = test_api_registration()
    
    # 步骤2: 检查注册的人脸
    print("\n" + "="*50)
    faces = check_registered_faces()
    
    # 步骤3: 测试人脸识别
    print("\n" + "="*50)
    test_image_path = Path(__file__).parent / "imgs" / "女.png"
    if test_image_path.exists():
        test_recognition(test_image_path)
    
    # 步骤4: 模拟MQTT注册
    print("\n" + "="*50)
    mqtt_uuid = simulate_mqtt_registration()
    
    # 步骤5: 再次检查注册的人脸
    print("\n" + "="*50)
    faces_after_mqtt = check_registered_faces()
    
    # 步骤6: 再次测试人脸识别
    print("\n" + "="*50)
    if test_image_path.exists():
        test_recognition(test_image_path)
    
    print("\n" + "="*50)
    print("调试总结:")
    print(f"API注册UUID: {api_uuid}")
    print(f"MQTT注册UUID: {mqtt_uuid}")
    print("请验证两种注册方式注册的人脸是否都能被识别")

if __name__ == "__main__":
    asyncio.run(main())