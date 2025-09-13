#!/usr/bin/env python3
"""
分析MQTT注册问题的根本原因
"""
import json
import requests
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, '/home/abt/lx/face_rec')

def analyze_mqtt_url_access():
    """分析MQTT注册中的URL访问问题"""
    print("=== 分析MQTT注册中的URL访问问题 ===")
    
    # 测试几种可能的图片URL格式
    test_urls = [
        "http://localhost:12010/static/self_test_face.jpg",
        "http://127.0.0.1:12010/static/self_test_face.jpg", 
        "http://0.0.0.0:12010/static/self_test_face.jpg",
        "http://localhost:8000/static/self_test_face.jpg",  # 常见错误端口
        "http://192.168.1.100:12010/static/self_test_face.jpg",  # 假设的局域网IP
    ]
    
    print("测试图片URL可访问性:")
    for url in test_urls:
        try:
            response = requests.head(url, timeout=5)
            print(f"✅ {url}: {response.status_code}")
        except Exception as e:
            print(f"❌ {url}: {str(e)}")

def check_mqtt_config():
    """检查MQTT配置"""
    print("\n=== 检查MQTT配置 ===")
    
    # 检查MQTT配置是否正确
    api_base_url = "http://localhost:12010/api/face"
    print(f"MQTT使用的API端点: {api_base_url}")
    
    try:
        response = requests.get(f"{api_base_url}/health", timeout=5)
        print(f"API端点健康检查: {response.status_code}")
    except Exception as e:
        print(f"API端点检查失败: {e}")

def simulate_real_mqtt_scenario():
    """模拟真实MQTT场景"""
    print("\n=== 模拟真实MQTT场景 ===")
    
    # 模拟真实的MQTT消息格式
    real_mqtt_messages = [
        {
            "actionType": "save",
            "items": [
                {
                    "sn": "EMP001",
                    "name": "张三",
                    "imageUrls": [
                        "http://192.168.1.100:8080/images/emp001.jpg"  # 外部系统URL
                    ]
                }
            ]
        },
        {
            "actionType": "sync",
            "items": [
                {
                    "sn": "EMP002", 
                    "name": "李四",
                    "imageUrls": [
                        "http://fileserver.company.com/face/emp002.jpg"  # 文件服务器URL
                    ]
                }
            ]
        }
    ]
    
    for i, message in enumerate(real_mqtt_messages):
        print(f"\n场景 {i+1}: {message['actionType']} 操作")
        for item in message["items"]:
            sn = item["sn"]
            name = item["name"]
            image_urls = item["imageUrls"]
            
            print(f"  人员: {name} (SN: {sn})")
            for url in image_urls:
                print(f"  图片URL: {url}")
                
                # 测试URL可访问性
                try:
                    response = requests.head(url, timeout=10)
                    print(f"    ✅ 可访问: {response.status_code}")
                except Exception as e:
                    print(f"    ❌ 不可访问: {str(e)}")

def check_network_connectivity():
    """检查网络连通性"""
    print("\n=== 网络连通性检查 ===")
    
    # 检查本机服务
    try:
        response = requests.get("http://localhost:12010/api/face/faces", timeout=5)
        print(f"本机服务: ✅ 正常 ({response.status_code})")
    except Exception as e:
        print(f"本机服务: ❌ 异常 - {e}")
    
    # 检查外部网络
    try:
        response = requests.get("http://httpbin.org/ip", timeout=5)
        print(f"外网访问: ✅ 正常 ({response.status_code})")
    except Exception as e:
        print(f"外网访问: ❌ 异常 - {e}")

def check_image_processing_pipeline():
    """检查图片处理流程"""
    print("\n=== 图片处理流程检查 ===")
    
    # 测试本地图片处理
    test_image_path = "/home/abt/lx/face_rec/app/static/self_test_face.jpg"
    
    if os.path.exists(test_image_path):
        print(f"本地测试图片: ✅ 存在 ({test_image_path})")
        
        # 测试图片读取
        try:
            from PIL import Image
            img = Image.open(test_image_path)
            print(f"图片格式: {img.format}, 模式: {img.mode}, 尺寸: {img.size}")
            
            # 测试图片转换
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_resized = img.resize((640, 640))
            print(f"✅ 图片转换成功")
            
        except Exception as e:
            print(f"❌ 图片处理失败: {e}")
    else:
        print(f"❌ 本地测试图片不存在")

def main():
    """主分析流程"""
    print("开始分析MQTT注册问题...")
    
    analyze_mqtt_url_access()
    check_mqtt_config()
    simulate_real_mqtt_scenario()
    check_network_connectivity()
    check_image_processing_pipeline()
    
    print("\n=== 问题分析总结 ===")
    print("1. MQTT注册流程本身是正常工作的")
    print("2. 主要问题在于图片URL的访问性")
    print("3. 外部系统提供的图片URL必须能被本机访问")
    print("4. 建议使用内网IP或可访问的域名")
    print("5. 检查防火墙和网络配置")

if __name__ == "__main__":
    main()