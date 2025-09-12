#!/usr/bin/env python3
"""
调试MQTT注册问题的脚本
"""
import sys
import os
sys.path.append('/home/abt/lx/face_rec')

from app.cfg.config import get_app_settings
from app.service.face_dao import LanceDBFaceDataDAO
import requests
import json

def check_api_endpoints():
    """检查API端点是否正确"""
    settings = get_app_settings()
    
    # 检查MQTT使用的API端点
    mqtt_api_url = f"http://localhost:{settings.server.port}/api/face"
    
    print("=== API端点检查 ===")
    print(f"MQTT使用的API端点: {mqtt_api_url}")
    print(f"实际服务端口: {settings.server.port}")
    
    # 测试API连通性
    try:
        response = requests.get(f"{mqtt_api_url}/faces", timeout=5)
        print(f"API连通性测试: {'✅ 成功' if response.status_code == 200 else '❌ 失败'}")
        if response.status_code == 200:
            faces = response.json().get("data", {}).get("faces", [])
            print(f"当前注册人脸数量: {len(faces)}")
            for face in faces:
                print(f"  - {face.get('name')} (SN: {face.get('sn')})")
    except Exception as e:
        print(f"❌ API连接失败: {e}")

def test_mqtt_registration():
    """测试MQTT注册流程"""
    settings = get_app_settings()
    
    # 模拟MQTT注册消息
    mqtt_message = {
        "actionType": "save",
        "items": [{
            "sn": "TEST_MQTT_001",
            "name": "MQTT测试用户",
            "imageUrls": ["https://via.placeholder.com/640x640.jpg"]
        }]
    }
    
    print("\n=== MQTT注册测试 ===")
    print("模拟MQTT注册消息:")
    print(json.dumps(mqtt_message, indent=2, ensure_ascii=False))
    
    # 检查API端点
    api_base_url = f"http://localhost:{settings.server.port}/api/face"
    print(f"将调用API端点: {api_base_url}/faces")

def check_lancedb_data():
    """检查LanceDB中的数据"""
    settings = get_app_settings()
    
    print("\n=== LanceDB数据检查 ===")
    try:
        face_dao = LanceDBFaceDataDAO(
            settings.inference.lancedb_uri,
            settings.inference.lancedb_table_name
        )
        
        # 获取所有数据
        all_data = face_dao.get_all()
        print(f"LanceDB中总记录数: {len(all_data)}")
        
        for data in all_data:
            print(f"  - ID: {data.id}, SN: {data.sn}, Name: {data.name}")
            
    except Exception as e:
        print(f"❌ LanceDB查询失败: {e}")

def main():
    """主函数"""
    print("🔍 开始调试MQTT注册问题...")
    
    check_api_endpoints()
    test_mqtt_registration()
    check_lancedb_data()
    
    print("\n=== 问题分析建议 ===")
    print("1. 检查MQTT使用的API端点是否与实际服务端口匹配")
    print("2. 验证MQTT注册时使用的图片URL是否可访问")
    print("3. 检查MQTT注册失败时的错误日志")
    print("4. 确认网络连接和防火墙设置")

if __name__ == "__main__":
    main()