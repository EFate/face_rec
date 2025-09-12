#!/usr/bin/env python3
"""
测试MQTT图片下载和处理
"""
import sys
import os
sys.path.append('/home/abt/lx/face_rec')

import requests
from PIL import Image
import io
import logging
import base64

def download_and_test_image(image_url):
    """下载并测试图片"""
    print(f"🌐 测试URL: {image_url}")
    
    try:
        # 设置请求头
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 下载图片
        response = requests.get(image_url, timeout=15, headers=headers)
        response.raise_for_status()
        
        print(f"✅ 下载成功: {len(response.content)} bytes")
        print(f"内容类型: {response.headers.get('content-type', 'unknown')}")
        
        # 检查内容
        if not response.content:
            print("❌ 内容为空")
            return False
            
        # 验证图片格式
        try:
            img = Image.open(io.BytesIO(response.content))
            print(f"格式: {img.format}, 模式: {img.mode}, 尺寸: {img.size}")
            
            # 转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("🔄 转换为RGB模式")
            
            # 调整大小
            resized = img.resize((640, 640))
            print(f"调整后尺寸: {resized.size}")
            
            # 检查是否为全黑
            img_array = resized.convert('L')
            pixels = list(img_array.getdata())
            avg_brightness = sum(pixels) / len(pixels)
            print(f"平均亮度: {avg_brightness:.2f} (0-255)")
            
            if avg_brightness < 10:
                print("⚠️ 图片可能过暗或全黑")
            else:
                print("✅ 图片亮度正常")
            
            # 保存检查
            resized.save("/tmp/mqtt_test_result.jpg", "JPEG", quality=95)
            print("📸 结果已保存到: /tmp/mqtt_test_result.jpg")
            
            return True
            
        except Exception as e:
            print(f"❌ 图片处理失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def test_common_issues():
    """测试常见问题"""
    print("🔍 测试常见图片URL问题")
    
    test_urls = [
        # 占位符图片
        "https://via.placeholder.com/640x640.jpg",
        "https://via.placeholder.com/640x640/ffffff/000000?text=Test+Face",
        
        # 真实图片
        "https://picsum.photos/640/640",
        
        # 测试图片
        "https://httpbin.org/image/jpeg",
        
        # 可能失败的URL
        "https://example.com/nonexistent.jpg",
        "http://via.placeholder.com/640x640.jpg",  # 非HTTPS
    ]
    
    for url in test_urls:
        print("\n" + "="*50)
        download_and_test_image(url)

def simulate_mqtt_save():
    """模拟MQTT save处理"""
    print("\n🔧 模拟MQTT save处理流程")
    
    # 模拟一个测试图片URL
    test_url = "https://via.placeholder.com/640x640/ffffff/000000?text=MQTT+Test"
    
    print("1. 模拟从MQTT消息获取URL...")
    print(f"URL: {test_url}")
    
    print("\n2. 下载和处理图片...")
    success = download_and_test_image(test_url)
    
    if success:
        print("\n✅ 图片处理成功，可以正常注册")
    else:
        print("\n❌ 图片处理失败，需要检查URL或网络")

def main():
    """主函数"""
    print("🚀 MQTT图片问题诊断工具")
    print("="*50)
    
    # 测试常见问题
    test_common_issues()
    
    # 模拟MQTT处理
    simulate_mqtt_save()
    
    print("\n" + "="*50)
    print("📋 诊断建议")
    print("="*50)
    print("1. 检查MQTT注册日志中的图片URL")
    print("2. 验证图片URL是否返回有效图片")
    print("3. 检查网络连接和防火墙设置")
    print("4. 测试图片URL是否支持直接访问")
    print("5. 考虑使用本地图片缓存机制")

if __name__ == "__main__":
    main()