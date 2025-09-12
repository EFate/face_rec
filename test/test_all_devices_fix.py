#!/usr/bin/env python3
"""
测试所有设备的关键点解析修复效果
"""

import sys
import os
sys.path.append('/home/abt/lx/face_rec')

import numpy as np
from PIL import Image
from app.inference.devices.hailo8.engine import Hailo8InferenceEngine
from app.inference.devices.rk3588.engine import RK3588InferenceEngine
from app.inference.models import InferenceInput

def test_device_landmarks(device_name, engine_class):
    """测试指定设备的关键点解析"""
    print(f"\n=== 测试 {device_name} 设备关键点解析 ===")
    
    try:
        # 初始化引擎
        engine = engine_class()
        
        # 加载测试图片
        image_path = "test/女.png"
        if not os.path.exists(image_path):
            print(f"测试图片 {image_path} 不存在")
            return False
        
        # 读取图片
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        
        # 创建推理输入
        input_data = InferenceInput(
            image=image,
            detection_threshold=0.5,
            extract_embeddings=True
        )
        
        # 执行推理
        result = engine.predict(input_data)
        
        if result.success and result.result.detections:
            print(f"{device_name} 检测到 {len(result.result.detections)} 张人脸")
            
            for i, detection in enumerate(result.result.detections):
                print(f"\n人脸 {i+1}:")
                print(f"  置信度: {detection.confidence:.3f}")
                print(f"  边界框: [{detection.bbox[0]:.1f}, {detection.bbox[1]:.1f}, {detection.bbox[2]:.1f}, {detection.bbox[3]:.1f}]")
                
                if detection.landmarks and len(detection.landmarks) >= 5:
                    print(f"  关键点数量: {len(detection.landmarks)}")
                    print("  关键点坐标:")
                    for j, landmark in enumerate(detection.landmarks):
                        print(f"    关键点 {j+1}: ({landmark[0]:.1f}, {landmark[1]:.1f})")
                    
                    if detection.embedding:
                        print(f"  特征向量维度: {len(detection.embedding)}")
                        print("  ✓ 关键点解析成功，人脸对齐功能正常")
                        return True
                    else:
                        print("  ✗ 无特征向量")
                        return False
                else:
                    print(f"  关键点数量: {len(detection.landmarks) if detection.landmarks else 0}")
                    print("  ✗ 关键点不足或无关键点，将使用裁剪方式")
                    return False
        else:
            print(f"{device_name} 检测失败: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"{device_name} 测试失败: {e}")
        return False

def simulate_landmarks_parsing():
    """模拟关键点解析逻辑"""
    print("\n=== 模拟关键点解析测试 ===")
    
    # 模拟检测模型的输出格式
    test_landmarks = [
        {"category_id": 0, "landmark": [315.4, 299.9], "score": 0.65},
        {"category_id": 1, "landmark": [464.1, 301.1], "score": 0.65},
        {"category_id": 2, "landmark": [415.7, 363.7], "score": 0.65},
        {"category_id": 3, "landmark": [322.5, 418.0], "score": 0.65},
        {"category_id": 4, "landmark": [453.5, 418.0], "score": 0.65}
    ]
    
    def parse_landmarks(raw_landmarks):
        """模拟解析逻辑"""
        landmarks = []
        if isinstance(raw_landmarks, list):
            for landmark in raw_landmarks:
                if isinstance(landmark, dict):
                    if 'landmark' in landmark and isinstance(landmark['landmark'], (list, tuple)) and len(landmark['landmark']) >= 2:
                        landmarks.append([float(landmark['landmark'][0]), float(landmark['landmark'][1])])
                    elif 'x' in landmark and 'y' in landmark:
                        landmarks.append([float(landmark['x']), float(landmark['y'])])
                elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                    landmarks.append([float(landmark[0]), float(landmark[1])])
        return landmarks
    
    parsed = parse_landmarks(test_landmarks)
    print(f"原始数据: {len(test_landmarks)} 个关键点")
    print(f"解析结果: {len(parsed)} 个关键点")
    for i, coord in enumerate(parsed):
        print(f"  关键点 {i+1}: ({coord[0]:.1f}, {coord[1]:.1f})")
    
    return len(parsed) >= 5

def main():
    """主测试函数"""
    print("开始测试所有设备的关键点解析修复...")
    
    # 模拟测试
    sim_success = simulate_landmarks_parsing()
    
    # 设备测试
    devices = [
        ("Hailo8", Hailo8InferenceEngine),
        ("RK3588", RK3588InferenceEngine)
    ]
    
    results = {}
    for device_name, engine_class in devices:
        results[device_name] = test_device_landmarks(device_name, engine_class)
    
    # 总结结果
    print("\n=== 测试结果总结 ===")
    print(f"模拟测试: {'✓ 通过' if sim_success else '✗ 失败'}")
    
    all_passed = True
    for device_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{device_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed and sim_success:
        print("\n🎉 所有设备的关键点解析修复成功！")
    else:
        print("\n❌ 部分设备测试失败，需要进一步调试")

if __name__ == "__main__":
    main()