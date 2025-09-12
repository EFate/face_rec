#!/usr/bin/env python3
"""
测试关键点修复效果
"""

import sys
import os
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import degirum as dg
    import degirum_tools
except ImportError:
    logger.error("DeGirum库未安装")
    sys.exit(1)

def test_keypoint_fix():
    """测试关键点修复效果"""
    print("=== 测试关键点修复效果 ===")
    
    # 模型路径
    zoo_path = "/home/abt/lx/face_rec/data/zoo"
    detection_model_name = "scrfd_10g--640x640_quant_hailort_hailo8_1"
    
    # 测试图片
    test_image = "/home/abt/lx/face_rec/test/imgs/女.png"
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图片不存在: {test_image}")
        return
    
    try:
        # 连接模型
        print("正在连接模型...")
        zoo = dg.connect(dg.LOCAL, zoo_path)
        model = zoo.load_model(detection_model_name)
        
        print(f"✅ 成功加载检测模型: {detection_model_name}")
        
        # 读取图片
        image = cv2.imread(test_image)
        if image is None:
            print("❌ 无法读取图片")
            return
        
        print(f"图片尺寸: {image.shape}")
        
        # 预处理
        input_size = (640, 640)
        processed_image = cv2.resize(image, input_size)
        
        # 执行检测
        print("执行检测...")
        results = model.predict(processed_image)
        
        # 获取检测结果
        detection_list = results.results if hasattr(results, 'results') else []
        print(f"检测到 {len(detection_list)} 张人脸")
        
        # 使用修复后的逻辑解析关键点
        for i, result in enumerate(detection_list):
            print(f"\n--- 人脸 {i+1} ---")
            
            # 模拟修复后的解析逻辑
            landmarks = None
            if 'landmarks' in result:
                raw_landmarks = result['landmarks']
                if isinstance(raw_landmarks, list):
                    landmarks = []
                    for landmark in raw_landmarks:
                        if isinstance(landmark, dict) and 'landmark' in landmark:
                            point = landmark['landmark']
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                landmarks.append([float(point[0]), float(point[1])])
                        elif isinstance(landmark, dict) and 'x' in landmark and 'y' in landmark:
                            landmarks.append([float(landmark['x']), float(landmark['y'])])
                        elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                            landmarks.append([float(landmark[0]), float(landmark[1])])
            
            print(f"原始关键点: {result.get('landmarks', '无')}")
            print(f"解析后关键点: {landmarks}")
            
            if landmarks and len(landmarks) >= 5:
                print("✅ 成功解析5个关键点")
                for j, point in enumerate(landmarks[:5]):
                    print(f"  关键点{j+1}: {point}")
            else:
                print("❌ 关键点解析失败或不足5个")
        
        model = None
        zoo = None
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_keypoint_fix()