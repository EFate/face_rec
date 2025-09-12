#!/usr/bin/env python3
"""
简化版调试检测模型输出格式
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

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

def debug_detection_format():
    """调试检测模型的实际输出格式"""
    print("=== 开始调试检测模型输出格式 ===")
    
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
        
        # 获取实际的检测结果列表
        detection_list = results.results if hasattr(results, 'results') else []
        print(f"检测到 {len(detection_list)} 个结果")
        
        # 详细分析每个结果
        for i, result in enumerate(detection_list):
            print(f"\n--- 结果 {i+1} ---")
            print("所有字段:")
            for key, value in result.items():
                print(f"  {key}: {value} (类型: {type(value)})")
                
                # 特别关注关键点相关字段
                if any(k in str(key).lower() for k in ['landmark', 'kpt', 'point', 'key']):
                    print(f"  🔍 关键点字段 {key}: {value}")
        
        # 检查是否有5个关键点
        for i, result in enumerate(detection_list):
            print(f"\n--- 结果 {i+1} 的关键点分析 ---")
            
            # 尝试不同的关键点字段名
            landmark_fields = ['landmarks', 'landmark', 'kpts', 'keypoints', 'points']
            
            for field in landmark_fields:
                if field in result:
                    landmarks = result[field]
                    print(f"找到关键点字段 '{field}': {landmarks}")
                    if isinstance(landmarks, list):
                        print(f"关键点数量: {len(landmarks)}")
                        if len(landmarks) >= 5:
                            print("✅ 找到5个或更多关键点")
                            # 显示具体的关键点坐标
                            for j, point in enumerate(landmarks[:5]):
                                print(f"  关键点{j+1}: {point}")
                        else:
                            print(f"❌ 只有 {len(landmarks)} 个关键点")
                    elif isinstance(landmarks, dict):
                        print(f"关键点是dict格式: {landmarks}")
                    break
            else:
                print("❌ 未找到任何关键点字段")
        
        model = None
        zoo = None
        
    except Exception as e:
        print(f"❌ 调试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_detection_format()