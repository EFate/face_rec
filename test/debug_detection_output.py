#!/usr/bin/env python3
"""
调试检测模型输出格式，验证关键点信息
"""

import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.inference.devices.hailo8.engine import Hailo8InferenceEngine
from app.inference.base import InferenceInput
from app.cfg.config import InferenceConfig
from app.cfg.logging import setup_logging

# 设置日志
setup_logging(None)  # 使用默认配置

def debug_detection_output():
    """调试检测模型的实际输出格式"""
    print("=== 开始调试检测模型输出格式 ===")
    
    # 配置
    config = InferenceConfig()
    config.device_type = "hailo8"
    config.hailo8.detection_model = "scrfd_10g--640x640_quant_hailort_hailo8_1"
    config.hailo8.recognition_model = "arcface_mobilefacenet--112x112_quant_hailort_hailo8_1"
    
    # 初始化引擎
    engine = Hailo8InferenceEngine(config)
    
    try:
        # 加载模型
        success = engine.initialize()
        if not success:
            print("❌ 模型初始化失败")
            return
        
        print("✅ 模型初始化成功")
        
        # 测试图片路径
        test_images = [
            "/home/abt/lx/face_rec/test/imgs/女.png",
            "/home/abt/lx/face_rec/test/imgs/女2.png"
        ]
        
        for img_path in test_images:
            if not os.path.exists(img_path):
                print(f"❌ 图片不存在: {img_path}")
                continue
            
            print(f"\n=== 处理图片: {os.path.basename(img_path)} ===")
            
            # 读取图片
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ 无法读取图片: {img_path}")
                continue
            
            # 执行检测（不提取特征向量，只看检测输出）
            input_data = InferenceInput(
                image=image,
                extract_embeddings=False,  # 只检测，不识别
                detection_threshold=0.2
            )
            
            output = engine.predict(input_data)
            
            if not output.success:
                print(f"❌ 检测失败: {output.error_message}")
                continue
            
            print(f"✅ 检测到 {len(output.result.faces)} 张人脸")
            
            # 详细分析每个检测结果
            for i, face in enumerate(output.result.faces):
                print(f"\n--- 人脸 {i+1} ---")
                print(f"边界框: {face.bbox}")
                print(f"置信度: {face.confidence}")
                print(f"关键点: {face.landmarks}")
                
                if face.landmarks:
                    print(f"关键点数量: {len(face.landmarks)}")
                    print(f"关键点类型: {type(face.landmarks)}")
                    if len(face.landmarks) > 0:
                        print(f"第一个关键点: {face.landmarks[0]} (类型: {type(face.landmarks[0])})")
                else:
                    print("❌ 关键点为空！")
            
            # 保存原始检测结果的详细信息
            try:
                # 获取原始检测结果
                detection_results = engine.detection_model.predict(
                    engine._preprocess_image(image, engine.detection_size)
                ).results
                
                print(f"\n--- 原始检测输出格式 ---")
                if detection_results:
                    for j, result in enumerate(detection_results):
                        print(f"结果 {j+1} 的键: {list(result.keys())}")
                        for key, value in result.items():
                            if 'landmark' in key.lower() or 'kpt' in key.lower():
                                print(f"  {key}: {value} (类型: {type(value)})")
                else:
                    print("无原始检测结果")
                    
            except Exception as e:
                print(f"获取原始检测结果失败: {e}")
    
    except Exception as e:
        print(f"❌ 调试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        try:
            engine.cleanup()
            print("\n✅ 资源清理完成")
        except Exception as e:
            print(f"清理资源时出错: {e}")

if __name__ == "__main__":
    debug_detection_output()