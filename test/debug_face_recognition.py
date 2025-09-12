#!/usr/bin/env python3
"""
调试人脸识别系统
验证项目配置与实际推理结果
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.cfg.config import get_app_settings
from app.inference.factory import InferenceEngineFactory
from app.inference.models import InferenceInput


def test_project_inference():
    """测试项目推理配置"""
    print("=== 测试项目推理配置 ===")
    
    # 获取配置
    settings = get_app_settings()
    print(f"设备类型: {settings.inference.device_type}")
    print(f"检测模型: {settings.inference.hailo8.detection_model}")
    print(f"识别模型: {settings.inference.hailo8.recognition_model}")
    print(f"检测阈值: {settings.inference.recognition_det_score_threshold}")
    print(f"相似度阈值: {settings.inference.recognition_similarity_threshold}")
    
    # 获取设备配置
    config = settings.inference.get_device_config()
    print(f"完整配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    return config


def test_model_loading():
    """测试模型加载"""
    print("\n=== 测试模型加载 ===")
    
    try:
        settings = get_app_settings()
        config = settings.inference.get_device_config()
        
        # 创建推理引擎
        engine = InferenceEngineFactory.create_engine("hailo8", config)
        
        # 初始化
        init_success = engine.initialize()
        print(f"初始化成功: {init_success}")
        
        if init_success:
            load_success = engine.load_models()
            print(f"模型加载成功: {load_success}")
            
            if load_success:
                return engine
                
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def test_image_processing(engine, image_path):
    """测试单张图片处理"""
    print(f"\n=== 测试图片处理: {image_path} ===")
    
    try:
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
            
        print(f"图片尺寸: {image.shape}")
        
        # 创建推理输入
        input_data = InferenceInput(
            image=image,
            detection_threshold=0.2,  # 使用较低的阈值
            extract_embeddings=True
        )
        
        # 执行推理
        result = engine.predict(input_data)
        
        print(f"推理成功: {result.success}")
        if result.success and result.result:
            print(f"检测到 {len(result.result.faces)} 张人脸")
            
            for i, face in enumerate(result.result.faces):
                print(f"人脸 {i+1}:")
                print(f"  置信度: {face.confidence}")
                print(f"  边界框: {face.bbox}")
                print(f"  关键点: {face.landmarks}")
                if face.embedding:
                    print(f"  特征向量长度: {len(face.embedding)}")
                    print(f"  特征向量前5个值: {face.embedding[:5]}")
                    
            return result
        else:
            print(f"推理失败: {result.error_message}")
            
    except Exception as e:
        print(f"图片处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def compare_images(engine, img1_path, img2_path):
    """比较两张图片"""
    print(f"\n=== 比较两张图片 ===")
    print(f"图片1: {img1_path}")
    print(f"图片2: {img2_path}")
    
    result1 = test_image_processing(engine, img1_path)
    result2 = test_image_processing(engine, img2_path)
    
    if result1 and result2 and result1.success and result2.success:
            # 获取第一张人脸的特征向量
            if result1.result.faces and result2.result.faces:
                emb1 = np.array(result1.result.faces[0].embedding)
                emb2 = np.array(result2.result.faces[0].embedding)
            
                # 计算余弦相似度
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                print(f"\n=== 相似度计算结果 ===")
                print(f"特征向量1范数: {np.linalg.norm(emb1)}")
                print(f"特征向量2范数: {np.linalg.norm(emb2)}")
                print(f"余弦相似度: {similarity}")
                
                # 使用项目配置的阈值
                settings = get_app_settings()
                threshold = settings.inference.recognition_similarity_threshold
                print(f"项目阈值: {threshold}")
                print(f"是否为同一人: {similarity >= threshold}")
                
                return {
                    "similarity": float(similarity),
                    "threshold": float(threshold),
                    "is_same_person": bool(similarity >= threshold),
                    "img1_detections": len(result1.result.faces),
                    "img2_detections": len(result2.result.faces)
                }
            else:
                print("未检测到足够的人脸进行比对")
    
    return None


def main():
    """主函数"""
    print("开始调试人脸识别系统...")
    
    # 测试配置
    config = test_project_inference()
    
    # 测试模型加载
    engine = test_model_loading()
    if not engine:
        print("模型加载失败，退出调试")
        return
    
    # 测试图片
    img1_path = project_root / "test" / "imgs" / "女.png"
    img2_path = project_root / "test" / "imgs" / "女2.png"
    
    if not img1_path.exists():
        print(f"图片不存在: {img1_path}")
        return
    
    if not img2_path.exists():
        print(f"图片不存在: {img2_path}")
        return
    
    # 比较图片
    result = compare_images(engine, str(img1_path), str(img2_path))
    
    if result:
        print(f"\n=== 最终调试结果 ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 清理
    engine.cleanup()
    print("\n调试完成")


if __name__ == "__main__":
    main()