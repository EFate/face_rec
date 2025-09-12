#!/usr/bin/env python3
"""
测试test/imgs目录中的图片
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cfg.config import get_app_settings

from app.inference.models import InferenceInput
from app.service.face_dao import LanceDBFaceDataDAO

def test_imgs_directory():
    """测试test/imgs目录"""
    
    settings = get_app_settings()
    
    # 创建DAO
    face_dao = LanceDBFaceDataDAO(
        db_uri=settings.inference.lancedb_uri,
        table_name=settings.inference.lancedb_table_name
    )
    
    # 创建推理引擎
    from app.inference.factory import InferenceEngineFactory
    engine = InferenceEngineFactory.create_engine(settings.inference.device_type, settings.inference.get_device_config())
    
    if not engine.initialize() or not engine.load_models():
        print("❌ 推理引擎初始化失败")
        return
    
    print("✅ 推理引擎初始化成功")
    
    # 检查test/imgs目录
    imgs_dir = Path("test/imgs")
    if not imgs_dir.exists():
        print("❌ test/imgs目录不存在")
        return
    
    print(f"📁 检查目录: {imgs_dir}")
    
    # 获取所有图片文件
    image_files = list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png")) + list(imgs_dir.glob("*.jpeg"))
    
    if not image_files:
        print("❌ 未找到图片文件")
        return
    
    print(f"📸 找到 {len(image_files)} 个图片文件")
    
    # 测试每个图片
    for img_path in image_files:
        print(f"\n🖼️ 测试图片: {img_path}")
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print("  ❌ 无法读取图片")
            continue
        
        print(f"  图片尺寸: {img.shape}")
        
        # 执行推理
        input_data = InferenceInput(
            image=img,
            detection_threshold=0.5,
            recognition_threshold=0.2
        )
        
        result = engine.predict(input_data)
        
        if not result.success:
            print(f"  ❌ 推理失败: {result.error_message}")
            continue
        
        faces = result.result.faces
        print(f"  ✅ 检测到 {len(faces)} 个人脸")
        
        for i, face in enumerate(faces):
            print(f"    人脸 {i+1}:")
            print(f"      置信度: {face.confidence:.3f}")
            print(f"      边界框: {face.bbox}")
            
            if face.embedding is not None:
                features = face.embedding
                print(f"      特征维度: {len(features)}")
                
                # 搜索相似人脸
                match_result = face_dao.search(features, threshold=0.2)
                if match_result:
                    name, sn, similarity = match_result
                    print(f"      ✅ 匹配成功: {name} (SN: {sn}, 相似度: {similarity:.3f})")
                else:
                    print("      ❌ 未找到匹配人脸")
            else:
                print("      ❌ 无特征向量")

if __name__ == "__main__":
    test_imgs_directory()