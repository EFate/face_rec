#!/usr/bin/env python3
"""
使用注册图片测试人脸识别系统
"""

import sys
import os
from pathlib import Path
import cv2

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cfg.config import get_app_settings
from app.inference.factory import InferenceEngineFactory
from app.inference.models import InferenceInput
from app.service.face_dao import LanceDBFaceDataDAO

def test_registered_faces():
    """使用注册图片测试人脸识别"""
    
    settings = get_app_settings()
    
    # 创建DAO
    face_dao = LanceDBFaceDataDAO(
        db_uri=settings.inference.lancedb_uri,
        table_name=settings.inference.lancedb_table_name
    )
    
    # 创建推理引擎
    engine = InferenceEngineFactory.create_engine(settings.inference.device_type, settings.inference.get_device_config())
    
    if not engine.initialize() or not engine.load_models():
        print("❌ 推理引擎初始化失败")
        return
    
    print("✅ 推理引擎初始化成功")
    
    # 获取所有注册的人脸
    all_faces = face_dao.get_all()
    print(f"📊 数据库中共有 {len(all_faces)} 个人脸记录")
    
    # 测试每个注册图片
    success_count = 0
    total_count = 0
    
    for face in all_faces:
        img_path = Path(face['image_path'])
        name = face['name']
        expected_sn = face['sn']
        
        print(f"\n🧪 测试: {name} ({expected_sn})")
        print(f"   图片: {img_path}")
        
        if not img_path.exists():
            print("   ❌ 图片文件不存在")
            continue
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print("   ❌ 无法读取图片")
            continue
        
        print(f"   图片尺寸: {img.shape}")
        
        # 执行推理
        input_data = InferenceInput(
            image=img,
            detection_threshold=0.5,
            recognition_threshold=0.2
        )
        
        result = engine.predict(input_data)
        
        if not result.success:
            print(f"   ❌ 推理失败: {result.error_message}")
            continue
        
        faces = result.result.faces
        print(f"   检测到 {len(faces)} 个人脸")
        
        if not faces:
            print("   ❌ 未检测到人脸")
            continue
        
        # 使用第一个检测到的人脸
        face_obj = faces[0]
        
        if face_obj.embedding is None:
            print("   ❌ 未提取到特征")
            continue
        
        features = face_obj.embedding
        print(f"   特征维度: {len(features)}")
        
        # 搜索匹配
        match_result = face_dao.search(features, threshold=0.2)
        
        total_count += 1
        
        if match_result:
            matched_name, matched_sn, similarity = match_result
            print(f"   ✅ 匹配成功: {matched_name} (SN: {matched_sn}, 相似度: {similarity:.3f})")
            
            if matched_sn == expected_sn:
                print("   🎯 正确识别")
                success_count += 1
            else:
                print(f"   ❌ 识别错误: 期望 {expected_sn}, 实际 {matched_sn}")
        else:
            print("   ❌ 未找到匹配")
    
    print(f"\n📈 测试结果:")
    print(f"   总测试数: {total_count}")
    print(f"   成功识别: {success_count}")
    print(f"   成功率: {success_count/total_count*100:.1f}%")

if __name__ == "__main__":
    test_registered_faces()