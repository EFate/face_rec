#!/usr/bin/env python3
"""
检查人脸注册和识别流程
"""
import sys
import os
sys.path.append('/home/abt/lx/face_rec')

from app.cfg.config import get_app_settings
from app.service.face_dao import LanceDBFaceDataDAO
from app.inference.factory import InferenceEngineFactory
import cv2
import numpy as np
from pathlib import Path

def check_registration():
    """检查人脸注册情况"""
    print("🔍 检查人脸注册和识别流程...")
    
    # 初始化配置和DAO
    settings = get_app_settings()
    face_dao = LanceDBFaceDataDAO(
        db_uri=settings.inference.lancedb_uri,
        table_name=settings.inference.lancedb_table_name
    )
    
    # 检查已注册的人脸
    print("\n📊 数据库中的人脸信息:")
    all_faces = face_dao.get_all()
    print(f"数据库中共有 {len(all_faces)} 个人脸记录")
    
    for i, face in enumerate(all_faces):
        print(f"  {i+1}. UUID: {face['uuid']}")
        print(f"     姓名: {face['name']}")
        print(f"     SN: {face['sn']}")
        print(f"     特征维度: {len(face['vector'])}")
        print(f"     图片路径: {face['image_path']}")
        
        # 检查图片是否存在
        img_path = Path(face['image_path'])
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                print(f"     图片尺寸: {img.shape}")
            else:
                print(f"     ❌ 无法读取图片")
        else:
            print(f"     ❌ 图片文件不存在: {img_path}")
    
    # 测试推理引擎
    print("\n🔄 测试推理引擎...")
    try:
        engine = InferenceEngineFactory.create_engine(
            device_type=settings.inference.device_type,
            config=settings.inference.get_device_config()
        )
        
        if not engine.initialize() or not engine.load_models():
            print("❌ 推理引擎初始化失败")
            return
        print("✅ 推理引擎初始化成功")
        
        # 测试data/faces目录中的图片
        faces_dir = Path("data/faces")
        if faces_dir.exists():
            print(f"\n📁 检查注册图片目录: {faces_dir}")
            for person_dir in faces_dir.iterdir():
                if person_dir.is_dir():
                    print(f"\n👤 人员目录: {person_dir.name}")
                    for img_file in person_dir.glob("*.jpg"):
                        print(f"  📸 测试图片: {img_file}")
                        
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            print(f"     图片尺寸: {img.shape}")
                            
                            # 检测人脸
                            # 执行人脸检测和识别
                            from app.inference.models import InferenceInput
                            input_data = InferenceInput(
                                image=img,
                                detection_threshold=0.5,
                                recognition_threshold=0.2
                            )
                            
                            result = engine.predict(input_data)
                            if not result.success or not result.result.faces:
                                print(f"     ❌ 推理失败: {result.error_message}")
                                continue
                            
                            faces = result.result.faces
                            print(f"     ✅ 检测到 {len(faces)} 个人脸")
                            
                            for j, face in enumerate(faces):
                                print(f"     人脸 {j+1}: 置信度={face.confidence:.3f}")
                                
                                if face.embedding is not None:
                                    features = face.embedding
                                    print(f"     特征维度: {len(features)}")
                                    
                                    # 搜索相似人脸
                                    match_result = face_dao.search(features, threshold=0.2)
                                    if match_result:
                                        name, sn, similarity = match_result
                                        print(f"     ✅ 匹配成功: {name} (SN: {sn}, 相似度: {similarity:.3f})")
                                    else:
                                        print("     ❌ 未找到匹配人脸")
                                else:
                                    print("     ❌ 特征提取失败")
                        else:
                            print(f"     ❌ 无法读取图片")
        
        engine.cleanup()
        
    except Exception as e:
        print(f"❌ 推理引擎测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_registration()