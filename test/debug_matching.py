#!/usr/bin/env python3
"""
调试人脸匹配问题
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cfg.config import get_app_settings
from app.service.face_dao import LanceDBFaceDataDAO

def debug_matching():
    """调试匹配问题"""
    
    settings = get_app_settings()
    
    # 创建DAO
    face_dao = LanceDBFaceDataDAO(
        db_uri=settings.inference.lancedb_uri,
        table_name=settings.inference.lancedb_table_name
    )
    
    print("🔍 调试人脸匹配问题...")
    
    # 获取所有注册的人脸
    all_faces = face_dao.get_all()
    print(f"📊 数据库中共有 {len(all_faces)} 个人脸记录")
    
    # 检查数据库中的特征向量
    for face in all_faces:
        print(f"\n👤 {face['name']} (SN: {face['sn']})")
        features = face['vector']
        print(f"   特征维度: {len(features)}")
        print(f"   特征均值: {np.mean(features):.4f}")
        print(f"   特征标准差: {np.std(features):.4f}")
        print(f"   特征范围: [{np.min(features):.4f}, {np.max(features):.4f}]")
        
        # 测试自匹配
        result = face_dao.search(features, threshold=0.2)
        if result:
            name, sn, similarity = result
            print(f"   ✅ 自匹配相似度: {similarity:.4f}")
        else:
            print(f"   ❌ 自匹配失败")
    
    # 检查test/imgs中的图片
    imgs_dir = Path("test/imgs")
    if imgs_dir.exists():
        print(f"\n📁 检查test/imgs目录...")
        
        image_files = list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png")) + list(imgs_dir.glob("*.jpeg"))
        
        for img_path in image_files:
            print(f"\n🖼️ 图片: {img_path.name}")
            
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print("   ❌ 无法读取图片")
                continue
            
            print(f"   图片尺寸: {img.shape}")
            
            # 这里我们模拟特征提取的结果
            # 在实际情况下，应该使用推理引擎提取特征
            
            # 创建一个随机特征向量用于测试
            dummy_features = np.random.randn(512).astype(np.float32)
            dummy_features = dummy_features / np.linalg.norm(dummy_features)  # 归一化
            
            print(f"   测试特征向量维度: {len(dummy_features)}")
            
            # 搜索相似人脸
            result = face_dao.search(dummy_features, threshold=0.2)
            if result:
                name, sn, similarity = result
                print(f"   ✅ 找到匹配: {name} (SN: {sn}, 相似度: {similarity:.4f})")
            else:
                print("   ❌ 未找到匹配")
                
                # 检查阈值设置
                print("   🔍 检查阈值设置...")
                
                # 计算与每个注册人脸的相似度
                for face in all_faces:
                    reg_features = np.array(face['vector'])
                    if len(reg_features) == len(dummy_features):
                        similarity = np.dot(dummy_features, reg_features)  # 余弦相似度
                        print(f"     与 {face['name']} 的相似度: {similarity:.4f}")

if __name__ == "__main__":
    debug_matching()