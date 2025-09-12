#!/usr/bin/env python3
"""
调试检测和识别流程
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.cfg.config import get_app_settings
from app.service.face_dao import LanceDBFaceDataDAO
from app.inference.factory import InferenceEngineFactory
from app.inference.base import InferenceInput

def debug_detection_flow():
    """调试检测流程"""
    print("🧪 开始调试检测和识别流程...")
    
    try:
        # 获取设置
        settings = get_app_settings()
        
        # 加载测试图片
        print("📸 加载测试图片...")
        test_image_path = "/home/abt/lx/face_rec/data/test.jpg"
        image = cv2.imread(test_image_path)
        if image is None:
            # 使用摄像头捕获测试图片
            print("📹 使用摄像头捕获测试图片...")
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = frame
                cv2.imwrite(test_image_path, image)
                print(f"✅ 摄像头捕获图片成功: {image.shape}")
            else:
                print("❌ 无法捕获摄像头图片，使用合成图片")
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:
            print(f"✅ 加载测试图片成功: {image.shape}")
        
        # 创建推理引擎
        print("🔄 创建推理引擎...")
        from app.core.inference_engine import InferenceEngine
        
        settings = get_app_settings()
        config = settings.inference.get_device_config()
        config.update({
            "image_db_path": str(settings.inference.image_db_path),
            "lancedb_uri": settings.inference.lancedb_uri,
            "lancedb_table_name": settings.inference.lancedb_table_name
        })
        
        engine = InferenceEngine(config)
        
        # 执行检测
        print("🔄 执行人脸检测...")
        inference_input = InferenceInput(
            image=image,
            threshold=0.5,
            max_faces=10
        )
        
        results = engine.infer(inference_input)
        print(f"✅ 检测到 {len(results)} 个人脸")
        
        if len(results) == 0:
            print("⚠️  未检测到任何人脸")
            return False
        
        # 打印检测结果
        for i, result in enumerate(results):
            print(f"人脸{i+1}: bbox={result.get('bbox')}, score={result.get('score')}")
            if 'keypoints' in result:
                print(f"  关键点: {len(result['keypoints'])} 个")
            if 'embedding' in result:
                print(f"  特征向量: {len(result['embedding'])} 维")
        
        # 检查人脸数据库
        print("🔄 检查人脸数据库...")
        dao = LanceDBFaceDataDAO(
            settings.insightface.lancedb_uri,
            settings.insightface.lancedb_table_name
        )
        
        faces = dao.get_all()
        print(f"数据库中有 {len(faces)} 个人脸记录")
        
        if len(faces) == 0:
            print("⚠️  人脸数据库为空")
            return False
        
        # 执行人脸识别
        print("🔄 执行人脸识别...")
        for i, result in enumerate(results):
            if 'embedding' in result:
                embedding = np.array(result['embedding'])
                match = dao.search(embedding, threshold=0.7)
                
                if match:
                    name, sn, similarity = match
                    print(f"人脸{i+1}: 识别为 {name} (相似度: {similarity:.3f})")
                else:
                    print(f"人脸{i+1}: 未匹配到已知身份")
            else:
                print(f"人脸{i+1}: 无特征向量，无法识别")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_detection_flow()
    
    if success:
        print("\n✅ 调试完成！")
    else:
        print("\n❌ 调试发现问题！")