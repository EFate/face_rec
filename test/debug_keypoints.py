#!/usr/bin/env python3
"""
调试关键点检测问题
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.cfg.config import get_app_settings
from app.inference.factory import InferenceEngineFactory
from app.inference.models import InferenceInput

def debug_keypoints():
    """调试关键点检测"""
    print("=== 调试关键点检测 ===")
    
    # 获取配置
    settings = get_app_settings()
    config = settings.inference.get_device_config()
    
    # 创建推理引擎
    engine = InferenceEngineFactory.create_engine("hailo8", config)
    
    # 初始化
    if not engine.initialize() or not engine.load_models():
        print("模型加载失败")
        return
    
    # 图片路径
    img_path = project_root / "test" / "imgs" / "女.png"
    
    # 读取图片
    import cv2
    image = cv2.imread(str(img_path))
    
    # 执行推理，不提取特征，只检测
    input_data = InferenceInput(
        image=image,
        detection_threshold=0.2,
        extract_embeddings=False  # 只检测，不提取特征
    )
    
    result = engine.predict(input_data)
    
    if result.success and result.result and result.result.faces:
        face = result.result.faces[0]
        print(f"检测到 {len(result.result.faces)} 张人脸")
        print(f"置信度: {face.confidence}")
        print(f"边界框: {face.bbox}")
        print(f"关键点数量: {len(face.landmarks) if face.landmarks else 0}")
        if face.landmarks:
            print(f"关键点: {face.landmarks}")
        else:
            print("关键点为空，将使用简单裁剪方式")
    else:
        print("检测失败")

if __name__ == "__main__":
    debug_keypoints()