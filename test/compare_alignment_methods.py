#!/usr/bin/env python3
"""
比较不同的人脸对齐方式对相似度的影响
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

def manual_align_and_crop(img, landmarks, image_size=112):
    """
    手动实现的人脸对齐，与hailo_face_recognition_degirium_test.py保持一致
    """
    # 定义ArcFace模型中使用的参考关键点
    _arcface_ref_kps = np.array([
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻子
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041],  # 右嘴角
    ], dtype=np.float32)

    # 确保输入的特征点数量恰好为5个
    assert len(landmarks) == 5

    # 根据目标图像尺寸调整缩放比例
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    # 对参考关键点应用缩放与偏移
    dst = _arcface_ref_kps * ratio
    dst[:, 0] += diff_x

    # 估计相似变换矩阵
    M, inliers = cv2.estimateAffinePartial2D(np.array(landmarks), dst, ransacReprojThreshold=1000)
    
    # 对输入图像应用仿射变换
    aligned_img = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

    return aligned_img

def test_with_manual_alignment():
    """使用手动对齐方式测试"""
    print("=== 使用手动对齐方式测试 ===")
    
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
    img1_path = project_root / "test" / "imgs" / "女.png"
    img2_path = project_root / "test" / "imgs" / "女2.png"
    
    # 读取图片
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    # 使用项目检测模型获取人脸信息
    input1 = InferenceInput(image=img1, detection_threshold=0.2, extract_embeddings=False)
    input2 = InferenceInput(image=img2, detection_threshold=0.2, extract_embeddings=False)
    
    result1 = engine.predict(input1)
    result2 = engine.predict(input2)
    
    if not (result1.success and result2.success and result1.result.faces and result2.result.faces):
        print("人脸检测失败")
        return
    
    # 获取检测信息
    face1 = result1.result.faces[0]
    face2 = result2.result.faces[0]
    
    # 手动对齐
    landmarks1 = [[face1.landmarks[i][0], face1.landmarks[i][1]] for i in range(min(5, len(face1.landmarks)))]
    landmarks2 = [[face2.landmarks[i][0], face2.landmarks[i][1]] for i in range(min(5, len(face2.landmarks)))]
    
    if len(landmarks1) < 5 or len(landmarks2) < 5:
        print("关键点不足")
        return
    
    aligned1 = manual_align_and_crop(img1, landmarks1, 112)
    aligned2 = manual_align_and_crop(img2, landmarks2, 112)
    
    # 使用项目识别模型提取特征
    input_aligned1 = InferenceInput(image=aligned1, detection_threshold=0.0, extract_embeddings=True)
    input_aligned2 = InferenceInput(image=aligned2, detection_threshold=0.0, extract_embeddings=True)
    
    emb_result1 = engine.predict(input_aligned1)
    emb_result2 = engine.predict(input_aligned2)
    
    if emb_result1.success and emb_result2.success and emb_result1.result.faces and emb_result2.result.faces:
        emb1 = np.array(emb_result1.result.faces[0].embedding)
        emb2 = np.array(emb_result2.result.faces[0].embedding)
        
        similarity_manual = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"手动对齐相似度: {similarity_manual:.4f}")
        
        return similarity_manual
    
    return None

def test_with_project_alignment():
    """使用项目内置对齐方式测试"""
    print("=== 使用项目内置对齐方式测试 ===")
    
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
    img1_path = project_root / "test" / "imgs" / "女.png"
    img2_path = project_root / "test" / "imgs" / "女2.png"
    
    # 读取图片
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    # 使用项目完整流程
    input1 = InferenceInput(image=img1, detection_threshold=0.2, extract_embeddings=True)
    input2 = InferenceInput(image=img2, detection_threshold=0.2, extract_embeddings=True)
    
    result1 = engine.predict(input1)
    result2 = engine.predict(input2)
    
    if result1.success and result2.success and result1.result.faces and result2.result.faces:
        emb1 = np.array(result1.result.faces[0].embedding)
        emb2 = np.array(result2.result.faces[0].embedding)
        
        similarity_project = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"项目对齐相似度: {similarity_project:.4f}")
        
        return similarity_project
    
    return None

def main():
    """主函数"""
    print("开始比较不同的人脸对齐方式...")
    
    similarity_manual = test_with_manual_alignment()
    similarity_project = test_with_project_alignment()
    
    if similarity_manual is not None and similarity_project is not None:
        print(f"\n=== 对比结果 ===")
        print(f"手动对齐相似度: {similarity_manual:.4f}")
        print(f"项目对齐相似度: {similarity_project:.4f}")
        print(f"差异: {abs(similarity_manual - similarity_project):.4f}")
        
        # 检查项目配置中的对齐参数
        settings = get_app_settings()
        print(f"\n项目配置:")
        print(f"检测阈值: {settings.inference.recognition_det_score_threshold}")
        print(f"相似度阈值: {settings.inference.recognition_similarity_threshold}")

if __name__ == "__main__":
    main()