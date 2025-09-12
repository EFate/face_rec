#!/usr/bin/env python3
"""
人脸识别测试脚本 - 使用DeGirum PySDK
完全参考test/人脸识别系统构建综合指南.md文档实现
使用Hailo8模型对test/imgs中的图片进行人脸检测和识别，验证两张图片是否为同一个人
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
import json
import degirum as dg
import degirum_tools

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_similarity(embedding1, embedding2):
    """
    计算两个特征嵌入之间的余弦相似度
    
    参数：
        embedding1 (np.ndarray)：第一个特征嵌入
        embedding2 (np.ndarray)：第二个特征嵌入
    
    返回：
        float：余弦相似度，范围在[-1, 1]之间，值越大表示越相似
    """
    # 计算点积
    dot_product = np.dot(embedding1, embedding2)
    
    # 计算L2范数
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    
    return similarity

def align_and_crop(img, landmarks, image_size=112):
    """
    根据给定的特征点，对图像中的人脸进行对齐与裁剪。

    参数：
        img (np.ndarray)：原始完整图像（非裁剪后的边界框图像），将对该图像进行变换。
        landmarks (List[np.ndarray])：5个关键点（特征点）的列表，格式为(x, y)坐标。这些关键点通常包括眼睛、鼻子、嘴巴。
        image_size (int, 可选)：图像调整后的尺寸，默认值为112。人脸识别模型常用尺寸通常为112或128。

    返回：
        Tuple[np.ndarray, np.ndarray]：对齐后的人脸图像与变换矩阵。
    """
    # 定义ArcFace模型中使用的参考关键点（基于典型面部特征点集）
    _arcface_ref_kps = np.array(
        [
            [38.2946, 51.6963],  # 左眼
            [73.5318, 51.5014],  # 右眼
            [56.0252, 71.7366],  # 鼻子
            [41.5493, 92.3655],  # 左嘴角
            [70.7299, 92.2041],  # 右嘴角
        ],
        dtype=np.float32,
    )

    # 确保输入的特征点数量恰好为5个（人脸对齐所需的标准数量）
    assert len(landmarks) == 5

    # 验证image_size是否可被112或128整除（人脸识别模型的常用图像尺寸）
    assert image_size % 112 == 0 or image_size % 128 == 0

    # 根据目标图像尺寸（112或128）调整缩放比例
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0  # 尺寸为112时无需水平偏移
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio  # 尺寸为128时需添加水平偏移

    # 对参考关键点应用缩放与偏移
    dst = _arcface_ref_kps * ratio
    dst[:, 0] += diff_x  # 应用水平偏移

    # 估计相似变换矩阵，使输入特征点与参考关键点对齐
    M, inliers = cv2.estimateAffinePartial2D(np.array(landmarks), dst, ransacReprojThreshold=1000)
    assert np.all(inliers == True)
    
    # 对输入图像应用仿射变换，实现人脸对齐
    aligned_img = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

    return aligned_img, M

def extract_face_embedding(image_path, face_det_model, face_rec_model):
    """
    从单张图像中提取人脸特征嵌入
    
    参数：
        image_path (str)：图像路径
        face_det_model：人脸检测模型
        face_rec_model：人脸识别模型
    
    返回：
        np.ndarray：人脸特征嵌入，如果检测失败则返回None
    """
    try:
        # 执行人脸检测
        detected_faces = face_det_model(image_path)
        
        # 检查是否检测到人脸
        if not detected_faces.results:
            logger.warning(f"在图像 {image_path} 中未检测到人脸")
            return None
        
        # 假设每张图像只有一个人脸
        face_detection = detected_faces.results[0]
        
        # 提取特征点并对齐人脸
        landmarks = [[landmark["landmark"][0], landmark["landmark"][1]] for landmark in face_detection["landmarks"]]
        aligned_face, _ = align_and_crop(detected_faces.image, landmarks)
        
        # 提取特征向量
        face_embedding = face_rec_model(aligned_face).results[0]["data"][0]
        
        return np.array(face_embedding)
        
    except Exception as e:
        logger.error(f"处理图像 {image_path} 时出错: {e}")
        return None

def main():
    """
    主函数：加载模型，处理图像，计算相似度
    """
    # 模型配置
    face_det_model_name = "scrfd_10g--640x640_quant_hailort_hailo8_1"
    face_rec_model_name = "arcface_mobilefacenet--112x112_quant_hailort_hailo8_1"
    
    # 推理配置
    inference_host_address = "@local"  # 使用本地推理
    zoo_url = "/home/abt/lx/face_rec/data/zoo"  # 本地模型库路径
    
    # 图像路径
    image_dir = "/home/abt/lx/face_rec/test/imgs"
    image_paths = [
        os.path.join(image_dir, "女.png"),
        os.path.join(image_dir, "女2.png")
    ]
    
    # 检查图像文件是否存在
    for path in image_paths:
        if not os.path.exists(path):
            logger.error(f"图像文件不存在: {path}")
            return
    
    # 获取访问令牌（本地推理时可以为空）
    try:
        token = degirum_tools.get_token()
    except:
        token = ""  # 本地推理时留空
    
    try:
        # 加载人脸检测模型
        logger.info("加载人脸检测模型...")
        face_det_model = dg.load_model(
            model_name=face_det_model_name,
            inference_host_address=inference_host_address,
            zoo_url=zoo_url,
            token=token,
            overlay_color=(0, 255, 0)  # 边界框颜色为绿色
        )
        
        # 加载人脸识别模型
        logger.info("加载人脸识别模型...")
        face_rec_model = dg.load_model(
            model_name=face_rec_model_name,
            inference_host_address=inference_host_address,
            zoo_url=zoo_url,
            token=token
        )
        
        # 处理图像
        embeddings = []
        for image_path in image_paths:
            logger.info(f"处理图像: {image_path}")
            embedding = extract_face_embedding(image_path, face_det_model, face_rec_model)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.error(f"无法提取图像 {image_path} 的特征")
                return
        
        # 确保成功提取了两张图像的特征
        if len(embeddings) != 2:
            logger.error(f"只成功提取了 {len(embeddings)} 张图像的特征，需要2张")
            return
        
        # 计算相似度
        similarity = calculate_similarity(embeddings[0], embeddings[1])
        logger.info(f"两张图像的相似度: {similarity:.4f}")
        
        # 判断是否为同一个人
        threshold = 0.6  # 相似度阈值，可根据实际情况调整
        if similarity >= threshold:
            logger.info(f"两张图像是同一个人（相似度 {similarity:.4f} >= {threshold}）")
        else:
            logger.info(f"两张图像不是同一个人（相似度 {similarity:.4f} < {threshold}）")
        
        # 保存结果
        result = {
            "image1": os.path.basename(image_paths[0]),
            "image2": os.path.basename(image_paths[1]),
            "similarity": float(similarity),
            "is_same_person": bool(similarity >= threshold)
        }
        
        # 保存结果到文件
        result_path = "/home/abt/lx/face_rec/test/result_degirium.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
        
        logger.info(f"结果已保存到: {result_path}")
        
    except Exception as e:
        logger.error(f"执行过程中出错: {e}")

if __name__ == "__main__":
    main()