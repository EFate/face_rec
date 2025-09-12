#!/usr/bin/env python3
"""
对比注册图片和test/imgs中的图片
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compare_images():
    """对比图片"""
    
    print("📸 对比注册图片和test/imgs图片...")
    
    # 注册图片
    reg_dir = Path("data/faces")
    registered_imgs = []
    
    for person_dir in reg_dir.iterdir():
        if person_dir.is_dir():
            for img_file in person_dir.glob("*.jpg"):
                registered_imgs.append(img_file)
    
    # test/imgs中的图片
    test_dir = Path("test/imgs")
    test_imgs = []
    if test_dir.exists():
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            test_imgs.extend(test_dir.glob(ext))
    
    print(f"📁 注册图片: {len(registered_imgs)} 张")
    for img in registered_imgs:
        print(f"   - {img}")
        
    print(f"📁 test/imgs图片: {len(test_imgs)} 张")
    for img in test_imgs:
        print(f"   - {img}")
    
    # 显示图片信息
    print("\n📊 图片信息:")
    
    for img_path in registered_imgs:
        img = cv2.imread(str(img_path))
        if img is not None:
            print(f"注册图片 {img_path.name}:")
            print(f"   尺寸: {img.shape}")
            print(f"   文件大小: {img_path.stat().st_size} bytes")
    
    for img_path in test_imgs:
        img = cv2.imread(str(img_path))
        if img is not None:
            print(f"测试图片 {img_path.name}:")
            print(f"   尺寸: {img.shape}")
            print(f"   文件大小: {img_path.stat().st_size} bytes")
    
    # 检查是否是同一张图片
    print("\n🔍 检查图片相似性...")
    
    for reg_img in registered_imgs:
        reg_data = cv2.imread(str(reg_img))
        if reg_data is None:
            continue
            
        for test_img in test_imgs:
            test_data = cv2.imread(str(test_img))
            if test_data is None:
                continue
                
            # 检查尺寸是否相同
            if reg_data.shape == test_data.shape:
                # 计算像素差异
                diff = cv2.absdiff(reg_data, test_data)
                diff_sum = np.sum(diff)
                
                if diff_sum == 0:
                    print(f"   ✅ {reg_img.name} 和 {test_img.name} 是同一张图片")
                else:
                    print(f"   ❌ {reg_img.name} 和 {test_img.name} 不是同一张图片 (差异: {diff_sum})")
            else:
                print(f"   ❌ {reg_img.name} 和 {test_img.name} 尺寸不同")

if __name__ == "__main__":
    compare_images()