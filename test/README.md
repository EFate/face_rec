# Hailo人脸识别测试脚本

这个目录包含使用Hailo8模型进行人脸识别的测试脚本。

## 文件说明

### 主要脚本
- `hailo_face_recognition_degirium_test.py` - 使用DeGirum PySDK的完整人脸识别测试脚本
- `hailo_face_recognition_test.py` - 原始测试脚本（存在函数未定义问题）

### 测试图像
- `imgs/女.png` - 第一张测试图像
- `imgs/女2.png` - 第二张测试图像

### 结果文件
- `result_degirium.json` - 测试结果（包含相似度和判断结果）

## 使用方法

1. 确保已安装依赖：
   ```bash
   pip install degirum opencv-python-headless numpy
   ```

2. 运行测试脚本：
   ```bash
   cd /home/abt/lx/face_rec
   python test/hailo_face_recognition_degirium_test.py
   ```

3. 查看结果：
   ```bash
   cat test/result_degirium.json
   ```

## 工作原理

该脚本按照以下步骤工作：

1. **人脸检测**：使用SCRFD模型检测图像中的人脸和关键点
2. **人脸对齐**：根据检测到的关键点对人脸进行对齐和裁剪
3. **特征提取**：使用ArcFace模型提取对齐人脸的特征向量
4. **相似度计算**：计算两个特征向量的余弦相似度
5. **结果判断**：根据相似度阈值判断是否同一个人

## 模型信息

- **人脸检测模型**：`scrfd_10g--640x640_quant_hailort_hailo8_1`
- **人脸识别模型**：`arcface_mobilefacenet--112x112_quant_hailort_hailo8_1`
- **模型路径**：`data/zoo/`目录下

## 测试输出示例

```
2025-09-12 16:23:49,321 - INFO - 加载人脸检测模型...
2025-09-12 16:23:49,422 - INFO - 加载人脸识别模型...
2025-09-12 16:23:50,250 - INFO - 处理图像: /home/abt/lx/face_rec/test/imgs/女.png
2025-09-12 16:23:50,278 - INFO - 两张图像的相似度: 0.6175
2025-09-12 16:23:50,278 - INFO - 两张图像是同一个人（相似度 0.6175 >= 0.6）
2025-09-12 16:23:50,279 - INFO - 结果已保存到: /home/abt/lx/face_rec/test/result_degirium.json
```

## 结果文件格式

```json
{
    "image1": "女.png",
    "image2": "女2.png",
    "similarity": 0.6174992185699952,
    "is_same_person": true
}
```