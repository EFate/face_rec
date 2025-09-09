# Hailo模型使用指南

## 1. 模型位置配置

### 模型存储位置
```python
# 模型文件存储在 data/zoo/ 目录下
MODEL_ZOO_DIR = "./data/zoo"

# 具体模型路径：
DETECTION_MODEL_PATH = "./data/zoo/scrfd_10g--640x640_quant_hailort_hailo8_1"
RECOGNITION_MODEL_PATH = "./data/zoo/arcface_mobilefacenet--112x112_quant_hailort_hailo8_1"
```

### 配置文件说明
每个模型目录包含：
- `.hef` 文件 - Hailo模型二进制文件
- `.json` 文件 - 模型配置和规格信息
- Python后处理脚本 - 输出数据处理

---

## 2. 模型输入输出格式

### SCRFD人脸检测模型
**输入格式：**
- 图像尺寸：`640x640` 像素
- 颜色格式：RGB三通道
- 数据类型：numpy array (uint8)

**输出格式：**
```python
[
    {
        "bbox": [x1, y1, x2, y2],  # 人脸边界框坐标
        "score": 0.95,              # 检测置信度
        "landmarks": [              # 5个人脸关键点
            {"landmark": [x, y]},   # 左眼
            {"landmark": [x, y]},   # 右眼
            {"landmark": [x, y]},   # 鼻子
            {"landmark": [x, y]},   # 左嘴角
            {"landmark": [x, y]}    # 右嘴角
        ]
    }
]
```

### ArcFace人脸识别模型
**输入格式：**
- 图像尺寸：`112x112` 像素（对齐后人脸）
- 颜色格式：RGB三通道
- 数据类型：numpy array (uint8)

**输出格式：**
- 512维浮点特征向量
- 形状：`(1, 512)`
- 数据类型：numpy array (float32)

---

## 3. 模型加载

### 单个模型加载
```python
import degirum as dg

def load_single_model(model_name, zoo_path):
    """加载单个Hailo模型"""
    model = dg.load_model(
        model_name=model_name,
        inference_host_address=dg.LOCAL,  # 本地推理
        zoo_url=f"file://{zoo_path}",     # 模型仓库路径
        image_backend='opencv'            # 使用OpenCV图像后端
    )
    return model

# 加载检测模型
detection_model = load_single_model(
    "scrfd_10g--640x640_quant_hailort_hailo8_1",
    "./data/zoo"
)

# 加载识别模型  
recognition_model = load_single_model(
    "arcface_mobilefacenet--112x112_quant_hailort_hailo8_1",
    "./data/zoo"
)
```

### 模型池加载（推荐）
```python
from app.core.model_manager import ModelPool
from app.cfg.config import get_app_settings

# 获取应用配置
settings = get_app_settings()

# 创建模型池（线程安全，支持多实例）
model_pool = ModelPool(settings, pool_size=3)

# 模型池自动加载两个模型：
# - detection_model: SCRFD人脸检测
# - recognition_model: ArcFace识别
```

---

## 4. 模型调用

### 获取模型实例
```python
# 从模型池获取一套模型
models = model_pool.acquire(timeout=0.1)
if models:
    det_model, rec_model = models
    # 使用模型进行推理...
```

### 人脸检测调用
```python
def detect_faces(frame, detection_model):
    """执行人脸检测"""
    # 输入：640x640 RGB图像
    # 输出：检测结果列表
    results = detection_model.predict(frame).results
    return results

# 使用示例
detection_results = detect_faces(input_frame, det_model)
```

### 人脸识别调用
```python
def extract_face_features(aligned_face, recognition_model):
    """提取人脸特征"""
    # 输入：112x112对齐后人脸图像
    # 输出：512维特征向量
    result = recognition_model.predict(aligned_face)
    embedding = np.array(result.results[0]['data'][0])
    return embedding

# 使用示例
face_embedding = extract_face_features(aligned_face, rec_model)
```

### 批量处理调用
```python
# 批量人脸识别（提高效率）
batch_results = recognition_model.predict_batch([face1, face2, face3])
```

---

## 5. 模型推理输出格式

### 人脸检测输出详解
```python
detection_result = [
    {
        "bbox": [100, 150, 200, 250],  # [x1, y1, x2, y2]
        "score": 0.98,                 # 置信度0.0-1.0
        "category_id": 0,              # 类别ID（0表示人脸）
        "label": "face",               # 类别标签
        "landmarks": [                 # 5个关键点
            {"landmark": [120, 170], "score": 0.98},  # 左眼
            {"landmark": [180, 170], "score": 0.97},  # 右眼
            {"landmark": [150, 190], "score": 0.96},  # 鼻子
            {"landmark": [130, 220], "score": 0.95},  # 左嘴角
            {"landmark": [170, 220], "score": 0.95}   # 右嘴角
        ]
    }
]
```

### 人脸识别输出详解
```python
# 原始输出格式
recognition_output = {
    "results": [
        {
            "data": [
                [0.12, -0.05, 0.08, ...]  # 512个浮点数
            ],
            "shape": [1, 512],
            "type": "DG_FLT"
        }
    ]
}

# 处理后的特征向量
face_embedding = np.array(recognition_output.results[0]['data'][0])
# 形状: (512,)
# 用途: 人脸比对、识别、检索
```

---

## 6. 模型硬件资源释放

### 归还模型到池中
```python
# 使用完成后及时归还模型
model_pool.release((det_model, rec_model))
```

### 强制资源清理
```python
import gc
import os
import signal
import psutil
from typing import Set
from logging import Logger

def get_all_degirum_worker_pids() -> Set[int]:
    """获取所有DeGirum工作进程的PID"""
    worker_pids = set()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and any("degirum/pproc_worker.py" in s for s in cmdline):
                worker_pids.add(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return worker_pids

def cleanup_degirum_workers_by_pids(pids_to_kill: Set[int], logger: Logger):
    """强制终止指定的DeGirum工作进程"""
    killed_count = 0
    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info(f"已终止PID {pid} 的DeGirum工作进程")
            killed_count += 1
        except ProcessLookupError:
            logger.warning(f"PID {pid} 进程已不存在")
        except Exception as e:
            logger.error(f"终止PID {pid} 时发生错误: {e}")
    
    if killed_count > 0:
        logger.info(f"成功终止了 {killed_count} 个DeGirum工作进程")

def cleanup_resources(model_pool, logger):
    """强制释放所有硬件资源"""
    # 1. 终止DeGirum工作进程
    pids = get_all_degirum_worker_pids()
    cleanup_degirum_workers_by_pids(pids, logger)
    
    # 2. 清空模型池
    model_pool.dispose()
    
    # 3. 垃圾回收
    gc.collect()
    logger.info("所有硬件资源已强制释放")

# 使用示例
from app.cfg.logging import app_logger
cleanup_resources(model_pool, app_logger)
```

### 完整的资源管理示例
```python
try:
    # 获取模型
    models = model_pool.acquire()
    if models:
        det_model, rec_model = models
        
        # 执行推理操作
        results = process_frame(frame, det_model, rec_model)
        
finally:
    # 确保资源释放
    if 'models' in locals():
        model_pool.release(models)
```


