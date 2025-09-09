# 异构算力卡推理引擎适配说明

## 概述

本项目已成功适配多种异构算力卡，支持CUDA、Hailo8和RK3588三种推理设备。通过统一的推理接口，可以在不同硬件平台上运行相同的人脸识别服务。

## 支持的设备类型

### 1. CUDA (默认)
- **推理库**: InsightFace + ONNX Runtime
- **模型格式**: ONNX
- **执行提供者**: CUDAExecutionProvider, CPUExecutionProvider
- **适用场景**: 有NVIDIA GPU的服务器或工作站

### 2. Hailo8
- **推理库**: Degirum
- **模型格式**: .hef (Hailo量化模型)
- **适用场景**: Hailo-8 AI加速器
- **模型位置**: `data/zoo/scrfd_10g--640x640_quant_hailort_hailo8_1/`

### 3. RK3588
- **推理库**: Degirum
- **模型格式**: .rknn (RKNN量化模型)
- **适用场景**: 瑞芯微RK3588芯片
- **模型位置**: `data/zoo/yolov8s_relu6_widerface_kpts--640x640_quant_rknn_rk3588_1/`

## 配置方法

### 环境变量配置

通过环境变量 `INFERENCE__DEVICE_TYPE` 指定推理设备类型：

```bash
# 使用CUDA
export INFERENCE__DEVICE_TYPE=cuda

# 使用Hailo8
export INFERENCE__DEVICE_TYPE=hailo8

# 使用RK3588
export INFERENCE__DEVICE_TYPE=rk3588
```

### 详细配置

#### CUDA配置
```bash
INFERENCE__CUDA__MODEL_PACK_NAME=buffalo_l
INFERENCE__CUDA__PROVIDERS=["CUDAExecutionProvider", "CPUExecutionProvider"]
INFERENCE__CUDA__DETECTION_SIZE=[640, 640]
INFERENCE__CUDA__DET_THRESH=0.4
INFERENCE__CUDA__HOME=./data/.insightface
```

#### Hailo8配置
```bash
INFERENCE__HAILO8__ZOO_PATH=./data/zoo
INFERENCE__HAILO8__DETECTION_MODEL=scrfd_10g--640x640_quant_hailort_hailo8_1
INFERENCE__HAILO8__RECOGNITION_MODEL=arcface_mobilefacenet--112x112_quant_hailort_hailo8_1
INFERENCE__HAILO8__DETECTION_SIZE=[640, 640]
INFERENCE__HAILO8__RECOGNITION_SIZE=[112, 112]
```

#### RK3588配置
```bash
INFERENCE__RK3588__ZOO_PATH=./data/zoo
INFERENCE__RK3588__DETECTION_MODEL=yolov8s_relu6_widerface_kpts--640x640_quant_rknn_rk3588_1
INFERENCE__RK3588__RECOGNITION_MODEL=mbf_w600k--112x112_float_rknn_rk3588_1
INFERENCE__RK3588__DETECTION_SIZE=[640, 640]
INFERENCE__RK3588__RECOGNITION_SIZE=[112, 112]
```

## 架构设计

### 统一抽象层
- `BaseInferenceEngine`: 推理引擎抽象基类
- `InferenceEngineFactory`: 推理引擎工厂
- `InferenceAdapter`: 推理适配器，统一不同引擎的接口

### 设备特定实现
- `CudaInferenceEngine`: CUDA推理引擎
- `Hailo8InferenceEngine`: Hailo8推理引擎
- `RK3588InferenceEngine`: RK3588推理引擎

### 标准化输入输出
- `InferenceInput`: 标准化输入数据
- `InferenceOutput`: 标准化输出数据
- `FaceDetection`: 人脸检测结果
- `InferenceResult`: 推理结果

## 资源管理

### Degirum资源清理
对于使用Degirum库的Hailo8和RK3588设备，系统会自动管理资源清理：

1. **自动清理**: 应用退出时自动清理Degirum工作进程
2. **信号处理**: 捕获SIGTERM和SIGINT信号进行清理
3. **进程监控**: 监控并终止残留的pproc_worker.py进程

### 模型池管理
- 支持多实例模型池
- 异步模型加载和释放
- 自动降级处理（如CUDA不可用时使用CPU）

## 使用方法

### 1. 启动服务
```bash
# 设置设备类型
export INFERENCE__DEVICE_TYPE=cuda  # 或 hailo8, rk3588

# 启动服务
python run.py
```

### 2. 测试推理引擎
```bash
# 测试所有推理引擎
python test_inference_engines.py

# 集成测试
python test_integration.py
```

### 3. API使用
API接口保持不变，系统会根据配置自动选择对应的推理引擎：

```bash
# 人脸注册
curl -X POST "http://localhost:8000/api/face/faces" \
  -F "name=张三" \
  -F "sn=EMP001" \
  -F "image_file=@test_face.jpg"

# 人脸识别
curl -X POST "http://localhost:8000/api/face/recognize" \
  -F "image_file=@test_image.jpg"

# 启动视频流
curl -X POST "http://localhost:8000/api/face/streams/start" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "0",
    "taskId": 1234567890,
    "appId": 31,
    "appName": "人脸应用",
    "domainName": "video.com"
  }'
```

## 性能优化

### 1. 模型预加载
- 启动时预加载模型到内存
- 支持多实例模型池
- 异步模型获取和释放

### 2. 批量处理
- 支持批量人脸识别
- 动态跳帧策略
- 智能缓冲区管理

### 3. 资源监控
- 实时监控模型池状态
- 自动清理过期资源
- 内存使用优化

## 故障排除

### 1. CUDA相关问题
```bash
# 检查CUDA环境
nvidia-smi

# 检查ONNX Runtime CUDA支持
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

### 2. Degirum相关问题
```bash
# 检查Degirum安装
python -c "import degirum; print('Degirum installed')"

# 手动清理残留进程
pkill -f "pproc_worker.py"
```

### 3. 模型文件问题
```bash
# 检查模型文件是否存在
ls -la data/zoo/*/
ls -la data/.insightface/
```

## 日志和监控

### 日志级别
- `INFO`: 一般信息
- `DEBUG`: 详细调试信息
- `ERROR`: 错误信息
- `WARNING`: 警告信息

### 关键日志
- 推理引擎初始化
- 模型加载状态
- 推理性能统计
- 资源清理状态

## 兼容性

### 向后兼容
- 保持原有API接口不变
- 支持传统InsightFace配置
- 自动降级处理

### 环境要求
- Python 3.8+
- 对应设备的推理库
- 足够的模型存储空间

## 开发指南

### 添加新设备类型
1. 继承 `BaseInferenceEngine`
2. 实现必要的抽象方法
3. 在 `InferenceEngineFactory` 中注册
4. 添加对应的配置类

### 自定义模型
1. 将模型文件放入对应目录
2. 更新配置文件中的模型路径
3. 调整输入输出处理逻辑

## 总结

异构算力卡适配已完成，系统现在支持：
- ✅ CUDA/CPU推理（InsightFace + ONNX Runtime）
- ✅ Hailo8推理（Degirum库）
- ✅ RK3588推理（Degirum库）
- ✅ 统一的推理接口
- ✅ 自动资源管理
- ✅ 完整的错误处理
- ✅ 向后兼容性

通过环境变量配置，可以轻松在不同硬件平台上部署相同的人脸识别服务。
