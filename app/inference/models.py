# app/inference/models.py
"""
推理引擎的标准化输入输出模型
使用Pydantic确保数据一致性和类型安全
"""

from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field, validator
import numpy as np


class FaceDetection(BaseModel):
    """单个人脸检测结果"""
    bbox: List[float] = Field(..., description="人脸边界框 [x1, y1, x2, y2]")
    confidence: float = Field(..., description="检测置信度 0.0-1.0")
    landmarks: Optional[List[List[float]]] = Field(None, description="人脸关键点坐标")
    embedding: Optional[List[float]] = Field(None, description="人脸特征向量")
    
    @validator('bbox')
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError('bbox must have exactly 4 coordinates [x1, y1, x2, y2]')
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError('Invalid bbox coordinates: x1 < x2 and y1 < y2')
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence must be between 0.0 and 1.0')
        return v
    
    class Config:
        arbitrary_types_allowed = True


class InferenceResult(BaseModel):
    """推理结果"""
    faces: List[FaceDetection] = Field(default_factory=list, description="检测到的人脸列表")
    processing_time: float = Field(0.0, description="推理处理时间（秒）")
    device_type: str = Field(..., description="使用的推理设备类型")
    
    @validator('processing_time')
    def validate_processing_time(cls, v):
        if v < 0:
            raise ValueError('processing_time must be non-negative')
        return v
    
    class Config:
        arbitrary_types_allowed = True


class InferenceInput(BaseModel):
    """推理输入数据"""
    image: np.ndarray = Field(..., description="输入图像数据")
    extract_embeddings: bool = Field(True, description="是否提取人脸特征向量")
    detection_threshold: float = Field(0.4, description="人脸检测置信度阈值")
    
    @validator('image')
    def validate_image(cls, v):
        if v is None:
            raise ValueError('image cannot be None')
        if len(v.shape) != 3:
            raise ValueError('image must be 3-dimensional (H, W, C)')
        if v.shape[2] != 3:
            raise ValueError('image must have 3 channels (RGB)')
        return v
    
    @validator('detection_threshold')
    def validate_detection_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('detection_threshold must be between 0.0 and 1.0')
        return v
    
    class Config:
        arbitrary_types_allowed = True


class InferenceOutput(BaseModel):
    """推理输出数据"""
    result: InferenceResult = Field(..., description="推理结果")
    success: bool = Field(True, description="推理是否成功")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    class Config:
        arbitrary_types_allowed = True


class ModelConfig(BaseModel):
    """模型配置"""
    device_type: str = Field(..., description="设备类型: cuda, hailo8, rk3588")
    model_path: str = Field(..., description="模型文件路径")
    input_size: List[int] = Field(..., description="模型输入尺寸 [width, height]")
    output_size: Optional[List[int]] = Field(None, description="模型输出尺寸")
    batch_size: int = Field(1, description="批处理大小")
    precision: str = Field("fp32", description="精度类型: fp32, fp16, int8")
    
    @validator('device_type')
    def validate_device_type(cls, v):
        supported_devices = ['cuda', 'hailo8', 'rk3588']
        if v not in supported_devices:
            raise ValueError(f'device_type must be one of: {supported_devices}')
        return v
    
    @validator('input_size')
    def validate_input_size(cls, v):
        if len(v) != 2:
            raise ValueError('input_size must have exactly 2 dimensions [width, height]')
        if any(dim <= 0 for dim in v):
            raise ValueError('input_size dimensions must be positive')
        return v
    
    class Config:
        arbitrary_types_allowed = True


class EngineInfo(BaseModel):
    """推理引擎信息"""
    device_type: str = Field(..., description="设备类型")
    initialized: bool = Field(False, description="是否已初始化")
    models_loaded: bool = Field(False, description="模型是否已加载")
    config: Dict[str, Any] = Field(default_factory=dict, description="引擎配置")
    
    class Config:
        arbitrary_types_allowed = True
