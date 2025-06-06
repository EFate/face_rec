# app/schema/face_schema.py
from pydantic import BaseModel, Field
from typing import List, Optional, TypeVar, Generic, Dict, Any
from datetime import datetime
import uuid

# --- 通用 API 响应模型 ---
T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    """标准API响应格式"""
    code: int = Field(0, description="响应状态码，0表示成功，其他值表示失败。")
    msg: str = Field("Success", description="响应消息。")
    data: Optional[T] = Field(None, description="响应数据。")

# --- 人脸元数据模型 (用于展示和响应) ---
class FaceInfo(BaseModel):
    """单个人脸的详细信息"""
    uuid: str = Field(..., description="人脸特征的唯一ID。")
    name: str = Field(..., description="人脸所属人员的姓名。")
    sn: str = Field(..., description="人脸所属人员的唯一标识SN。")
    registration_time: datetime = Field(..., description="人脸注册时间。")
    image_path: str = Field(..., description="注册图像在文件系统中的路径。")
    extra_info: Optional[Dict[str, Any]] = Field(None, description="预留的额外信息字段。")

    class Config:
        orm_mode = True # 兼容 SQLAlchemy 模型

# --- API 请求体模型 ---
class FaceRegisterRequest(BaseModel):
    """人脸注册请求体"""
    name: str = Field(..., description="人员姓名。", example="张三")
    sn: str = Field(..., description="人员唯一标识SN，例如工号或学号。", example="EMP001")

class UpdateFaceRequest(BaseModel):
    """更新人脸信息请求体"""
    name: Optional[str] = Field(None, description="新的姓名。", example="李四")
    # 这里可以添加更多希望能够更新的字段，例如部门、职位等
    # extra_info: Optional[Dict[str, Any]] = Field(None, description="更新或添加额外信息。")

# --- API 响应数据体模型 ---
class FaceRegisterResponseData(BaseModel):
    """注册人脸的响应数据"""
    face_info: FaceInfo = Field(..., description="注册成功的人脸信息。")

class FaceRecognitionResult(BaseModel):
    """单次人脸识别结果"""
    name: str = Field(..., description="识别到的人脸姓名。")
    sn: str = Field(..., description="识别到的人脸SN。")
    distance: float = Field(..., description="与已知人脸特征的距离，值越小越相似。")
    confidence: Optional[float] = Field(None, description="识别置信度（可选）。")
    box: Optional[List[int]] = Field(None, description="人脸在图像中的边界框 [x, y, w, h]。")

class GetAllFacesResponseData(BaseModel):
    """获取所有人脸列表的响应数据"""
    count: int = Field(..., description="人脸总数。")
    faces: List[FaceInfo] = Field(..., description="已注册人脸的列表。")

class DeleteFaceResponseData(BaseModel):
    """删除人脸的响应数据"""
    sn: str = Field(..., description="被删除的人员SN。")
    deleted_count: int = Field(..., description="成功删除的人脸特征数量。")

class UpdateFaceResponseData(BaseModel):
    """更新人脸信息的响应数据"""
    sn: str = Field(..., description="被更新的人员SN。")
    updated_count: int = Field(..., description="成功更新的人脸特征数量。")
    face_info: FaceInfo = Field(..., description="更新后的人脸信息。")

class HealthCheckResponseData(BaseModel):
    """健康检查响应数据"""
    status: str = Field("ok", description="服务状态。")
    message: str = Field("人脸识别服务正常运行。", description="服务状态信息。")