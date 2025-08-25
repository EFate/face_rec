# app/schema/detection_schema.py
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.schema.face_schema import ApiResponse


class DetectionRecordResponse(BaseModel):
    """检测记录响应"""
    id: int = Field(..., description="记录ID")
    create_time: datetime = Field(..., description="创建时间")
    update_time: Optional[datetime] = Field(None, description="更新时间")
    sn: str = Field(..., description="人员SN")
    name: Optional[str] = Field(None, description="人员姓名")
    similarity: Optional[float] = Field(None, description="相似度分数")
    image_url: str = Field(..., description="检测图片URL")

    class Config:
        from_attributes = True


class DetectionListResponse(BaseModel):
    """检测记录列表响应"""
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页记录数")
    total_pages: int = Field(..., description="总页数")
    records: List[DetectionRecordResponse] = Field(..., description="检测记录列表")


class DetectionRecordInfo(BaseModel):
    """检测记录信息"""
    id: int = Field(..., description="记录ID")
    create_time: datetime = Field(..., description="创建时间")
    detected_at: Optional[datetime] = Field(None, description="检测时间（兼容前端）")
    update_time: Optional[datetime] = Field(None, description="更新时间")
    sn: str = Field(..., description="人员SN")
    name: Optional[str] = Field(None, description="人员姓名")
    similarity: Optional[float] = Field(None, description="相似度分数")
    image_url: str = Field(..., description="检测图片URL")
    image_path: Optional[str] = Field(None, description="图片路径（兼容前端）")

    class Config:
        from_attributes = True

    def __init__(self, **data):
        # 设置兼容字段
        if 'create_time' in data and 'detected_at' not in data:
            data['detected_at'] = data['create_time']
        if 'image_url' in data and 'image_path' not in data:
            data['image_path'] = data['image_url']
        super().__init__(**data)


class DetectionQueryParams(BaseModel):
    """检测记录查询参数"""
    page: int = Field(1, ge=1, description="页码，从1开始")
    page_size: int = Field(20, ge=1, le=100, description="每页记录数，最大100")
    sn: Optional[str] = Field(None, description="按SN筛选")
    name: Optional[str] = Field(None, description="按姓名筛选")
    start_date: Optional[datetime] = Field(None, description="开始时间")
    end_date: Optional[datetime] = Field(None, description="结束时间")


class DetectionListResponseData(BaseModel):
    """检测记录列表响应数据"""
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页记录数")
    total_pages: int = Field(..., description="总页数")
    records: List[DetectionRecordInfo] = Field(..., description="检测记录列表")
    detections: List[DetectionRecordInfo] = Field(..., description="检测记录列表（兼容前端）")


class TrendDataPoint(BaseModel):
    """趋势数据点"""
    date: str = Field(..., description="日期，格式为YYYY-MM-DD")
    count: int = Field(..., description="检测次数")


class WeeklyTrendResponseData(BaseModel):
    """每周趋势响应数据"""
    trend_data: List[TrendDataPoint] = Field(..., description="最近七天的检测趋势数据")


class DetectionStatsResponseData(BaseModel):
    """检测统计响应数据"""
    total_detections: int = Field(..., description="总检测次数")
    unique_persons: int = Field(..., description="检测到的不同人员数")
    today_detections: int = Field(..., description="今日检测次数")
    recent_detections: List[DetectionRecordInfo] = Field(..., description="最近检测记录")


class PersonDetectionPieData(BaseModel):
    """人员检测饼图数据点"""
    name: str = Field(..., description="人员姓名")
    sn: str = Field(..., description="人员SN")
    count: int = Field(..., description="检测次数")
    percentage: float = Field(..., description="占比百分比")


class PersonDetectionPieResponseData(BaseModel):
    """人员检测饼图响应数据"""
    total_detections: int = Field(..., description="总检测次数")
    total_persons: int = Field(..., description="总人员数")
    pie_data: List[PersonDetectionPieData] = Field(..., description="饼图数据")


class HourlyDetectionData(BaseModel):
    """按小时检测数据点"""
    hour: int = Field(..., description="小时(0-23)")
    count: int = Field(..., description="检测次数")


class HourlyDetectionResponseData(BaseModel):
    """按小时检测响应数据"""
    hourly_data: List[HourlyDetectionData] = Field(..., description="24小时检测数据")


class TopPersonData(BaseModel):
    """检测排行数据点"""
    name: str = Field(..., description="人员姓名")
    sn: str = Field(..., description="人员SN")
    count: int = Field(..., description="检测次数")
    rank: int = Field(..., description="排名")


class TopPersonsResponseData(BaseModel):
    """检测排行响应数据"""
    top_persons: List[TopPersonData] = Field(..., description="检测排行数据")
