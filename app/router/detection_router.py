# app/router/detection_router.py
from typing import Optional
from datetime import datetime, date
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.core.database.database import get_db_session
from app.core.database.query_ops import DetectionQueryOps
from app.schema.face_schema import ApiResponse
from app.schema.detection_schema import (
    DetectionListResponseData, DetectionStatsResponseData,
    DetectionQueryParams, DetectionRecordInfo, WeeklyTrendResponseData,
    PersonDetectionPieResponseData, HourlyDetectionResponseData, TopPersonsResponseData
)

router = APIRouter()


@router.get(
    "/records",
    response_model=ApiResponse[DetectionListResponseData],
    summary="获取检测记录列表",
    tags=["检测记录"]
)
async def get_detection_records(
        request: Request,
        page: int = Query(1, ge=1, description="页码，从1开始"),
        page_size: int = Query(20, ge=1, le=100, description="每页记录数，最大100"),
        sn: Optional[str] = Query(None, description="按SN筛选"),
        name: Optional[str] = Query(None, description="按姓名筛选"),
        start_date: Optional[datetime] = Query(None, description="开始时间"),
        end_date: Optional[datetime] = Query(None, description="结束时间"),
        db: Session = Depends(get_db_session)
):
    """
    分页查询检测记录
    - **page**: 页码，从1开始
    - **page_size**: 每页记录数，最大100
    - **sn**: 按人员SN筛选（可选）
    - **name**: 按人员姓名筛选（可选）
    - **start_date**: 开始时间筛选（可选）
    - **end_date**: 结束时间筛选（可选）
    """
    query_params = DetectionQueryParams(
        page=page,
        page_size=page_size,
        sn=sn,
        name=name,
        start_date=start_date,
        end_date=end_date
    )

    query_service = DetectionQueryOps(db)
    result = query_service.get_detection_records(
        page=page,
        page_size=page_size,
        name=name,
        sn=sn,
        start_date=start_date,
        end_date=end_date
    )

    # 从环境变量获取HOST_IP和SERVER_PORT构建完整URL
    import os
    host_ip = os.getenv('HOST__IP', '127.0.0.1')
    server_port = os.getenv('SERVER__PORT', '12010')
    base_url = f"http://{host_ip}:{server_port}"

    for record in result.records:
        if record.image_url and not record.image_url.startswith('http'):
            # 统一处理图片URL：确保所有图片URL都是完整的HTTP地址
            if record.image_url.startswith('/data/'):
                record.image_url = f"{base_url}{record.image_url}"
            else:
                # 处理相对路径，提取文件名并构建正确路径
                filename = record.image_url.split('/')[-1] if '/' in record.image_url else record.image_url
                record.image_url = f"{base_url}/data/detected_imgs/{filename}"

    # 直接使用DetectionListResponseData格式返回
    response_data = DetectionListResponseData(
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
        records=result.records,
        detections=result.records  # Dashboard组件需要detections字段
    )

    return ApiResponse(data=response_data)


@router.get(
    "/stats",
    response_model=ApiResponse[DetectionStatsResponseData],
    summary="获取检测统计信息",
    tags=["检测记录"]
)
async def get_detection_stats(
        request: Request,
        db: Session = Depends(get_db_session)
):
    """获取检测统计信息，包括总检测次数、不同人员数、今日检测次数等"""
    query_service = DetectionQueryOps(db)
    result = query_service.get_detection_stats()

    # 从环境变量获取HOST_IP和SERVER_PORT构建完整URL
    import os
    host_ip = os.getenv('HOST__IP', '127.0.0.1')
    server_port = os.getenv('SERVER__PORT', '12010')
    base_url = f"http://{host_ip}:{server_port}"

    # 处理统计数据中的图片URL
    for record in result.recent_detections:
        if record.image_url and not record.image_url.startswith('http'):
            if record.image_url.startswith('/data/'):
                record.image_url = f"{base_url}{record.image_url}"
            else:
                filename = record.image_url.split('/')[-1] if '/' in record.image_url else record.image_url
                record.image_url = f"{base_url}/data/detected_imgs/{filename}"

    return ApiResponse(data=result)


@router.get(
    "/person-pie",
    response_model=ApiResponse[PersonDetectionPieResponseData],
    summary="获取人员检测饼图数据",
    tags=["检测统计"]
)
async def get_person_detection_pie(
        request: Request,
        db: Session = Depends(get_db_session)
):
    """获取不同检测人员的占比数据，用于饼图展示"""
    query_service = DetectionQueryOps(db)
    result = query_service.get_person_detection_pie_data()
    
    return ApiResponse(data=result)


@router.get(
    "/hourly-trend",
    response_model=ApiResponse[HourlyDetectionResponseData],
    summary="获取按小时检测趋势",
    tags=["检测统计"]
)
async def get_hourly_detection_trend(
        request: Request,
        db: Session = Depends(get_db_session)
):
    """获取24小时检测活跃度数据，用于热力图或柱状图展示"""
    query_service = DetectionQueryOps(db)
    result = query_service.get_hourly_detection_data()
    
    return ApiResponse(data=result)


@router.get(
    "/top-persons",
    response_model=ApiResponse[TopPersonsResponseData],
    summary="获取检测排行榜",
    tags=["检测统计"]
)
async def get_top_persons(
        request: Request,
        limit: int = Query(10, ge=1, le=50, description="返回排行数量，最大50"),
        db: Session = Depends(get_db_session)
):
    """获取检测次数最多的人员排行，用于排行榜展示"""
    query_service = DetectionQueryOps(db)
    result = query_service.get_top_persons_data(limit=limit)
    
    return ApiResponse(data=result)


@router.get(
    "/weekly-trend",
    response_model=ApiResponse[WeeklyTrendResponseData],
    summary="获取最近七天的检测趋势",
    tags=["检测统计"]
)
async def get_weekly_trend(
        request: Request,
        db: Session = Depends(get_db_session)
):
    """获取最近七天的检测趋势数据，用于仪表盘展示"""
    query_service = DetectionQueryOps(db)
    result = query_service.get_weekly_trend()

    return ApiResponse(data=result)


@router.get(
    "/records/{record_id}",
    response_model=ApiResponse[DetectionRecordInfo],
    summary="获取单个检测记录详情",
    tags=["检测记录"]
)
async def get_detection_record(
        record_id: int,
        request: Request,
        db: Session = Depends(get_db_session)
):
    """根据记录ID获取单个检测记录的详细信息"""
    query_service = DetectionQueryOps(db)
    record = query_service.get_detection_record_by_id(record_id)

    if not record:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="检测记录不存在")

    # 从环境变量获取HOST_IP和SERVER_PORT构建完整URL
    import os
    host_ip = os.getenv('HOST__IP', '127.0.0.1')
    server_port = os.getenv('SERVER__PORT', '12010')
    base_url = f"http://{host_ip}:{server_port}"

    if record.image_url and not record.image_url.startswith('http'):
        # 统一处理图片URL：确保所有图片URL都是完整的HTTP地址
        if record.image_url.startswith('/data/'):
            record.image_url = f"{base_url}{record.image_url}"
        else:
            # 处理相对路径，提取文件名并构建正确路径
            filename = record.image_url.split('/')[-1] if '/' in record.image_url else record.image_url
            record.image_url = f"{base_url}/data/detected_imgs/{filename}"

    return ApiResponse(data=record)