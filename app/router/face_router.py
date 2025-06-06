# app/router/face_router.py
from typing import List, Optional
from fastapi import (
    APIRouter, Depends, status, File, UploadFile, Form, 
    HTTPException, Request, Query, Path as FastApiPath
)
from fastapi.responses import JSONResponse, StreamingResponse

from app.schema.face_schema import (
    ApiResponse, FaceRegisterResponseData, FaceRecognitionResult,
    GetAllFacesResponseData, DeleteFaceResponseData, HealthCheckResponseData,
    UpdateFaceRequest, UpdateFaceResponseData, FaceRegisterRequest, FaceInfo
)
from app.service.face_service import FaceService

router = APIRouter()

# 依赖注入服务实例
def get_face_service(request: Request) -> FaceService:
    if not hasattr(request.app.state, 'face_service') or request.app.state.face_service is None:
        raise HTTPException(status_code=503, detail="人脸识别服务当前不可用。")
    return request.app.state.face_service

@router.get(
    "/health", 
    response_model=ApiResponse[HealthCheckResponseData], 
    summary="健康检查",
    tags=["系统"]
)
async def health_check():
    """检查服务是否正常运行。"""
    return ApiResponse(data=HealthCheckResponseData())

@router.post(
    "/faces", 
    response_model=ApiResponse[FaceRegisterResponseData], 
    status_code=status.HTTP_201_CREATED, 
    summary="注册一张新的人脸",
    tags=["人脸管理"]
)
async def register_face(
    form_data: FaceRegisterRequest = Depends(),
    image_file: UploadFile = File(..., description="上传的人脸图像文件。"),
    face_service: FaceService = Depends(get_face_service)
):
    """
    上传一张图片并关联到指定的人员信息（姓名和SN）。
    - **name**: 人员姓名
    - **sn**: 人员唯一标识 (如工号)
    - **image_file**: 包含清晰人脸的图片文件 (jpg, png等)
    """
    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传的图像文件为空。")
    
    face_info = await face_service.register_face(form_data.name, form_data.sn, image_bytes)
    return ApiResponse(data=FaceRegisterResponseData(face_info=face_info))

@router.post(
    "/recognize", 
    response_model=ApiResponse[List[FaceRecognitionResult]], 
    summary="识别图像中的人脸",
    tags=["人脸识别"]
)
async def recognize_face(
    image_file: UploadFile = File(..., description="待识别人脸的图像文件。"),
    face_service: FaceService = Depends(get_face_service)
):
    """
    上传一张图片，服务将识别图中的所有人脸，并返回最匹配的已知人员信息。
    """
    image_bytes = await image_file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传的图像文件为空。")
        
    results = await face_service.recognize_face(image_bytes)
    if not results:
        # 使用标准的HTTP状态码来表示“未找到”
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ApiResponse(code=404, msg="未在图像中匹配到任何已知人脸。", data=[]).model_dump()
        )
    return ApiResponse(data=results)

@router.get(
    "/faces", 
    response_model=ApiResponse[GetAllFacesResponseData], 
    summary="获取所有已注册的人脸信息",
    tags=["人脸管理"]
)
async def get_all_faces(face_service: FaceService = Depends(get_face_service)):
    """获取数据库中所有已注册人脸的元数据列表。"""
    faces = await face_service.get_all_faces()
    return ApiResponse(data=GetAllFacesResponseData(count=len(faces), faces=faces))

@router.get(
    "/faces/{sn}", 
    response_model=ApiResponse[List[FaceInfo]], 
    summary="根据SN获取特定人员的人脸信息",
    tags=["人脸管理"]
)
async def get_face_by_sn(
    sn: str = FastApiPath(..., description="要查询的人员唯一标识SN。", example="EMP001"),
    face_service: FaceService = Depends(get_face_service)
):
    """根据SN获取一个人的所有已注册人脸信息。"""
    faces = await face_service.get_face_by_sn(sn)
    return ApiResponse(data=faces)

@router.put(
    "/faces/{sn}", 
    response_model=ApiResponse[UpdateFaceResponseData], 
    summary="更新指定SN的人员信息",
    tags=["人脸管理"]
)
async def update_face_info(
    sn: str,
    update_data: UpdateFaceRequest,
    face_service: FaceService = Depends(get_face_service)
):
    """根据SN更新人员的姓名或其他信息。"""
    updated_face = await face_service.update_face_by_sn(sn, update_data)
    return ApiResponse(
        msg=f"成功更新SN为 '{sn}' 的人员信息。",
        data=UpdateFaceResponseData(sn=sn, updated_count=1, face_info=updated_face)
    )

@router.delete(
    "/faces/{sn}", 
    response_model=ApiResponse[DeleteFaceResponseData], 
    summary="删除指定SN的所有人脸记录",
    tags=["人脸管理"]
)
async def delete_face(
    sn: str, 
    face_service: FaceService = Depends(get_face_service)
):
    """根据SN删除一个人的所有相关人脸数据和图片文件。"""
    deleted_count = await face_service.delete_face_by_sn(sn)
    return ApiResponse(
        msg=f"成功删除SN为 '{sn}' 的 {deleted_count} 条人脸记录。", 
        data=DeleteFaceResponseData(sn=sn, deleted_count=deleted_count)
    )

@router.get(
    "/stream", 
    summary="实时视频流人脸识别与展示",
    tags=["人脸识别"],
    # 此接口的响应比较特殊，不在OpenAPI文档中定义模型
    responses={
        200: {
            "content": {"multipart/x-mixed-replace": {}},
            "description": "一个包含实时识别结果的JPEG视频流。可以直接在HTML的<img>标签中使用。",
        }
    }
)
async def live_recognition_stream(
    source: str = Query("0", description="视频源。可以是摄像头ID(如 '0') 或视频文件路径/URL。"),
    face_service: FaceService = Depends(get_face_service)
):
    """
    启动实时人脸识别视频流。
    
    服务会从指定的视频源（摄像头或文件）读取视频，
    对每一帧进行人脸识别，并将结果（边界框和姓名）绘制在画面上，
    然后将处理后的视频流式传输给客户端。
    
    **使用方法**:
    在HTML页面中，可以直接将此接口作为`<img>`标签的`src`属性值来播放视频流：
    `<img src="http://127.0.0.1:8000/api/face/stream?source=0" />`
    """
    return StreamingResponse(
        face_service.stream_video_recognition(video_source=source),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )