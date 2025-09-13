# app/main.py
import asyncio
import queue
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException

from app.cfg.config import get_app_settings
from app.cfg.logging import app_logger, setup_logging
from app.core.model_manager import model_manager, load_models_on_startup, release_models_on_shutdown
from app.core.database.database import init_db
from app.router.face_router import router as face_router
from app.router.detection_router import router as detection_router
from app.service.face_service import FaceService
from app.schema.face_schema import ApiResponse
from app.cfg.config import DATA_DIR
# --- 导入新的服务 ---
from app.core.result_processor import ResultPersistenceProcessor
from app.cfg.mqtt_manager import MQTTManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器
    """
    # --- 启动任务 ---
    settings = get_app_settings()
    setup_logging(settings)  # 设置日志配置
    app.state.settings = settings
    app_logger.info("应用程序启动... 开始执行启动任务。")
    
    app_logger.info("应用启动，开始初始化数据库...")
    init_db()
    app_logger.info("数据库初始化完成。")

    # 1. 加载机器学习模型到模型池
    await load_models_on_startup()
    app_logger.info("✅ 模型池加载完成。")

    # 2. 创建用于结果持久化的共享队列
    result_persistence_queue = queue.Queue(maxsize=256)

    try:
        # 3. 创建MQTT管理器
        mqtt_manager = MQTTManager(settings.mqtt, settings.server)
        app.state.mqtt_manager = mqtt_manager
        
        # 4. 创建结果持久化服务
        result_service = ResultPersistenceProcessor(settings=settings, result_queue=result_persistence_queue, mqtt_manager=mqtt_manager)
        app.state.result_service = result_service
        
        # 5. 初始化FaceService(传入所有必需参数)
        face_service = FaceService(
            settings=settings,
            model_manager=model_manager,
            result_queue=result_persistence_queue,
            mqtt_manager=mqtt_manager
        )
        app.state.face_service = face_service
        
        # 初始化FaceService
        await face_service.initialize()
        app_logger.info("✅ FaceService 初始化完成。")
        
        # 启动MQTT管理器
        mqtt_manager.start()
        app_logger.info("✅ MQTT管理器已启动。")
        
        # 启动结果持久化服务
        result_service.start()
        app_logger.info("✅ 结果持久化服务已启动。")
        
    except Exception as e:
        app_logger.error(f"服务初始化失败: {str(e)}", exc_info=True)
        raise
    app_logger.info("✅ FaceService 初始化完成。")

    # 6. 启动周期性清理过期流的后台任务
    cleanup_task = asyncio.create_task(face_service.cleanup_expired_streams())
    app.state.cleanup_task = cleanup_task
    app_logger.info("✅ 启动了周期性清理过期视频流的后台任务。")

    app_logger.info("🎉 所有启动任务完成，应用程序准备就绪。")
    yield
    # --- 关闭任务 ---
    app_logger.info("应用程序正在关闭... 开始执行清理任务。")

    # 1. 停止周期性任务
    if hasattr(app.state, 'cleanup_task') and not app.state.cleanup_task.done():
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            pass
        app_logger.info("✅ 视频流清理任务已取消。")

    # 2. 停止MQTT管理器
    if hasattr(app.state, 'mqtt_manager'):
        app.state.mqtt_manager.stop()

    # 3. 停止结果持久化服务 (在停止视频流之前)
    if hasattr(app.state, 'result_service'):
        app.state.result_service.stop()

    # 4. 关闭所有活动的视频流
    if hasattr(app.state, 'face_service'):
        try:
            await app.state.face_service.stop_all_streams()
            app_logger.info("✅ 所有视频流已停止。")
        except Exception as e:
            app_logger.error(f"停止视频流时出错: {e}", exc_info=True)

    # 5. 释放模型资源
    try:
        await release_models_on_shutdown()
        app_logger.info("✅ 模型资源已释放。")
    except Exception as e:
        app_logger.error(f"释放模型资源时出错: {e}", exc_info=True)

    app_logger.info("✅ 所有清理任务完成。")


def create_app() -> FastAPI:
    app_settings = get_app_settings()
    app = FastAPI(
        lifespan=lifespan,
        title=app_settings.app.title,
        description=app_settings.app.description,
        version=app_settings.app.version,
        debug=app_settings.app.debug,
        docs_url=None,
        redoc_url=None,
    )

    @app.middleware("http")
    async def log_request_response(request: Request, call_next):
        """请求-响应日志中间件，记录所有HTTP请求的详细信息"""
        # 记录请求信息
        start_time = asyncio.get_event_loop().time()
        app_logger.info(
            f"请求: {request.method} {request.url.path}",
            extra={
                "client": request.client.host if request.client else "unknown",
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params)
            }
        )
        
        # 处理请求
        try:
            response = await call_next(request)
            # 计算请求处理时间
            process_time = asyncio.get_event_loop().time() - start_time
            
            # 记录响应信息
            app_logger.info(
                f"响应: {request.method} {request.url.path} {response.status_code} ({process_time:.3f}s)",
                extra={
                    "client": request.client.host if request.client else "unknown",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            return response
        except Exception as e:
            # 记录异常信息
            process_time = asyncio.get_event_loop().time() - start_time
            app_logger.error(
                f"请求处理异常: {request.method} {request.url.path} - {str(e)} ({process_time:.3f}s)",
                exc_info=True,
                extra={
                    "client": request.client.host if request.client else "unknown",
                    "method": request.method,
                    "path": request.url.path,
                    "process_time": process_time
                }
            )
            raise


    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content=ApiResponse(code=exc.status_code, msg=exc.detail).model_dump())

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        # 检查是否是流媒体相关的异常
        if "stream" in str(request.url).lower() or "feed" in str(request.url).lower():
            app_logger.warning(f"流媒体请求异常: {request.url} - {exc}")
            # 对于流媒体请求，不返回JSON响应，让客户端自然断开
            return JSONResponse(status_code=500, content=ApiResponse(code=500, msg="流媒体服务暂时不可用").model_dump())
        else:
            app_logger.exception(f"未处理的服务器内部错误: {exc}")
            return JSONResponse(status_code=500, content=ApiResponse(code=500, msg="服务器内部错误").model_dump())

    app.include_router(face_router, prefix="/api/face", tags=["人脸服务"])
    app.include_router(detection_router, prefix="/api/detection", tags=["检测结果信息"])

    STATIC_FILES_DIR = Path("app/static")
    if STATIC_FILES_DIR.exists(): app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")
    if DATA_DIR.exists(): app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
    if DATA_DIR.exists():
        faces_dir = DATA_DIR / "faces"
        if faces_dir.exists():
            app.mount("/api/static/faces", StaticFiles(directory=faces_dir), name="faces")

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - API Docs",
                                   swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
                                   swagger_css_url="/static/swagger-ui/swagger-ui.css")

    return app


app = create_app()