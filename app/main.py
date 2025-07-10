# app/main.py
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import HTTPException

from app.cfg.config import AppSettings, get_app_settings
from app.cfg.logging import app_logger
from app.core.model_manager import model_manager, load_models_on_startup, release_models_on_shutdown
from app.router.face_router import router as face_router
from app.service.face_service import FaceService
from app.schema.face_schema import ApiResponse
from app.cfg.config import DATA_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器 (已更新以匹配新的服务)
    """
    # --- 启动任务 ---
    app_logger.info("应用程序启动... 开始执行启动任务。")
    settings = get_app_settings()
    app.state.settings = settings

    # 1. 加载主进程所需的机器学习模型
    await load_models_on_startup()
    app_logger.info("✅ 主进程模型加载完成。")

    # 2. 初始化服务，注入模型和配置
    face_service = FaceService(settings=settings, model=model_manager.get_model())
    app.state.face_service = face_service

    # 3. 【修正】调用服务自身的初始化方法，而不是旧的缓存加载方法
    await face_service.initialize()
    app_logger.info("✅ FaceService 初始化完成。")

    # 4. 启动周期性清理过期流的后台任务
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

    # 2. 优雅地关闭所有活动的视频流
    if hasattr(app.state, 'face_service'):
        await app.state.face_service.stop_all_streams()

    # 3. 释放模型资源
    await release_models_on_shutdown()

    app_logger.info("✅ 所有清理任务完成。")


# create_app 函数和其余部分保持不变
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

    # 注册全局异常处理器
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code,
                            content=ApiResponse(code=exc.status_code, msg=exc.detail).model_dump())

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        app_logger.exception(f"未处理的服务器内部错误: {exc}")
        return JSONResponse(status_code=500, content=ApiResponse(code=500, msg="服务器内部错误").model_dump())

    # 包含API路由
    app.include_router(face_router, prefix="/api/face", tags=["人脸服务"])

    # 挂载静态文件和数据目录
    STATIC_FILES_DIR = Path("app/static")
    if STATIC_FILES_DIR.exists(): app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")
    if DATA_DIR.exists(): app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

    # 自定义Swagger UI
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " - API Docs",
                                   swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
                                   swagger_css_url="/static/swagger-ui/swagger-ui.css")

    return app


app = create_app()