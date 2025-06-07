# app/core/model_manager.py
import os
import asyncio
from typing import Optional
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis

from app.cfg.config import AppSettings, get_app_settings
from app.cfg.logging import app_logger

class ModelManager:
    _instance = None
    _model: Optional[FaceAnalysis] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.settings: AppSettings = get_app_settings()
            cls._instance._initialize_insightface_env()
        return cls._instance

    def _initialize_insightface_env(self):
        """初始化 InsightFace 相关的环境变量和目录。"""
        model_home = str(self.settings.insightface.home)
        os.environ["INSIGHTFACE_HOME"] = model_home
        Path(model_home).mkdir(parents=True, exist_ok=True)
        app_logger.info(f"InsightFace HOME 环境变量设置为: {model_home}")

    async def load_insightface_model(self):
        """
        异步加载 InsightFace 模型。此方法应在应用启动时调用。
        """
        if self._model is None:
            try:
                app_logger.info(
                    f"正在加载 InsightFace 模型包: '{self.settings.insightface.model_pack_name}' "
                    f"使用 providers: {self.settings.insightface.providers}"
                )

                loop = asyncio.get_running_loop()
                # 将阻塞的模型加载操作放入线程池中执行
                model = await loop.run_in_executor(
                    None,  # 使用默认的线程池执行器
                    self._build_and_prepare_model
                )
                self._model = model
                app_logger.info("✅ InsightFace 模型加载并准备成功。")
            except Exception as e:
                app_logger.exception(f"❌ InsightFace 模型加载失败: {e}")
                raise RuntimeError(f"InsightFace 模型初始化失败: {e}")
        else:
            app_logger.info("InsightFace 模型已加载，跳过重复加载。")

    def _build_and_prepare_model(self) -> FaceAnalysis:
        """同步执行的模型构建和准备函数。"""
        # 1. 构建模型
        model = FaceAnalysis(
            name=self.settings.insightface.model_pack_name,
            root=self.settings.insightface.home,
            providers=self.settings.insightface.providers
        )
        # 2. 准备模型 (指定上下文ID和检测尺寸)
        # ctx_id < 0 表示 CPU, >= 0 表示 GPU 设备 ID
        # 如果 providers 中有 'CUDAExecutionProvider'，则 ctx_id 设为 0 使用第一个GPU
        # 否则设为 -1 使用CPU
        ctx_id = 0 if 'CUDAExecutionProvider' in self.settings.insightface.providers else -1
        model.prepare(ctx_id=ctx_id, det_size=(640, 640))
        return model

    def get_model(self) -> FaceAnalysis:
        """获取已加载的模型实例。如果模型未加载，则抛出异常。"""
        if self._model is None:
            raise RuntimeError("InsightFace 模型尚未加载。请在应用启动时调用 load_insightface_model。")
        return self._model

    async def release_resources(self):
        """
        释放模型资源。
        """
        app_logger.info("正在释放模型资源...")
        self._model = None
        app_logger.info("模型资源已释放。")

# 单例实例
model_manager = ModelManager()

# 用于FastAPI启动时加载模型
async def load_models_on_startup():
    await model_manager.load_insightface_model()

# 用于FastAPI关闭时释放资源
async def release_models_on_shutdown():
    await model_manager.release_resources()