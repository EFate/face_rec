# app/core/model_manager.py
import os
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from deepface import DeepFace
from app.cfg.config import AppSettings, get_app_settings, DEEPFACE_MODELS_DIR

class ModelManager:
    _instance = None
    _deepface_model = None # 用于存储DeepFace模型实例或指示其已加载

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.settings: AppSettings = get_app_settings()
            cls._instance._initialize_deepface_env()
        return cls._instance

    def _initialize_deepface_env(self):
        """初始化DeepFace相关的环境变量和目录。"""
        os.environ["DEEPFACE_HOME"] = str(self.settings.deepface.home)
        print(f"DeepFace HOME 环境变量设置为: {os.environ['DEEPFACE_HOME']}")
        deepface_weights_dir = Path(os.environ["DEEPFACE_HOME"]) / ".deepface" / "weights"
        deepface_weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"DeepFace 模型权重目录已确保存在: {deepface_weights_dir}")

    async def load_deepface_model(self):
        """
        异步加载DeepFace模型。确保模型只加载一次。
        """
        if self._deepface_model is None:
            try:
                print(f"正在加载DeepFace模型: {self.settings.deepface.model_name}...")
                
                ## 优化点：将阻塞的模型加载操作放入线程池中执行
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    DeepFace.build_model,
                    self.settings.deepface.model_name
                )
                
                self._deepface_model = True # 标记为已加载
                print("DeepFace模型加载成功。")
            except Exception as e:
                print(f"DeepFace模型加载失败: {e}")
                raise RuntimeError(f"DeepFace模型初始化失败: {e}")
        else:
            print("DeepFace模型已加载，跳过重复加载。")

    async def release_resources(self):
        """
        异步释放模型资源（如果DeepFace提供了相应的API）。
        """
        print("尝试释放模型资源...")
        print("模型资源释放完成（如果DeepFace支持）。")

# 单例实例
model_manager = ModelManager()

# 用于FastAPI启动时加载模型
async def load_models_on_startup():
    await model_manager.load_deepface_model()

# 用于FastAPI关闭时释放资源
async def release_models_on_shutdown():
    await model_manager.release_resources()