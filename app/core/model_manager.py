# app/core/model_manager.py
import os
import asyncio
import queue
from typing import Optional, List
from pathlib import Path
import cv2
from insightface.app import FaceAnalysis

from app.cfg.config import AppSettings, get_app_settings, BASE_DIR
from app.cfg.logging import app_logger


def create_face_analysis_model(settings: AppSettings) -> FaceAnalysis:
    """创建并初始化FaceAnalysis模型实例"""
    app_logger.info(f"创建InsightFace模型: {settings.insightface.model_pack_name}")
    
    try:
        # 创建模型实例
        model = FaceAnalysis(
            name=settings.insightface.model_pack_name,
            root=str(settings.insightface.home),
            providers=settings.insightface.providers
        )
        
        # 准备模型参数
        ctx_id = 0 if 'CUDAExecutionProvider' in settings.insightface.providers else -1
        det_thresh = settings.insightface.recognition_det_score_threshold
        det_size = tuple(settings.insightface.detection_size)
        
        # 初始化模型
        model.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        
        app_logger.info("InsightFace模型创建成功")
        return model
        
    except Exception as e:
        app_logger.error(f"创建InsightFace模型失败: {e}")
        if "CUDA" in str(e):
            app_logger.error("GPU环境配置错误，请检查CUDA、cuDNN版本或使用CPU模式")
        raise RuntimeError(f"模型创建失败: {e}") from e


class ModelManager:
    """模型管理器 - 单例模式管理模型池"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.settings = get_app_settings()
        self.pool_size = self.settings.app.max_concurrent_tasks
        self._pool: queue.Queue = queue.Queue(maxsize=self.pool_size)
        self._setup_environment()
        self._initialized = True
    
    def _setup_environment(self):
        """设置InsightFace环境变量"""
        model_home = str(self.settings.insightface.home)
        os.environ["INSIGHTFACE_HOME"] = model_home
        app_logger.info(f"设置INSIGHTFACE_HOME: {model_home}")
    
    async def load_models(self):
        """异步加载模型到池中"""
        if self._pool.full():
            app_logger.info("模型池已满，跳过加载")
            return
        
        try:
            app_logger.info(f"开始加载{self.pool_size}个模型实例到池中")
            
            # 并发创建模型
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(None, create_face_analysis_model, self.settings)
                for _ in range(self.pool_size)
            ]
            models = await asyncio.gather(*tasks)
            
            # 将模型放入池中
            for model in models:
                self._pool.put_nowait(model)
            
            app_logger.info(f"成功加载{self.pool_size}个模型到池中")
            
            # 执行启动自检
            await self._startup_test()
            
        except Exception as e:
            app_logger.error(f"模型加载失败: {e}")
            raise
    
    async def _startup_test(self):
        """启动自检"""
        test_image_path = BASE_DIR / "app" / "static" / "self_test_face.jpg"
        
        if not test_image_path.exists():
            app_logger.warning(f"未找到测试图片: {test_image_path}，跳过自检")
            return
        
        app_logger.info("开始启动自检")
        model = self.acquire_model()
        
        try:
            # 异步执行检测
            loop = asyncio.get_running_loop()
            faces = await loop.run_in_executor(
                None, self._detect_faces, test_image_path, model
            )
            
            if faces:
                app_logger.info(f"自检成功：检测到{len(faces)}张人脸")
            else:
                app_logger.error("自检失败：未检测到人脸")
                raise RuntimeError("模型自检失败")
                
        finally:
            self.release_model(model)
    
    def _detect_faces(self, image_path: Path, model: FaceAnalysis) -> List:
        """执行人脸检测"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        return model.get(img)
    
    def acquire_model(self) -> FaceAnalysis:
        """从池中获取模型（同步）"""
        app_logger.debug(f"获取模型 (可用: {self._pool.qsize()}/{self.pool_size})")
        model = self._pool.get()
        app_logger.debug(f"模型已获取 (剩余: {self._pool.qsize()}/{self.pool_size})")
        return model
    
    def release_model(self, model: FaceAnalysis):
        """释放模型到池中（同步）"""
        self._pool.put_nowait(model)
        app_logger.debug(f"模型已释放 (可用: {self._pool.qsize()}/{self.pool_size})")
    
    async def acquire_model_async(self) -> FaceAnalysis:
        """异步获取模型"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.acquire_model)
    
    async def release_model_async(self, model: FaceAnalysis):
        """异步释放模型"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.release_model, model)
    
    async def cleanup(self):
        """清理模型池资源"""
        app_logger.info("清理模型池资源")
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break
        app_logger.info("模型池已清空")


# 全局模型管理器实例
model_manager = ModelManager()


# 兼容性函数
async def load_models_on_startup():
    """启动时加载模型"""
    await model_manager.load_models()


async def release_models_on_shutdown():
    """关闭时释放模型"""
    await model_manager.cleanup()