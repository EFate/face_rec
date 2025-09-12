import os
import asyncio
import queue
import shutil
import subprocess
import sys
from typing import Optional, List
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from app.cfg.config import AppSettings, get_app_settings, BASE_DIR
from app.cfg.logging import app_logger

# 导入新的推理引擎
try:
    from app.inference.models import InferenceInput, InferenceOutput
    from app.inference.factory import InferenceEngineFactory
    from app.inference.base import BaseInferenceEngine
    NEW_INFERENCE_AVAILABLE = True
except ImportError as e:
    NEW_INFERENCE_AVAILABLE = False
    app_logger.warning(f"新推理引擎不可用，使用传统InsightFace: {e}")


def create_face_analysis_model(settings: AppSettings) -> FaceAnalysis:
    """创建并初始化FaceAnalysis模型实例"""
    app_logger.info(f"创建InsightFace模型: {settings.insightface.model_pack_name}")
    
    try:
        # 创建模型实例 - 设置download=True以便在需要时自动下载
        model = FaceAnalysis(
            name=settings.insightface.model_pack_name,
            root=str(settings.insightface.home),
            providers=settings.insightface.providers,
            download=True  # 启用自动下载，仅在模型不存在时下载
        )
        
        # 准备模型参数
        ctx_id = 0 if 'CUDAExecutionProvider' in settings.insightface.providers else -1
        det_thresh = settings.insightface.recognition_det_score_threshold
        det_size = tuple(settings.insightface.detection_size)
        
        # 初始化模型 - 如果模型不存在，这一步会触发下载
        model.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        
        app_logger.info(f"模型下载成功: {settings.insightface.model_pack_name}")
        return model
        
    except Exception as e:
        app_logger.error(f"创建InsightFace模型失败: {e}")
        if "CUDA" in str(e):
            app_logger.error("GPU环境配置错误，请检查CUDA、cuDNN版本或使用CPU模式")
        raise RuntimeError(f"模型创建失败: {e}") from e


class ModelManager:
    """模型管理器 - 单例模式管理模型池，支持异构算力卡"""
    
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
        self._use_new_inference = NEW_INFERENCE_AVAILABLE and hasattr(self.settings, 'inference')
        self._engine_config = None
        if self._use_new_inference:
            self._engine_config = self._build_engine_config()
            app_logger.info(f"使用新推理引擎，设备类型: {self.settings.inference.device_type}")
        else:
            app_logger.info("使用传统InsightFace推理引擎")
        self._setup_environment()
        self._initialized = True
    
    def _build_engine_config(self) -> dict:
        """构建推理引擎配置"""
        if not self._use_new_inference:
            return {}
        return self.settings.inference.get_device_config()
    
    def _setup_environment(self):
        """设置InsightFace环境变量"""
        model_home = str(self.settings.insightface.home)
        os.environ["INSIGHTFACE_HOME"] = model_home
        app_logger.info(f"设置INSIGHTFACE_HOME: {model_home}")
        
        # 确保模型目录存在
        Path(model_home).mkdir(parents=True, exist_ok=True)
    
    async def load_models(self):
        """异步加载模型到池中"""
        if self._use_new_inference:
            await self._load_new_inference_models()
        else:
            await self._load_insightface_models()
    
    async def _load_new_inference_models(self):
        """加载新推理引擎模型"""
        if self._pool.full():
            app_logger.info("模型池已满，跳过加载")
            return
        
        try:
            app_logger.info(f"开始加载{self.pool_size}个推理引擎实例到池中")
            app_logger.info(f"使用设备类型: {self.settings.inference.device_type}")
            
            # 先尝试创建一个引擎，确保模型文件存在
            first_engine = await self._create_inference_engine()
            self._pool.put_nowait(first_engine)
            app_logger.info("第一个推理引擎加载成功，继续加载剩余引擎")
            
            # 并发创建剩余引擎
            remaining_count = self.pool_size - 1
            if remaining_count > 0:
                tasks = [
                    self._create_inference_engine()
                    for _ in range(remaining_count)
                ]
                engines = await asyncio.gather(*tasks)
                
                # 将引擎放入池中
                for engine in engines:
                    self._pool.put_nowait(engine)
            
            app_logger.info(f"成功加载{self.pool_size}个推理引擎到池中")
            
            # 执行启动自检
            await self._startup_test()
            
        except Exception as e:
            app_logger.error(f"推理引擎加载失败: {e}")
            raise
    
    async def _load_insightface_models(self):
        """加载传统InsightFace模型"""
        if self._pool.full():
            app_logger.info("模型池已满，跳过加载")
            return
        
        try:
            app_logger.info(f"开始加载{self.pool_size}个InsightFace模型实例到池中")
            app_logger.info(f"使用模型: {self.settings.insightface.model_pack_name}")
            app_logger.info(f"使用提供者: {self.settings.insightface.providers}")
            
            # 先尝试创建一个模型，确保模型文件存在
            first_model = create_face_analysis_model(self.settings)
            self._pool.put_nowait(first_model)
            app_logger.info("第一个InsightFace模型加载成功，继续加载剩余模型")
            
            # 并发创建剩余模型
            remaining_count = self.pool_size - 1
            if remaining_count > 0:
                loop = asyncio.get_running_loop()
                tasks = [
                    loop.run_in_executor(None, create_face_analysis_model, self.settings)
                    for _ in range(remaining_count)
                ]
                models = await asyncio.gather(*tasks)
                
                # 将模型放入池中
                for model in models:
                    self._pool.put_nowait(model)
            
            app_logger.info(f"成功加载{self.pool_size}个InsightFace模型到池中")
            
            # 执行启动自检
            await self._startup_test()
            
        except Exception as e:
            app_logger.error(f"InsightFace模型加载失败: {e}")
            raise
    
    async def _create_inference_engine(self) -> BaseInferenceEngine:
        """创建推理引擎实例"""
        try:
            # 使用工厂创建引擎
            engine = InferenceEngineFactory.create_engine_with_fallback(
                self.settings.inference.device_type,
                self._engine_config
            )
            
            # 初始化引擎
            if not engine.initialize():
                raise RuntimeError("推理引擎初始化失败")
            
            # 加载模型
            if not engine.load_models():
                raise RuntimeError("模型加载失败")
            
            app_logger.info(f"推理引擎创建成功: {engine.device_type}")
            return engine
            
        except Exception as e:
            app_logger.error(f"创建推理引擎失败: {e}")
            raise
    
    async def _startup_test(self):
        """启动自检"""
        test_image_path = BASE_DIR / "app" / "static" / "self_test_face.jpg"
        
        if not test_image_path.exists():
            app_logger.warning(f"未找到测试图片: {test_image_path}，跳过自检")
            return
        
        app_logger.info("开始启动自检")
        
        if self._use_new_inference:
            await self._startup_test_new_inference(test_image_path)
        else:
            await self._startup_test_insightface(test_image_path)
    
    async def _startup_test_new_inference(self, test_image_path: Path):
        """新推理引擎启动自检"""
        engine = self.acquire_model()
        
        try:
            # 读取测试图像
            img = cv2.imread(str(test_image_path))
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {test_image_path}")
            
            # 创建输入数据
            input_data = InferenceInput(
                image=img,
                extract_embeddings=True,
                detection_threshold=self.settings.inference.recognition_det_score_threshold
            )
            
            # 执行推理
            output = engine.predict(input_data)
            
            if output.success and output.result.faces:
                app_logger.info(f"自检成功：检测到{len(output.result.faces)}张人脸")
            else:
                app_logger.error("自检失败：未检测到人脸")
                if not output.success:
                    app_logger.error(f"推理失败: {output.error_message}")
                raise RuntimeError("模型自检失败")
                
        finally:
            self.release_model(engine)
    
    async def _startup_test_insightface(self, test_image_path: Path):
        """InsightFace启动自检"""
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
    
    def acquire_model(self):
        """从池中获取模型（同步）"""
        app_logger.debug(f"获取模型 (可用: {self._pool.qsize()}/{self.pool_size})")
        
        # 如果池为空，等待或抛出异常，避免在同步方法中创建异步对象
        if self._pool.empty():
            app_logger.error("模型池为空，无法获取模型。请确保在启动时正确加载了模型。")
            raise RuntimeError("无法获取模型：模型池为空。请确保在启动时正确加载了模型。")
        
        model = self._pool.get()
        app_logger.debug(f"模型已获取 (剩余: {self._pool.qsize()}/{self.pool_size})")
        return model
    
    def release_model(self, model):
        """释放模型到池中（同步）"""
        if self._pool.full():
            app_logger.debug("模型池已满，丢弃模型")
            return
            
        self._pool.put_nowait(model)
        app_logger.debug(f"模型已释放 (可用: {self._pool.qsize()}/{self.pool_size})")
    
    async def acquire_model_async(self):
        """异步获取模型"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.acquire_model)
    
    async def release_model_async(self, model):
        """异步释放模型"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.release_model, model)
    
    async def cleanup(self):
        """清理模型池资源"""
        app_logger.info("清理模型池资源")
        
        # 清理所有模型/引擎
        while not self._pool.empty():
            try:
                model = self._pool.get_nowait()
                if self._use_new_inference and hasattr(model, 'cleanup'):
                    # 清理推理引擎
                    model.cleanup()
            except queue.Empty:
                break
            except Exception as e:
                app_logger.error(f"清理模型时出错: {e}")
        
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