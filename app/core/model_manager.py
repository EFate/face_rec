# app/core/model_manager.py
import os
import asyncio
import queue
from typing import Optional, List
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np

from app.cfg.config import AppSettings, get_app_settings, BASE_DIR
from app.cfg.logging import app_logger


def create_face_analysis_model(settings: AppSettings) -> FaceAnalysis:
    """
    根据提供的配置，构建并准备一个 FaceAnalysis 模型实例。
    """
    app_logger.info("--- 正在创建 InsightFace 模型实例 ---")
    app_logger.info(f"  - 模型包 (Name): '{settings.insightface.model_pack_name}'")
    app_logger.info(f"  - 模型根目录 (Root): '{settings.insightface.home}'")
    app_logger.info(f"  - 执行提供者 (Providers): {settings.insightface.providers}")

    try:
        model = FaceAnalysis(
            name=settings.insightface.model_pack_name,
            root=str(settings.insightface.home),
            providers=settings.insightface.providers
        )
    except Exception as e:
        app_logger.exception(f"❌ 创建 FaceAnalysis 实例失败: {e}")
        raise RuntimeError(f"创建 FaceAnalysis 实例时出错: {e}") from e

    ctx_id = 0 if 'CUDAExecutionProvider' in settings.insightface.providers else -1
    det_thresh = settings.insightface.recognition_det_score_threshold
    det_size = tuple(settings.insightface.detection_size)

    app_logger.info(
        f"  - 准备模型参数: 上下文ID(ctx_id)={ctx_id}, 检测阈值(det_thresh)={det_thresh}, 检测尺寸(det_size)={det_size}")

    try:
        model.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
    except Exception as e:
        app_logger.exception(f"❌ 模型 prepare() 阶段失败: {e}")
        raise RuntimeError(f"模型准备(prepare)失败: {e}") from e

    app_logger.info("--- InsightFace 模型实例创建并准备完毕 ---")
    return model


class ModelManager:
    _instance = None
    _pool: Optional[queue.Queue] = None
    pool_size: int = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            settings: AppSettings = get_app_settings()
            cls._instance.settings = settings
            cls._instance.pool_size = settings.app.max_concurrent_tasks
            cls._instance._pool = queue.Queue(maxsize=cls._instance.pool_size)
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
        异步加载 InsightFace 模型，并根据配置创建指定数量的实例放入模型池。
        """
        if self._pool.full():
            app_logger.info("模型池已满，跳过重复加载。")
            return

        try:
            app_logger.info(f"主进程正在创建并加载 {self.pool_size} 个 InsightFace 模型实例到池中...")
            loop = asyncio.get_running_loop()

            creation_tasks = [
                loop.run_in_executor(None, create_face_analysis_model, self.settings)
                for _ in range(self.pool_size)
            ]
            models = await asyncio.gather(*creation_tasks)

            for model in models:
                self._pool.put_nowait(model)

            app_logger.info(f"✅ {self.pool_size} 个模型已成功加载并放入池中。")
            await self._run_startup_self_test()

        except Exception as e:
            error_message = str(e)
            if "CUDA wasn't able to be loaded" in error_message or "LoadLibrary failed with error 126" in error_message:
                app_logger.critical("❌ ONNX Runtime无法加载CUDA执行提供者！这是GPU环境配置问题。")
                guide_message = (
                    "GPU调用失败，请严格按照以下步骤检查您的环境：\n"
                    "1. [驱动检查] 运行 `nvidia-smi` 命令，确保NVIDIA驱动已正确安装。\n"
                    "2. [版本匹配] 确认 `onnxruntime-gpu`、`CUDA Toolkit`、`cuDNN` 三者的版本是严格匹配的。请查阅ONNX Runtime官方文档获取版本对应关系。\n"
                    "3. [文件核对] 确保已将cuDNN压缩包中的 `bin`, `lib`, `include` 目录下的所有文件，正确复制到CUDA Toolkit的安装目录中。\n"
                    "4. [临时方案] 如果暂时无法解决，可修改配置文件，将执行提供者改为 `['CPUExecutionProvider']` 以使用CPU运行。"
                )
                app_logger.error(guide_message)
                raise RuntimeError("GPU环境配置错误，无法启动服务。请查看上方日志获取详细指引。") from e
            else:
                app_logger.exception(f"❌ InsightFace 模型加载过程中发生未知错误: {e}")
                raise RuntimeError(f"InsightFace 模型初始化失败，请检查日志。") from e

    async def _run_startup_self_test(self):
        """
        【启动自检函数。从模型池中获取一个模型进行测试，然后立即归还。
        """
        app_logger.info("--- 正在执行启动自检 ---")
        test_image_path = BASE_DIR / "app" / "static" / "self_test_face.jpg"
        if not test_image_path.exists():
            app_logger.warning(f"自检失败：未找到测试图片 {test_image_path}。跳过自检。")
            app_logger.warning(
                "强烈建议在 'app/static/' 目录下放置一张名为 'self_test_face.jpg' 的清晰人脸图用于启动自检。")
            return

        model = self.acquire_model()
        app_logger.info("自检程序已获取模型，开始测试...")
        try:
            loop = asyncio.get_running_loop()
            faces = await loop.run_in_executor(None, self._perform_detection, test_image_path, model)

            if not faces:
                app_logger.critical("❌ 【启动自检失败】模型无法在测试图片中检测到任何人脸！")
                error_guide = (
                    "这个问题非常严重，意味着模型无法正常工作。请按以下步骤排查：\n"
                    "1. [首要建议] 删除 'data/.insightface' 目录，然后重启程序，让系统重新下载所有模型文件。\n"
                    "2. [图片检查] 确认 'app/static/self_test_face.jpg' 是一张清晰、单人、正面的人脸图片。\n"
                    "3. [环境检查] 如果问题依旧，请核对您的 onnxruntime, CUDA, cuDNN 版本是否匹配。\n"
                    "4. [日志分析] 仔细查看上方的模型加载日志，寻找其他错误信息。"
                )
                app_logger.error(error_guide)
                raise RuntimeError("InsightFace模型自检失败，服务无法启动。")
            else:
                app_logger.info(f"✅ 【启动自检成功】在测试图片中成功检测到 {len(faces)} 张人脸。模型工作正常。")
        except Exception as e:
            app_logger.exception(f"❌ 自检过程中发生意外错误: {e}")
            raise RuntimeError(f"模型自检过程中出错: {e}") from e
        finally:
            self.release_model(model)
            app_logger.info("自检程序已释放模型。")

    def _perform_detection(self, image_path: Path, model: FaceAnalysis) -> List:
        """同步执行的检测函数，现在需要传入模型实例。"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"无法读取测试图片: {image_path}")
        return model.get(img)

    def acquire_model(self) -> FaceAnalysis:
        """从模型池中获取一个模型实例（同步，阻塞）。"""
        app_logger.debug(f"尝试从模型池中获取模型 (可用: {self._pool.qsize()}/{self.pool_size})...")
        model = self._pool.get()
        app_logger.debug(f"成功获取模型。 (剩余: {self._pool.qsize()}/{self.pool_size})")
        return model

    def release_model(self, model: FaceAnalysis):
        """将一个模型实例归还到池中（同步）。"""
        self._pool.put_nowait(model)
        app_logger.debug(f"模型已释放回池中。 (可用: {self._pool.qsize()}/{self.pool_size})")

    async def acquire_model_async(self) -> FaceAnalysis:
        """异步方式从模型池中获取模型。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.acquire_model)

    async def release_model_async(self, model: FaceAnalysis):
        """异步方式将模型归还到池中。"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.release_model, model)

    async def release_resources(self):
        """释放模型池中的所有资源。"""
        app_logger.info("正在清空模型池并释放资源...")
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break
        app_logger.info("模型池已清空。")


model_manager = ModelManager()


async def load_models_on_startup():
    await model_manager.load_insightface_model()


async def release_models_on_shutdown():
    await model_manager.release_resources()