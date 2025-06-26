# app/core/model_manager.py
import os
import asyncio
from typing import Optional
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np

from app.cfg.config import AppSettings, get_app_settings, BASE_DIR
from app.cfg.logging import app_logger


def create_face_analysis_model(settings: AppSettings) -> FaceAnalysis:
    """
    【统一模型创建函数】
    根据提供的配置，构建并准备一个 FaceAnalysis 模型实例。
    主进程和子进程都应调用此函数以确保一致性。
    """
    # 1. 记录关键配置
    app_logger.info("--- 正在创建 InsightFace 模型实例 ---")
    app_logger.info(f"  - 模型包 (Name): '{settings.insightface.model_pack_name}'")
    app_logger.info(f"  - 模型根目录 (Root): '{settings.insightface.home}'")
    app_logger.info(f"  - 执行提供者 (Providers): {settings.insightface.providers}")

    # 2. 构建模型
    try:
        model = FaceAnalysis(
            name=settings.insightface.model_pack_name,
            root=str(settings.insightface.home),
            providers=settings.insightface.providers
        )
    except Exception as e:
        app_logger.exception(f"❌ 创建 FaceAnalysis 实例失败: {e}")
        raise RuntimeError(f"创建 FaceAnalysis 实例时出错: {e}") from e

    # 3. 准备模型
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
        主要用于主进程的静态图片识别功能。
        """
        if self._model is None:
            try:
                app_logger.info("主进程正在加载 InsightFace 模型...")
                loop = asyncio.get_running_loop()
                model = await loop.run_in_executor(
                    None,  # 使用默认的线程池执行器
                    create_face_analysis_model,
                    self.settings
                )
                self._model = model
                app_logger.info("✅ 主进程 InsightFace 模型加载并准备成功。")

                # ---【核心修复】执行启动自检 ---
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
        else:
            app_logger.info("主进程 InsightFace 模型已加载，跳过重复加载。")

    async def _run_startup_self_test(self):
        """
        【新增】启动自检函数。
        使用一张内置的测试图片来验证模型是否能正常工作。
        """
        app_logger.info("--- 正在执行启动自检 ---")
        test_image_path = BASE_DIR / "app" / "static" / "self_test_face.jpg"
        if not test_image_path.exists():
            app_logger.warning(f"自检失败：未找到测试图片 {test_image_path}。跳过自检。")
            app_logger.warning(
                "强烈建议在 'app/static/' 目录下放置一张名为 'self_test_face.jpg' 的清晰人脸图用于启动自检。")
            return

        try:
            loop = asyncio.get_running_loop()
            # 在线程池中执行IO和CPU密集型操作
            faces = await loop.run_in_executor(
                None, self._perform_detection, test_image_path
            )

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

    def _perform_detection(self, image_path: Path) -> list:
        """同步执行的检测函数"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"无法读取测试图片: {image_path}")
        return self._model.get(img)

    def get_model(self) -> FaceAnalysis:
        """获取已加载的模型实例。如果模型未加载，则抛出异常。"""
        if self._model is None:
            raise RuntimeError("主进程的 InsightFace 模型尚未加载。请在应用启动时调用 load_insightface_model。")
        return self._model

    async def release_resources(self):
        """
        释放模型资源。
        """
        app_logger.info("正在释放主进程模型资源...")
        self._model = None
        app_logger.info("主进程模型资源已释放。")


# 单例实例
model_manager = ModelManager()


# 用于FastAPI启动时加载模型
async def load_models_on_startup():
    await model_manager.load_insightface_model()


# 用于FastAPI关闭时释放资源
async def release_models_on_shutdown():
    await model_manager.release_resources()