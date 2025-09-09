# app/inference/devices/hailo8/engine.py
"""
Hailo8推理引擎实现
基于Degirum库
"""

import time
import asyncio
import os
import atexit
import signal
import psutil
from typing import List, Optional, Dict, Any
import numpy as np
import cv2

from ...base import BaseInferenceEngine
from ...models import InferenceInput, InferenceOutput, InferenceResult, FaceDetection
from app.cfg.logging import app_logger

# 导入Degirum库
try:
    import degirum as dg
    DEGIRUM_AVAILABLE = True
except ImportError:
    DEGIRUM_AVAILABLE = False
    app_logger.warning("Degirum库未安装，Hailo8推理引擎不可用")


class Hailo8InferenceEngine(BaseInferenceEngine):
    """Hailo8推理引擎实现"""
    
    def __init__(self, device_type: str, config: Dict[str, Any]):
        """
        初始化Hailo8推理引擎
        
        Args:
            device_type: 设备类型
            config: 配置参数，包含：
                - zoo_path: 模型仓库路径
                - detection_model: 检测模型名称
                - recognition_model: 识别模型名称
                - detection_size: 检测模型输入尺寸
                - recognition_size: 识别模型输入尺寸
        """
        super().__init__(device_type, config)
        
        if not DEGIRUM_AVAILABLE:
            raise ImportError("Degirum库未安装，无法使用Hailo8推理引擎")
        
        self.zoo_path = config.get("zoo_path", "./data/zoo")
        self.detection_model_name = config.get("detection_model", "scrfd_10g--640x640_quant_hailort_hailo8_1")
        self.recognition_model_name = config.get("recognition_model", "arcface_mobilefacenet--112x112_quant_hailort_hailo8_1")
        self.detection_size = tuple(config.get("detection_size", [640, 640]))
        self.recognition_size = tuple(config.get("recognition_size", [112, 112]))
        
        # 模型实例
        self.detection_model = None
        self.recognition_model = None
        self.zoo = None
        
        # 注册清理函数
        atexit.register(self._cleanup_on_exit)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    async def initialize(self) -> bool:
        """
        初始化推理引擎
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            app_logger.info(f"初始化Hailo8推理引擎: {self.zoo_path}")
            
            # 检查模型仓库路径
            if not os.path.exists(self.zoo_path):
                raise FileNotFoundError(f"模型仓库路径不存在: {self.zoo_path}")
            
            # 检查模型文件
            detection_model_path = os.path.join(self.zoo_path, self.detection_model_name)
            recognition_model_path = os.path.join(self.zoo_path, self.recognition_model_name)
            
            if not os.path.exists(detection_model_path):
                raise FileNotFoundError(f"检测模型路径不存在: {detection_model_path}")
            
            if not os.path.exists(recognition_model_path):
                raise FileNotFoundError(f"识别模型路径不存在: {recognition_model_path}")
            
            self._initialized = True
            self._initialization_time = time.time()
            app_logger.info("Hailo8推理引擎初始化成功")
            return True
            
        except Exception as e:
            app_logger.error(f"Hailo8推理引擎初始化失败: {e}")
            return False
    
    async def load_models(self) -> bool:
        """
        加载Hailo8模型
        
        Returns:
            bool: 模型加载是否成功
        """
        try:
            if not self._initialized:
                raise RuntimeError("推理引擎未初始化")
            
            app_logger.info(f"加载Hailo8模型: {self.detection_model_name}, {self.recognition_model_name}")
            
            # 连接模型仓库
            self.zoo = dg.connect(
                inference_host_address=dg.LOCAL,
                zoo_url=f"file://{self.zoo_path}"
            )
            
            # 加载检测模型
            self.detection_model = self.zoo.load_model(self.detection_model_name)
            app_logger.info(f"检测模型加载成功: {self.detection_model_name}")
            
            # 加载识别模型
            self.recognition_model = self.zoo.load_model(self.recognition_model_name)
            app_logger.info(f"识别模型加载成功: {self.recognition_model_name}")
            
            self._models_loaded = True
            self._model_loading_time = time.time()
            app_logger.info("Hailo8模型加载成功")
            return True
            
        except Exception as e:
            app_logger.error(f"Hailo8模型加载失败: {e}")
            return False
    
    async def predict(self, input_data: InferenceInput) -> InferenceOutput:
        """
        执行人脸检测和识别
        
        Args:
            input_data: 输入数据
            
        Returns:
            InferenceOutput: 推理结果
        """
        start_time = time.time()
        
        try:
            if not self._models_loaded:
                raise RuntimeError("模型未加载")
            
            if not self._validate_input(input_data):
                raise ValueError("输入数据验证失败")
            
            # 预处理图像
            processed_image = self._preprocess_image(input_data.image, self.detection_size)
            
            # 执行人脸检测
            detection_results = self.detection_model.predict(processed_image).results
            
            face_detections = []
            for result in detection_results:
                # 检查检测置信度
                if result.get('score', 0) < input_data.detection_threshold:
                    continue
                
                # 获取边界框
                bbox = result.get('bbox', [0, 0, 0, 0])
                
                # 后处理边界框坐标
                original_size = (input_data.image.shape[1], input_data.image.shape[0])
                processed_bbox = self._postprocess_bbox(bbox, original_size, self.detection_size)
                
                # 获取关键点
                landmarks = None
                if 'landmarks' in result:
                    raw_landmarks = result['landmarks']
                    # 转换landmarks格式：从dict转换为List[List[float]]
                    if isinstance(raw_landmarks, list):
                        landmarks = []
                        for landmark in raw_landmarks:
                            if isinstance(landmark, dict) and 'x' in landmark and 'y' in landmark:
                                landmarks.append([float(landmark['x']), float(landmark['y'])])
                            elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                                landmarks.append([float(landmark[0]), float(landmark[1])])
                    elif isinstance(raw_landmarks, dict):
                        # 如果是单个dict，尝试提取坐标
                        if 'x' in raw_landmarks and 'y' in raw_landmarks:
                            landmarks = [[float(raw_landmarks['x']), float(raw_landmarks['y'])]]
                
                # 提取人脸特征向量
                embedding = None
                if input_data.extract_embeddings:
                    embedding = await self._extract_face_embedding(
                        input_data.image, processed_bbox, landmarks
                    )
                
                # 创建人脸检测结果
                face_detection = self._create_face_detection(
                    bbox=processed_bbox,
                    confidence=float(result.get('score', 0)),
                    landmarks=landmarks,
                    embedding=embedding
                )
                face_detections.append(face_detection)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 创建推理结果
            result = self._create_inference_result(face_detections, processing_time)
            
            # 创建输出
            output = self._create_inference_output(result, success=True)
            
            app_logger.debug(f"Hailo8推理完成: 检测到{len(face_detections)}张人脸, 耗时{processing_time:.3f}s")
            return output
            
        except Exception as e:
            processing_time = time.time() - start_time
            app_logger.error(f"Hailo8推理失败: {e}")
            
            # 创建错误输出
            result = self._create_inference_result([], processing_time)
            output = self._create_inference_output(result, success=False, error_message=str(e))
            return output
    
    async def _extract_face_embedding(self, image: np.ndarray, bbox: List[float], landmarks: Optional[List]) -> Optional[List[float]]:
        """
        提取人脸特征向量
        
        Args:
            image: 原始图像
            bbox: 人脸边界框
            landmarks: 人脸关键点
            
        Returns:
            Optional[List[float]]: 特征向量
        """
        try:
            # 裁剪人脸区域
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # 调整到识别模型输入尺寸
            face_resized = cv2.resize(face_crop, self.recognition_size)
            
            # 执行识别推理
            recognition_result = self.recognition_model.predict(face_resized)
            
            # 提取特征向量
            if recognition_result.results and len(recognition_result.results) > 0:
                embedding_data = recognition_result.results[0].get('data', [])
                if embedding_data and len(embedding_data) > 0:
                    # 归一化特征向量
                    embedding = self._normalize_embedding(embedding_data[0])
                    return embedding
            
            return None
            
        except Exception as e:
            app_logger.debug(f"提取人脸特征向量失败: {e}")
            return None
    
    async def cleanup(self) -> bool:
        """
        清理资源
        
        Returns:
            bool: 清理是否成功
        """
        try:
            app_logger.info("清理Hailo8推理引擎资源")
            
            # 清理模型实例
            if self.recognition_model is not None:
                self.recognition_model = None
            
            if self.detection_model is not None:
                self.detection_model = None
            
            if self.zoo is not None:
                self.zoo = None
            
            # 强制清理Degirum工作进程
            self._cleanup_degirum_workers()
            
            self._models_loaded = False
            self._initialized = False
            
            app_logger.info("Hailo8推理引擎资源清理完成")
            return True
            
        except Exception as e:
            app_logger.error(f"Hailo8推理引擎资源清理失败: {e}")
            return False
    
    def _cleanup_degirum_workers(self):
        """清理Degirum工作进程"""
        try:
            current_pid = os.getpid()
            worker_pids = set()
            
            # 查找所有Degirum工作进程
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline')
                    if cmdline and any("degirum/pproc_worker.py" in s for s in cmdline):
                        # 检查是否是当前进程的子进程
                        if any(f"--parent_pid {current_pid}" in s for s in cmdline):
                            worker_pids.add(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # 使用pkill命令清理，更可靠
            if worker_pids:
                try:
                    # 使用pkill按父进程ID清理
                    import subprocess
                    result = subprocess.run(
                        f"pkill -f 'pproc_worker.py.*--parent_pid {current_pid}'",
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        app_logger.info(f"成功清理了 {len(worker_pids)} 个Degirum工作进程")
                    else:
                        app_logger.warning(f"pkill命令执行失败: {result.stderr}")
                except Exception as e:
                    app_logger.warning(f"使用pkill清理失败，尝试手动清理: {e}")
                    
                    # 手动清理作为备选方案
                    for pid in worker_pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            app_logger.info(f"已强制终止Degirum工作进程 PID: {pid}")
                        except ProcessLookupError:
                            app_logger.debug(f"进程 PID {pid} 已不存在")
                        except Exception as e:
                            app_logger.warning(f"终止进程 PID {pid} 时出错: {e}")
                
        except Exception as e:
            app_logger.error(f"清理Degirum工作进程时出错: {e}")
    
    def _cleanup_on_exit(self):
        """程序退出时的清理函数"""
        try:
            if self._initialized:
                # 直接清理资源，不使用asyncio.run()
                self._cleanup_degirum_workers()
                self._models_loaded = False
                self._initialized = False
                app_logger.info("Hailo8推理引擎资源清理完成")
        except Exception as e:
            app_logger.error(f"程序退出时清理资源失败: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        app_logger.info(f"收到信号 {signum}，开始清理资源")
        self._cleanup_on_exit()
        exit(0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "device_type": self.device_type,
            "zoo_path": self.zoo_path,
            "detection_model": self.detection_model_name,
            "recognition_model": self.recognition_model_name,
            "detection_size": self.detection_size,
            "recognition_size": self.recognition_size,
            "initialized": self._initialized,
            "models_loaded": self._models_loaded,
            "initialization_time": self._initialization_time,
            "model_loading_time": self._model_loading_time
        }
