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
    
    def initialize(self) -> bool:
        """
        初始化推理引擎
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            app_logger.debug(f"初始化Hailo8推理引擎: {self.zoo_path}")
            
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
            app_logger.debug("Hailo8推理引擎初始化成功")
            return True
            
        except Exception as e:
            app_logger.error(f"Hailo8推理引擎初始化失败: {e}")
            return False
    
    def load_models(self) -> bool:
        """
        加载Hailo8模型
        
        Returns:
            bool: 模型加载是否成功
        """
        try:
            if not self._initialized:
                raise RuntimeError("推理引擎未初始化")
            
            app_logger.debug(f"加载Hailo8模型: {self.detection_model_name}, {self.recognition_model_name}")
            
            # 连接模型仓库
            self.zoo = dg.connect(
                inference_host_address=dg.LOCAL,
                zoo_url=f"file://{self.zoo_path}"
            )
            
            # 加载检测模型
            self.detection_model = self.zoo.load_model(self.detection_model_name)
            app_logger.debug(f"检测模型加载成功: {self.detection_model_name}")
            
            # 加载识别模型
            self.recognition_model = self.zoo.load_model(self.recognition_model_name)
            app_logger.debug(f"识别模型加载成功: {self.recognition_model_name}")
            
            self._models_loaded = True
            self._model_loading_time = time.time()
            app_logger.debug("Hailo8模型加载成功")
            return True
            
        except Exception as e:
            app_logger.error(f"Hailo8模型加载失败: {e}")
            return False
    
    def predict(self, input_data: InferenceInput) -> InferenceOutput:
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
                    # 转换landmarks格式：从复杂结构提取坐标
                    if isinstance(raw_landmarks, list):
                        landmarks = []
                        for landmark in raw_landmarks:
                            if isinstance(landmark, dict):
                                # 处理 {"landmark": [x, y], "score": ..., "category_id": ...} 格式
                                if 'landmark' in landmark and isinstance(landmark['landmark'], (list, tuple)) and len(landmark['landmark']) >= 2:
                                    landmarks.append([float(landmark['landmark'][0]), float(landmark['landmark'][1])])
                                # 处理 {"x": x, "y": y} 格式
                                elif 'x' in landmark and 'y' in landmark:
                                    landmarks.append([float(landmark['x']), float(landmark['y'])])
                            elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                                # 处理 [x, y] 简单格式
                                landmarks.append([float(landmark[0]), float(landmark[1])])
                    elif isinstance(raw_landmarks, dict):
                        # 处理单个关键点的情况
                        if 'landmark' in raw_landmarks and isinstance(raw_landmarks['landmark'], (list, tuple)) and len(raw_landmarks['landmark']) >= 2:
                            landmarks = [[float(raw_landmarks['landmark'][0]), float(raw_landmarks['landmark'][1])]]
                        elif 'x' in raw_landmarks and 'y' in raw_landmarks:
                            landmarks = [[float(raw_landmarks['x']), float(raw_landmarks['y'])]]
                
                # 提取人脸特征向量
                embedding = None
                if input_data.extract_embeddings:
                    embedding = self._extract_face_embedding(
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
    
    def _align_face(self, img: np.ndarray, landmarks: List, image_size: int = 112) -> tuple:
        """
        根据给定的特征点，对图像中的人脸进行对齐与裁剪。
        此方法基于人脸识别系统构建综合指南中的实现，针对侧脸识别进行了优化。
        
        Args:
            img: 原始完整图像，将对该图像进行变换
            landmarks: 5个关键点（特征点）的列表，格式为(x, y)坐标
            image_size: 图像调整后的尺寸，默认值为112
            
        Returns:
            tuple: 对齐后的人脸图像与变换矩阵
        """
        try:
            # 定义ArcFace模型中使用的参考关键点（基于典型面部特征点集）
            _arcface_ref_kps = np.array(
                [
                    [38.2946, 51.6963],  # 左眼
                    [73.5318, 51.5014],  # 右眼
                    [56.0252, 71.7366],  # 鼻子
                    [41.5493, 92.3655],  # 左嘴角
                    [70.7299, 92.2041],  # 右嘴角
                ],
                dtype=np.float32,
            )

            # 确保输入的特征点数量恰好为5个（人脸对齐所需的标准数量）
            if len(landmarks) != 5:
                app_logger.warning(f"人脸对齐需要5个关键点，但提供了{len(landmarks)}个")
                return None, None

            # 验证image_size是否可被112或128整除（人脸识别模型的常用图像尺寸）
            if image_size % 112 != 0 and image_size % 128 != 0:
                app_logger.warning(f"图像尺寸{image_size}不是112或128的倍数")
                return None, None

            # 将landmarks转换为numpy数组
            landmarks_array = np.array(landmarks, dtype=np.float32)

            # 根据目标图像尺寸（112或128）调整缩放比例
            if image_size % 112 == 0:
                ratio = float(image_size) / 112.0
                diff_x = 0  # 尺寸为112时无需水平偏移
            else:
                ratio = float(image_size) / 128.0
                diff_x = 8.0 * ratio  # 尺寸为128时需添加水平偏移

            # 对参考关键点应用缩放与偏移
            dst = _arcface_ref_kps * ratio
            dst[:, 0] += diff_x  # 应用水平偏移

            # 估计相似变换矩阵，使输入特征点与参考关键点对齐
            # 使用较低的RANSAC阈值以提高对侧脸的适应性
            M, inliers = cv2.estimateAffinePartial2D(
                landmarks_array, 
                dst, 
                method=cv2.RANSAC, 
                ransacReprojThreshold=20.0  # 增加阈值以适应侧脸
            )
            
            if M is None:
                app_logger.warning("无法估计变换矩阵，尝试使用仿射变换")
                # 如果相似变换失败，尝试使用仿射变换
                M, inliers = cv2.estimateAffine2D(
                    landmarks_array, 
                    dst, 
                    method=cv2.RANSAC, 
                    ransacReprojThreshold=20.0
                )
                
            if M is None:
                app_logger.warning("无法估计变换矩阵")
                return None, None
                
            # 对输入图像应用仿射变换，实现人脸对齐
            aligned_img = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

            return aligned_img, M
            
        except Exception as e:
            app_logger.error(f"人脸对齐失败: {e}")
            return None, None
    
    def _extract_face_embedding(self, image: np.ndarray, bbox: List[float], landmarks: Optional[List]) -> Optional[List[float]]:
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
            # 使用完整图像进行人脸矫正（如果有关键点）
            aligned_face = None
            if landmarks and len(landmarks) >= 5:
                app_logger.debug(f"使用关键点进行人脸对齐，关键点数量: {len(landmarks)}")
                aligned_face, _ = self._align_face(image, landmarks[:5], self.recognition_size[0])
            
            # 如果矫正成功，使用矫正后的人脸；否则裁剪人脸区域
            if aligned_face is not None and aligned_face.size > 0:
                app_logger.debug("人脸对齐成功，使用对齐后的人脸")
                face_crop = aligned_face
            else:
                app_logger.debug("人脸对齐失败或无关键点，使用裁剪方法")
                # 裁剪人脸区域（备选方案）
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                face_crop = image[y1:y2, x1:x2]
                if face_crop.size == 0:
                    app_logger.warning("裁剪后的人脸区域为空")
                    return None
                
                # 调整到识别模型输入尺寸
                face_crop = cv2.resize(face_crop, self.recognition_size)
            
            # 执行识别推理
            recognition_result = self.recognition_model.predict(face_crop)
            
            # 提取特征向量
            if recognition_result.results and len(recognition_result.results) > 0:
                embedding_data = recognition_result.results[0].get('data', [])
                if embedding_data and len(embedding_data) > 0:
                    embedding = embedding_data[0]
                    # 归一化特征向量
                    embedding_array = np.array(embedding, dtype=np.float32)
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        normalized_embedding = embedding_array / norm
                        return normalized_embedding.tolist()
                    return embedding
            
            app_logger.warning("无法从识别模型获取有效的特征向量")
            return None
            
        except Exception as e:
            app_logger.error(f"提取人脸特征向量失败: {e}")
            return None
    
    def cleanup(self) -> bool:
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
            import os
            current_pid = os.getpid()
            app_logger.info("开始清理Degirum工作进程")
            
            # 1. 首先尝试使用Python的os模块和signal模块来清理进程
            import os
            import signal
            import time
            import psutil
            
            try:
                # 查找所有degirum工作进程
                degirum_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info['cmdline']
                        if cmdline and any('degirum/pproc_worker.py' in cmd for cmd in cmdline):
                            # 检查是否是当前进程创建的工作进程
                            parent_pid = None
                            for arg in cmdline:
                                if not arg.startswith('--parent_pid'):
                                    continue
                                    
                                try:
                                    parts = arg.split('=', 1)
                                    if len(parts) == 2:
                                        # 格式: --parent_pid=123
                                        parent_pid = int(parts[1])
                                    else:
                                        # 格式: --parent_pid 123
                                        idx = cmdline.index(arg)
                                        if idx + 1 < len(cmdline) and cmdline[idx+1].isdigit():
                                            parent_pid = int(cmdline[idx + 1])
                                    break
                                except (IndexError, ValueError) as e:
                                    app_logger.warning(f"解析parent_pid参数失败: {arg}, 错误: {e}")
                                    continue
                                    
                            # 如果没有找到parent_pid，则跳过该进程
                            if parent_pid is None:
                                continue
                            
                            if parent_pid is None or parent_pid == current_pid:
                                degirum_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
                
                if degirum_processes:
                    app_logger.debug(f"找到 {len(degirum_processes)} 个Degirum工作进程")
                    
                    # 先尝试优雅终止
                    for proc in degirum_processes:
                        try:
                            os.kill(proc.info['pid'], signal.SIGTERM)
                            app_logger.debug(f"发送SIGTERM信号到进程 {proc.info['pid']}")
                        except Exception as e:
                            app_logger.error(f"发送SIGTERM到进程 {proc.info['pid']} 时出错: {e}")
                    
                    # 等待进程终止
                    time.sleep(1.0)
                    
                    # 检查是否还有残留进程
                    remaining_processes = []
                    for proc in degirum_processes:
                        try:
                            if psutil.pid_exists(proc.info['pid']):
                                remaining_processes.append(proc)
                        except Exception:
                            pass
                    
                    if remaining_processes:
                        # 还有残留进程，强制终止
                        app_logger.warning(f"发现 {len(remaining_processes)} 个残留的Degirum工作进程，强制终止")
                        for proc in remaining_processes:
                            try:
                                os.kill(proc.info['pid'], signal.SIGKILL)
                                app_logger.debug(f"发送SIGKILL信号到进程 {proc.info['pid']}")
                            except Exception as e:
                                app_logger.error(f"发送SIGKILL到进程 {proc.info['pid']} 时出错: {e}")
                    
                    # 最终检查
                    time.sleep(0.5)
                    final_remaining = []
                    for proc in degirum_processes:
                        try:
                            if psutil.pid_exists(proc.info['pid']):
                                final_remaining.append(proc.info['pid'])
                        except Exception:
                            pass
                    
                    if final_remaining:
                        # 2. 如果Python方法清理不完全，尝试使用subprocess调用pkill命令
                        app_logger.warning(f"Python方法清理后仍有Degirum进程: {final_remaining}，尝试使用pkill命令")
                        self._cleanup_with_pkill()
                        
                        # 再次检查残留进程
                        time.sleep(1.0)  # 增加等待时间确保进程终止
                        final_remaining = []
                        for pid in final_remaining:
                            try:
                                if psutil.pid_exists(pid):
                                    final_remaining.append(pid)
                                    proc = psutil.Process(pid)
                                    app_logger.error(f"残留进程详细信息[PID={pid}]: "
                                                    f"cmdline={proc.cmdline()}, "
                                                    f"status={proc.status()}, "
                                                    f"create_time={proc.create_time()}, "
                                                    f"ppid={proc.ppid()}")
                            except Exception as e:
                                app_logger.error(f"检查进程{pid}失败: {e}")
                        
                        if final_remaining:
                            app_logger.error(f"最终清理后仍有{len(final_remaining)}个残留进程")
                            # 尝试终极清理方案
                            for pid in final_remaining:
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                    time.sleep(0.1)
                                    if psutil.pid_exists(pid):
                                        app_logger.critical(f"无法终止的顽固进程: {pid}")
                                except Exception as e:
                                    app_logger.error(f"终止进程{pid}失败: {e}")
                        
                        # 3. 终极清理方案 - 强制杀死所有相关进程
                        time.sleep(0.5)
                        still_running = []
                        for pid in final_remaining:
                            if psutil.pid_exists(pid):
                                still_running.append(pid)
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                except:
                                    pass
                        
                        if still_running:
                            app_logger.error(f"无法完全清理的Degirum进程: {still_running}")
                    else:
                        app_logger.debug("所有Degirum工作进程已清理完成")
                else:
                    app_logger.debug("未找到Degirum工作进程")
                    
            except Exception as e:
                app_logger.error(f"使用Python方法清理Degirum工作进程时出错: {e}")
                # 尝试使用pkill命令作为备选方案
                self._cleanup_with_pkill()
                import traceback
                app_logger.error(f"错误详情: {traceback.format_exc()}")
            
        except Exception as e:
            app_logger.error(f"清理Degirum工作进程时出错: {e}")
    
    def _cleanup_with_pkill(self):
        """使用pkill命令清理Degirum工作进程的备选方案"""
        try:
            import os
            current_pid = os.getpid()
            import subprocess
            import time
            
            app_logger.debug("使用pkill命令尝试清理Degirum工作进程")
            
            # 先尝试优雅终止
            try:
                result = subprocess.run(
                    f"pkill -TERM -f 'pproc_worker.py.*--parent_pid {current_pid}'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                app_logger.debug("发送SIGTERM信号到所有Degirum工作进程")
                
                # 等待进程终止
                time.sleep(1.0)
                
                # 检查是否还有残留进程
                check_result = subprocess.run(
                    f"pgrep -f 'pproc_worker.py.*--parent_pid {current_pid}'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if check_result.returncode == 0 and check_result.stdout.strip():
                    # 还有残留进程，强制终止
                    app_logger.warning("发现残留的Degirum工作进程，强制终止")
                    kill_result = subprocess.run(
                        f"pkill -KILL -f 'pproc_worker.py.*--parent_pid {current_pid}'",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    app_logger.debug("强制终止所有残留的Degirum工作进程")
                
                # 最终检查
                time.sleep(0.5)
                final_check = subprocess.run(
                    f"pgrep -f 'pproc_worker.py.*--parent_pid {current_pid}'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if final_check.returncode == 0 and final_check.stdout.strip():
                    app_logger.warning(f"pkill清理后仍有Degirum进程: {final_check.stdout.strip()}")
                else:
                    app_logger.debug("pkill成功清理所有Degirum工作进程")
                    
            except subprocess.TimeoutExpired:
                app_logger.error("使用pkill清理Degirum工作进程超时")
            except Exception as e:
                app_logger.error(f"使用pkill清理Degirum工作进程时出错: {e}")
        except Exception as e:
            app_logger.error(f"执行pkill备选方案时出错: {e}")
    
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
