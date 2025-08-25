# app/core/pipeline.py
import queue
import threading
import time
from datetime import datetime
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from insightface.app.common import Face
from insightface.app import FaceAnalysis  # 引入 FaceAnalysis 类型注解

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.service.face_dao import LanceDBFaceDataDAO, FaceDataDAO


def _draw_results_on_frame(frame: np.ndarray, results: List[Dict[str, Any]]):
    """在帧上绘制识别结果（边界框和标签）"""
    if not results:
        return
    
    # 第一步：使用OpenCV绘制所有人脸边界框
    for res in results:
        box = res['box'].astype(int)
        name = res['name'] if res['name'] else "Unknown"
        
        # 设置边界框颜色
        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # 绘制人脸边界框
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)
    
    # 第二步：转换为PIL图像进行文本绘制
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    # 使用配置中的中文字体
    font = None
    try:
        from app.cfg.config import get_app_settings
        settings = get_app_settings()
        font_path = settings.app.chinese_font_path
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 28)
    except Exception as e:
        app_logger.debug(f"加载配置字体失败: {e}")
    
    # 如果配置字体加载失败，使用默认字体
    if font is None:
        font = ImageFont.load_default()
    
    # 第三步：在PIL图像上绘制文本标识
    for res in results:
        box = res['box'].astype(int)
        name = res['name'] if res['name'] else "Unknown"
        
        # 构建标签文本
        if res['similarity'] is not None:
            label = f"{name} ({res['similarity']:.2f})"
        else:
            label = name
        
        # 设置文本颜色与框颜色一致 (PIL使用RGB)
        text_color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
        
        # 计算人脸框宽度
        box_width = box[2] - box[0]
        
        # 计算文本尺寸并确保不超过框宽度
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 如果文本宽度超过框宽度，截断文本
        if text_width > box_width:
            # 逐步缩短文本直到适合框宽度
            if res['similarity'] is not None:
                # 先尝试只显示名字和简化的相似度
                short_label = f"{name} ({res['similarity']:.1f})"
                bbox = draw.textbbox((0, 0), short_label, font=font)
                if bbox[2] - bbox[0] <= box_width:
                    label = short_label
                    text_width = bbox[2] - bbox[0]
                else:
                    # 如果还是太长，只显示名字
                    label = name
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    
                    # 如果名字还是太长，截断名字
                    if text_width > box_width:
                        max_chars = int(box_width / (text_width / len(name))) - 1
                        if max_chars > 0:
                            label = name[:max_chars] + "..."
                            bbox = draw.textbbox((0, 0), label, font=font)
                            text_width = bbox[2] - bbox[0]
        
        # 计算文本位置，居中对齐到人脸框
        text_x = box[0] + (box_width - text_width) // 2  # 居中对齐
        text_y = box[1] - text_height - 8  # 文本在框上方
        
        # 如果文本会超出图像顶部，放到框下方
        if text_y < 0:
            text_y = box[3] + 5
        
        # 确保文本不会超出图像左右边界
        text_x = max(0, min(text_x, frame.shape[1] - text_width))
        
        # 直接绘制文本，颜色与框一致，无阴影
        draw.text((text_x, text_y), label, font=font, fill=text_color)
    
    # 第四步：转换回OpenCV格式，保留边界框
    frame_with_text = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    # 将绘制了文本的图像复制回原始frame
    frame[:] = frame_with_text


class FaceStreamPipeline:
    """
    封装一个视频流的完整四级处理流水线。
    每个实例对应一个独立的视频源，并管理其下的所有处理线程。
    """
    # --- 修改 __init__ 方法以接收持久化队列 ---
    def __init__(self, settings: AppSettings, stream_id: str, video_source: str, output_queue: queue.Queue, model: FaceAnalysis, result_persistence_queue: queue.Queue):
        self.settings = settings
        self.stream_id = stream_id
        self.video_source = video_source
        self.output_queue = output_queue
        self.model = model
        self.result_persistence_queue = result_persistence_queue # 接收结果持久化队列

        app_logger.info(f"【流水线 {self.stream_id}】正在初始化...")


        self.face_dao: FaceDataDAO = LanceDBFaceDataDAO(
            db_uri=self.settings.insightface.lancedb_uri,
            table_name=self.settings.insightface.lancedb_table_name
        )
        self.settings.app.detected_imgs_path.mkdir(parents=True, exist_ok=True)
        self.preprocess_queue = queue.Queue(maxsize=4)
        self.inference_queue = queue.Queue(maxsize=4)
        self.postprocess_queue = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []
        self.cap = None

    def start(self):
        """启动所有流水线工作线程。"""
        app_logger.info(f"【流水线 {self.stream_id}】正在启动...")
        try:
            self._start_threads()
        except Exception as e:
            app_logger.error(f"❌【流水线 {self.stream_id}】启动或运行时失败: {e}", exc_info=True)
            raise

    def stop(self):
        """停止所有流水线工作线程并释放资源。"""
        if self.stop_event.is_set(): return
        app_logger.info(f"【流水线 {self.stream_id}】正在停止...")
        self.stop_event.set()

        for t in self.threads: t.join(timeout=2.0)
        if self.cap and self.cap.isOpened(): self.cap.release()
        for q in [self.preprocess_queue, self.inference_queue, self.postprocess_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        if self.face_dao:
            self.face_dao.dispose()
        app_logger.info(f"✅【流水线 {self.stream_id}】已安全停止。")

    def _start_threads(self):
        source_for_cv = int(self.video_source) if self.video_source.isdigit() else self.video_source
        self.cap = cv2.VideoCapture(source_for_cv)
        if not self.cap.isOpened(): raise RuntimeError(f"无法打开视频源: {self.video_source}")

        self.threads = [
            threading.Thread(target=self._reader_thread, name=f"Reader-{self.stream_id}"),
            threading.Thread(target=self._preprocessor_thread, name=f"Preprocessor-{self.stream_id}"),
            threading.Thread(target=self._inference_thread, name=f"Inference-{self.stream_id}"),
            threading.Thread(target=self._postprocessor_thread, name=f"Postprocessor-{self.stream_id}")
        ]
        for t in self.threads: t.start()

    def _reader_thread(self):
        """
        [T1: 读帧] 智能读帧线程，支持自适应帧率和缓冲区管理。
        """
        app_logger.info(f"【T1:读帧 {self.stream_id}】启动。")
        
        consecutive_failures = 0
        max_failures = 10
        adaptive_sleep = 0.01
        
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    app_logger.error(f"【T1:读帧 {self.stream_id}】连续{max_failures}次读帧失败，流可能已断开。")
                    break
                time.sleep(min(0.1 * consecutive_failures, 1.0))
                continue
            
            consecutive_failures = 0
            
            # 智能缓冲区管理：保持队列中只有最新的帧
            queue_size = self.preprocess_queue.qsize()
            if queue_size >= 3:  # 队列积压严重时
                # 清空队列，只保留最新帧
                while not self.preprocess_queue.empty():
                    try:
                        self.preprocess_queue.get_nowait()
                    except queue.Empty:
                        break
                adaptive_sleep = max(0.005, adaptive_sleep * 0.9)  # 减少休眠时间
            elif queue_size == 0:
                adaptive_sleep = min(0.02, adaptive_sleep * 1.1)   # 增加休眠时间
            
            try:
                self.preprocess_queue.put_nowait(frame)
            except queue.Full:
                # 队列满时丢弃当前帧，保持实时性
                pass
            
            # 自适应休眠
            time.sleep(adaptive_sleep)

        self.preprocess_queue.put(None)
        app_logger.info(f"【T1:读帧 {self.stream_id}】已停止。")

    def _preprocessor_thread(self):
        """[T2: 预处理] 从预处理队列取帧，放入推理队列。"""
        app_logger.info(f"【T2:预处理 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            try:
                frame = self.preprocess_queue.get(timeout=1.0)
                if frame is None:
                    self.inference_queue.put(None)
                    break
                self.inference_queue.put(frame)
            except queue.Empty:
                continue
        app_logger.info(f"【T2:预处理 {self.stream_id}】已停止。")

    def _inference_thread(self):
        """[T3: 推理] 执行模型推理，支持动态跳帧策略。"""
        app_logger.info(f"【T3:推理 {self.stream_id}】启动。")
        
        skip_frames = 0
        
        while not self.stop_event.is_set():
            try:
                frame = self.inference_queue.get(timeout=1.0)
                if frame is None:
                    self.postprocess_queue.put(None)
                    break
                
                # 动态跳帧策略：如果推理队列积压严重，跳过一些帧
                if self.inference_queue.qsize() > 2:
                    skip_frames += 1
                    if skip_frames % 2 == 0:  # 每2帧跳1帧
                        continue
                
                # 执行推理
                detected_faces: List[Face] = self.model.get(frame)
                
                # 非阻塞放入后处理队列
                try:
                    self.postprocess_queue.put_nowait((frame, detected_faces))
                except queue.Full:
                    # 后处理队列满时，丢弃最旧的结果
                    try:
                        self.postprocess_queue.get_nowait()
                        self.postprocess_queue.put_nowait((frame, detected_faces))
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【T3:推理 {self.stream_id}】发生错误: {e}", exc_info=True)
                
        app_logger.info(f"【T3:推理 {self.stream_id}】已停止。")

    def _postprocessor_thread(self):
        """
        [T4: 后处理/识别] 快速处理人脸比对和结果输出，异步处理数据保存。
        """
        app_logger.info(f"【T4:后处理 {self.stream_id}】启动。")

        threshold = self.settings.insightface.recognition_similarity_threshold

        while not self.stop_event.is_set():
            try:
                data = self.postprocess_queue.get(timeout=1.0)
                if data is None: break

                original_frame, detected_faces = data
                results = []
                
                if detected_faces:
                    for face in detected_faces:
                        # 快速执行人脸搜索
                        search_res = self.face_dao.search(face.normed_embedding, threshold)
                        result_item = {"box": face.bbox, "name": "Unknown", "similarity": None}
                        
                        if search_res:
                            name, sn, similarity = search_res
                            result_item.update({"name": name, "sn": sn, "similarity": similarity})

                            # 异步处理结果保存 - 不阻塞推理流程
                            try:
                                box = face.bbox.astype(int)
                                y1_c, y2_c = max(0, box[1]), min(original_frame.shape[0], box[3])
                                x1_c, x2_c = max(0, box[0]), min(original_frame.shape[1], box[2])
                                face_crop = original_frame[y1_c:y2_c, x1_c:x2_c]

                                if face_crop.size > 0:
                                    # 准备持久化数据
                                    persistence_data = {
                                        "sn": sn,
                                        "name": name,
                                        "similarity": similarity,
                                        "face_crop": face_crop.copy(),  # 复制数据避免引用问题
                                        "timestamp": datetime.now()
                                    }
                                    # 非阻塞放入队列，满了就丢弃
                                    self.result_persistence_queue.put_nowait(persistence_data)
                            except queue.Full:
                                # 队列满时静默丢弃，不影响推理性能
                                pass
                            except Exception as e:
                                app_logger.debug(f"准备持久化数据时出错: {e}")
                        
                        results.append(result_item)

                # 快速绘制结果并输出
                _draw_results_on_frame(original_frame, results)
                flag, encoded_image = cv2.imencode(".jpg", original_frame, 
                                                 [cv2.IMWRITE_JPEG_QUALITY, 85])  # 降低质量提升编码速度
                if flag:
                    try:
                        self.output_queue.put_nowait(encoded_image.tobytes())
                    except queue.Full:
                        # 输出队列满时丢弃旧帧，保持实时性
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【T4:后处理 {self.stream_id}】发生错误: {e}", exc_info=True)
                
        # 清理工作
        try:
            self.output_queue.put_nowait(None)
        except (queue.Full, ValueError):
            pass
        app_logger.info(f"【T4:后处理 {self.stream_id}】已停止。")
