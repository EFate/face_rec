# app/core/pipeline.py
import asyncio
import queue
import threading
import time
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from insightface.app.common import Face

from app.cfg.config import AppSettings
from app.cfg.logging import app_logger
from app.core.model_manager import create_face_analysis_model
from app.service.face_dao import LanceDBFaceDataDAO, FaceDataDAO


def _draw_results_on_frame(frame: np.ndarray, results: List[Dict[str, Any]]):
    """在帧上绘制识别结果（边界框和标签）"""
    for res in results:
        box = res['box'].astype(int)
        label = f"{res['name']}"
        if res['similarity'] is not None:
            label += f" ({res['similarity']:.2f})"
        color = (0, 255, 0) if res['name'] != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (box[0], box[1] - lh - 10), (box[0] + lw, box[1]), color, cv2.FILLED)
        cv2.putText(frame, label, (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


class FaceStreamPipeline:
    """
    封装一个视频流的完整四级处理流水线。
    每个实例对应一个独立的视频源，并管理其下的所有处理线程。
    """

    def __init__(self, settings: AppSettings, stream_id: str, video_source: str, output_queue: asyncio.Queue):
        self.settings = settings
        self.stream_id = stream_id
        self.video_source = video_source
        self.output_queue = output_queue

        app_logger.info(f"【流水线 {self.stream_id}】正在初始化...")

        self.model = create_face_analysis_model(settings)

        # 每个子进程的流水线将拥有一个贯穿整个生命周期的DAO实例。
        self.face_dao: FaceDataDAO = LanceDBFaceDataDAO(
            db_uri=self.settings.insightface.lancedb_uri,
            table_name=self.settings.insightface.lancedb_table_name
        )


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
            while not self.stop_event.is_set():
                time.sleep(1)
        except Exception as e:
            app_logger.error(f"❌【流水线 {self.stream_id}】启动或运行时失败: {e}", exc_info=True)
        finally:
            self.stop()

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
        [T1: 读帧] 以最快速度读取帧，并在每次循环后短暂休眠以降低CPU占用。
        同时，当队列满时会自动丢弃最旧的帧以保证处理最新帧。
        """
        app_logger.info(f"【T1:读帧 {self.stream_id}】启动 (无FPS限制模式)。")

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                app_logger.warning(f"【T1:读帧 {self.stream_id}】无法读取帧，流结束。")
                break

            if self.preprocess_queue.full():
                try:
                    self.preprocess_queue.get_nowait()
                except queue.Empty:
                    pass
            self.preprocess_queue.put(frame)

            time.sleep(0.01)

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
        """[T3: 推理] 执行模型推理，结果放入后处理队列。"""
        app_logger.info(f"【T3:推理 {self.stream_id}】启动。")
        while not self.stop_event.is_set():
            try:
                frame = self.inference_queue.get(timeout=1.0)
                if frame is None:
                    self.postprocess_queue.put(None)
                    break
                detected_faces: List[Face] = self.model.get(frame)
                self.postprocess_queue.put((frame, detected_faces))
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【T3:推理 {self.stream_id}】发生错误: {e}", exc_info=True)
        app_logger.info(f"【T3:推理 {self.stream_id}】已停止。")

    def _postprocessor_thread(self):
        """
        [T4: 后处理/识别] 进行人脸比对、绘制并放入最终输出队列。
        """
        app_logger.info(f"【T4:后处理 {self.stream_id}】启动。")

        # 调用 self.face_dao.search() 查询数据库文件


        threshold = self.settings.insightface.recognition_similarity_threshold

        while not self.stop_event.is_set():
            try:
                data = self.postprocess_queue.get(timeout=1.0)
                if data is None: break

                original_frame, detected_faces = data
                results = []
                if detected_faces:
                    for face in detected_faces:
                        # 每次搜索都直接访问数据库
                        search_res = self.face_dao.search(face.normed_embedding, threshold)
                        result_item = {"box": face.bbox, "name": "Unknown", "similarity": None}
                        if search_res:
                            name, sn, similarity = search_res
                            result_item.update({"name": name, "sn": sn, "similarity": similarity})
                        results.append(result_item)

                _draw_results_on_frame(original_frame, results)

                (flag, encodedImage) = cv2.imencode(".jpg", original_frame)
                if flag:
                    try:
                        self.output_queue.put_nowait(encodedImage.tobytes())
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"【T4:后处理 {self.stream_id}】发生错误: {e}", exc_info=True)

        try:
            self.output_queue.put_nowait(None)
        except (queue.Full, ValueError):
            pass
        app_logger.info(f"【T4:后处理 {self.stream_id}】已停止。")