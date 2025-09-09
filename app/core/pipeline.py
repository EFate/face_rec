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

# 全局字体配置（仅加载路径和默认参数，不固定字体大小）
try:
    from app.cfg.config import get_app_settings
    settings = get_app_settings()
    FONT_PATH = str(settings.app.chinese_font_path) if settings.app.chinese_font_path.exists() else None
except Exception as e:
    app_logger.debug(f"加载字体配置失败: {e}")
    FONT_PATH = None

# 字体大小边界（确保可读性：最小12，最大20）
MIN_FONT_SIZE = 12
MAX_FONT_SIZE = 20

def _draw_results_on_frame(frame: np.ndarray, results: List[Dict[str, Any]]):
    """在帧上绘制识别结果（名称+相似度整体保留，动态调整字体大小）"""
    if not results:
        return
    
    draw_items = []
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    
    # 工具函数1：创建指定大小的字体（复用路径，避免重复加载）
    def create_font(font_size: int) -> ImageFont.FreeTypeFont:
        try:
            if FONT_PATH and font_size >= MIN_FONT_SIZE:
                return ImageFont.truetype(FONT_PATH, font_size)
            else:
                # 字体路径无效或尺寸过小，用默认字体
                return ImageFont.load_default()
        except Exception as e:
            app_logger.debug(f"创建字体（大小{font_size}）失败: {e}")
            return ImageFont.load_default()
    
    # 工具函数2：计算文本宽度（基于指定字体）
    def calc_text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]  # 宽度 = 右坐标 - 左坐标
    
    # 工具函数3：计算文本高度（基于指定字体）
    def calc_text_height(text: str, font: ImageFont.FreeTypeFont) -> int:
        bbox = font.getbbox(text)
        return bbox[3] - bbox[1]  # 高度 = 下坐标 - 上坐标
    
    # 常量配置：边界保护与间距
    MIN_BOX_WIDTH = 50  # 最小人脸框宽度（小于此值不显示文本）
    TEXT_MARGIN_TOP = 5  # 文本与框上方的间距
    TEXT_MARGIN_BOTTOM = 3  # 文本与框下方的间距

    for res in results:
        # 1. 边界框处理：仅一次类型转换，过滤过窄框
        box = res['box'].astype(int)  # (x1, y1, x2, y2)
        box_width = box[2] - box[0]
        
        # 过滤过窄框：仅画框不显示文本（避免字体过度压缩）
        if box_width < MIN_BOX_WIDTH:
            draw_items.append({
                "box": box,
                "color": (255, 0, 0) if res['name'] == "Unknown" else (0, 255, 0),
                "label": "",
                "font": create_font(MAX_FONT_SIZE),
                "text_x": 0,
                "text_y": 0
            })
            continue

        # 2. 基础信息：构建完整文本（名称+相似度整体）
        name = res['name'] if res['name'] else "Unknown"
        color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)  # PIL用RGB
        similarity = res['similarity'] if res['similarity'] is not None else 0.0
        # 强制保留整体格式（如“张三 (0.98)”，无相似度时仍显“Unknown (0.00)”）
        label = f"{name} ({similarity:.2f})"

        # 3. 核心逻辑：动态调整字体大小（确保文本宽度 ≤ 人脸框宽度）
        target_width = box_width  # 文本最大允许宽度（等于人脸框宽度）
        current_font_size = MAX_FONT_SIZE  # 初始字体大小（从最大开始尝试）
        current_font = create_font(current_font_size)
        current_text_width = calc_text_width(label, current_font)

        # 逐步减小字体大小，直到宽度达标或达到最小尺寸
        while current_text_width > target_width and current_font_size > MIN_FONT_SIZE:
            current_font_size -= 1  # 字体大小递减1
            current_font = create_font(current_font_size)
            current_text_width = calc_text_width(label, current_font)

        # 4. 兜底策略：若字体已最小仍超宽，截断名称（保留相似度）
        if current_text_width > target_width:
            # 计算名称可占用的最大宽度（总宽 - 相似度宽度 - 空格宽度）
            sim_text = f"({similarity:.2f})"
            sim_width = calc_text_width(sim_text, current_font)
            space_width = calc_text_width(" ", current_font)
            max_name_width = target_width - sim_width - space_width

            # 截断名称（确保总宽度达标）
            truncated_name = name
            while calc_text_width(truncated_name, current_font) > max_name_width and len(truncated_name) > 1:
                truncated_name = truncated_name[:-1]  # 每次删最后1个字符
            # 名称至少保留1个字符，加省略号标识截断
            if len(truncated_name) < len(name):
                truncated_name += "..."
            # 重新组合标签（截断后的名称 + 相似度）
            label = f"{truncated_name} {sim_text}"
            # 重新计算最终宽度（确保达标）
            current_text_width = calc_text_width(label, current_font)

        # 5. 文本位置计算（基于调整后的字体大小）
        text_height = calc_text_height(label, current_font)
        # 水平居中：(框宽 - 文本宽) / 2，且不超出帧左右边界
        text_x = box[0] + (target_width - current_text_width) // 2
        text_x = max(0, min(text_x, frame_width - current_text_width))
        # 垂直位置：优先放框上方，超顶则放下方
        text_y = box[1] - text_height - TEXT_MARGIN_TOP
        if text_y < 0:
            text_y = box[3] + TEXT_MARGIN_BOTTOM

        # 收集绘制信息（含调整后的字体）
        draw_items.append({
            "box": box,
            "color": color,
            "label": label,
            "font": current_font,
            "text_x": text_x,
            "text_y": text_y
        })

    # 6. 统一绘制：PIL一次完成边界框和文本
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    for item in draw_items:
        # 绘制边界框（线宽2，与原逻辑一致）
        draw.rectangle(
            xy=[item["box"][0], item["box"][1], item["box"][2], item["box"][3]],
            outline=item["color"],
            width=2
        )
        # 绘制文本（用调整后的字体）
        if item["label"]:
            draw.text(
                xy=(item["text_x"], item["text_y"]),
                text=item["label"],
                font=item["font"],
                fill=item["color"]
            )

    # 7. 格式回传：PIL → OpenCV，覆盖原帧
    frame_with_results = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    frame[:] = frame_with_results


class FaceStreamPipeline:
    """
    封装一个视频流的完整四级处理流水线。
    每个实例对应一个独立的视频源，并管理其下的所有处理线程。
    """
    # --- 修改 __init__ 方法以接收持久化队列 ---
    def __init__(self, settings: AppSettings, stream_id: str, video_source: str, output_queue: queue.Queue, model: FaceAnalysis, result_persistence_queue: queue.Queue, task_id: int, app_id: int, app_name: str, domain_name: str, mqtt_manager=None):
        self.settings = settings
        self.stream_id = stream_id
        self.video_source = video_source
        self.output_queue = output_queue
        self.model = model
        self.result_persistence_queue = result_persistence_queue # 接收结果持久化队列
        self.mqtt_manager = mqtt_manager  # 接收MQTT管理器
        self.task_id = task_id
        self.app_id = app_id
        self.app_name = app_name
        self.domain_name = domain_name

        app_logger.info(f"【流水线 {self.stream_id}】正在初始化...")


        self.face_dao: FaceDataDAO = LanceDBFaceDataDAO(
            db_uri=self.settings.insightface.lancedb_uri,
            table_name=self.settings.insightface.lancedb_table_name
        )
        self.settings.app.detected_imgs_path.mkdir(parents=True, exist_ok=True)
        self.inference_queue = queue.Queue(maxsize=8)  # 增加推理队列大小
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
        if self.stop_event.is_set(): 
            return
        app_logger.info(f"【流水线 {self.stream_id}】正在停止...")
        self.stop_event.set()

        # 等待所有线程安全退出
        for t in self.threads: 
            if t.is_alive():
                t.join(timeout=3.0)
                if t.is_alive():
                    app_logger.warning(f"【流水线 {self.stream_id}】线程 {t.name} 未能及时退出")
        
        # 安全释放视频捕获资源
        if self.cap and self.cap.isOpened(): 
            try:
                self.cap.release()
            except Exception as e:
                app_logger.warning(f"【流水线 {self.stream_id}】释放视频捕获资源时出错: {e}")
        
        # 清空队列，避免内存泄漏
        for q in [self.inference_queue, self.postprocess_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # 安全释放数据库连接
        if self.face_dao:
            try:
                self.face_dao.dispose()
            except Exception as e:
                app_logger.warning(f"【流水线 {self.stream_id}】释放数据库连接时出错: {e}")
        
        app_logger.info(f"✅【流水线 {self.stream_id}】已安全停止。")

    def _start_threads(self):
        try:
            source_for_cv = int(self.video_source) if self.video_source.isdigit() else self.video_source
            self.cap = cv2.VideoCapture(source_for_cv)
            if not self.cap.isOpened(): 
                raise RuntimeError(f"无法打开视频源: {self.video_source}")

            self.threads = [
                threading.Thread(target=self._reader_thread, name=f"Reader-{self.stream_id}"),
                threading.Thread(target=self._inference_thread, name=f"Inference-{self.stream_id}"),
                threading.Thread(target=self._postprocessor_thread, name=f"Postprocessor-{self.stream_id}")
            ]
            for t in self.threads: 
                t.start()
        except Exception as e:
            app_logger.error(f"【流水线 {self.stream_id}】启动线程失败: {e}", exc_info=True)
            raise

    def _reader_thread(self):
        """
        [T1: 读帧] 智能读帧线程，支持自适应帧率和缓冲区管理。
        """
        app_logger.info(f"【T1:读帧 {self.stream_id}】启动。")
        
        consecutive_failures = 0
        max_failures = 10
        adaptive_sleep = 0.01
        
        try:
            while not self.stop_event.is_set():
                try:
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
                    queue_size = self.inference_queue.qsize()
                    if queue_size >= 6:  # 队列积压严重时
                        # 清空队列，只保留最新帧
                        while not self.inference_queue.empty():
                            try:
                                self.inference_queue.get_nowait()
                            except queue.Empty:
                                break
                        adaptive_sleep = max(0.005, adaptive_sleep * 0.9)  # 减少休眠时间
                    elif queue_size == 0:
                        adaptive_sleep = min(0.02, adaptive_sleep * 1.1)   # 增加休眠时间
                    
                    try:
                        self.inference_queue.put_nowait(frame)
                    except queue.Full:
                        # 队列满时丢弃当前帧，保持实时性
                        pass
                    
                    # 自适应休眠
                    time.sleep(adaptive_sleep)
                    
                except Exception as e:
                    app_logger.error(f"【T1:读帧 {self.stream_id}】读帧过程中发生错误: {e}", exc_info=True)
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        break
                    time.sleep(0.1)

        except Exception as e:
            app_logger.error(f"【T1:读帧 {self.stream_id}】线程发生致命错误: {e}", exc_info=True)
        finally:
            try:
                self.inference_queue.put(None)
            except Exception as e:
                app_logger.error(f"【T1:读帧 {self.stream_id}】发送停止信号失败: {e}")
            app_logger.info(f"【T1:读帧 {self.stream_id}】已停止。")


    def _inference_thread(self):
        """[T2: 推理] 执行模型推理，支持动态跳帧策略。"""
        app_logger.info(f"【T2:推理 {self.stream_id}】启动。")
        
        skip_frames = 0
        
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.inference_queue.get(timeout=1.0)
                    if frame is None:
                        try:
                            self.postprocess_queue.put(None)
                        except Exception as e:
                            app_logger.error(f"【T2:推理 {self.stream_id}】发送停止信号失败: {e}")
                        break
                    
                    # 动态跳帧策略：如果推理队列积压严重，跳过一些帧
                    if self.inference_queue.qsize() > 2:
                        skip_frames += 1
                        if skip_frames % 2 == 0:  # 每2帧跳1帧
                            continue
                    
                    # 执行推理
                    try:
                        detected_faces: List[Face] = self.model.get(frame)
                    except Exception as e:
                        app_logger.error(f"【T2:推理 {self.stream_id}】模型推理失败: {e}", exc_info=True)
                        continue
                    
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
                    app_logger.error(f"【T2:推理 {self.stream_id}】处理帧时发生错误: {e}", exc_info=True)
                    continue
                    
        except Exception as e:
            app_logger.error(f"【T2:推理 {self.stream_id}】线程发生致命错误: {e}", exc_info=True)
        finally:
            app_logger.info(f"【T2:推理 {self.stream_id}】已停止。")

    def _postprocessor_thread(self):
        """
        [T3: 后处理/识别] 快速处理人脸比对和结果输出，异步处理数据保存。
        """
        app_logger.info(f"【T3:后处理 {self.stream_id}】启动。")

        threshold = self.settings.insightface.recognition_similarity_threshold

        try:
            while not self.stop_event.is_set():
                try:
                    data = self.postprocess_queue.get(timeout=1.0)
                    if data is None: 
                        break

                    original_frame, detected_faces = data
                    results = []
                    
                    if detected_faces:
                        for face in detected_faces:
                            try:
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
                                                "timestamp": datetime.now(),
                                                "task_id": self.task_id,
                                                "app_id": self.app_id,
                                                "app_name": self.app_name,
                                                "domain_name": self.domain_name
                                            }
                                            # 非阻塞放入队列，满了就丢弃
                                            self.result_persistence_queue.put_nowait(persistence_data)
                                    except queue.Full:
                                        # 队列满时静默丢弃，不影响推理性能
                                        pass
                                    except Exception as e:
                                        app_logger.debug(f"准备持久化数据时出错: {e}")
                                
                                results.append(result_item)
                            except Exception as e:
                                app_logger.error(f"【T3:后处理 {self.stream_id}】处理单个人脸时出错: {e}", exc_info=True)
                                # 继续处理其他人脸
                                continue

                    # 快速绘制结果并输出
                    try:
                        _draw_results_on_frame(original_frame, results)
                        flag, encoded_image = cv2.imencode(".jpg", original_frame, 
                                                         [cv2.IMWRITE_JPEG_QUALITY, 85])  # 降低质量提升编码速度
                        if flag:
                            try:
                                self.output_queue.put_nowait(encoded_image.tobytes())
                            except queue.Full:
                                # 输出队列满时丢弃旧帧，保持实时性
                                pass
                    except Exception as e:
                        app_logger.error(f"【T3:后处理 {self.stream_id}】编码图像时出错: {e}", exc_info=True)
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    app_logger.error(f"【T3:后处理 {self.stream_id}】处理帧数据时发生错误: {e}", exc_info=True)
                    continue
                    
        except Exception as e:
            app_logger.error(f"【T3:后处理 {self.stream_id}】线程发生致命错误: {e}", exc_info=True)
        finally:
            # 清理工作
            try:
                self.output_queue.put_nowait(None)
            except (queue.Full, ValueError):
                pass
            app_logger.info(f"【T3:后处理 {self.stream_id}】已停止。")
