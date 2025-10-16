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
    def __init__(self, settings: AppSettings, stream_id: str, video_source: str, output_queue: queue.Queue, model, result_persistence_queue: queue.Queue, task_id: int, app_id: int, app_name: str, domain_name: str, mqtt_manager=None, original_output_queue: queue.Queue = None):
        self.settings = settings
        self.stream_id = stream_id
        self.video_source = video_source
        self.output_queue = output_queue
        self.original_output_queue = original_output_queue  # 添加原始视频输出队列
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
        self.inference_queue = queue.Queue(maxsize=3)  # 减小推理队列大小以降低延迟
        self.postprocess_queue = queue.Queue(maxsize=2)  # 减小后处理队列大小
        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []
        self.cap = None
        self._cap_released = False  # 资源释放标志
        self._resources_cleaned = False  # 清理完成标志

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
        
        try:
            # 1. 设置停止事件，让线程优雅退出
            self.stop_event.set()
            
            # 2. 发送停止信号到所有队列，确保线程能收到
            try:
                for q in [self.inference_queue, self.postprocess_queue, self.output_queue]:
                    try:
                        # 多次尝试发送停止信号
                        for _ in range(3):
                            try:
                                q.put_nowait(None)
                                break
                            except queue.Full:
                                # 清空队列后重试
                                try:
                                    q.get_nowait()
                                except queue.Empty:
                                    break
                    except (ValueError, AttributeError):
                        # 队列已关闭，忽略
                        pass
            except Exception as e:
                app_logger.debug(f"【流水线 {self.stream_id}】发送停止信号时出错: {e}")
            
            # 3. 等待所有线程退出，使用更长的超时时间
            if hasattr(self, 'threads'):
                for t in self.threads:
                    if t.is_alive():
                        t.join(timeout=2.0)
                        if t.is_alive():
                            app_logger.warning(f"【流水线 {self.stream_id}】线程 {t.name} 超时未退出")
            
            # 4. 安全释放视频捕获资源
            try:
                if hasattr(self, 'cap') and self.cap is not None:
                    # 检查是否已释放
                    if hasattr(self, '_cap_released') and self._cap_released:
                        pass
                    else:
                        self.cap.release()
                        self._cap_released = True
                        app_logger.debug(f"【流水线 {self.stream_id}】视频捕获资源已释放")
            except (AttributeError, cv2.error):
                # 资源可能已释放或无效
                pass
            
            # 5. 清空所有队列（使用安全方式）
            for q_name, q in [("inference", self.inference_queue), 
                             ("postprocess", self.postprocess_queue), 
                             ("output", self.output_queue)]:
                try:
                    while True:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                        except (ValueError, AttributeError):
                            break
                except Exception:
                    pass
            
            # 6. 释放数据库连接
            if hasattr(self, 'face_dao') and self.face_dao:
                try:
                    self.face_dao.dispose()
                except Exception as e:
                    app_logger.debug(f"【流水线 {self.stream_id}】释放数据库连接时出错: {e}")
            
            app_logger.info(f"✅【流水线 {self.stream_id}】已安全停止。")
            
        except Exception as e:
            app_logger.error(f"❌【流水线 {self.stream_id}】停止时发生错误: {e}")
            # 最后的资源清理尝试
            try:
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()
            except:
                pass

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
        [T1: 读帧] 读取视频帧线程，保持最新帧，无跳帧逻辑。
        """
        app_logger.info(f"【T1:读帧 {self.stream_id}】启动。")
        
        consecutive_failures = 0
        max_failures = 10
        
        try:
            while not self.stop_event.is_set():
                try:
                    # 检查视频捕获对象是否有效
                    if not hasattr(self, 'cap') or self.cap is None:
                        app_logger.error(f"【T1:读帧 {self.stream_id}】视频捕获对象无效")
                        break
                    
                    if not self.cap.isOpened():
                        app_logger.error(f"【T1:读帧 {self.stream_id}】视频捕获已关闭")
                        break
                    
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            app_logger.error(f"【T1:读帧 {self.stream_id}】连续{max_failures}次读帧失败，流可能已断开。")
                            break
                        time.sleep(min(0.1 * consecutive_failures, 1.0))
                        continue
                    
                    consecutive_failures = 0
                    
                    # 优化：只丢弃一帧而不是清空整个队列，减少CPU开销
                    try:
                        # 如果队列已满，丢弃最旧的一帧
                        if self.inference_queue.full():
                            try:
                                self.inference_queue.get_nowait()
                            except queue.Empty:
                                pass
                        
                        # 放入当前帧
                        self.inference_queue.put_nowait(frame)
                    except queue.Full:
                        # 队列仍然满，丢弃当前帧
                        pass
                    
                except Exception as e:
                    if self.stop_event.is_set():
                        break
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
                if not self.stop_event.is_set():
                    app_logger.debug(f"【T1:读帧 {self.stream_id}】发送停止信号时出错: {e}")
            app_logger.info(f"【T1:读帧 {self.stream_id}】已停止。")


    def _inference_thread(self):
        """[T2: 推理] 执行模型推理。"""
        app_logger.info(f"【T2:推理 {self.stream_id}】启动。")
        
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
                    
                    # 执行推理
                    try:
                        # 检查是否使用新的推理引擎
                        if hasattr(self.model, 'predict'):
                            # 新推理引擎 - 直接调用同步方法
                            from app.inference.models import InferenceInput
                            
                            input_data = InferenceInput(
                                image=frame,
                                extract_embeddings=True,
                                detection_threshold=self.settings.insightface.recognition_det_score_threshold
                            )
                            output = self.model.predict(input_data)
                            if output.success:
                                # 转换为InsightFace的Face对象格式
                                detected_faces = []
                                for face_detection in output.result.faces:
                                    face = self._convert_to_insightface_face(face_detection, frame.shape)
                                    detected_faces.append(face)
                            else:
                                app_logger.error(f"【T2:推理 {self.stream_id}】推理失败: {output.error_message}")
                                detected_faces = []
                        else:
                            # 传统InsightFace
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

        threshold = self.settings.inference.recognition_similarity_threshold

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
                                # 尝试获取embedding，优先使用embedding属性
                                embedding = getattr(face, 'embedding', None)
                                if embedding is None:
                                    embedding = getattr(face, 'normed_embedding', None)
                                
                                result_item = {"box": face.bbox, "name": "Unknown", "similarity": None}
                                
                                if embedding is not None:
                                    # 使用更短的超时时间进行人脸搜索，减少阻塞
                                    try:
                                        search_res = self.face_dao.search(embedding, threshold)
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
                                                    # 使用中国时区 (Asia/Shanghai)
                                                    import pytz
                                                    china_tz = pytz.timezone('Asia/Shanghai')
                                                    # 准备持久化数据
                                                    persistence_data = {
                                                        "sn": sn,
                                                        "name": name,
                                                        "similarity": similarity,
                                                        "face_crop": face_crop.copy(),  # 复制数据避免引用问题
                                                        "timestamp": datetime.now(china_tz),
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
                                    except Exception as e:
                                        # 人脸搜索失败，继续使用默认的"Unknown"结果
                                        app_logger.debug(f"人脸搜索失败: {e}")
                                
                                results.append(result_item)
                            except Exception as e:
                                if not self.stop_event.is_set():
                                    app_logger.error(f"【T3:后处理 {self.stream_id}】处理单个人脸时出错: {e}", exc_info=True)
                                # 继续处理其他人脸
                                continue

                    # 快速绘制结果并输出
                    try:
                        # 如果有原始视频输出队列，先输出原始帧
                        if self.original_output_queue is not None:
                            flag_orig, encoded_orig_image = cv2.imencode(".jpg", original_frame, 
                                                                     [cv2.IMWRITE_JPEG_QUALITY, 70])  # 降低质量提高速度
                            if flag_orig:
                                try:
                                    self.original_output_queue.put_nowait(encoded_orig_image.tobytes())
                                except queue.Full:
                                    # 输出队列满时丢弃旧帧，保持实时性
                                    pass
                        
                        # 绘制推理结果
                        _draw_results_on_frame(original_frame, results)
                        flag, encoded_image = cv2.imencode(".jpg", original_frame, 
                                                         [cv2.IMWRITE_JPEG_QUALITY, 70])  # 降低质量提高速度
                        if flag:
                            try:
                                self.output_queue.put_nowait(encoded_image.tobytes())
                            except queue.Full:
                                # 输出队列满时丢弃旧帧，保持实时性
                                pass
                    except Exception as e:
                        if not self.stop_event.is_set():
                            app_logger.error(f"【T3:后处理 {self.stream_id}】编码图像时出错: {e}", exc_info=True)
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        app_logger.error(f"【T3:后处理 {self.stream_id}】处理帧数据时发生错误: {e}", exc_info=True)
                    continue
                    
        except Exception as e:
            if not self.stop_event.is_set():
                app_logger.error(f"【T3:后处理 {self.stream_id}】线程发生致命错误: {e}", exc_info=True)
        finally:
            # 清理工作
            try:
                self.output_queue.put_nowait(None)
            except (queue.Full, ValueError):
                pass
            app_logger.info(f"【T3:后处理 {self.stream_id}】已停止。")
    
    def _convert_to_insightface_face(self, face_detection, image_shape) -> Face:
        """将推理引擎的结果转换为InsightFace的Face对象"""
        # 创建Face对象
        face = Face()
        
        # 设置边界框
        face.bbox = np.array(face_detection.bbox, dtype=np.float32)
        
        # 设置检测置信度
        face.det_score = face_detection.confidence
        
        # 设置关键点
        if face_detection.landmarks:
            face.landmark_2d_106 = np.array(face_detection.landmarks, dtype=np.float32)
        
        # 设置特征向量 - 使用embedding属性而不是normed_embedding
        if face_detection.embedding:
            try:
                face.embedding = np.array(face_detection.embedding, dtype=np.float32)
            except AttributeError:
                # 如果embedding属性不可写，尝试使用normed_embedding
                try:
                    face.normed_embedding = np.array(face_detection.embedding, dtype=np.float32)
                except AttributeError:
                    # 如果都不可写，使用setattr
                    setattr(face, 'embedding', np.array(face_detection.embedding, dtype=np.float32))
        
        # 设置其他属性
        face.img_shape = image_shape[:2]  # (height, width)
        
        return face
