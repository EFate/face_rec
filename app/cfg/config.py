# app/cfg/config.py
import os
from pathlib import Path
from typing import List, Any, Dict
from functools import lru_cache
from pydantic import BaseModel, Field, BeforeValidator, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 路径配置
def get_base_dir() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent.parent

BASE_DIR = get_base_dir()
ENV_FILE = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# 获取环境变量
IMAGE_URL_IP = os.getenv("HOST__IP", "localhost")

# 自定义类型
FilePath = Annotated[Path, BeforeValidator(lambda v: Path(v) if isinstance(v, str) else v)]


class AppConfig(BaseModel):
    """应用程序配置"""
    title: str = Field("FastAPI人脸识别服务", description="应用程序名称")
    description: str = Field("基于FastAPI、InsightFace和LanceDB构建的高性能人脸识别服务", description="应用程序描述")
    version: str = Field("4.5.0-Final", description="应用程序版本")
    debug: bool = Field(False, description="是否开启调试模式")
    host_ip: str = Field(f"{IMAGE_URL_IP}", description="宿主机ip，用于访问保存的图片资源")
    max_concurrent_tasks: int = Field(3, description="最大并发AI任务数（模型池大小）")
    detected_imgs_path: FilePath = Field(DATA_DIR / "detected_imgs", description="检测图片保存路径")
    stream_default_lifetime_minutes: int = Field(10, description="视频流默认生命周期（分钟）")
    stream_cleanup_interval_seconds: int = Field(60, description="清理过期视频流的间隔（秒）")
    recognition_save_interval_seconds: float = Field(0.6, description="同一人识别结果的最小保存间隔（秒）")
    recognition_frame_interval: int = Field(10, description="同一人识别结果的最小帧间隔（每隔多少帧保存一次）")
    chinese_font_path: FilePath = Field(BASE_DIR / "app" / "static" / "SimHei.ttf", description="中文字体文件路径")

    def model_post_init__(self, __context: Any) -> None:
        """模型初始化后的处理"""
        if self.detected_imgs_path:
            self.detected_imgs_path.mkdir(parents=True, exist_ok=True)


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = Field("0.0.0.0", description="服务器监听地址")
    port: int = Field(8000, description="服务器监听端口")
    reload: bool = Field(False, description="是否开启热重载")


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field("ERROR", description="日志级别")
    file_path: FilePath = Field(LOGS_DIR / "app.log", description="日志文件路径")
    max_bytes: int = Field(50 * 1024 * 1024, description="单个日志文件最大字节数（50MB）")
    backup_count: int = Field(10, description="日志文件备份数量")

    def model_post_init__(self, __context: Any) -> None:
        """模型初始化后的处理"""
        if self.file_path:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseModel):
    """数据库配置"""
    url: str = Field("sqlite:///./data/app.db", description="数据库连接URL")
    echo: bool = Field(False, description="是否打印SQL语句")


class CudaConfig(BaseModel):
    """CUDA推理配置"""
    model_pack_name: str = Field("buffalo_l", description="InsightFace模型包名称")
    providers: List[str] = Field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        description="ONNX Runtime执行提供者列表"
    )
    detection_size: List[int] = Field([640, 640], description="人脸检测模型输入尺寸")
    det_thresh: float = Field(0.4, description="人脸检测置信度阈值")
    home: FilePath = Field(DATA_DIR / ".insightface", description="InsightFace模型存储目录")

    def model_post_init__(self, __context: Any) -> None:
        """模型初始化后的处理"""
        if self.home:
            self.home.mkdir(parents=True, exist_ok=True)


class Hailo8Config(BaseModel):
    """Hailo8推理配置"""
    zoo_path: FilePath = Field(DATA_DIR / "zoo", description="模型仓库路径")
    detection_model: str = Field("scrfd_10g--640x640_quant_hailort_hailo8_1", description="检测模型名称")
    recognition_model: str = Field("arcface_mobilefacenet--112x112_quant_hailort_hailo8_1", description="识别模型名称")
    detection_size: List[int] = Field([640, 640], description="检测模型输入尺寸")
    recognition_size: List[int] = Field([112, 112], description="识别模型输入尺寸")

    def model_post_init__(self, __context: Any) -> None:
        """模型初始化后的处理"""
        if self.zoo_path:
            self.zoo_path.mkdir(parents=True, exist_ok=True)


class RK3588Config(BaseModel):
    """RK3588推理配置"""
    zoo_path: FilePath = Field(DATA_DIR / "zoo", description="模型仓库路径")
    detection_model: str = Field("yolov8s_relu6_widerface_kpts--640x640_quant_rknn_rk3588_1", description="检测模型名称")
    recognition_model: str = Field("mbf_w600k--112x112_float_rknn_rk3588_1", description="识别模型名称")
    detection_size: List[int] = Field([640, 640], description="检测模型输入尺寸")
    recognition_size: List[int] = Field([112, 112], description="识别模型输入尺寸")

    def model_post_init__(self, __context: Any) -> None:
        """模型初始化后的处理"""
        if self.zoo_path:
            self.zoo_path.mkdir(parents=True, exist_ok=True)


class InferenceConfig(BaseModel):
    """推理引擎配置"""
    device_type: str = Field("hailo8", description="推理设备类型: cuda, hailo8, rk3588")
    cuda: CudaConfig = Field(default_factory=CudaConfig, description="CUDA推理配置")
    hailo8: Hailo8Config = Field(default_factory=Hailo8Config, description="Hailo8推理配置")
    rk3588: RK3588Config = Field(default_factory=RK3588Config, description="RK3588推理配置")
    recognition_similarity_threshold: float = Field(0.3, description="人脸识别相似度阈值")
    recognition_det_score_threshold: float = Field(0.2, description="人脸检测置信度阈值（识别时使用）")
    registration_det_score_threshold: float = Field(0.2, description="人脸检测置信度阈值（注册时使用，更宽松）")
    image_db_path: FilePath = Field(DATA_DIR / "faces", description="注册人脸图像存储目录")
    lancedb_uri: str = Field(str(DATA_DIR / "lancedb"), description="LanceDB数据库存储目录")
    lancedb_table_name: str = Field("faces_v2", description="人脸特征表名")

    @validator('device_type')
    def validate_device_type(cls, v):
        """验证设备类型"""
        supported_devices = ['cuda', 'hailo8', 'rk3588']
        if v not in supported_devices:
            raise ValueError(f'device_type must be one of: {supported_devices}')
        return v
    
    def get_device_config(self) -> Dict[str, Any]:
        """获取当前设备的配置"""
        if self.device_type == 'cuda':
            return {
                'device_type': self.device_type,
                'model_pack_name': self.cuda.model_pack_name,
                'providers': self.cuda.providers,
                'detection_size': self.cuda.detection_size,
                'recognition_size': self.cuda.recognition_size,
                'recognition_similarity_threshold': self.recognition_similarity_threshold,
                'recognition_det_score_threshold': self.recognition_det_score_threshold,
                'registration_det_score_threshold': self.registration_det_score_threshold,
                'image_db_path': str(self.image_db_path),
                'lancedb_uri': self.lancedb_uri,
                'lancedb_table_name': self.lancedb_table_name,
                'home': str(self.cuda.home)
            }
        elif self.device_type == 'hailo8':
            return {
                'device_type': self.device_type,
                'zoo_path': str(self.hailo8.zoo_path),
                'detection_model': self.hailo8.detection_model,
                'recognition_model': self.hailo8.recognition_model,
                'detection_size': self.hailo8.detection_size,
                'recognition_size': self.hailo8.recognition_size,
                'recognition_similarity_threshold': self.recognition_similarity_threshold,
                'recognition_det_score_threshold': self.recognition_det_score_threshold,
                'registration_det_score_threshold': self.registration_det_score_threshold,
                'image_db_path': str(self.image_db_path),
                'lancedb_uri': self.lancedb_uri,
                'lancedb_table_name': self.lancedb_table_name
            }
        elif self.device_type == 'rk3588':
            return {
                'device_type': self.device_type,
                'zoo_path': str(self.rk3588.zoo_path),
                'detection_model': self.rk3588.detection_model,
                'recognition_model': self.rk3588.recognition_model,
                'detection_size': self.rk3588.detection_size,
                'recognition_size': self.rk3588.recognition_size,
                'recognition_similarity_threshold': self.recognition_similarity_threshold,
                'recognition_det_score_threshold': self.recognition_det_score_threshold,
                'registration_det_score_threshold': self.registration_det_score_threshold,
                'image_db_path': str(self.image_db_path),
                'lancedb_uri': self.lancedb_uri,
                'lancedb_table_name': self.lancedb_table_name
            }
        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

    def model_post_init__(self, __context: Any) -> None:
        """模型初始化后的处理"""
        if self.image_db_path:
            self.image_db_path.mkdir(parents=True, exist_ok=True)
        Path(self.lancedb_uri).mkdir(parents=True, exist_ok=True)
    
    def get_device_config(self) -> Dict[str, Any]:
        """获取当前设备类型的配置"""
        if self.device_type == "cuda":
            return {
                "model_pack_name": self.cuda.model_pack_name,
                "providers": self.cuda.providers,
                "home": str(self.cuda.home),
                "detection_size": self.cuda.detection_size,
                "det_thresh": self.cuda.det_thresh
            }
        elif self.device_type == "hailo8":
            return {
                "zoo_path": str(self.hailo8.zoo_path),
                "detection_model": self.hailo8.detection_model,
                "recognition_model": self.hailo8.recognition_model,
                "detection_size": self.hailo8.detection_size,
                "recognition_size": self.hailo8.recognition_size
            }
        elif self.device_type == "rk3588":
            return {
                "zoo_path": str(self.rk3588.zoo_path),
                "detection_model": self.rk3588.detection_model,
                "recognition_model": self.rk3588.recognition_model,
                "detection_size": self.rk3588.detection_size,
                "recognition_size": self.rk3588.recognition_size
            }
        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")


# 保持向后兼容性
class InsightFaceConfig(BaseModel):
    """InsightFace配置（向后兼容）"""
    model_pack_name: str = Field("buffalo_l", description="InsightFace模型包名称")
    providers: List[str] = Field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        description="ONNX Runtime执行提供者列表"
    )
    recognition_similarity_threshold: float = Field(0.5, description="人脸识别相似度阈值")
    recognition_det_score_threshold: float = Field(0.4, description="人脸检测置信度阈值（识别时使用）")
    registration_det_score_threshold: float = Field(0.2, description="人脸检测置信度阈值（注册时使用，更宽松）")
    detection_size: List[int] = Field([640, 640], description="人脸检测模型输入尺寸")
    home: FilePath = Field(DATA_DIR / ".insightface", description="InsightFace模型存储目录")
    image_db_path: FilePath = Field(DATA_DIR / "faces", description="注册人脸图像存储目录")
    lancedb_uri: str = Field(str(DATA_DIR / "lancedb"), description="LanceDB数据库存储目录")
    lancedb_table_name: str = Field("faces_v2", description="人脸特征表名")

    def model_post_init__(self, __context: Any) -> None:
        """模型初始化后的处理"""
        if self.home:
            self.home.mkdir(parents=True, exist_ok=True)
        if self.image_db_path:
            self.image_db_path.mkdir(parents=True, exist_ok=True)
        Path(self.lancedb_uri).mkdir(parents=True, exist_ok=True)


class MQTTConfig(BaseModel):
    """MQTT配置"""
    enabled: bool = Field(True, description="是否启用MQTT功能")
    broker_host: str = Field("172.16.104.108", description="MQTT服务器地址")
    broker_port: int = Field(1883, description="MQTT服务器端口")
    username: str = Field("abtnet", description="MQTT用户名")
    password: str = Field("Abt@Rabbit#123", description="MQTT密码")
    keepalive: int = Field(60, description="保持连接的时间间隔(秒)")
    topic_prefix: str = Field("abt/visio/face", description="MQTT主题前缀")
    device_address: str = Field(default_factory=lambda: os.getenv("HOST__IP", "172.16.104.111"), description="设备地址，从环境变量HOST__IP获取")
    app_type: str = Field("FACE_DETECT", description="应用类型")
    max_queue_size: int = Field(1000, description="MQTT消息队列最大大小")
    publish_interval: float = Field(0.1, description="MQTT消息发布间隔(秒)")
    
    def get_detection_topic(self) -> str:
        """获取检测结果主题，格式：abt/visio/face/ip"""
        return f"{self.topic_prefix}/{self.device_address}"


class AppSettings(BaseSettings):
    """应用程序设置 - 完全通过环境变量配置"""
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    insightface: InsightFaceConfig = Field(default_factory=InsightFaceConfig)  # 保持向后兼容
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
        env_nested_delimiter="__",
    )


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    """获取应用程序设置（单例模式）
    完全通过环境变量配置，优先级：环境变量 > 默认值
    """
    return AppSettings()
