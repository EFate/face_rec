# app/cfg/config.py
import os
from enum import Enum
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache
from pydantic import BaseModel, Field, BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated


# --- 路径定义 ---
def get_base_dir() -> Path:
    """计算项目根目录。"""
    return Path(__file__).resolve().parent.parent.parent

BASE_DIR = get_base_dir()
ENV_FILE = BASE_DIR / ".env"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "app" / "cfg"
DATA_DIR = BASE_DIR / "data"
# InsightFace 模型默认下载目录在用户主目录的 .insightface 下，
# 我们通过环境变量在 model_manager 中重定向
INSIGHTFACE_MODELS_DIR = BASE_DIR / "data" / ".insightface"


# --- 自定义类型 ---
FilePath = Annotated[Path, BeforeValidator(lambda v: Path(v) if isinstance(v, str) else v)]

# --- 存储类型枚举 ---
class StorageType(str, Enum):
    SQLITE = "sqlite"
    CSV = "csv"

# --- 配置模型定义 ---
class AppConfig(BaseModel):
    title: str = Field("高性能人脸识别服务", description="应用程序名称。")
    description: str = Field("基于FastAPI和DeepFace构建的高性能、生产级应用", description="应用程序描述。")
    version: str = Field("2.0.0", description="应用程序版本。")
    debug: bool = Field(False, description="是否开启调试模式。")
    video_width: int = Field(640, description="视频流宽度。")
    video_height: int = Field(480, description="视频流高度。")
    stream_default_lifetime_minutes: int = Field(10, description="视频流默认生命周期（分钟）。-1表示永久。")
    stream_cleanup_interval_seconds: int = Field(30, description="清理过期视频流的后台任务运行间隔（秒）。")

class ServerConfig(BaseModel):
    host: str = Field("0.0.0.0", description="服务器监听地址。")
    port: int = Field(8000, description="服务器监听端口。")
    reload: bool = Field(False, description="是否开启热重载（仅开发环境）。")

class LoggingConfig(BaseModel):
    level: str = Field("INFO", description="日志级别。")
    file_path: FilePath = Field(LOGS_DIR / "app.log", description="日志文件绝对路径。")
    max_bytes: int = Field(10 * 1024 * 1024, description="单个日志文件最大字节数（10MB）。")
    backup_count: int = Field(5, description="日志文件备份数量。")

    def model_post_init__(self, __context: Any) -> None:
        """确保日志文件目录存在。"""
        if self.file_path:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

class DatabaseConfig(BaseModel):
    url: str = Field(f"sqlite:///{DATA_DIR / 'face_features.db'}", description="数据库连接URL。")
    echo: bool = Field(False, description="是否打印SQL语句。")

    def model_post_init__(self, __context: Any) -> None:
        """为 SQLite 数据库确保数据目录存在。"""
        if self.url.startswith("sqlite:///") and ":memory:" not in self.url:
            db_path_str = self.url.split("sqlite:///")[1]
            if not Path(db_path_str).is_absolute():
                db_path = BASE_DIR / db_path_str
            else:
                db_path = Path(db_path_str)
            db_path.parent.mkdir(parents=True, exist_ok=True)

class SecurityConfig(BaseModel):
    secret_key: str = Field("a_very_secure_default_secret_key_change_me", description="用于签名和加密的密钥。")
    algorithm: str = Field("HS256", description="JWT签名算法。")
    access_token_expire_minutes: int = Field(30, description="访问令牌过期时间（分钟）。")


class InsightFaceConfig(BaseModel):
    model_pack_name: str = Field(
        "buffalo_l",
        description="InsightFace 模型包名称 (例如: 'buffalo_l', 'buffalo_s', 'antelopev2')。"
    )
    providers: List[str] = Field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        description="ONNX Runtime 执行提供者列表。例如: ['CUDAExecutionProvider', 'CPUExecutionProvider']。"
    )
    recognition_threshold: float = Field(
        0.5,
        description="人脸识别余弦距离阈值，距离小于此值认为匹配成功。"
    )
    home: FilePath = Field(
        INSIGHTFACE_MODELS_DIR,
        description="InsightFace 模型下载和缓存的根目录。"
    )
    image_db_path: FilePath = Field(
        DATA_DIR / "faces",
        description="用于存储注册人脸图像的根目录。"
    )
    features_file_name: str = Field(
        "face_features",
        description="存储人脸特征的文件名（不含扩展名）。"
    )
    storage_type: StorageType = Field(
        StorageType.SQLITE,
        description="选择存储后端：sqlite (生产推荐) 或 csv (仅原型)。"
    )
    # --- 视频流分析的配置开关 ---
    enable_stream_analysis: bool = Field(
        False,
        description="是否在视频流中开启附加分析（年龄、性别等）。默认为False以节省资源。"
    )
    stream_analysis_actions: List[str] = Field(
        default_factory=lambda: ["age", "gender"],
        description="如果开启分析，要执行的项目列表。"
    )


# --- 主配置类 ---
class AppSettings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    insightface: InsightFaceConfig = Field(default_factory=InsightFaceConfig)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
        env_nested_delimiter="__",
    )


# --- YAML 配置加载器 ---
class ConfigLoader:
    @staticmethod
    def load_yaml_configs(env: Optional[str] = None) -> Dict[str, Any]:
        current_env = env or os.getenv("APP_ENV", "development").lower()
        config: Dict[str, Any] = {}

        default_path = CONFIG_DIR / "default.yaml"
        if default_path.exists():
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"警告：加载默认配置文件 {default_path} 失败: {e}")

        env_path = CONFIG_DIR / f"{current_env}.yaml"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    env_config = yaml.safe_load(f) or {}
                    config = ConfigLoader._deep_merge_dicts(config, env_config)
            except Exception as e:
                print(f"警告：加载环境特定配置文件 {env_path} 失败: {e}")
        return config

    @staticmethod
    def _deep_merge_dicts(base: Dict, updates: Dict) -> Dict:
        merged = base.copy()
        for key, value in updates.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigLoader._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged


# --- 配置加载接口 (单例模式) ---
@lru_cache(maxsize=1)
def get_app_settings(env_override: Optional[str] = None) -> AppSettings:
    current_env = env_override or os.getenv("APP_ENV", "development")
    yaml_config = ConfigLoader.load_yaml_configs(current_env)
    base_settings = AppSettings.model_validate(yaml_config)
    env_settings = AppSettings()
    
    # 优化合并逻辑，确保环境变量优先
    env_data_to_merge = env_settings.model_dump(exclude_unset=True)
    final_merged_data = ConfigLoader._deep_merge_dicts(base_settings.model_dump(), env_data_to_merge)
    settings = AppSettings.model_validate(final_merged_data)
    
    return settings