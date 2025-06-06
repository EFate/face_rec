import os
from pathlib import Path
import socket
import typer
from contextlib import contextmanager
from typing import Annotated, Optional, Any

import uvicorn
from dotenv import load_dotenv

# 导入配置和日志模块
from app.cfg.config import get_app_settings, AppSettings
from app.cfg.logging import app_logger as logger, setup_logging
# 导入 create_app 函数。我们不再需要直接导入 app 实例，Uvicorn 会自己导入。
# from app.main import create_app, app as fastapi_app_instance # 移除 fastapi_app_instance 导入


# 加载 .env 文件
# 确保在任何配置或日志初始化之前加载环境变量
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# 创建 Typer 应用实例
app = typer.Typer(
    pretty_exceptions_enable=False,  # 禁用 Typer 默认的漂亮异常处理，由 Loguru 接管
    context_settings={"help_option_names": ["-h", "--help"]},
)


@contextmanager
def _apply_env_and_config(env: Optional[str] = None):
    """
    上下文管理器：设置 APP_ENV 环境变量，并加载配置和初始化日志。
    确保在此上下文块中，配置和日志系统是完全准备好的。
    """
    original_env = os.getenv("APP_ENV") # 备份原始 APP_ENV
    if env:
        # 验证环境名称
        if env not in ["development", "testing", "production"]:
            logger.error(f"❌ 无效环境名称: '{env}'。有效值为 development/testing/production")
            raise typer.Exit(code=1)
        os.environ["APP_ENV"] = env # 设置新的 APP_ENV

    try:
        # 清除配置缓存并重新加载配置
        # 这确保了每次 `_apply_env_and_config` 被调用时，
        # 都会根据当前 `APP_ENV` 重新加载最新配置。
        get_app_settings.cache_clear()
        current_settings: AppSettings = get_app_settings()

        # 设置日志配置
        # 必须在配置加载后调用，因为日志配置依赖于 `current_settings`
        setup_logging(current_settings)

        # 记录环境信息
        logger.info(f"⚙️ 应用程序环境已设置为: {current_settings.app.title} v{current_settings.app.version}")
        yield current_settings # 将配置实例传递给上下文块

    finally:
        # 恢复原始环境变量，避免影响 Typer 外部或其他后续操作
        if original_env is not None:
            os.environ["APP_ENV"] = original_env
        else:
            os.unsetenv("APP_ENV")


def get_local_ip() -> str:
    """
    获取本机的可用外网IP地址（备用方案）。
    通过连接到一个公共DNS服务（如 Google DNS）来获取本地连接的IP地址。
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # 使用 Google DNS 保证连接成功
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1" # 获取失败时返回环回地址


def _print_config_details(settings_instance: AppSettings):
    """
    打印当前应用的详细配置信息。
    敏感信息（如密钥）会进行脱敏处理。
    """
    logger.info("\n--- 📦 应用配置 (AppConfig) ---")
    logger.info(f"  📝 应用标题 (title): {settings_instance.app.title}")
    logger.info(f"  📄 描述 (description): {settings_instance.app.description}")
    logger.info(f"  🔧 版本 (version): {settings_instance.app.version}")
    logger.info(f"  🔐 调试模式 (debug): {'✅ 开启' if settings_instance.app.debug else '❌ 关闭'}")

    logger.info("\n--- ⚙️ 服务器配置 (ServerConfig) ---")
    logger.info(f"  🌐 主机 (host): {settings_instance.server.host}")
    logger.info(f"  🔁 端口 (port): {settings_instance.server.port}")
    logger.info(f"  🔄 热重载 (reload): {'✅ 开启' if settings_instance.server.reload else '❌ 关闭'}")

    logger.info("\n--- 📂 日志配置 (LoggingConfig) ---")
    logger.info(f"  📜 日志级别 (level): {settings_instance.logging.level.upper()}")
    logger.info(f"  🗃️ 日志文件路径 (file_path): {settings_instance.logging.file_path.resolve()}")
    logger.info(f"  🧱 单个日志文件大小 (max_bytes): {settings_instance.logging.max_bytes} Bytes ({settings_instance.logging.max_bytes / (1024*1024):.2f} MB)")
    logger.info(f"  🔁 备份数量 (backup_count): {settings_instance.logging.backup_count}")

    logger.info("\n--- 🗄️ 数据库配置 (DatabaseConfig) ---")
    logger.info(f"  🔌 数据库连接 URL (url): {settings_instance.database.url}")
    logger.info(f"  📜 打印 SQL 语句 (echo): {'✅ 开启' if settings_instance.database.echo else '❌ 关闭'}")

    logger.info("\n--- 🔐 安全配置 (SecurityConfig) ---")
    secret_key = settings_instance.security.secret_key
    masked_key = secret_key[:8] + "..." + secret_key[-4:] if len(secret_key) > 12 else "****"
    logger.info(f"  🔑 安全密钥 (secret_key): {masked_key}")
    logger.info(f"  🔒 JWT 算法 (algorithm): {settings_instance.security.algorithm}")
    logger.info(f"  ⏳ 令牌过期时间 (access_token_expire_minutes): {settings_instance.security.access_token_expire_minutes} 分钟")


@app.callback()
def _main_callback(
    env: Annotated[
        Optional[str],
        typer.Option(
            "--env",
            "-e",
            help="指定运行环境 (development, testing, production)。",
            envvar="APP_ENV", # 从环境变量 APP_ENV 读取
            show_envvar=True, # 在帮助信息中显示环境变量
        ),
    ] = None,
    show_config: Annotated[
        bool,
        typer.Option(
            "--show-config",
            "-s",
            help="显示当前环境的详细配置信息。",
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="显示应用版本信息。",
            is_eager=True, # 立即处理此选项
            is_flag=True, # 这是一个布尔标志
        ),
    ] = False,
):
    """
    FastAPI 应用的命令行接口主回调函数。
    处理全局选项，如环境设置、显示配置或版本信息。
    """
    if show_config:
        with _apply_env_and_config(env) as settings:
            _print_config_details(settings)
        raise typer.Exit() # 显示完配置后退出

    if version:
        with _apply_env_and_config(env) as settings:
            print(f"{settings.app.title} 版本: {settings.app.version}")
        raise typer.Exit() # 显示完版本后退出


@app.command(name="start")
def start_server(
    env: Annotated[
        Optional[str],
        typer.Option(
            "--env",
            "-e",
            help="指定运行环境 (development, testing, production)。",
            envvar="APP_ENV",
            show_envvar=True,
        ),
    ] = None,
):
    """
    启动 FastAPI Uvicorn 服务器。
    此命令会根据指定的或默认的环境加载配置，并初始化日志系统，然后启动Uvicorn。
    """
    with _apply_env_and_config(env) as settings:
        local_ip = get_local_ip()
        logger.info(f"\n🚀 准备启动服务器：{settings.app.title} v{settings.app.version}")
        logger.info("🔧 配置概览:")
        logger.info(f"  - 主机 (Host): {settings.server.host}")
        logger.info(f"  - 端口 (Port): {settings.server.port}")
        logger.info(f"  - 本机IP (Local IP): {local_ip}")
        logger.info(f"  - 调试模式 (Debug): {'✅ 开启' if settings.app.debug else '❌ 关闭'}")
        logger.info(f"  - 热重载 (Reload): {'✅ 开启' if settings.server.reload else '❌ 关闭'}")

        try:
            # 启动 Uvicorn 服务器
            # 注意：这里将应用传入字符串 "app.main:app"
            # 这使得 Uvicorn 可以在 reload 模式下正确地重新加载应用程序。
            # app.main 模块被导入时，其顶层的 if app is None: 逻辑会执行 create_app。
            uvicorn.run(
                "app.main:app", # 使用导入字符串
                host=settings.server.host,
                port=settings.server.port,
                reload=settings.server.reload,
                log_level=settings.logging.level.lower(),
                # Loguru 会接管日志，这里 Uvicorn 的 log_config 可以设为 None，
                # 但为了兼容性，保留 log_level。
                log_config=None # 禁用 Uvicorn 自身的日志配置，完全由 Loguru 接管
            )
        except Exception as e:
            logger.critical(f"⚠️ Uvicorn 服务器启动失败: {e}", exc_info=True)
            typer.Exit(code=1)


@app.command(name="config")
def show_current_config_command():
    """
    显示当前激活环境的详细配置信息。
    这是一个独立的命令，用于方便地查看配置。
    """
    with _apply_env_and_config() as settings:
        _print_config_details(settings)


if __name__ == "__main__":
    app()