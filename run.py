# run.py
import socket
from typing import Optional
import typer
import uvicorn

from app.cfg.config import get_app_settings, AppSettings
from app.cfg.logging import app_logger as logger, setup_logging

# 创建 Typer 应用实例
app = typer.Typer(
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def init_app_state() -> AppSettings:
    """初始化应用状态：加载配置并配置日志"""
    # 加载配置
    settings = get_app_settings()
    
    # 设置日志系统
    setup_logging(settings)
    
    logger.info("应用配置加载完成")
    return settings


def get_local_ip() -> str:
    """获取本机IP地址"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    show_config: bool = typer.Option(False, "--show-config", help="显示配置信息"),
    version: bool = typer.Option(False, "--version", "-v", help="显示版本信息"),
):
    """FastAPI人脸识别服务 - 命令行接口"""
    # 初始化应用状态
    settings = init_app_state()
    ctx.obj = settings

    if version:
        typer.echo(f"{settings.app.title} - Version: {settings.app.version}")
        raise typer.Exit()

    if show_config:
        logger.info("应用配置:")
        logger.info(f"  标题: {settings.app.title}")
        logger.info(f"  版本: {settings.app.version}")
        logger.info(f"  调试模式: {settings.app.debug}")
        logger.info(f"  服务器: {settings.server.host}:{settings.server.port}")
        logger.info(f"  日志级别: {settings.logging.level}")
        logger.info(f"  数据库: {settings.database.url}")
        logger.info(f"  模型包: {settings.insightface.model_pack_name}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo("使用 'start' 启动服务，或 '--help' 查看帮助")


@app.command(name="start")
def start_server(
    ctx: typer.Context,
    host: Optional[str] = typer.Option(None, "--host", help="服务器主机地址"),
    port: Optional[int] = typer.Option(None, "--port", help="服务器端口"),
):
    """启动 FastAPI 服务器"""
    settings: AppSettings = ctx.obj

    # 命令行参数优先级最高
    final_host = host or settings.server.host
    final_port = port or settings.server.port
    final_reload = settings.server.reload

    logger.info(f"启动服务器: {settings.app.title} v{settings.app.version}")
    logger.info(f"监听地址: http://{final_host}:{final_port}")
    
    if final_host == "0.0.0.0":
        local_ip = get_local_ip()
        logger.info(f"本机访问: http://127.0.0.1:{final_port}")
        logger.info(f"局域网访问: http://{local_ip}:{final_port}")
    
    logger.info(f"热重载: {'开启' if final_reload else '关闭'}")

    try:
        uvicorn.run(
            "app.main:app",
            host=final_host,
            port=final_port,
            reload=final_reload,
            log_level=settings.logging.level.lower(),
            log_config=None
        )
    except Exception as e:
        logger.critical(f"服务器启动失败: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    # 设置多进程启动方式
    # import multiprocessing as mp
    # try:
    #     mp.set_start_method('spawn', force=True)
    #     logger.info("多进程启动方式设置为 'spawn'")
    # except RuntimeError:
    #     pass
    
    app()