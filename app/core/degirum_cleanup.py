# app/core/degirum_cleanup.py
"""
Degirum资源清理管理器
确保Degirum工作进程在应用退出时正确清理
"""

import os
import signal
import atexit
import psutil
import asyncio
from typing import Set
from app.cfg.logging import app_logger


class DegirumCleanupManager:
    """Degirum资源清理管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._cleanup_registered = False
        self._initialized = True
        app_logger.info("Degirum资源清理管理器已初始化")
    
    def register_cleanup(self):
        """注册清理函数"""
        if self._cleanup_registered:
            return
        
        # 注册程序退出时的清理函数
        atexit.register(self._cleanup_on_exit)
        
        # 注册信号处理器
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self._cleanup_registered = True
        app_logger.info("Degirum资源清理函数已注册")
    
    def get_all_degirum_worker_pids(self) -> Set[int]:
        """获取所有DeGirum工作进程的PID"""
        worker_pids = set()
        current_pid = os.getpid()
        
        try:
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline')
                    if cmdline and any("degirum/pproc_worker.py" in s for s in cmdline):
                        # 检查是否是当前进程的子进程
                        if any(f"--parent_pid {current_pid}" in s for s in cmdline):
                            worker_pids.add(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            app_logger.error(f"获取Degirum工作进程PID时出错: {e}")
        
        return worker_pids
    
    def get_all_degirum_worker_pids_global(self) -> Set[int]:
        """获取所有DeGirum工作进程的PID（全局）"""
        worker_pids = set()
        
        try:
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline')
                    if cmdline and any("degirum/pproc_worker.py" in s for s in cmdline):
                        worker_pids.add(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            app_logger.error(f"获取全局Degirum工作进程PID时出错: {e}")
        
        return worker_pids
    
    def cleanup_degirum_workers_by_pids(self, pids_to_kill: Set[int]):
        """强制终止指定的DeGirum工作进程"""
        killed_count = 0
        
        for pid in pids_to_kill:
            try:
                os.kill(pid, signal.SIGKILL)
                app_logger.info(f"已终止Degirum工作进程 PID: {pid}")
                killed_count += 1
            except ProcessLookupError:
                app_logger.debug(f"进程 PID {pid} 已不存在")
            except Exception as e:
                app_logger.error(f"终止进程 PID {pid} 时出错: {e}")
        
        if killed_count > 0:
            app_logger.info(f"成功终止了 {killed_count} 个Degirum工作进程")
    
    def cleanup_all_degirum_workers(self):
        """清理所有Degirum工作进程"""
        try:
            app_logger.info("开始清理Degirum工作进程")
            
            # 获取所有Degirum工作进程
            worker_pids = self.get_all_degirum_worker_pids()
            
            if worker_pids:
                app_logger.info(f"发现 {len(worker_pids)} 个Degirum工作进程需要清理")
                self.cleanup_degirum_workers_by_pids(worker_pids)
            else:
                app_logger.info("未发现需要清理的Degirum工作进程")
                
        except Exception as e:
            app_logger.error(f"清理Degirum工作进程时出错: {e}")
    
    def cleanup_all_degirum_workers_global(self):
        """清理所有Degirum工作进程（全局）"""
        try:
            app_logger.info("开始全局清理Degirum工作进程")
            
            # 获取所有Degirum工作进程
            worker_pids = self.get_all_degirum_worker_pids_global()
            
            if worker_pids:
                app_logger.info(f"发现 {len(worker_pids)} 个全局Degirum工作进程需要清理")
                self.cleanup_degirum_workers_by_pids(worker_pids)
            else:
                app_logger.info("未发现需要清理的全局Degirum工作进程")
                
        except Exception as e:
            app_logger.error(f"全局清理Degirum工作进程时出错: {e}")
    
    def _cleanup_on_exit(self):
        """程序退出时的清理函数"""
        try:
            app_logger.info("程序退出，开始清理Degirum资源")
            self.cleanup_all_degirum_workers()
        except Exception as e:
            app_logger.error(f"程序退出时清理Degirum资源失败: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        app_logger.info(f"收到信号 {signum}，开始清理Degirum资源")
        self._cleanup_on_exit()
        exit(0)
    
    def force_cleanup(self):
        """强制清理所有Degirum资源"""
        try:
            app_logger.info("执行强制Degirum资源清理")
            
            # 清理工作进程
            self.cleanup_all_degirum_workers()
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            app_logger.info("Degirum资源强制清理完成")
            
        except Exception as e:
            app_logger.error(f"强制清理Degirum资源时出错: {e}")


# 全局清理管理器实例
degirum_cleanup_manager = DegirumCleanupManager()


def register_degirum_cleanup():
    """注册Degirum资源清理"""
    degirum_cleanup_manager.register_cleanup()


def cleanup_degirum_resources():
    """清理Degirum资源"""
    degirum_cleanup_manager.force_cleanup()
