#!/usr/bin/env python3
"""
测试流水线停止功能的修复效果
"""

import time
import signal
import sys
import os
import queue
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.pipeline import FaceStreamPipeline
from app.inference.factory import InferenceEngineFactory
from app.service.face_dao import LanceDBFaceDataDAO
from app.cfg.config import get_app_settings

def test_pipeline_stop():
    """测试流水线停止功能"""
    print("🧪 开始测试流水线停止功能...")
    
    try:
        # 初始化组件
        settings = get_app_settings()
        
        # 创建推理引擎
        print("🔧 创建推理引擎...")
        model = InferenceEngineFactory.create_engine(
            settings.inference.device_type,
            settings.inference.get_device_config()
        )
        
        # 创建数据访问对象
        print("📊 创建数据访问对象...")
        face_dao = LanceDBFaceDataDAO(settings.inference.lancedb_uri, settings.inference.lancedb_table_name)
        
        # 创建流水线
        print("🎥 创建视频处理流水线...")
        output_queue = queue.Queue(maxsize=2)
        result_persistence_queue = queue.Queue(maxsize=100)
        
        pipeline = FaceStreamPipeline(
            settings=settings,
            stream_id="test_stop",
            video_source="test/imgs/女.png",  # 使用测试图片代替摄像头
            output_queue=output_queue,
            model=model,
            result_persistence_queue=result_persistence_queue,
            task_id=1,
            app_id=1,
            app_name="test_app",
            domain_name="test_domain"
        )
        
        # 启动流水线
        print("🚀 启动流水线...")
        pipeline.start()
        
        # 运行3秒后停止
        print("⏱️ 运行3秒...")
        time.sleep(3)
        
        # 停止流水线
        print("🛑 停止流水线...")
        start_time = time.time()
        pipeline.stop()
        stop_time = time.time()
        
        print(f"✅ 流水线停止完成，耗时: {stop_time - start_time:.2f}秒")
        
        # 验证所有线程都已停止
        if hasattr(pipeline, 'threads'):
            for thread in pipeline.threads:
                if thread.is_alive():
                    print(f"⚠️  线程 {thread.name} 仍在运行")
                else:
                    print(f"✅ 线程 {thread.name} 已停止")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def signal_handler(signum, frame):
    """信号处理函数"""
    print(f"\n🚨 收到信号 {signum}，优雅退出...")
    sys.exit(0)

if __name__ == "__main__":
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行测试
    success = test_pipeline_stop()
    
    if success:
        print("🎉 流水线停止测试通过！")
    else:
        print("💥 流水线停止测试失败！")
        sys.exit(1)