#!/usr/bin/env python3
"""
简化测试：验证线程停止逻辑
"""

import time
import threading
import queue
import signal
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.pipeline import FaceStreamPipeline
from app.cfg.config import get_app_settings

def test_thread_stop():
    """测试线程停止逻辑"""
    print("🧪 开始测试线程停止逻辑...")
    
    try:
        # 创建模拟队列
        output_queue = queue.Queue()
        result_persistence_queue = queue.Queue()
        
        # 获取设置
        settings = get_app_settings()
        
        # 创建模拟模型对象
        class MockModel:
            def __init__(self):
                self.name = "mock_model"
        
        # 创建流水线实例（使用测试参数）
        pipeline = FaceStreamPipeline(
            settings=settings,
            stream_id="test_stop",
            video_source="test.mp4",  # 使用不存在的视频文件
            output_queue=output_queue,
            model=MockModel(),
            result_persistence_queue=result_persistence_queue,
            task_id=1,
            app_id=1,
            app_name="test_app",
            domain_name="test_domain"
        )
        
        print("✅ 流水线实例创建成功")
        
        # 模拟线程创建
        pipeline.threads = []
        
        # 创建测试线程
        def test_worker():
            count = 0
            while not pipeline.stop_event.is_set():
                time.sleep(0.1)
                count += 1
                if count > 50:  # 最多运行5秒
                    break
        
        test_thread = threading.Thread(target=test_worker, name="TestWorker")
        pipeline.threads.append(test_thread)
        
        # 启动线程
        test_thread.start()
        print("✅ 测试线程启动成功")
        
        # 运行2秒后停止
        time.sleep(2)
        
        # 测试停止逻辑
        print("🛑 正在停止流水线...")
        start_time = time.time()
        
        # 调用停止方法
        pipeline.stop()
        
        stop_time = time.time()
        print(f"✅ 停止完成，耗时: {stop_time - start_time:.2f}秒")
        
        # 验证线程已停止
        test_thread.join(timeout=1.0)
        if test_thread.is_alive():
            print("⚠️  测试线程仍在运行")
            return False
        else:
            print("✅ 测试线程已停止")
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
    success = test_thread_stop()
    
    if success:
        print("🎉 线程停止测试通过！")
    else:
        print("💥 线程停止测试失败！")
        sys.exit(1)