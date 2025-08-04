#!/usr/bin/env python3
"""
myStock 系统完整启动脚本
同时启动后端API服务器和前端开发服务器
"""

import os
import sys
import time
import subprocess
import threading
import signal
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemLauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True

    def start_backend(self):
        """启动后端API服务器"""
        logger.info("🚀 启动后端API服务器...")
        
        try:
            # 设置环境变量确保使用修复后的配置
            env = os.environ.copy()
            env['PRIMARY_DATA_PROVIDER'] = 'eastmoney'
            env['EASTMONEY_ENABLED'] = 'true'
            env['YAHOO_FINANCE_ENABLED'] = 'true'
            
            # 启动FastAPI应用
            cmd = [
                sys.executable, '-c',
                """
from myQuant.interfaces.api.monolith_api import create_app
import uvicorn

print("=== myStock 后端API服务器 ===")
print("✅ 数据源: EastMoney (主) + Yahoo Finance (备用)")
print("✅ 支持真实股票数据获取")
print("✅ API文档: http://localhost:8000/docs")
print("=" * 50)

app = create_app()
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
                """.strip()
            ]
            
            self.backend_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 启动后端日志输出线程
            threading.Thread(
                target=self._stream_backend_logs,
                daemon=True
            ).start()
            
            logger.info("✅ 后端API服务器启动成功 - http://localhost:8000")
            
        except Exception as e:
            logger.error(f"❌ 后端启动失败: {e}")
            return False
            
        return True

    def start_frontend(self):
        """启动前端开发服务器"""
        logger.info("🎨 启动前端开发服务器...")
        
        try:
            frontend_dir = Path(__file__).parent / "frontend"
            
            if not frontend_dir.exists():
                logger.error(f"❌ 前端目录不存在: {frontend_dir}")
                return False
            
            # 检查是否有node_modules
            if not (frontend_dir / "node_modules").exists():
                logger.info("📦 安装前端依赖...")
                install_result = subprocess.run(
                    ["npm", "install"],
                    cwd=frontend_dir,
                    capture_output=True,
                    text=True
                )
                
                if install_result.returncode != 0:
                    logger.error(f"❌ 前端依赖安装失败: {install_result.stderr}")
                    return False
            
            # 启动前端开发服务器
            env = os.environ.copy()
            env['REACT_APP_API_BASE_URL'] = 'http://localhost:8000'
            
            self.frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd=frontend_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 启动前端日志输出线程
            threading.Thread(
                target=self._stream_frontend_logs,
                daemon=True
            ).start()
            
            logger.info("✅ 前端开发服务器启动成功 - http://localhost:3000")
            
        except Exception as e:
            logger.error(f"❌ 前端启动失败: {e}")
            return False
            
        return True

    def _stream_backend_logs(self):
        """输出后端日志"""
        if not self.backend_process:
            return
            
        for line in iter(self.backend_process.stdout.readline, ''):
            if not self.running:
                break
            print(f"[BACKEND] {line.rstrip()}")

    def _stream_frontend_logs(self):
        """输出前端日志"""
        if not self.frontend_process:
            return
            
        for line in iter(self.frontend_process.stdout.readline, ''):
            if not self.running:
                break
            print(f"[FRONTEND] {line.rstrip()}")

    def stop(self):
        """停止所有服务"""
        logger.info("🛑 正在停止系统...")
        self.running = False
        
        if self.backend_process:
            logger.info("停止后端服务器...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process:
            logger.info("停止前端服务器...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        logger.info("✅ 系统已停止")

    def run(self):
        """运行完整系统"""
        print("🚀 myStock 量化交易平台启动中...")
        print("=" * 60)
        
        # 信号处理
        def signal_handler(signum, frame):
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # 启动后端
            if not self.start_backend():
                logger.error("❌ 后端启动失败，退出")
                return
            
            # 等待后端启动
            logger.info("⏳ 等待后端服务器就绪...")
            time.sleep(8)
            
            # 启动前端
            if not self.start_frontend():
                logger.error("❌ 前端启动失败，但后端继续运行")
            
            print("\n" + "=" * 60)
            print("🎉 myStock 系统启动完成!")
            print("📊 后端API: http://localhost:8000")
            print("🎨 前端界面: http://localhost:3000") 
            print("📖 API文档: http://localhost:8000/docs")
            print("=" * 60)
            print("💡 在前端Dashboard页面可以看到实时数据验证器")
            print("🔧 数据修复已完成，现在显示真实股票价格")
            print("⚠️  按 Ctrl+C 停止系统")
            print("=" * 60)
            
            # 保持运行
            while self.running:
                time.sleep(1)
                
                # 检查进程状态
                if self.backend_process and self.backend_process.poll() is not None:
                    logger.error("❌ 后端服务器意外停止")
                    break
                    
                if self.frontend_process and self.frontend_process.poll() is not None:
                    logger.warning("⚠️ 前端服务器意外停止")
        
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            self.stop()

def main():
    """主函数"""
    # 检查依赖
    logger.info("🔍 检查系统依赖...")
    
    # 检查Python环境
    if sys.version_info < (3, 8):
        logger.error("❌ 需要Python 3.8或更高版本")
        return
    
    # 检查是否在项目根目录
    if not Path("myQuant").exists():
        logger.error("❌ 请在项目根目录运行此脚本")
        return
    
    # 启动系统
    launcher = SystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main()