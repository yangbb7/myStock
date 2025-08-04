@echo off
echo ========================================
echo myQuant 量化交易系统启动脚本
echo ========================================

echo 正在启动系统...

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.11+
    pause
    exit /b 1
)

REM 创建必要目录
if not exist "data" mkdir data
if not exist "logs" mkdir logs

echo 启动后端API服务...
start "myQuant Backend" python main.py --api-server

echo 等待后端服务启动...
timeout /t 5 /nobreak >nul

echo 启动前端Web界面...
cd frontend-web-interface
if exist "node_modules" (
    start "myQuant Frontend" npm run dev
) else (
    echo 安装前端依赖...
    npm install
    start "myQuant Frontend" npm run dev
)

echo.
echo ========================================
echo 系统启动完成!
echo ========================================
echo 前端界面: http://localhost:3000
echo 后端API:  http://localhost:8000
echo API文档:  http://localhost:8000/docs
echo ========================================
echo 按任意键关闭此窗口...
pause >nul