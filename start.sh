#!/bin/bash

echo "========================================"
echo "myQuant 量化交易系统启动脚本"
echo "========================================"

echo "正在启动系统..."

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.11+"
    exit 1
fi

# 创建必要目录
mkdir -p data logs

echo "启动后端API服务..."
python3 main.py --api-server &
BACKEND_PID=$!

echo "等待后端服务启动..."
sleep 5

echo "启动前端Web界面..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "系统启动完成!"
echo "========================================"
echo "前端界面: http://localhost:3000"
echo "后端API:  http://localhost:8000"
echo "API文档:  http://localhost:8000/docs"
echo "========================================"
echo "按 Ctrl+C 停止系统"

# 等待用户中断
trap 'echo "正在停止系统..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo "系统已停止"; exit 0' INT

# 保持脚本运行
wait