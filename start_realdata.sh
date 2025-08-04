#!/bin/bash

echo "========================================"
echo "myQuant 真实数据版本启动脚本"
echo "========================================"

# 设置环境变量启用真实数据
export PRIMARY_DATA_PROVIDER=yahoo
export YAHOO_FINANCE_ENABLED=true
export EASTMONEY_ENABLED=true
export LOG_LEVEL=INFO

echo "📊 数据源配置:"
echo "  - 主要数据源: $PRIMARY_DATA_PROVIDER"
echo "  - Yahoo Finance: 启用"
echo "  - 东方财富: 启用"
echo ""

echo "🔗 测试数据连接..."
python3 -c "
from myQuant.infrastructure.data.providers.real_data_provider import RealDataProvider
import os

config = {
    'primary_provider': os.getenv('PRIMARY_DATA_PROVIDER', 'yahoo'),
    'fallback_providers': ['eastmoney'],
    'yahoo': {'enabled': True},
    'eastmoney': {'enabled': True}
}

provider = RealDataProvider(config)
data = provider.get_realtime_data(['000001.SZ'])
if data and '000001.SZ' in data:
    price = data['000001.SZ']['current_price']
    print(f'✅ 数据连接成功 - 平安银行: ¥{price:.2f}')
else:
    print('❌ 数据连接失败')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 数据源测试失败，请检查网络连接"
    exit 1
fi

echo ""
echo "🚀 启动后端服务（真实数据）..."

# 启动后端
python3 main.py --api-server &
BACKEND_PID=$!

echo "等待后端服务启动..."
sleep 5

echo ""
echo "========================================"
echo "✅ 系统启动完成（真实数据模式）!"
echo "========================================"
echo "🌐 后端API:  http://localhost:8000"
echo "📚 API文档:  http://localhost:8000/docs"
echo "🔌 WebSocket: ws://localhost:8000/socket.io/"
echo ""
echo "📊 现在显示真实股票价格:"
echo "  - 平安银行 (000001.SZ)"
echo "  - 万科A (000002.SZ)"  
echo "  - 浦发银行 (600000.SH)"
echo "  - 招商银行 (600036.SH)"
echo "  - 五粮液 (000858.SZ)"
echo ""
echo "🔄 数据每3秒自动更新"
echo "========================================"
echo "按 Ctrl+C 停止系统"

# 等待用户中断
trap 'echo "正在停止系统..."; kill $BACKEND_PID 2>/dev/null; echo "系统已停止"; exit 0' INT

# 保持脚本运行
wait