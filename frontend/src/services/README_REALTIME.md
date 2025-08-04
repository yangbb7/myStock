# 实时数据通信系统

本文档描述了前端实时数据通信系统的实现，包括WebSocket服务、React Hooks和相关组件。

## 架构概览

实时数据通信系统由以下几个核心部分组成：

1. **WebSocketService** - 底层WebSocket连接管理
2. **useWebSocket Hook** - React Hook封装
3. **useRealTime Hooks** - 高级业务逻辑Hooks
4. **监控组件** - UI组件展示实时数据

## 核心功能

### 1. WebSocket连接管理

- 自动连接和重连机制
- 连接状态监控
- 错误处理和恢复
- 订阅/取消订阅管理

### 2. 数据缓冲和节流

- 消息缓冲机制，避免频繁更新
- 可配置的节流时间
- 数据去重（特别是市场数据）
- 批处理优化

### 3. 实时数据类型

支持以下类型的实时数据：

- **market_data** - 市场数据推送
- **system_status** - 系统状态监控
- **order_update** - 订单状态更新
- **risk_alert** - 风险告警

## 使用方法

### 基础WebSocket连接

```typescript
import { useWebSocket } from '../hooks/useWebSocket';

const MyComponent = () => {
  const { isConnected, subscribe, unsubscribe } = useWebSocket({
    url: 'ws://localhost:8000',
    autoConnect: true,
  });

  useEffect(() => {
    const subscriptionId = subscribe('market_data', (data) => {
      console.log('Market data:', data);
    });

    return () => unsubscribe(subscriptionId);
  }, []);

  return <div>Connected: {isConnected ? 'Yes' : 'No'}</div>;
};
```

### 市场数据监控

```typescript
import { useMarketData } from '../hooks/useRealTime';

const MarketDataComponent = () => {
  const { data, isConnected, subscribe } = useMarketData({
    symbols: ['000001.SZ', '600000.SH'],
    throttle: 100, // 100ms节流
    autoSubscribe: true,
  });

  return (
    <div>
      {Object.entries(data).map(([symbol, marketData]) => (
        <div key={symbol}>
          {symbol}: ¥{marketData.price} ({marketData.change >= 0 ? '+' : ''}{marketData.change})
        </div>
      ))}
    </div>
  );
};
```

### 系统状态监控

```typescript
import { useSystemStatus } from '../hooks/useRealTime';

const SystemStatusComponent = () => {
  const { data, isConnected, lastUpdated } = useSystemStatus();

  if (!data) return <div>Loading...</div>;

  return (
    <div>
      <h3>系统状态: {data.systemRunning ? '运行中' : '已停止'}</h3>
      <p>运行时间: {data.uptimeSeconds}秒</p>
      <div>
        模块状态:
        {Object.entries(data.modules).map(([name, module]) => (
          <div key={name}>
            {name}: {module.initialized ? '正常' : '异常'}
          </div>
        ))}
      </div>
      <small>最后更新: {lastUpdated?.toLocaleTimeString()}</small>
    </div>
  );
};
```

### 风险告警监控

```typescript
import { useRiskAlerts } from '../hooks/useRealTime';

const RiskAlertsComponent = () => {
  const { alerts, latestAlert, clearAlerts, dismissAlert } = useRiskAlerts({
    maxAlertsSize: 50,
  });

  return (
    <div>
      <h3>风险告警 ({alerts.length})</h3>
      <button onClick={clearAlerts}>清空所有告警</button>
      
      {latestAlert && (
        <div className="latest-alert">
          <strong>最新告警:</strong> {latestAlert.message}
        </div>
      )}
      
      {alerts.map((alert, index) => (
        <div key={index} className={`alert alert-${alert.level}`}>
          <span>{alert.message}</span>
          <button onClick={() => dismissAlert(index)}>忽略</button>
        </div>
      ))}
    </div>
  );
};
```

### 综合实时监控

```typescript
import { useRealTime } from '../hooks/useRealTime';

const RealTimeDashboard = () => {
  const { marketData, systemStatus, riskAlerts, isConnected } = useRealTime({
    marketData: { 
      symbols: ['000001.SZ', '600000.SH'],
      throttle: 200 
    },
    systemStatus: { autoSubscribe: true },
    riskAlerts: { maxAlertsSize: 20 },
  });

  return (
    <div>
      <h2>实时监控面板</h2>
      <div>连接状态: {isConnected ? '已连接' : '连接断开'}</div>
      
      {/* 市场数据 */}
      <section>
        <h3>市场数据</h3>
        {Object.entries(marketData.data).map(([symbol, data]) => (
          <div key={symbol}>{symbol}: ¥{data.price}</div>
        ))}
      </section>
      
      {/* 系统状态 */}
      <section>
        <h3>系统状态</h3>
        {systemStatus.data && (
          <div>状态: {systemStatus.data.systemRunning ? '运行中' : '已停止'}</div>
        )}
      </section>
      
      {/* 风险告警 */}
      <section>
        <h3>风险告警</h3>
        <div>告警数量: {riskAlerts.alerts.length}</div>
        {riskAlerts.latestAlert && (
          <div>最新告警: {riskAlerts.latestAlert.message}</div>
        )}
      </section>
    </div>
  );
};
```

## 配置选项

### WebSocket配置

```typescript
interface WebSocketConfig {
  url: string;                    // WebSocket服务器地址
  autoConnect?: boolean;          // 自动连接 (默认: true)
  reconnection?: boolean;         // 自动重连 (默认: true)
  reconnectionAttempts?: number;  // 重连尝试次数 (默认: 5)
  reconnectionDelay?: number;     // 重连延迟 (默认: 1000ms)
  timeout?: number;               // 连接超时 (默认: 20000ms)
}
```

### 订阅选项

```typescript
interface SubscriptionOptions {
  symbols?: string[];    // 股票代码过滤
  throttle?: number;     // 节流时间 (ms)
  bufferSize?: number;   // 缓冲区大小
}
```

## 错误处理

系统提供了完善的错误处理机制：

1. **连接错误** - 自动重连，状态通知
2. **消息错误** - 错误日志，继续处理其他消息
3. **订阅错误** - 重新订阅，错误回调
4. **数据格式错误** - 数据验证，降级处理

## 性能优化

1. **消息节流** - 避免频繁UI更新
2. **数据缓冲** - 批量处理消息
3. **去重机制** - 避免重复数据处理
4. **内存管理** - 自动清理过期数据
5. **连接复用** - 单例WebSocket连接

## 测试

系统包含完整的单元测试：

```bash
# 运行WebSocket服务测试
npm test src/services/__tests__/websocket.test.ts

# 运行实时Hooks测试
npm test src/hooks/__tests__/useRealTime.test.tsx

# 运行所有测试
npm test
```

## 环境变量

```env
# WebSocket服务器地址
VITE_WS_URL=ws://localhost:8000

# 开发模式下的调试选项
VITE_WS_DEBUG=true
```

## 故障排除

### 常见问题

1. **连接失败**
   - 检查WebSocket服务器是否运行
   - 确认URL配置正确
   - 检查网络连接

2. **数据不更新**
   - 确认订阅是否成功
   - 检查过滤条件
   - 验证数据格式

3. **性能问题**
   - 调整节流时间
   - 减少订阅数量
   - 优化缓冲区大小

### 调试工具

```typescript
// 启用调试模式
const service = createWebSocketService({
  url: 'ws://localhost:8000',
  debug: true, // 启用调试日志
});

// 监听连接状态变化
service.onStateChange((state) => {
  console.log('WebSocket state changed:', state);
});
```

## 扩展开发

### 添加新的消息类型

1. 在 `types.ts` 中定义消息类型
2. 在 `WebSocketService` 中添加处理逻辑
3. 创建对应的Hook
4. 编写测试用例

### 自定义组件

参考 `SystemStatusMonitor.tsx` 和 `RiskAlertsMonitor.tsx` 创建自定义监控组件。

## 最佳实践

1. **合理设置节流时间** - 平衡实时性和性能
2. **适当的缓冲区大小** - 避免内存溢出
3. **错误边界处理** - 防止单个组件错误影响整个应用
4. **资源清理** - 组件卸载时取消订阅
5. **状态管理** - 合理使用本地状态和全局状态