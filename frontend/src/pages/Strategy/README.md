# 策略管理页面 (Strategy Management Page)

## 概述

策略管理页面是myQuant量化交易系统的核心功能模块，提供完整的策略生命周期管理功能，包括策略配置、性能监控、操作管理等。

## 功能特性

### 1. 策略配置表单 (StrategyConfigForm)
- **策略模板选择**: 提供预定义的策略模板（移动平均、RSI反转、布林带等）
- **参数配置**: 支持基础参数和高级参数配置
- **实时验证**: 表单参数实时验证和错误提示
- **配置预览**: 高级模式下提供JSON格式配置预览
- **风险控制**: 止损止盈参数配置

#### 主要功能
- 策略名称和交易标的配置
- 初始资金、风险容忍度、最大仓位设置
- 技术指标和策略参数配置
- 表单验证和错误处理

### 2. 策略性能监控 (StrategyPerformanceMonitor)
- **性能排行榜**: 策略性能实时排名和对比
- **关键指标**: 胜率、盈利因子、夏普比率、最大回撤等
- **可视化图表**: 策略对比图表和性能趋势分析
- **实时更新**: 5秒间隔自动刷新数据

#### 主要功能
- 策略性能表格展示
- 多策略对比分析
- 性能趋势图表
- 排序和筛选功能

### 3. 策略操作管理 (StrategyOperations)
- **策略控制**: 启动、停止、重启策略
- **配置管理**: 编辑策略参数和配置
- **日志查看**: 实时策略运行日志
- **备份恢复**: 策略配置备份和导出

#### 主要功能
- 策略状态监控和控制
- 策略编辑和参数更新
- 运行日志查看和导出
- 策略备份和删除

### 4. 综合管理界面 (StrategyManagementPage)
- **概览仪表板**: 策略总体统计和关键指标
- **标签页导航**: 配置、监控、操作功能分离
- **快速操作**: 常用功能快速访问
- **状态展示**: 最佳/最差策略展示

## 技术实现

### 技术栈
- **React 18**: 现代化前端框架
- **TypeScript**: 类型安全开发
- **Ant Design 5**: 企业级UI组件库
- **React Query**: 服务端状态管理
- **ECharts**: 数据可视化图表

### 组件架构
```
StrategyManagementPage (主页面)
├── StrategyConfigForm (策略配置)
├── StrategyPerformanceMonitor (性能监控)
└── StrategyOperations (操作管理)
```

### API集成
- 策略CRUD操作 (`/strategy/*`)
- 实时性能数据获取
- 策略状态控制
- 配置验证和更新

## 文件结构

```
src/pages/Strategy/
├── index.tsx                    # 主页面组件
├── StrategyConfigForm.tsx       # 策略配置表单
├── StrategyPerformanceMonitor.tsx # 性能监控组件
├── StrategyOperations.tsx       # 操作管理组件
├── components.ts                # 组件导出
├── __tests__/                   # 测试文件
│   └── StrategyManagement.test.tsx
└── README.md                    # 文档说明
```

## 使用方法

### 基本使用
```tsx
import { StrategyManagementPage } from '@/pages/Strategy';

function App() {
  return <StrategyManagementPage />;
}
```

### 单独使用组件
```tsx
import { 
  StrategyConfigForm, 
  StrategyPerformanceMonitor, 
  StrategyOperations 
} from '@/pages/Strategy/components';

// 策略配置表单
<StrategyConfigForm 
  mode="create" 
  onSuccess={(strategy) => console.log('Strategy added:', strategy)} 
/>

// 性能监控
<StrategyPerformanceMonitor />

// 操作管理
<StrategyOperations 
  onEditStrategy={(name) => console.log('Edit strategy:', name)} 
/>
```

## 测试

组件包含完整的单元测试和集成测试：

```bash
# 运行策略管理测试
npm test -- src/pages/Strategy/__tests__/StrategyManagement.test.tsx
```

测试覆盖：
- 组件渲染和基本功能
- API集成和错误处理
- 用户交互和状态管理
- 数据展示和更新

## 配置选项

### 策略模板
- **移动平均策略**: 基于移动平均线交叉
- **RSI反转策略**: 基于RSI指标的均值回归
- **布林带策略**: 基于布林带的突破和回归
- **自定义策略**: 用户自定义参数配置

### 性能指标
- **基础指标**: 信号数量、成功交易、总盈亏
- **风险指标**: 胜率、盈利因子、最大回撤
- **高级指标**: 夏普比率、波动率、VaR

## 注意事项

1. **实时数据**: 组件依赖实时API数据，确保后端服务正常运行
2. **权限控制**: 策略操作需要相应权限，注意权限验证
3. **数据验证**: 策略参数需要严格验证，避免无效配置
4. **性能优化**: 大量策略时注意分页和虚拟滚动
5. **错误处理**: 完善的错误处理和用户反馈机制

## 扩展功能

未来可扩展的功能：
- 策略模板市场
- 策略组合管理
- 高级回测功能
- 策略分享和协作
- 移动端适配

## 更新日志

### v1.0.0 (2025-01-19)
- 初始版本发布
- 完整的策略管理功能
- 策略配置、监控、操作三大模块
- 完善的测试覆盖
- 响应式设计和错误处理