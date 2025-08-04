# 代码风格和约定

## Python 代码规范
- 使用 Type Hints 进行类型注解
- 所有类和函数都需要 docstring（中文注释）
- 遵循 PEP 8 规范
- 使用 `logging` 模块进行日志记录
- 异步函数使用 `async/await` 语法

### 示例
```python
from typing import List, Dict, Optional

class DataManager:
    """数据管理器
    
    负责管理市场数据的获取、缓存和分发
    """
    
    async def get_market_data(
        self, 
        symbols: List[str], 
        start_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """获取市场数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            
        Returns:
            包含各股票数据的字典
        """
        pass
```

## TypeScript/React 代码规范
- 使用函数组件和 Hooks
- 严格的 TypeScript 类型定义
- 组件文件使用 .tsx 扩展名
- 使用 Ant Design 组件库
- 状态管理使用 Zustand

### 示例
```typescript
interface StrategyConfig {
  name: string;
  symbols: string[];
  params: Record<string, any>;
}

const StrategyConfigForm: React.FC<{
  onSubmit: (config: StrategyConfig) => void;
}> = ({ onSubmit }) => {
  // 组件实现
};
```

## 文件命名
- Python: snake_case.py
- React组件: PascalCase.tsx
- 工具函数: camelCase.ts
- 测试文件: *.test.py 或 *.test.tsx

## 项目结构
- 核心业务逻辑在 `myQuant/core/`
- API接口在 `myQuant/interfaces/api/`
- 前端页面在 `frontend/src/pages/`
- 共享组件在 `frontend/src/components/`