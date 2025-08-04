# 建议的开发命令

## 项目启动
```bash
# 后端启动
python main.py                # 交互式界面
python main.py --api-server   # API服务器模式
python main.py --production   # 生产模式

# 前端启动
cd frontend
npm run dev                   # 开发模式
npm run build                 # 构建生产版本
```

## 测试命令
```bash
# 后端测试
pytest                        # 运行所有测试
pytest --cov                  # 测试覆盖率
pytest tests/unit/           # 单元测试
pytest tests/integration/    # 集成测试

# 前端测试
cd frontend
npm run test                 # 运行测试
npm run test:coverage        # 测试覆盖率
npm run test:e2e            # E2E测试
```

## 代码质量
```bash
# 后端
ruff check .                 # 代码检查
ruff format .                # 代码格式化
mypy .                       # 类型检查

# 前端
npm run lint                 # ESLint检查
npm run lint:fix            # 自动修复
npm run format              # Prettier格式化
npm run type-check          # TypeScript类型检查
```

## 依赖管理
```bash
# 使用 uv (推荐)
uv sync                      # 安装依赖
uv add <package>            # 添加新依赖

# 使用 pip
pip install -r requirements.txt
```

## 系统工具
```bash
git status                   # 查看Git状态
git diff                     # 查看更改
ls -la                       # 列出文件（macOS）
```