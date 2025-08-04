# 任务完成检查清单

完成代码修改后，请按以下步骤进行验证：

## 后端开发
1. **代码质量检查**
   ```bash
   ruff check .              # 代码风格检查
   ruff format .             # 代码格式化
   mypy .                    # 类型检查
   ```

2. **运行测试**
   ```bash
   pytest                    # 运行所有测试
   pytest --cov             # 检查测试覆盖率
   ```

3. **手动测试**
   - 启动 API 服务器：`python main.py --api-server`
   - 测试相关功能是否正常工作

## 前端开发
1. **代码质量检查**
   ```bash
   cd frontend
   npm run lint             # ESLint 检查
   npm run type-check       # TypeScript 类型检查
   npm run format:check     # 格式检查
   ```

2. **运行测试**
   ```bash
   npm run test            # 运行单元测试
   ```

3. **构建验证**
   ```bash
   npm run build:check     # 构建并进行类型检查
   ```

## 通用检查
1. 确保没有引入新的依赖而未更新 pyproject.toml 或 package.json
2. 检查是否需要更新相关文档
3. 确保代码符合项目的命名规范和风格指南
4. 验证新功能是否有适当的错误处理
5. 检查是否有适当的日志记录

## Git 提交前
1. 运行 `git status` 查看所有更改
2. 运行 `git diff` 审查具体改动
3. 确保只提交相关的文件更改