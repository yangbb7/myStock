"""
pytest配置文件和共享fixtures
用于解决硬编码路径问题和提供通用测试配置
"""

import os
import sys
from pathlib import Path


def setup_test_environment():
    """设置测试环境路径"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    myquant_path = project_root / "myQuant"
    
    # 添加到Python路径（如果还没有的话）
    if str(myquant_path) not in sys.path:
        sys.path.insert(0, str(myquant_path))


# 在导入时自动设置环境
setup_test_environment()