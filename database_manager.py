#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库管理工具 - 全局最优解的统一入口
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from myQuant.infrastructure.database import DatabaseManager, DatabaseConfig, MigrationManager
from myQuant.infrastructure.database.database_config import Environment


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_ROOT / "logs" / "database_manager.log")
        ]
    )


def cmd_init(args):
    """初始化数据库"""
    print("🔧 Initializing unified database system...")
    
    # 设置环境
    if args.test:
        config = DatabaseConfig(environment=Environment.TEST)
    else:
        config = DatabaseConfig.from_environment()
    
    manager = DatabaseManager(config)
    manager.initialize()
    
    print("✅ Database system initialized successfully!")
    print(f"📍 Database location: {config.database_path}")
    print(f"📁 Data directory: {config.data_path}")


def cmd_migrate(args):
    """迁移现有数据库"""
    print("🔄 Migrating existing databases to unified architecture...")
    
    config = DatabaseConfig.from_environment()
    manager = DatabaseManager(config)
    
    # 先初始化新系统
    manager.initialize()
    
    # 迁移现有数据
    manager.migrate_existing_databases()
    
    print("✅ Database migration completed successfully!")
    
    # 显示迁移结果
    info = manager.get_database_info()
    print("\n📊 Migration Results:")
    print(f"   Main database: {info['main_database']['path']}")
    print(f"   Record count: {sum(table['record_count'] for table in info['main_database']['tables'])}")
    print(f"   Database size: {info['main_database']['size'] / 1024:.1f} KB")


def cmd_info(args):
    """显示数据库信息"""
    print("📊 Database System Information")
    print("=" * 50)
    
    config = DatabaseConfig.from_environment()
    manager = DatabaseManager(config)
    
    info = manager.get_database_info()
    
    print(f"Environment: {info['environment']}")
    print(f"Database Type: {info['config']['database_type']}")
    print(f"Database URL: {info['config']['database_url']}")
    print(f"Data Path: {info['config']['data_path']}")
    
    print("\n📈 Main Database:")
    main_db = info['main_database']
    print(f"   Path: {main_db['path']}")
    print(f"   Exists: {'✅' if main_db['exists'] else '❌'}")
    print(f"   Size: {main_db['size'] / 1024:.1f} KB")
    
    if main_db['tables']:
        print("   Tables:")
        for table in main_db['tables']:
            print(f"     - {table['name']}: {table['record_count']} records")
    
    print("\n📚 Shards:")
    if info['shards']:
        for shard_name, shard_info in info['shards'].items():
            print(f"   {shard_name}:")
            print(f"     - Files: {shard_info['file_count']}")
            print(f"     - Size: {shard_info['total_size'] / 1024:.1f} KB")
    else:
        print("   No shards found")


def cmd_health(args):
    """检查数据库健康状态"""
    print("🏥 Database Health Check")
    print("=" * 50)
    
    config = DatabaseConfig.from_environment()
    manager = DatabaseManager(config)
    
    health = manager.health_check()
    
    status_icon = "✅" if health['status'] == 'healthy' else "⚠️" if health['status'] == 'warning' else "❌"
    print(f"Status: {status_icon} {health['status'].upper()}")
    
    if health['issues']:
        print("\n🔍 Issues Found:")
        for issue in health['issues']:
            print(f"   - {issue}")
    
    if health['recommendations']:
        print("\n💡 Recommendations:")
        for rec in health['recommendations']:
            print(f"   - {rec}")
    
    if health['status'] == 'healthy':
        print("\n🎉 Database system is healthy!")


def cmd_optimize(args):
    """优化数据库性能"""
    print("⚡ Optimizing database performance...")
    
    config = DatabaseConfig.from_environment()
    manager = DatabaseManager(config)
    
    manager.optimize_database()
    
    print("✅ Database optimization completed!")


def cmd_backup(args):
    """备份数据库"""
    print("💾 Creating database backup...")
    
    config = DatabaseConfig.from_environment()
    manager = DatabaseManager(config)
    
    backup_path = manager.backup_database(args.name)
    
    print(f"✅ Database backed up to: {backup_path}")


def cmd_restore(args):
    """恢复数据库"""
    print(f"🔄 Restoring database from: {args.backup_path}")
    
    config = DatabaseConfig.from_environment()
    manager = DatabaseManager(config)
    
    backup_path = Path(args.backup_path)
    manager.restore_database(backup_path)
    
    print("✅ Database restored successfully!")


def cmd_clean(args):
    """清理旧数据库文件"""
    print("🧹 Cleaning up old database files...")
    
    # 查找所有可能的旧数据库文件
    old_files = []
    
    # myQuant/data/myquant.db
    old_myquant_db = PROJECT_ROOT / "myQuant" / "data" / "myquant.db"
    if old_myquant_db.exists():
        old_files.append(old_myquant_db)
    
    # tests/data/myquant.db
    old_tests_db = PROJECT_ROOT / "tests" / "data" / "myquant.db"
    if old_tests_db.exists():
        old_files.append(old_tests_db)
    
    if not old_files:
        print("✅ No old database files found to clean up")
        return
    
    print(f"Found {len(old_files)} old database files:")
    for file_path in old_files:
        print(f"   - {file_path}")
    
    if not args.force:
        confirm = input("\nDo you want to proceed with cleanup? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Cleanup cancelled")
            return
    
    # 创建备份并删除旧文件
    config = DatabaseConfig.from_environment()
    manager = DatabaseManager(config)
    
    manager._cleanup_old_databases(old_files)
    
    print("✅ Old database files cleaned up successfully!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="myQuant Database Manager - 统一数据库管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python database_manager.py init           # 初始化数据库系统
  python database_manager.py migrate        # 迁移现有数据库
  python database_manager.py info           # 显示数据库信息
  python database_manager.py health         # 检查数据库健康状态
  python database_manager.py optimize       # 优化数据库性能
  python database_manager.py backup         # 备份数据库
  python database_manager.py clean --force  # 清理旧数据库文件
        """
    )
    
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')
    parser.add_argument('--test', action='store_true', help='使用测试环境')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # init 命令
    parser_init = subparsers.add_parser('init', help='初始化数据库系统')
    parser_init.set_defaults(func=cmd_init)
    
    # migrate 命令
    parser_migrate = subparsers.add_parser('migrate', help='迁移现有数据库')
    parser_migrate.set_defaults(func=cmd_migrate)
    
    # info 命令
    parser_info = subparsers.add_parser('info', help='显示数据库信息')
    parser_info.set_defaults(func=cmd_info)
    
    # health 命令
    parser_health = subparsers.add_parser('health', help='检查数据库健康状态')
    parser_health.set_defaults(func=cmd_health)
    
    # optimize 命令
    parser_optimize = subparsers.add_parser('optimize', help='优化数据库性能')
    parser_optimize.set_defaults(func=cmd_optimize)
    
    # backup 命令
    parser_backup = subparsers.add_parser('backup', help='备份数据库')
    parser_backup.add_argument('--name', help='备份名称')
    parser_backup.set_defaults(func=cmd_backup)
    
    # restore 命令
    parser_restore = subparsers.add_parser('restore', help='恢复数据库')
    parser_restore.add_argument('backup_path', help='备份文件路径')
    parser_restore.set_defaults(func=cmd_restore)
    
    # clean 命令
    parser_clean = subparsers.add_parser('clean', help='清理旧数据库文件')
    parser_clean.add_argument('--force', action='store_true', help='强制清理，不询问确认')
    parser_clean.set_defaults(func=cmd_clean)
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 创建日志目录
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()