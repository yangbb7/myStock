#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
集成测试运行器
提供完整的集成测试套件运行和报告功能
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "myQuant"))

from myQuant import setup_logging, get_version


class IntegrationTestRunner:
    """集成测试运行器"""
    
    def __init__(self, test_dir: Optional[str] = None):
        self.test_dir = test_dir or Path(__file__).parent
        self.logger = setup_logging(level='INFO')
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def discover_tests(self) -> List[str]:
        """发现集成测试文件"""
        test_files = []
        test_dir = Path(self.test_dir)
        
        for file_path in test_dir.glob("test_*.py"):
            if file_path.name != "test_runner.py":
                test_files.append(str(file_path))
        
        return sorted(test_files)
    
    def run_test_file(self, test_file: str) -> Dict:
        """运行单个测试文件"""
        self.logger.info(f"运行测试文件: {test_file}")
        
        start_time = time.time()
        
        try:
            # 使用pytest运行测试
            cmd = [
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                "--maxfail=5",
                "-x",  # 遇到第一个失败就停止
                "--durations=10"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=str(project_root)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                'file': test_file,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': duration,
                'success': result.returncode == 0
            }
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                'file': test_file,
                'return_code': -1,
                'stdout': "",
                'stderr': str(e),
                'duration': duration,
                'success': False
            }
    
    def run_all_tests(self) -> Dict:
        """运行所有集成测试"""
        self.logger.info("开始运行集成测试套件")
        self.start_time = time.time()
        
        test_files = self.discover_tests()
        self.logger.info(f"发现 {len(test_files)} 个测试文件")
        
        results = {}
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_file in test_files:
            result = self.run_test_file(test_file)
            results[test_file] = result
            
            if result['success']:
                passed_tests += 1
                self.logger.info(f"✅ {Path(test_file).name} - 通过 ({result['duration']:.2f}s)")
            else:
                failed_tests += 1
                self.logger.error(f"❌ {Path(test_file).name} - 失败 ({result['duration']:.2f}s)")
            
            total_tests += 1
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        self.results = {
            'summary': {
                'total_files': total_tests,
                'passed_files': passed_tests,
                'failed_files': failed_tests,
                'total_duration': total_duration,
                'start_time': self.start_time,
                'end_time': self.end_time
            },
            'details': results
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """生成测试报告"""
        if not self.results:
            return "没有测试结果"
        
        summary = self.results['summary']
        details = self.results['details']
        
        report = []
        report.append("=" * 80)
        report.append("myStock 集成测试报告")
        report.append("=" * 80)
        report.append(f"版本: {get_version()}")
        report.append(f"测试时间: {datetime.fromtimestamp(summary['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试耗时: {summary['total_duration']:.2f}秒")
        report.append("")
        
        # 总结
        report.append("📊 测试总结")
        report.append("-" * 40)
        report.append(f"总测试文件: {summary['total_files']}")
        report.append(f"通过文件: {summary['passed_files']}")
        report.append(f"失败文件: {summary['failed_files']}")
        
        success_rate = (summary['passed_files'] / summary['total_files']) * 100 if summary['total_files'] > 0 else 0
        report.append(f"成功率: {success_rate:.1f}%")
        report.append("")
        
        # 详细结果
        report.append("📋 详细结果")
        report.append("-" * 40)
        
        for test_file, result in details.items():
            file_name = Path(test_file).name
            status = "✅ 通过" if result['success'] else "❌ 失败"
            report.append(f"{status} {file_name} ({result['duration']:.2f}s)")
            
            if not result['success']:
                report.append(f"  错误码: {result['return_code']}")
                if result['stderr']:
                    # 只显示错误信息的前几行
                    error_lines = result['stderr'].split('\n')[:5]
                    for line in error_lines:
                        if line.strip():
                            report.append(f"  错误: {line}")
        
        report.append("")
        
        # 性能统计
        report.append("⚡ 性能统计")
        report.append("-" * 40)
        
        durations = [result['duration'] for result in details.values()]
        if durations:
            report.append(f"平均耗时: {sum(durations) / len(durations):.2f}秒")
            report.append(f"最快测试: {min(durations):.2f}秒")
            report.append(f"最慢测试: {max(durations):.2f}秒")
        
        report.append("")
        
        # 建议
        if summary['failed_files'] > 0:
            report.append("💡 建议")
            report.append("-" * 40)
            report.append("- 检查失败的测试文件详细错误信息")
            report.append("- 确保所有依赖项已正确安装")
            report.append("- 验证测试环境配置")
            report.append("- 考虑单独运行失败的测试进行调试")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, report_path: Optional[str] = None) -> str:
        """保存测试报告"""
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"integration_test_report_{timestamp}.txt"
        
        report_content = self.generate_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
    
    def save_json_report(self, json_path: Optional[str] = None) -> str:
        """保存JSON格式的测试报告"""
        if json_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = f"integration_test_results_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        return json_path


def run_specific_tests(test_patterns: List[str]) -> Dict:
    """运行特定的测试"""
    runner = IntegrationTestRunner()
    
    test_files = runner.discover_tests()
    filtered_files = []
    
    for pattern in test_patterns:
        for test_file in test_files:
            if pattern in test_file:
                filtered_files.append(test_file)
    
    if not filtered_files:
        print(f"没有找到匹配的测试文件: {test_patterns}")
        return {}
    
    results = {}
    for test_file in filtered_files:
        result = runner.run_test_file(test_file)
        results[test_file] = result
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='myStock 集成测试运行器')
    parser.add_argument('--test-dir', type=str, help='测试目录路径')
    parser.add_argument('--output', type=str, help='报告输出路径')
    parser.add_argument('--json', type=str, help='JSON报告输出路径')
    parser.add_argument('--tests', nargs='+', help='指定要运行的测试文件模式')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--quick', '-q', action='store_true', help='快速模式（跳过耗时测试）')
    
    args = parser.parse_args()
    
    if args.tests:
        # 运行特定测试
        results = run_specific_tests(args.tests)
        
        for test_file, result in results.items():
            print(f"{Path(test_file).name}: {'通过' if result['success'] else '失败'}")
            if not result['success']:
                print(f"  错误: {result['stderr']}")
    else:
        # 运行所有测试
        runner = IntegrationTestRunner(args.test_dir)
        
        if args.verbose:
            runner.logger.setLevel('DEBUG')
        
        # 设置环境变量
        if args.quick:
            os.environ['MYSTOCK_QUICK_TEST'] = '1'
        
        results = runner.run_all_tests()
        
        # 生成报告
        report = runner.generate_report()
        print(report)
        
        # 保存报告
        if args.output:
            report_path = runner.save_report(args.output)
            print(f"\n报告已保存到: {report_path}")
        
        if args.json:
            json_path = runner.save_json_report(args.json)
            print(f"JSON报告已保存到: {json_path}")
        
        # 根据测试结果设置退出码
        if results['summary']['failed_files'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()