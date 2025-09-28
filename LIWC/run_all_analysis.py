#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有LIWC分析脚本的批处理文件
"""

import subprocess
import sys
import os

def run_script(script_name):
    """运行指定的Python脚本"""
    print(f"\n{'='*60}")
    print(f"正在运行: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"✅ {script_name} 运行成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} 运行失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 运行 {script_name} 时发生错误: {e}")
        return False

def main():
    """主函数"""
    print("开始运行所有LIWC分析脚本...")
    
    # 确保output目录存在
    if not os.path.exists('output'):
        os.makedirs('output')
        print("创建了output目录")
    
    # 要运行的脚本列表（按推荐顺序）
    scripts = [
        'empathy_score.py',
        'liwc_Anova_results_table.py', 
        'liwc_Anova_graph.py',
        'liwc_anovaTest.py'
    ]
    
    success_count = 0
    total_count = len(scripts)
    
    for script in scripts:
        if os.path.exists(script):
            if run_script(script):
                success_count += 1
        else:
            print(f"❌ 脚本文件不存在: {script}")
    
    print(f"\n{'='*60}")
    print(f"批处理完成！")
    print(f"成功运行: {success_count}/{total_count} 个脚本")
    print(f"所有输出文件已保存到 output/ 目录")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()