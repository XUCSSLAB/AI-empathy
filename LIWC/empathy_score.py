#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empathy Score Calculator
根据指定公式计算empathy score: 
(% 第二人称) × 1.5 + (% 负面情绪) × 1.0 + (% 认知过程 + 洞察词) × 1.2 - (% 第一人称单数) × 2.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """加载数据"""
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    return df

def calculate_new_empathy_score(df):
    """
    根据新公式计算empathy score:
    (% 第二人称) × 1.5 + (% 负面情绪) × 1.0 + (% 认知过程 + 洞察词) × 1.2 - (% 第一人称单数) × 2.0
    """
    print("正在计算新的empathy score...")
    
    # 创建副本避免修改原数据
    df_new = df.copy()
    
    # 计算新的empathy score
    df_new['new_empathy_score'] = (
        df_new['second_person'] * 1.5 +           # 第二人称 × 1.5
        df_new['negative_emotion'] * 1.0 +        # 负面情绪 × 1.0
        (df_new['cognitive_processes'] + df_new['insight']) * 1.2 -  # (认知过程 + 洞察词) × 1.2
        df_new['first_person_singular'] * 2.0     # 第一人称单数 × 2.0
    )
    
    # 计算各个组成部分的贡献
    df_new['second_person_contribution'] = df_new['second_person'] * 1.5
    df_new['negative_emotion_contribution'] = df_new['negative_emotion'] * 1.0
    df_new['cognitive_insight_contribution'] = (df_new['cognitive_processes'] + df_new['insight']) * 1.2
    df_new['first_person_penalty'] = df_new['first_person_singular'] * 2.0
    
    print(f"新empathy score计算完成")
    print(f"新empathy score范围: {df_new['new_empathy_score'].min():.2f} 到 {df_new['new_empathy_score'].max():.2f}")
    print(f"新empathy score平均值: {df_new['new_empathy_score'].mean():.2f}")
    
    return df_new

def create_comparison_analysis(df):
    """创建新旧empathy score的对比分析"""
    print("正在创建对比分析...")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Empathy Score Analysis - New Formula vs Original', fontsize=16, fontweight='bold')
    
    # 1. 新旧score分布对比
    ax1 = axes[0, 0]
    ax1.hist(df['empathy_score'], alpha=0.7, bins=30, label='Original Score', color='blue')
    ax1.hist(df['new_empathy_score'], alpha=0.7, bins=30, label='New Score', color='red')
    ax1.set_xlabel('Empathy Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 新旧score散点图
    ax2 = axes[0, 1]
    ax2.scatter(df['empathy_score'], df['new_empathy_score'], alpha=0.6, s=20)
    ax2.set_xlabel('Original Empathy Score')
    ax2.set_ylabel('New Empathy Score')
    ax2.set_title('Original vs New Score Correlation')
    ax2.grid(True, alpha=0.3)
    
    # 添加对角线
    min_val = min(df['empathy_score'].min(), df['new_empathy_score'].min())
    max_val = max(df['empathy_score'].max(), df['new_empathy_score'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='y=x')
    ax2.legend()
    
    # 3. 各组成部分的贡献分析
    ax3 = axes[1, 0]
    contributions = [
        df['second_person_contribution'].mean(),
        df['negative_emotion_contribution'].mean(),
        df['cognitive_insight_contribution'].mean(),
        -df['first_person_penalty'].mean()  # 负值表示惩罚
    ]
    labels = ['Second Person\n(×1.5)', 'Negative Emotion\n(×1.0)', 
              'Cognitive+Insight\n(×1.2)', 'First Person Penalty\n(×-2.0)']
    colors = ['green', 'orange', 'blue', 'red']
    
    bars = ax3.bar(labels, contributions, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Contribution')
    ax3.set_title('Component Contributions to New Score')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars, contributions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # 4. 按empathy_type分组的新score分析
    ax4 = axes[1, 1]
    empathy_types = df['empathy_type'].unique()
    new_scores_by_type = [df[df['empathy_type'] == et]['new_empathy_score'].values for et in empathy_types]
    
    box_plot = ax4.boxplot(new_scores_by_type, labels=empathy_types, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('New Empathy Score')
    ax4.set_title('New Score Distribution by Empathy Type')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/empathy_score_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_statistics(df):
    """创建详细的统计分析"""
    print("正在生成详细统计...")
    
    # 基本统计
    stats_summary = {
        'Original Score': {
            'Mean': df['empathy_score'].mean(),
            'Std': df['empathy_score'].std(),
            'Min': df['empathy_score'].min(),
            'Max': df['empathy_score'].max(),
            'Median': df['empathy_score'].median()
        },
        'New Score': {
            'Mean': df['new_empathy_score'].mean(),
            'Std': df['new_empathy_score'].std(),
            'Min': df['new_empathy_score'].min(),
            'Max': df['new_empathy_score'].max(),
            'Median': df['new_empathy_score'].median()
        }
    }
    
    # 按empathy_type分组统计
    empathy_type_stats = df.groupby('empathy_type').agg({
        'empathy_score': ['mean', 'std', 'count'],
        'new_empathy_score': ['mean', 'std', 'count'],
        'second_person': 'mean',
        'negative_emotion': 'mean',
        'cognitive_processes': 'mean',
        'insight': 'mean',
        'first_person_singular': 'mean'
    }).round(3)
    
    # 按attribute_type分组统计
    attribute_type_stats = df.groupby('attribute_type').agg({
        'empathy_score': ['mean', 'std', 'count'],
        'new_empathy_score': ['mean', 'std', 'count']
    }).round(3)
    
    # 相关性分析
    correlation = df['empathy_score'].corr(df['new_empathy_score'])
    
    return stats_summary, empathy_type_stats, attribute_type_stats, correlation

def save_results(df, stats_summary, empathy_type_stats, attribute_type_stats, correlation):
    """保存结果到文件"""
    print("正在保存结果...")
    
    # 保存新的数据文件
    output_file = 'output/empathy_scores_with_new_formula.csv'
    df.to_csv(output_file, index=False)
    print(f"新数据已保存到: {output_file}")
    
    # 保存统计报告
    report_file = 'output/empathy_score_analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("EMPATHY SCORE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("新公式: (% 第二人称) × 1.5 + (% 负面情绪) × 1.0 + (% 认知过程 + 洞察词) × 1.2 - (% 第一人称单数) × 2.0\n\n")
        
        f.write("1. 基本统计对比\n")
        f.write("-" * 20 + "\n")
        for score_type, stats in stats_summary.items():
            f.write(f"\n{score_type}:\n")
            for stat_name, value in stats.items():
                f.write(f"  {stat_name}: {value:.3f}\n")
        
        f.write(f"\n原始分数与新分数的相关系数: {correlation:.3f}\n\n")
        
        f.write("2. 按共情类型分组统计\n")
        f.write("-" * 25 + "\n")
        f.write(empathy_type_stats.to_string())
        
        f.write("\n\n3. 按属性类型分组统计\n")
        f.write("-" * 25 + "\n")
        f.write(attribute_type_stats.to_string())
        
        f.write("\n\n4. 主要发现\n")
        f.write("-" * 15 + "\n")
        
        # 找出新分数最高和最低的empathy_type
        empathy_means = df.groupby('empathy_type')['new_empathy_score'].mean()
        best_empathy = empathy_means.idxmax()
        worst_empathy = empathy_means.idxmin()
        
        f.write(f"- 新公式下表现最好的共情类型: {best_empathy} (平均分: {empathy_means[best_empathy]:.2f})\n")
        f.write(f"- 新公式下表现最差的共情类型: {worst_empathy} (平均分: {empathy_means[worst_empathy]:.2f})\n")
        
        # 找出新分数最高和最低的attribute_type
        attr_means = df.groupby('attribute_type')['new_empathy_score'].mean()
        best_attr = attr_means.idxmax()
        worst_attr = attr_means.idxmin()
        
        f.write(f"- 新公式下表现最好的属性类型: {best_attr} (平均分: {attr_means[best_attr]:.2f})\n")
        f.write(f"- 新公式下表现最差的属性类型: {worst_attr} (平均分: {attr_means[worst_attr]:.2f})\n")
        
        # 组成部分分析
        f.write(f"\n5. 新公式各组成部分平均贡献\n")
        f.write("-" * 30 + "\n")
        f.write(f"- 第二人称贡献 (×1.5): {df['second_person_contribution'].mean():.2f}\n")
        f.write(f"- 负面情绪贡献 (×1.0): {df['negative_emotion_contribution'].mean():.2f}\n")
        f.write(f"- 认知+洞察贡献 (×1.2): {df['cognitive_insight_contribution'].mean():.2f}\n")
        f.write(f"- 第一人称惩罚 (×2.0): -{df['first_person_penalty'].mean():.2f}\n")
    
    print(f"分析报告已保存到: {report_file}")

def main():
    """主函数"""
    print("开始Empathy Score分析")
    print("=" * 50)
    
    # 加载数据
    data_file = 'liwc_results.csv'
    df = load_data(data_file)
    
    # 计算新的empathy score
    df_with_new_score = calculate_new_empathy_score(df)
    
    # 创建对比分析图表
    fig = create_comparison_analysis(df_with_new_score)
    
    # 生成详细统计
    stats_summary, empathy_type_stats, attribute_type_stats, correlation = create_detailed_statistics(df_with_new_score)
    
    # 保存结果
    save_results(df_with_new_score, stats_summary, empathy_type_stats, attribute_type_stats, correlation)
    
    print("\n" + "=" * 50)
    print("分析完成！")
    print("生成的文件:")
    print("- empathy_scores_with_new_formula.csv (包含新分数的数据)")
    print("- empathy_score_comparison.png (对比分析图表)")
    print("- output/empathy_score_analysis_report.txt (详细分析报告)")

if __name__ == "__main__":
    main()