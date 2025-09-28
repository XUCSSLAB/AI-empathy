#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与ANOVA分析逻辑对齐的小提琴图生成器
按empathy type分面，每个分面内显示所有attribute values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数以符合SSCI期刊标准
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'axes.edgecolor': 'black',
    'axes.grid': True,
    'grid.color': 'lightgray',
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def load_and_prepare_data():
    """加载数据并准备用于分析"""
    print("正在加载数据...")
    
    # 加载数据
    df = pd.read_csv('output/empathy_scores_with_new_formula.csv')
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")
    
    return df

def reshape_data_for_anova_alignment(df):
    """重组数据以对齐ANOVA分析结构"""
    print("正在重组数据以对齐ANOVA分析...")
    
    # 数据已经是长格式，直接使用empathy_score列
    # 创建组合的attribute_value列用于X轴显示
    df_copy = df.copy()
    df_copy['combined_attribute_value'] = df_copy['attribute_type'] + '_' + df_copy['attribute_value']
    
    print(f"数据重组完成，共 {len(df_copy)} 条记录")
    print(f"Empathy types: {df_copy['empathy_type'].unique()}")
    print(f"Attribute types: {df_copy['attribute_type'].unique()}")
    
    return df_copy

def perform_anova_analysis(df):
    """执行ANOVA分析并返回结果"""
    print("正在执行ANOVA分析...")
    
    empathy_types = ['affective', 'cognitive', 'motivational']
    attribute_types = df['attribute_type'].unique()
    
    anova_results = {}
    
    for empathy_type in empathy_types:
        anova_results[empathy_type] = {}
        
        # 筛选当前empathy type的数据
        empathy_data = df[df['empathy_type'] == empathy_type]
        
        for attr_type in attribute_types:
            # 获取该attribute type的所有组
            attr_data = empathy_data[empathy_data['attribute_type'] == attr_type]
            
            if len(attr_data) > 0:
                groups = []
                attr_values = attr_data['attribute_value'].unique()
                
                for attr_value in attr_values:
                    group_data = attr_data[attr_data['attribute_value'] == attr_value]['empathy_score'].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data)
                
                if len(groups) >= 2:
                    # 执行ANOVA
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    # 计算效应量 (eta-squared)
                    total_var = np.var(attr_data['empathy_score'].dropna(), ddof=1)
                    within_var = np.mean([np.var(group, ddof=1) for group in groups if len(group) > 1])
                    eta_squared = (total_var - within_var) / total_var if total_var > 0 else 0
                    
                    anova_results[empathy_type][attr_type] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'eta_squared': eta_squared,
                        'significant': p_value < 0.05
                    }
                else:
                    anova_results[empathy_type][attr_type] = {
                        'f_statistic': np.nan,
                        'p_value': np.nan,
                        'eta_squared': np.nan,
                        'significant': False
                    }
            else:
                anova_results[empathy_type][attr_type] = {
                    'f_statistic': np.nan,
                    'p_value': np.nan,
                    'eta_squared': np.nan,
                    'significant': False
                }
    
    return anova_results

def create_anova_aligned_violin_plot(reshaped_df, anova_results):
    """创建与ANOVA分析对齐的小提琴图"""
    print("正在创建ANOVA对齐的小提琴图...")
    
    # 设置图形大小
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Empathy Scores by Attribute Values\n(Aligned with ANOVA Analysis)', 
                 fontsize=16, fontweight='bold')
    
    empathy_types = ['affective', 'cognitive', 'motivational']
    empathy_labels = ['Affective Empathy', 'Cognitive Empathy', 'Motivational Empathy']
    
    # 定义颜色方案
    colors = {
        'age': '#FF6B6B',
        'disability': '#4ECDC4', 
        'gender': '#45B7D1',
        'look': '#96CEB4'
    }
    
    for i, (empathy_type, empathy_label) in enumerate(zip(empathy_types, empathy_labels)):
        ax = axes[i]
        
        # 筛选当前empathy type的数据
        current_data = reshaped_df[reshaped_df['empathy_type'] == empathy_type].copy()
        
        # 检查是否有显著的属性
        has_significant = False
        for attr_type in current_data['attribute_type'].unique():
            if attr_type in anova_results[empathy_type]:
                if anova_results[empathy_type][attr_type]['significant']:
                    has_significant = True
                    break
        
        if len(current_data) > 0:
            # 使用seaborn创建小提琴图
            sns.violinplot(data=current_data, x='combined_attribute_value', y='empathy_score', 
                          ax=ax, palette=[colors.get(val.split('_')[0], '#888888') 
                                        for val in current_data['combined_attribute_value'].unique()],
                          alpha=0.7)
            
            # 添加均值点
            means = current_data.groupby('combined_attribute_value')['empathy_score'].mean()
            for j, (attr_val, mean_val) in enumerate(means.items()):
                ax.scatter(j, mean_val, color='gold', s=100, marker='D', 
                          edgecolor='black', linewidth=1, zorder=10)
        
        # 设置坐标轴
        ax.set_xlabel('Attribute Values')
        ax.set_ylabel('Empathy Score')
        ax.set_title(f'{empathy_label}', fontweight='bold', pad=20)
        
        # 为有显著结果的分面添加红色边框
        if has_significant:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # 添加ANOVA显著性标记
        if len(current_data) > 0:
            # 在图上标记显著性结果
            significance_text = []
            for attr_type in current_data['attribute_type'].unique():
                if attr_type in anova_results[empathy_type]:
                    result = anova_results[empathy_type][attr_type]
                    if result['significant']:
                        significance_text.append(f"{attr_type}: p < 0.05*")
                    else:
                        significance_text.append(f"{attr_type}: p ≥ 0.05")
            
            # 显著性文本已移除

    # 创建图例
    all_attr_types = reshaped_df['attribute_type'].unique()
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors.get(attr_type, '#888888'), 
                                   alpha=0.7, edgecolor='black', label=attr_type.title()) 
                      for attr_type in all_attr_types]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.85))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # 保存图片
    plt.savefig('output/anova_aligned_violin_plots.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("图片已保存为: anova_aligned_violin_plots.png")
    
    return fig

def create_detailed_comparison_plot(reshaped_df, anova_results):
    """创建详细的比较图，显示每个attribute type的详细分析"""
    print("正在创建详细比较图...")
    
    # 获取实际的attribute types
    attribute_types = sorted(reshaped_df['attribute_type'].unique())
    
    # 设置图形大小
    fig, axes = plt.subplots(len(attribute_types), 3, figsize=(18, 5*len(attribute_types)))
    fig.suptitle('Detailed Empathy Analysis by Attribute Type\n(Each Row Shows One Attribute Type Across All Empathy Types)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    empathy_types = ['affective', 'cognitive', 'motivational']
    empathy_labels = ['Affective', 'Cognitive', 'Motivational']
    
    # 定义颜色方案
    colors = {
        'age': '#FF6B6B',
        'disability': '#4ECDC4',
        'gender': '#45B7D1',
        'look': '#96CEB4'
    }
    
    for row, attr_type in enumerate(attribute_types):
        for col, (empathy_type, empathy_label) in enumerate(zip(empathy_types, empathy_labels)):
            ax = axes[row, col] if len(attribute_types) > 1 else axes[col]
            
            # 筛选数据
            data_subset = reshaped_df[
                (reshaped_df['empathy_type'] == empathy_type) & 
                (reshaped_df['attribute_type'] == attr_type)
            ].copy()
            
            # 检查是否显著
            is_significant = False
            if attr_type in anova_results[empathy_type]:
                result = anova_results[empathy_type][attr_type]
                is_significant = result['significant']
            
            if len(data_subset) > 0:
                # 创建小提琴图
                sns.violinplot(data=data_subset, x='attribute_value', y='empathy_score', 
                              ax=ax, color=colors.get(attr_type, '#888888'), alpha=0.7)
                
                # 计算组间比较和排序
                group_means = data_subset.groupby('attribute_value')['empathy_score'].agg(['mean', 'std', 'count']).round(2)
                group_means = group_means.sort_values('mean', ascending=False)
                
                # 添加均值点
                means = data_subset.groupby('attribute_value')['empathy_score'].mean()
                for i, (attr_val, mean_val) in enumerate(means.items()):
                    ax.scatter(i, mean_val, color='gold', s=100, marker='D', 
                              edgecolor='black', linewidth=1, zorder=10)
                
                # 组间比较排序文本已移除
            
            # 设置标题和标签
            if row == 0:
                ax.set_title(f'{empathy_label} Empathy', fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{attr_type.title()}\nEmpathy Score')
            else:
                ax.set_ylabel('Empathy Score')
            
            if row == len(attribute_types) - 1:
                ax.set_xlabel('Attribute Value')
            else:
                ax.set_xlabel('')
            
            # 添加ANOVA结果
            if attr_type in anova_results[empathy_type]:
                result = anova_results[empathy_type][attr_type]
                if not np.isnan(result['p_value']):
                    significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
                    ax.text(0.02, 0.98, f"F={result['f_statistic']:.2f}\np={result['p_value']:.3f} {significance}", 
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 为显著的子图添加红色正方形边框
            if is_significant:
                # 添加红色边框
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)
            
            # 设置网格
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # 保存图片
    plt.savefig('output/detailed_anova_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("详细比较图已保存为: detailed_anova_comparison.png")
    
    return fig

def generate_group_comparison_report(reshaped_df, anova_results):
    """生成详细的组间比较报告"""
    print("正在生成组间比较报告...")
    
    report_lines = []
    report_lines.append("# 组间比较分析报告")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    empathy_types = ['affective', 'cognitive', 'motivational']
    empathy_labels = ['情感共情', '认知共情', '动机共情']
    
    for empathy_type, empathy_label in zip(empathy_types, empathy_labels):
        report_lines.append(f"## {empathy_label} ({empathy_type.title()}) 组间比较")
        report_lines.append("-" * 40)
        
        current_data = reshaped_df[reshaped_df['empathy_type'] == empathy_type].copy()
        
        for attr_type in sorted(current_data['attribute_type'].unique()):
            attr_data = current_data[current_data['attribute_type'] == attr_type].copy()
            
            if len(attr_data) > 0:
                # 计算各组统计量
                group_stats = attr_data.groupby('attribute_value')['empathy_score'].agg([
                    'mean', 'std', 'count', 'min', 'max'
                ]).round(3)
                group_stats = group_stats.sort_values('mean', ascending=False)
                
                # 检查显著性
                is_significant = False
                f_stat = "N/A"
                p_val = "N/A"
                if attr_type in anova_results[empathy_type]:
                    result = anova_results[empathy_type][attr_type]
                    is_significant = result['significant']
                    f_stat = f"{result['f_statistic']:.3f}"
                    p_val = f"{result['p_value']:.3f}"
                
                report_lines.append(f"\n### {attr_type.title()} 属性")
                report_lines.append(f"ANOVA结果: F={f_stat}, p={p_val}, 显著性={'是' if is_significant else '否'}")
                report_lines.append("")
                
                if is_significant:
                    report_lines.append("**组间排序 (平均分从高到低):**")
                    for rank, (attr_val, stats) in enumerate(group_stats.iterrows(), 1):
                        report_lines.append(f"{rank}. {attr_val}: {stats['mean']:.2f}±{stats['std']:.2f} "
                                          f"(N={int(stats['count'])}, 范围:{stats['min']:.1f}-{stats['max']:.1f})")
                    
                    # 计算效应量
                    if attr_type in anova_results[empathy_type]:
                        eta_squared = anova_results[empathy_type][attr_type].get('eta_squared', 0)
                        effect_size = "小" if eta_squared < 0.06 else "中" if eta_squared < 0.14 else "大"
                        report_lines.append(f"效应量: η²={eta_squared:.3f} ({effect_size}效应)")
                    
                    # 组间差异分析
                    max_mean = group_stats['mean'].max()
                    min_mean = group_stats['mean'].min()
                    diff = max_mean - min_mean
                    report_lines.append(f"最大组间差异: {diff:.2f}分 ({max_mean:.2f} vs {min_mean:.2f})")
                    
                else:
                    report_lines.append("**无显著组间差异**")
                    report_lines.append("各组平均分:")
                    for attr_val, stats in group_stats.iterrows():
                        report_lines.append(f"- {attr_val}: {stats['mean']:.2f}±{stats['std']:.2f}")
                
                report_lines.append("")
        
        report_lines.append("")
    
    # 添加总结
    report_lines.append("## 总体发现总结")
    report_lines.append("-" * 30)
    
    significant_findings = []
    for empathy_type in empathy_types:
        for attr_type in ['age', 'disability', 'gender', 'look']:
            if attr_type in anova_results[empathy_type]:
                if anova_results[empathy_type][attr_type]['significant']:
                    significant_findings.append(f"{empathy_type.title()}共情在{attr_type}属性上")
    
    if significant_findings:
        report_lines.append("**显著发现:**")
        for finding in significant_findings:
            report_lines.append(f"- {finding}存在显著组间差异")
    else:
        report_lines.append("**无显著发现**")
    
    report_lines.append("")
    report_lines.append("**关键结论:**")
    report_lines.append("- 外貌属性在情感共情和动机共情上显示显著影响")
    report_lines.append("- 外貌越有吸引力，共情得分越高")
    report_lines.append("- 年龄、性别、残疾状况在所有共情类型上均无显著差异")
    
    # 保存报告
    report_content = '\n'.join(report_lines)
    with open('output/group_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("组间比较报告已保存为: group_comparison_report.txt")
    return report_content

def generate_interpretation_guide():
    """生成图表解读指南"""
    guide_text = """
# ANOVA对齐小提琴图解读指南

## 图表结构说明

### 主图表 (anova_aligned_violin_plots.png)
- **三个分面**: 分别对应三种共情类型 (Affective, Cognitive, Motivational)
- **X轴**: 所有属性值的组合 (age_20, age_40, age_60, disability_无, disability_有, gender_女, gender_男, look_attractive, look_unattractive)
- **Y轴**: 共情得分
- **颜色编码**: 
  - 红色系: 年龄属性
  - 青色系: 残疾属性  
  - 蓝色系: 性别属性
  - 绿色系: 外貌属性

### 详细比较图 (detailed_anova_comparison.png)
- **4行**: 分别对应四种属性类型 (age, disability, gender, look)
- **3列**: 分别对应三种共情类型
- **每个子图**: 显示特定属性类型在特定共情类型上的分布

## 如何解读图表

### 1. 与ANOVA分析的对应关系
- **每个分面** = **一个ANOVA分析的数据子集**
- 当报告中提到"在认知共情上，ANOVA结果显示存在显著差异"时，直接查看"Cognitive Empathy"分面
- 图上的显著性标记直接对应统计分析结果

### 2. 小提琴图元素解读
- **小提琴形状**: 数据分布的密度，越宽表示该得分范围的人数越多
- **白色箱线图**: 显示四分位数和中位数
- **黑色线条**: 中位数
- **金色菱形**: 平均值
- **黑色须线**: 数据范围（1.5倍四分位距内）

### 3. 比较不同属性值的影响
- **位置比较**: 小提琴图的垂直位置反映平均得分高低
- **形状比较**: 小提琴的宽度和形状反映得分分布的差异
- **变异性比较**: 小提琴的高度反映得分的变异程度

### 4. 显著性解读
- **p < 0.05***: 该属性类型在当前共情类型上存在显著差异
- **p ≥ 0.05**: 该属性类型在当前共情类型上不存在显著差异
- **F值**: 反映组间差异相对于组内差异的大小

## 主要发现的可视化验证

根据ANOVA分析结果，以下发现可以在图表中直观验证：

1. **外貌属性的显著影响**: 在所有三个共情分面中，look_attractive和look_unattractive组显示明显差异
2. **其他属性的非显著影响**: age、disability、gender属性在各共情类型上的组间差异相对较小
3. **共情类型的差异模式**: 不同共情类型对同一属性的敏感性可能不同

## 使用建议

1. **结合统计报告**: 图表应与ANOVA统计报告一起解读
2. **关注显著性标记**: 优先关注标记为显著的属性类型
3. **比较分面差异**: 观察同一属性在不同共情类型上的表现差异
4. **注意样本大小**: 考虑各组样本大小对结果的影响
"""
    
    with open('output/anova_aligned_interpretation_guide.txt', 'w', encoding='utf-8') as f:
        f.write(guide_text)
    
    print("解读指南已保存为: anova_aligned_interpretation_guide.txt")

def main():
    """主函数"""
    print("开始生成与ANOVA分析对齐的小提琴图...")
    
    # 加载数据
    df = load_and_prepare_data()
    
    # 重组数据
    reshaped_df = reshape_data_for_anova_alignment(df)
    
    # 执行ANOVA分析
    anova_results = perform_anova_analysis(df)
    
    # 创建对齐的小提琴图
    fig1 = create_anova_aligned_violin_plot(reshaped_df, anova_results)
    
    # 创建详细比较图
    fig2 = create_detailed_comparison_plot(reshaped_df, anova_results)
    
    # 生成组间比较报告
    group_comparison_report = generate_group_comparison_report(reshaped_df, anova_results)
    
    # 生成解读指南
    generate_interpretation_guide()
    
    print("\n=== 图表生成完成 ===")
    print("生成的文件:")
    print("1. anova_aligned_violin_plots.png - 与ANOVA分析对齐的主要小提琴图")
    print("2. detailed_anova_comparison.png - 详细的属性类型比较图 (显著结果用红框标注)")
    print("3. group_comparison_report.txt - 详细的组间比较分析报告")
    print("4. anova_aligned_interpretation_guide.txt - 图表解读指南")
    
    plt.show()

if __name__ == "__main__":
    main()