#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empathy Score and Attribute Value Relationship Analysis
分析empathy score与attribute_value的关系，按empathy_type分组
包含ANOVA分析和符合SSCI规范的可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

# 设置科学期刊标准的图表样式
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'normal',
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True
})

def load_and_prepare_data(file_path):
    """加载数据并进行预处理"""
    print("正在加载数据...")
    df = pd.read_csv(file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"Empathy类型: {sorted(df['empathy_type'].unique())}")
    print(f"Attribute类型: {sorted(df['attribute_type'].unique())}")
    
    # 查看每个attribute_type下的attribute_value
    print("\n各attribute_type下的attribute_value:")
    for attr_type in sorted(df['attribute_type'].unique()):
        values = sorted(df[df['attribute_type'] == attr_type]['attribute_value'].unique())
        print(f"  {attr_type}: {values}")
    
    return df

def perform_anova_analysis(df):
    """进行ANOVA分析"""
    print("\n" + "="*60)
    print("ANOVA分析结果")
    print("="*60)
    
    anova_results = {}
    
    # 按empathy_type分组进行ANOVA分析
    empathy_types = sorted(df['empathy_type'].unique())
    
    for empathy_type in empathy_types:
        print(f"\n{empathy_type.upper()} EMPATHY:")
        print("-" * 40)
        
        # 筛选当前empathy_type的数据
        subset = df[df['empathy_type'] == empathy_type]
        
        # 按attribute_type进行ANOVA
        attr_results = {}
        
        for attr_type in sorted(subset['attribute_type'].unique()):
            attr_subset = subset[subset['attribute_type'] == attr_type]
            
            # 获取不同attribute_value的empathy score
            groups = []
            group_names = []
            
            for attr_value in sorted(attr_subset['attribute_value'].unique()):
                group_data = attr_subset[attr_subset['attribute_value'] == attr_value]['new_empathy_score']
                if len(group_data) > 0:
                    groups.append(group_data)
                    group_names.append(attr_value)
            
            if len(groups) >= 2:  # 至少需要2组进行比较
                # 进行ANOVA
                f_stat, p_value = f_oneway(*groups)
                
                # 计算效应量 (eta-squared)
                total_var = np.var(attr_subset['new_empathy_score'], ddof=1)
                group_means = [np.mean(group) for group in groups]
                group_sizes = [len(group) for group in groups]
                overall_mean = np.mean(attr_subset['new_empathy_score'])
                
                between_var = sum(size * (mean - overall_mean)**2 for size, mean in zip(group_sizes, group_means))
                eta_squared = between_var / (len(attr_subset) * total_var) if total_var > 0 else 0
                
                attr_results[attr_type] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'eta_squared': eta_squared,
                    'groups': group_names,
                    'group_means': group_means,
                    'group_sizes': group_sizes
                }
                
                # 输出结果
                print(f"  {attr_type}:")
                print(f"    F统计量: {f_stat:.4f}")
                print(f"    p值: {p_value:.6f}")
                print(f"    效应量(η²): {eta_squared:.4f}")
                
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                
                print(f"    显著性: {significance}")
                
                # 输出各组均值
                print(f"    各组均值:")
                for name, mean, size in zip(group_names, group_means, group_sizes):
                    print(f"      {name}: {mean:.2f} (n={size})")
        
        anova_results[empathy_type] = attr_results
    
    return anova_results

def create_violin_plots(df):
    """创建符合SSCI规范的小提琴图"""
    print("\n正在创建小提琴图...")
    
    # 设置图表大小和布局
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Empathy Scores by Attribute Values Across Empathy Types\n(Violin Plots with Box Plots)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    empathy_types = ['cognitive', 'affective', 'motivational']
    empathy_labels = ['Cognitive Empathy', 'Affective Empathy', 'Motivational Empathy']
    
    # 定义颜色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (empathy_type, label, color) in enumerate(zip(empathy_types, empathy_labels, colors)):
        ax = axes[idx]
        
        # 筛选当前empathy_type的数据
        subset = df[df['empathy_type'] == empathy_type]
        
        # 创建小提琴图
        violin_parts = ax.violinplot([subset[subset['attribute_type'] == attr]['new_empathy_score'].values 
                                     for attr in sorted(subset['attribute_type'].unique())],
                                    positions=range(len(subset['attribute_type'].unique())),
                                    showmeans=False, showmedians=False, showextrema=False)
        
        # 设置小提琴图颜色
        for pc in violin_parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # 添加箱线图
        box_data = [subset[subset['attribute_type'] == attr]['new_empathy_score'].values 
                   for attr in sorted(subset['attribute_type'].unique())]
        
        box_parts = ax.boxplot(box_data, positions=range(len(subset['attribute_type'].unique())),
                              widths=0.3, patch_artist=True, 
                              boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.5),
                              medianprops=dict(color='red', linewidth=2),
                              whiskerprops=dict(linewidth=1.5),
                              capprops=dict(linewidth=1.5),
                              flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.6))
        
        # 设置坐标轴
        ax.set_xticks(range(len(subset['attribute_type'].unique())))
        ax.set_xticklabels([attr.capitalize() for attr in sorted(subset['attribute_type'].unique())], 
                          fontweight='bold')
        ax.set_ylabel('New Empathy Score', fontweight='bold', fontsize=14)
        ax.set_title(label, fontweight='bold', fontsize=16, pad=20)
        
        # 设置y轴范围
        ax.set_ylim(df['new_empathy_score'].min() - 2, df['new_empathy_score'].max() + 2)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # 添加统计信息
        y_pos = ax.get_ylim()[1] - 3
        for i, attr in enumerate(sorted(subset['attribute_type'].unique())):
            attr_data = subset[subset['attribute_type'] == attr]['new_empathy_score']
            mean_val = attr_data.mean()
            std_val = attr_data.std()
            n_val = len(attr_data)
            ax.text(i, y_pos, f'M={mean_val:.1f}\nSD={std_val:.1f}\nn={n_val}', 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('output/empathy_attribute_violin_plots.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_violin_plots_by_attribute(df):
    """为每个attribute_type创建详细的小提琴图"""
    print("\n正在创建按attribute_type分组的详细小提琴图...")
    
    attribute_types = sorted(df['attribute_type'].unique())
    n_attrs = len(attribute_types)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Empathy Scores by Attribute Values\n(Detailed Analysis by Attribute Type)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    # 定义颜色方案
    empathy_colors = {'cognitive': '#2E86AB', 'affective': '#A23B72', 'motivational': '#F18F01'}
    
    for idx, attr_type in enumerate(attribute_types):
        if idx < len(axes):
            ax = axes[idx]
            
            # 筛选当前attribute_type的数据
            attr_subset = df[df['attribute_type'] == attr_type]
            
            # 准备数据
            empathy_types = sorted(attr_subset['empathy_type'].unique())
            attr_values = sorted(attr_subset['attribute_value'].unique())
            
            # 创建位置
            x_positions = []
            data_for_violin = []
            colors_for_violin = []
            labels_for_legend = []
            
            pos = 0
            for attr_val in attr_values:
                for emp_type in empathy_types:
                    subset_data = attr_subset[
                        (attr_subset['attribute_value'] == attr_val) & 
                        (attr_subset['empathy_type'] == emp_type)
                    ]['new_empathy_score'].values
                    
                    if len(subset_data) > 0:
                        data_for_violin.append(subset_data)
                        x_positions.append(pos)
                        colors_for_violin.append(empathy_colors[emp_type])
                        if attr_val == attr_values[0]:  # 只在第一个attr_value时添加图例标签
                            labels_for_legend.append(emp_type.capitalize())
                    pos += 1
                pos += 0.5  # 在不同attribute_value之间添加间隔
            
            # 创建小提琴图
            if data_for_violin:
                violin_parts = ax.violinplot(data_for_violin, positions=x_positions,
                                           showmeans=False, showmedians=False, showextrema=False)
                
                # 设置颜色
                for pc, color in zip(violin_parts['bodies'], colors_for_violin):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1)
                
                # 添加箱线图
                box_parts = ax.boxplot(data_for_violin, positions=x_positions,
                                      widths=0.2, patch_artist=True,
                                      boxprops=dict(facecolor='white', alpha=0.8, linewidth=1),
                                      medianprops=dict(color='red', linewidth=2),
                                      whiskerprops=dict(linewidth=1),
                                      capprops=dict(linewidth=1),
                                      flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.6))
            
            # 设置坐标轴
            ax.set_title(f'{attr_type.capitalize()} Attribute', fontweight='bold', fontsize=14)
            ax.set_ylabel('New Empathy Score', fontweight='bold')
            
            # 设置x轴标签
            if attr_values:
                tick_positions = []
                tick_labels = []
                pos = 0
                for attr_val in attr_values:
                    # 计算每个attribute_value组的中心位置
                    center_pos = pos + (len(empathy_types) - 1) / 2
                    tick_positions.append(center_pos)
                    tick_labels.append(str(attr_val))
                    pos += len(empathy_types) + 0.5
                
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontweight='bold')
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
    
    # 隐藏多余的子图
    for idx in range(len(attribute_types), len(axes)):
        axes[idx].set_visible(False)
    
    # 添加图例
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=empathy_colors[emp_type], alpha=0.7, 
                                   edgecolor='black', label=emp_type.capitalize()) 
                      for emp_type in ['cognitive', 'affective', 'motivational']]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
              title='Empathy Type', title_fontsize=12, fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('output/empathy_detailed_violin_plots.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_comprehensive_report(df, anova_results):
    """生成综合分析报告"""
    print("\n正在生成综合分析报告...")
    
    report_file = 'output/empathy_attribute_analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("EMPATHY SCORE AND ATTRIBUTE VALUE RELATIONSHIP ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("研究目的: 探究empathy score与attribute_value的关系，按empathy_type分组分析\n")
        f.write("分析方法: ANOVA分析 + 小提琴图可视化\n\n")
        
        # 数据概述
        f.write("1. 数据概述\n")
        f.write("-" * 20 + "\n")
        f.write(f"总样本量: {len(df)}\n")
        f.write(f"Empathy类型: {', '.join(sorted(df['empathy_type'].unique()))}\n")
        f.write(f"Attribute类型: {', '.join(sorted(df['attribute_type'].unique()))}\n\n")
        
        # 各组样本量
        f.write("各组样本量分布:\n")
        sample_counts = df.groupby(['empathy_type', 'attribute_type', 'attribute_value']).size().unstack(fill_value=0)
        f.write(sample_counts.to_string())
        f.write("\n\n")
        
        # ANOVA结果
        f.write("2. ANOVA分析结果\n")
        f.write("-" * 25 + "\n")
        
        for empathy_type, results in anova_results.items():
            f.write(f"\n{empathy_type.upper()} EMPATHY:\n")
            f.write("-" * 40 + "\n")
            
            for attr_type, stats in results.items():
                f.write(f"\n  {attr_type}属性:\n")
                f.write(f"    F统计量: {stats['f_statistic']:.4f}\n")
                f.write(f"    p值: {stats['p_value']:.6f}\n")
                f.write(f"    效应量(η²): {stats['eta_squared']:.4f}\n")
                
                # 显著性判断
                if stats['p_value'] < 0.001:
                    significance = "极显著 (***)"
                elif stats['p_value'] < 0.01:
                    significance = "非常显著 (**)"
                elif stats['p_value'] < 0.05:
                    significance = "显著 (*)"
                else:
                    significance = "不显著 (ns)"
                
                f.write(f"    显著性: {significance}\n")
                
                # 效应量解释
                if stats['eta_squared'] >= 0.14:
                    effect_size = "大效应"
                elif stats['eta_squared'] >= 0.06:
                    effect_size = "中等效应"
                elif stats['eta_squared'] >= 0.01:
                    effect_size = "小效应"
                else:
                    effect_size = "无效应"
                
                f.write(f"    效应量大小: {effect_size}\n")
                
                # 各组均值
                f.write(f"    各组均值:\n")
                for name, mean, size in zip(stats['groups'], stats['group_means'], stats['group_sizes']):
                    f.write(f"      {name}: {mean:.2f} (n={size})\n")
        
        # 描述性统计
        f.write("\n\n3. 描述性统计\n")
        f.write("-" * 20 + "\n")
        
        desc_stats = df.groupby(['empathy_type', 'attribute_type', 'attribute_value'])['new_empathy_score'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        
        f.write(desc_stats.to_string())
        
        # 主要发现
        f.write("\n\n4. 主要发现\n")
        f.write("-" * 15 + "\n")
        
        # 找出显著差异
        significant_findings = []
        for empathy_type, results in anova_results.items():
            for attr_type, stats in results.items():
                if stats['p_value'] < 0.05:
                    significant_findings.append(f"{empathy_type} empathy在{attr_type}属性上存在显著差异 (p={stats['p_value']:.4f})")
        
        if significant_findings:
            f.write("显著性发现:\n")
            for finding in significant_findings:
                f.write(f"- {finding}\n")
        else:
            f.write("- 未发现显著的组间差异\n")
        
        # 最高和最低分组
        f.write(f"\n各empathy类型的最高平均分组:\n")
        for empathy_type in sorted(df['empathy_type'].unique()):
            subset = df[df['empathy_type'] == empathy_type]
            group_means = subset.groupby(['attribute_type', 'attribute_value'])['new_empathy_score'].mean()
            best_group = group_means.idxmax()
            best_score = group_means.max()
            f.write(f"- {empathy_type}: {best_group[0]}-{best_group[1]} (平均分: {best_score:.2f})\n")
        
        f.write(f"\n各empathy类型的最低平均分组:\n")
        for empathy_type in sorted(df['empathy_type'].unique()):
            subset = df[df['empathy_type'] == empathy_type]
            group_means = subset.groupby(['attribute_type', 'attribute_value'])['new_empathy_score'].mean()
            worst_group = group_means.idxmin()
            worst_score = group_means.min()
            f.write(f"- {empathy_type}: {worst_group[0]}-{worst_group[1]} (平均分: {worst_score:.2f})\n")
        
        f.write("\n\n5. 研究建议\n")
        f.write("-" * 15 + "\n")
        f.write("- 基于ANOVA结果，建议进一步进行事后检验(如Tukey HSD)来确定具体的组间差异\n")
        f.write("- 考虑进行多元方差分析(MANOVA)来同时考虑多个因变量\n")
        f.write("- 建议增加样本量以提高统计检验力\n")
        f.write("- 可以考虑进行非参数检验(如Kruskal-Wallis)作为补充分析\n")
    
    print(f"综合分析报告已保存到: {report_file}")

def main():
    """主函数"""
    print("开始Empathy Score与Attribute Value关系分析")
    print("=" * 70)
    
    # 加载数据
    data_file = 'output/empathy_scores_with_new_formula.csv'
    df = load_and_prepare_data(data_file)
    
    # 进行ANOVA分析
    anova_results = perform_anova_analysis(df)
    
    # 创建小提琴图
    fig1 = create_violin_plots(df)
    
    # 创建详细的小提琴图
    fig2 = create_detailed_violin_plots_by_attribute(df)
    
    # 生成综合报告
    generate_comprehensive_report(df, anova_results)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("生成的文件:")
    print("- empathy_attribute_violin_plots.png (按empathy_type分面的小提琴图)")
    print("- empathy_detailed_violin_plots.png (按attribute_type分组的详细小提琴图)")
    print("- empathy_attribute_analysis_report.txt (综合分析报告)")

if __name__ == "__main__":
    main()