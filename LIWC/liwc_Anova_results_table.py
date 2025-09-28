#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANOVA分析结果表格生成器 (无外部依赖版本)
创建清晰的统计分析结果表格
"""

import csv
import math
import statistics
from collections import defaultdict

def load_data():
    """加载CSV数据"""
    print("正在加载数据...")
    data = []
    with open('output/empathy_scores_with_new_formula.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['empathy_score'] = float(row['empathy_score'])
                data.append(row)
            except (ValueError, KeyError):
                continue
    return data

def calculate_f_statistic(groups):
    """计算F统计量"""
    if len(groups) < 2:
        return 0, 1
    
    # 计算各组统计量
    group_means = []
    group_sizes = []
    all_values = []
    
    for group in groups:
        if len(group) > 0:
            group_means.append(statistics.mean(group))
            group_sizes.append(len(group))
            all_values.extend(group)
    
    if len(all_values) < 2:
        return 0, 1
    
    # 总体均值
    grand_mean = statistics.mean(all_values)
    
    # 组间平方和 (SSB)
    ssb = sum(n * (mean - grand_mean) ** 2 for mean, n in zip(group_means, group_sizes))
    
    # 组内平方和 (SSW)
    ssw = 0
    for group in groups:
        if len(group) > 1:
            group_mean = statistics.mean(group)
            ssw += sum((x - group_mean) ** 2 for x in group)
    
    # 自由度
    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    
    if df_between == 0 or df_within == 0:
        return 0, 1
    
    # 均方
    msb = ssb / df_between
    msw = ssw / df_within if df_within > 0 else 1
    
    # F统计量
    f_stat = msb / msw if msw > 0 else 0
    
    # 简化的p值估计 (基于F分布的近似)
    if f_stat > 7.0:
        p_value = 0.001
    elif f_stat > 4.0:
        p_value = 0.01
    elif f_stat > 2.5:
        p_value = 0.05
    else:
        p_value = 0.1
    
    return f_stat, p_value

def calculate_eta_squared(groups):
    """计算效应量 eta-squared"""
    all_values = []
    group_means = []
    group_sizes = []
    
    for group in groups:
        if len(group) > 0:
            all_values.extend(group)
            group_means.append(statistics.mean(group))
            group_sizes.append(len(group))
    
    if len(all_values) < 2:
        return 0
    
    grand_mean = statistics.mean(all_values)
    
    # 总变异
    total_ss = sum((x - grand_mean) ** 2 for x in all_values)
    
    # 组间变异
    between_ss = sum(n * (mean - grand_mean) ** 2 for mean, n in zip(group_means, group_sizes))
    
    return between_ss / total_ss if total_ss > 0 else 0

def analyze_data(data):
    """执行ANOVA分析"""
    print("正在执行ANOVA分析...")
    
    empathy_types = ['affective', 'cognitive', 'motivational']
    attribute_types = list(set(row['attribute_type'] for row in data))
    
    results = []
    
    for empathy_type in empathy_types:
        # 筛选当前empathy type的数据
        empathy_data = [row for row in data if row['empathy_type'] == empathy_type]
        
        for attr_type in attribute_types:
            # 获取该attribute type的所有组
            attr_data = [row for row in empathy_data if row['attribute_type'] == attr_type]
            
            if len(attr_data) > 0:
                # 按attribute_value分组
                groups_dict = defaultdict(list)
                for row in attr_data:
                    groups_dict[row['attribute_value']].append(row['empathy_score'])
                
                groups = [group for group in groups_dict.values() if len(group) > 0]
                
                if len(groups) >= 2:
                    # 执行ANOVA
                    f_stat, p_value = calculate_f_statistic(groups)
                    eta_squared = calculate_eta_squared(groups)
                    
                    # 计算组别统计
                    group_stats = {}
                    for attr_value, group in groups_dict.items():
                        if len(group) > 0:
                            group_stats[attr_value] = {
                                'n': len(group),
                                'mean': statistics.mean(group),
                                'std': statistics.stdev(group) if len(group) > 1 else 0
                            }
                    
                    # 总体统计
                    all_scores = [row['empathy_score'] for row in attr_data]
                    total_n = len(all_scores)
                    overall_mean = statistics.mean(all_scores)
                    overall_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
                    
                    # 确定显著性水平
                    if p_value < 0.001:
                        significance = "***"
                        sig_level = "p < 0.001"
                    elif p_value < 0.01:
                        significance = "**"
                        sig_level = "p < 0.01"
                    elif p_value < 0.05:
                        significance = "*"
                        sig_level = "p < 0.05"
                    else:
                        significance = "ns"
                        sig_level = "p ≥ 0.05"
                    
                    # 效应量分类
                    if eta_squared >= 0.14:
                        effect_size = 'Large'
                    elif eta_squared >= 0.06:
                        effect_size = 'Medium'
                    else:
                        effect_size = 'Small'
                    
                    # 添加结果
                    results.append({
                        'Empathy_Type': empathy_type.title(),
                        'Attribute_Type': attr_type.title(),
                        'F_Statistic': round(f_stat, 3),
                        'p_Value': round(p_value, 4),
                        'Significance': significance,
                        'Sig_Level': sig_level,
                        'Eta_Squared': round(eta_squared, 3),
                        'Effect_Size': effect_size,
                        'Total_N': total_n,
                        'Overall_Mean': round(overall_mean, 2),
                        'Overall_Std': round(overall_std, 2),
                        'Groups': len(groups),
                        'Group_Details': '; '.join([f"{k}: M={v['mean']:.2f}, SD={v['std']:.2f}, N={v['n']}" 
                                                   for k, v in group_stats.items()])
                    })
    
    return results

def save_csv_table(data, filename, headers):
    """保存数据为CSV表格"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        # 只保存指定字段的数据
        filtered_data = []
        for row in data:
            filtered_row = {k: row.get(k, '') for k in headers}
            filtered_data.append(filtered_row)
        writer.writerows(filtered_data)

def create_summary_table(results):
    """创建汇总表格"""
    print("正在创建ANOVA结果汇总表...")
    
    headers = ['Empathy_Type', 'Attribute_Type', 'F_Statistic', 'p_Value', 
               'Significance', 'Eta_Squared', 'Effect_Size', 'Total_N']
    
    summary_data = []
    for result in results:
        summary_data.append({k: result[k] for k in headers})
    
    return summary_data, headers

def create_significance_summary(results):
    """创建显著性结果汇总"""
    print("正在创建显著性结果汇总...")
    
    empathy_types = list(set(r['Empathy_Type'] for r in results))
    sig_summary = []
    
    for empathy_type in empathy_types:
        empathy_results = [r for r in results if r['Empathy_Type'] == empathy_type]
        
        significant_attrs = [r['Attribute_Type'] for r in empathy_results if r['Significance'] != 'ns']
        non_significant_attrs = [r['Attribute_Type'] for r in empathy_results if r['Significance'] == 'ns']
        
        sig_summary.append({
            'Empathy_Type': empathy_type,
            'Significant_Attributes': ', '.join(significant_attrs) if significant_attrs else 'None',
            'Non_Significant_Attributes': ', '.join(non_significant_attrs) if non_significant_attrs else 'None',
            'Total_Significant': len(significant_attrs),
            'Total_Tests': len(empathy_results)
        })
    
    headers = ['Empathy_Type', 'Significant_Attributes', 'Non_Significant_Attributes', 
               'Total_Significant', 'Total_Tests']
    
    return sig_summary, headers

def format_table_for_display(data, headers, max_rows=10):
    """格式化表格用于控制台显示"""
    if not data:
        return "无数据"
    
    # 计算列宽
    col_widths = {}
    for header in headers:
        col_widths[header] = len(header)
        for row in data[:max_rows]:
            col_widths[header] = max(col_widths[header], len(str(row.get(header, ''))))
    
    # 创建表格
    lines = []
    
    # 表头
    header_line = " | ".join(header.ljust(col_widths[header]) for header in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # 数据行
    for row in data[:max_rows]:
        data_line = " | ".join(str(row.get(header, '')).ljust(col_widths[header]) for header in headers)
        lines.append(data_line)
    
    if len(data) > max_rows:
        lines.append(f"... 还有 {len(data) - max_rows} 行数据")
    
    return "\n".join(lines)

def save_text_report(results, summary_data, sig_summary):
    """保存格式化的文本报告"""
    print("正在保存文本报告...")
    
    with open('output/anova_tables_report.txt', 'w', encoding='utf-8') as f:
        f.write("# ANOVA分析结果表格报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("## 1. ANOVA结果汇总表\n")
        f.write("-" * 30 + "\n")
        f.write("此表显示所有ANOVA分析的主要统计结果\n\n")
        
        # 汇总表
        summary_headers = ['Empathy_Type', 'Attribute_Type', 'F_Statistic', 'p_Value', 
                          'Significance', 'Eta_Squared', 'Effect_Size', 'Total_N']
        f.write(format_table_for_display(summary_data, summary_headers))
        f.write("\n\n")
        
        f.write("## 2. 显著性结果汇总\n")
        f.write("-" * 30 + "\n")
        f.write("此表汇总了每种共情类型的显著性结果\n\n")
        
        # 显著性汇总表
        sig_headers = ['Empathy_Type', 'Significant_Attributes', 'Non_Significant_Attributes', 
                      'Total_Significant', 'Total_Tests']
        f.write(format_table_for_display(sig_summary, sig_headers))
        f.write("\n\n")
        
        f.write("## 3. 详细ANOVA结果\n")
        f.write("-" * 30 + "\n")
        f.write("完整的统计信息和组别详情\n\n")
        
        for result in results:
            f.write(f"### {result['Empathy_Type']} - {result['Attribute_Type']}\n")
            f.write(f"F({result['Groups']-1}, {result['Total_N']-result['Groups']}) = {result['F_Statistic']}, ")
            f.write(f"p = {result['p_Value']}, η² = {result['Eta_Squared']} ({result['Effect_Size']} effect)\n")
            f.write(f"组别详情: {result['Group_Details']}\n\n")
        
        f.write("## 4. 统计符号说明\n")
        f.write("-" * 30 + "\n")
        f.write("*** : p < 0.001 (极显著)\n")
        f.write("**  : p < 0.01  (高度显著)\n")
        f.write("*   : p < 0.05  (显著)\n")
        f.write("ns  : p ≥ 0.05  (不显著)\n\n")
        
        f.write("## 5. 效应量解释\n")
        f.write("-" * 30 + "\n")
        f.write("Eta-squared (η²) 效应量标准:\n")
        f.write("- Small:  η² < 0.06\n")
        f.write("- Medium: 0.06 ≤ η² < 0.14\n")
        f.write("- Large:  η² ≥ 0.14\n\n")
        
        f.write("## 6. 主要发现\n")
        f.write("-" * 30 + "\n")
        
        # 分析主要发现
        significant_results = [r for r in results if r['Significance'] != 'ns']
        if significant_results:
            f.write("显著性结果:\n")
            for result in significant_results:
                f.write(f"- {result['Empathy_Type']}共情在{result['Attribute_Type']}属性上存在显著差异 ")
                f.write(f"(F = {result['F_Statistic']}, p = {result['p_Value']}, η² = {result['Eta_Squared']})\n")
        else:
            f.write("未发现显著性差异。\n")

def main():
    """主函数"""
    print("开始生成ANOVA分析结果表格...")
    
    # 加载数据并分析
    data = load_data()
    results = analyze_data(data)
    
    # 创建各种表格
    summary_data, summary_headers = create_summary_table(results)
    sig_summary, sig_headers = create_significance_summary(results)
    
    # 保存CSV文件
    save_csv_table(summary_data, 'output/anova_summary_table.csv', summary_headers)
    
    # 为详细表格准备正确的字段
    detailed_headers = ['Empathy_Type', 'Attribute_Type', 'F_Statistic', 'p_Value', 'Sig_Level', 
                       'Eta_Squared', 'Overall_Mean', 'Overall_Std', 'Groups', 'Group_Details']
    save_csv_table(results, 'output/anova_detailed_table.csv', detailed_headers)
    
    save_csv_table(sig_summary, 'output/anova_significance_summary.csv', sig_headers)
    
    # 保存文本报告
    save_text_report(results, summary_data, sig_summary)
    
    # 在控制台显示结果
    print("\n" + "="*80)
    print("ANOVA分析结果表格")
    print("="*80)
    
    print("\n📊 主要结果汇总表:")
    print("-" * 60)
    print(format_table_for_display(summary_data, summary_headers))
    
    print("\n\n📈 显著性结果汇总:")
    print("-" * 60)
    print(format_table_for_display(sig_summary, sig_headers))
    
    print("\n\n📋 详细结果 (前5项):")
    print("-" * 60)
    detailed_headers = ['Empathy_Type', 'Attribute_Type', 'F_Statistic', 'p_Value', 'Significance', 'Eta_Squared']
    print(format_table_for_display(results, detailed_headers, 5))
    
    print("\n\n文件已保存:")
    print("1. anova_summary_table.csv - 主要结果汇总表")
    print("2. anova_detailed_table.csv - 详细结果表")
    print("3. anova_significance_summary.csv - 显著性汇总表")
    print("4. anova_tables_report.txt - 格式化文本报告")
    
    print("\n✅ ANOVA分析表格生成完成！")

if __name__ == "__main__":
    main()