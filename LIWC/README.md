# LIWC Empathy Analysis Scripts

这个文件夹包含了用于分析empathy score与各种属性关系的Python脚本。

## 文件说明

### 数据文件
- `liwc_results.csv` - 原始LIWC分析结果数据

### 分析脚本
1. `empathy_score.py` - 计算新的empathy score并生成对比分析
2. `liwc_Anova_results_table.py` - 生成ANOVA分析结果表格
3. `liwc_Anova_graph.py` - 生成与ANOVA分析对齐的小提琴图
4. `liwc_anovaTest.py` - 执行详细的ANOVA测试和可视化

### 输出目录
- `output/` - 所有生成的文件都保存在这个目录中

## 使用方法

### 1. 运行empathy score分析
```bash
python empathy_score.py
```
生成文件：
- `output/empathy_scores_with_new_formula.csv` - 包含新计算的empathy score的数据
- `output/empathy_score_comparison.png` - 对比分析图表
- `output/empathy_score_analysis_report.txt` - 详细分析报告

### 2. 生成ANOVA分析表格
```bash
python liwc_Anova_results_table.py
```
生成文件：
- `output/anova_summary_table.csv` - 主要结果汇总表
- `output/anova_detailed_table.csv` - 详细结果表
- `output/anova_significance_summary.csv` - 显著性汇总表
- `output/anova_tables_report.txt` - 格式化文本报告

### 3. 生成ANOVA对齐的可视化图表
```bash
python liwc_Anova_graph.py
```
生成文件：
- `output/anova_aligned_violin_plots.png` - 与ANOVA分析对齐的主要小提琴图
- `output/detailed_anova_comparison.png` - 详细的属性类型比较图
- `output/group_comparison_report.txt` - 详细的组间比较分析报告
- `output/anova_aligned_interpretation_guide.txt` - 图表解读指南

### 4. 执行详细的ANOVA测试
```bash
python liwc_anovaTest.py
```
生成文件：
- `output/empathy_attribute_violin_plots.png` - 按empathy_type分面的小提琴图
- `output/empathy_detailed_violin_plots.png` - 按attribute_type分组的详细小提琴图
- `output/empathy_attribute_analysis_report.txt` - 综合分析报告

## 运行顺序建议

1. 首先运行 `empathy_score.py` 生成新的empathy score数据
2. 然后可以运行其他三个脚本进行不同类型的分析

## 依赖包

确保安装了以下Python包：
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## 注意事项

- 所有输出文件都保存在 `output/` 目录中
- 脚本使用相对路径，适合在GitHub等版本控制系统中使用
- 如果需要修改输入数据文件，请更新相应脚本中的文件路径