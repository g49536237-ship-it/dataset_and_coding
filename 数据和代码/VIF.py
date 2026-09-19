import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# =========================================================
# 全局字体设置：Arial（普通文本 + 数学文本）
# =========================================================
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 16,

    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',

    'xtick.labelsize': 16,
    'ytick.labelsize': 16,

    'legend.fontsize': 14,

    # 数学文本（关键）
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'mathtext.default': 'regular'
})


def calculate_vif(X):
    """
    计算每个输入特征的方差膨胀因子（VIF）
    """
    X_scaled = StandardScaler().fit_transform(X)

    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X_scaled, i)
        for i in range(X.shape[1])
    ]

    return vif_data.sort_values(by='VIF', ascending=True).reset_index(drop=True)


def plot_vif(vif_df):
    """
    绘制 VIF 水平条形图
    """
    plt.figure(figsize=(11, 8))

    # =====================================================
    # 特征名称映射（正体 + 正体上下标）
    # =====================================================
    feature_names_map = {
        'Fe2+/Fe3+': r'$\mathrm{Fe}^{\mathregular{2+}}$/$\mathrm{Fe}^{\mathregular{3+}}$',
        'EHOMO': r'$\mathrm{E}_{\mathregular{HOMO}}$',
        'ELUMO': r'$\mathrm{E}_{\mathregular{LUMO}}$',
        'Egap': r'$\mathrm{E}_{\mathregular{gap}}$',
        'Mulliken electronegativity': r'$\chi_{\mathregular{M}}$',
        'Hardness': r'$\eta$'
    }

    # 使用映射表生成 Y 轴标签
    feature_labels = [
        feature_names_map.get(f, f) for f in vif_df['Feature']
    ]

    colors = sns.color_palette("Spectral_r", len(vif_df))
    bars = plt.barh(feature_labels, vif_df['VIF'], color=colors)

    # Y 轴刻度字体（真正 normal）
    plt.yticks(fontsize=16, fontweight='normal')

    # 数值标注
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}',
            ha='left',
            va='center',
            fontsize=14
        )

    plt.xlabel('Variance Inflation Factor (VIF)', fontsize=20)
    # plt.title(
    #     'Variance Inflation Factor (VIF) of Each Feature (Sorted by VIF Ascending)',
    #     fontsize=20
    # )

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(r"D:\pytharm\shijie\picture\VIF.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    filepath = '3-2.csv'
    data = pd.read_csv(filepath)

    # 仅保留数值型列，清理 NaN / Inf
    data = data.select_dtypes(include=[np.number])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # =====================================================
    # 排除最后一列（通常为因变量）
    # =====================================================
    data = data.iloc[:, :-1]

    vif_result = calculate_vif(data)
    print(vif_result)

    plot_vif(vif_result)
