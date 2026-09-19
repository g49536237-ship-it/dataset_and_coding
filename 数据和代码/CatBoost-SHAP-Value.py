import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import os

# =======================
# 1. 读取与数据清洗
# =======================
data = pd.read_csv('3-2.csv', encoding='gbk')

# 转换百分比列
for col in data.columns:
    if data[col].dtype == 'object' and any('%' in str(val) for val in data[col]):
        data[col] = data[col].str.replace('%', '').astype(float)

# 删除最后一列为0的行
data = data[data.iloc[:, -1] != 0]

# =======================
# 2. 特征映射（你的特定格式）
# =======================
feature_names_map = {
    'Fe2+/Fe3+': r'$\mathrm{Fe}^{\mathregular{2+}}$/$\mathrm{Fe}^{\mathregular{3+}}$',
    'EHOMO':  r'$\mathrm{E}_{\mathregular{HOMO}}$',
    'ELUMO':  r'$\mathrm{E}_{\mathregular{LUMO}}$',
    'Egap':  r'$\mathrm{E}_{\mathregular{gap}}$',
    'Mulliken electronegativity': r'$\chi_{\mathregular{M}}$',
    'Hardness': r'$\eta$'  # 希腊字母必须包裹在 $ 中

}

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# 3. 模型训练
# =======================
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

grid_params = {
    'iterations': [500, 1000],
    'learning_rate': [0.05, 0.1],
    'depth': [6, 8]
}

model = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0)
grid_search_result = model.grid_search(grid_params, train_pool, cv=3, verbose=False)

best_params = grid_search_result['params']
best_model = CatBoostRegressor(**best_params, loss_function='RMSE', random_seed=42, verbose=0)
best_model.fit(train_pool, eval_set=test_pool)

# =======================
# 4. SHAP 分析与绘图（统一字体 & 坐标格式）
# =======================

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)

# 特征名映射为 LaTeX
X_train_latex = X_train.rename(columns=feature_names_map)
feature_names_latex = X_train_latex.columns.tolist()

# 【新增】6. 自定义浅蓝 → 浅粉 SHAP 颜色映射
# =======================
from matplotlib.colors import LinearSegmentedColormap  # 【新增】

blue_to_pink_cmap = LinearSegmentedColormap.from_list(
    'blue_to_pink',
    ['#5888be', '#9a221c']
)



# -------- A. SHAP 点图（dot summary plot）--------
plt.figure(figsize=(10, 16))

shap.summary_plot(
    shap_values,
    X_train_latex,
    plot_type="dot",
    max_display=len(feature_names_latex),
    # cmap=blue_to_pink_cmap,
    cmap="coolwarm",
    show=False
)

ax = plt.gca()

# ===== 横纵坐标刻度字号 =====
ax.tick_params(
    axis='x',
    labelsize=20,
    direction='in'
)
ax.tick_params(
    axis='y',
    labelsize=20,
    direction='in'
)

# ===== 坐标轴标签字号 =====
ax.set_xlabel("SHAP value (impact on model output)", fontsize=24)

# ===== colorbar 字体控制 =====
# SHAP 的 colorbar 是最后一个 axes
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=16)
cbar.set_ylabel("Feature value", fontsize=18)

plt.tight_layout()
plt.savefig(r"D:\pycharm\shijie\photo\SHAP_dot_CatBoost.png", dpi=600)
plt.show()

# # B. 条形图 (Summary Bar Plot)
# plt.figure(figsize=(10, 8))
# shap.summary_plot(
#     shap_values,
#     X_train_latex, # 传入重命名后的 DataFrame
#     plot_type="bar",
#     max_display=len(feature_names_latex),
#     show=False
# )
# plt.xlabel("mean(|SHAP value|)", fontsize=18)
# plt.tight_layout()
# # plt.show()
#
# # =======================
# # 5. 归一化重要性 (%)
# # =======================
# mean_abs_shap = np.abs(shap_values).mean(axis=0)
# rel_imp_pct = mean_abs_shap / mean_abs_shap.sum() * 100
#
# # 包装成 DataFrame 以便排序
# df_imp = pd.DataFrame({
#     'Feature': feature_names_latex,
#     'Importance': rel_imp_pct
# }).sort_values(by='Importance', ascending=True)
#
# plt.figure(figsize=(8, len(feature_names_latex) * 0.4))
# plt.barh(df_imp['Feature'], df_imp['Importance'], color='#1f77b4', edgecolor='black')
# plt.xlabel("Relative importance (%)", fontsize=14)
# plt.title("Feature Importance Percentage", fontsize=16)
# plt.tight_layout()
# # plt.show()

# =======================
# # 6. 打印结果
# # =======================
# print("\n--- 各特征重要性评分 ---")
# for index, row in df_imp.sort_values(by='Importance', ascending=False).iterrows():
#     # 打印时也会显示 LaTeX 源码，但在图中会渲染为符号
#     print(f"{row['Feature']}: {row['Importance']:.2f}%")