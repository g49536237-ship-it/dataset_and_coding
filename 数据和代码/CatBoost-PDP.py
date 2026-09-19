import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from sklearn.inspection import partial_dependence
import os
import re
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# =======================
# 1. 读取数据集
# =======================
data = pd.read_csv('3-2.csv', encoding='gbk')

for col in data.columns:
    if data[col].dtype == 'object' and any('%' in str(val) for val in data[col]):
        data[col] = data[col].str.replace('%', '').astype(float)

feature_names = list(data.columns[:-1])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# =======================
# 2. 划分数据集
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43
)

# =======================
# 3. CatBoost 训练
# =======================
param_grid = {
    'iterations': [500, 1000],
    'learning_rate': [0.05, 0.1],
    'depth': [6, 8]
}

cb_model = CatBoostRegressor(loss_function='RMSE', random_state=43, verbose=0)
grid_search = GridSearchCV(cb_model, param_grid, cv=3, scoring='r2', verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
cb_model = CatBoostRegressor(**best_params, loss_function='RMSE', random_state=43, verbose=0)
cb_model.fit(X_train, y_train)

# =======================
# 4. PDP 提取与绘图 (修复报错部分)
# =======================
save_fig_path = r'D:\pycharm\shijie\picture\PDP_fig_cb'
save_data_path = r'D:\pycharm\shijie\picture\PDP_data_cb'
os.makedirs(save_fig_path, exist_ok=True)
os.makedirs(save_data_path, exist_ok=True)

results = []


def clean_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', '', filename).strip()


num_top_features_to_plot = min(12, len(feature_names))
top_features = feature_names[:num_top_features_to_plot]

print("\n正在生成 PDP 数据...")

for feature in top_features:
    # 获取真实唯一值
    raw_values = X_train[feature].values
    unique_real_values = np.sort(np.unique(raw_values))

    # 计算偏依赖
    pdp_results = partial_dependence(
        cb_model,
        X_train,
        features=[feature],
        kind='average',
        percentiles=(0, 1),
        grid_resolution=len(unique_real_values)
    )

    # --- 兼容性修复：判断返回对象的键名 ---
    # 旧版本返回 'values'，新版本可能在某些结构下返回 'grid_values'
    # 或者直接作为元组返回 (average, values)

    if isinstance(pdp_results, dict) or hasattr(pdp_results, 'keys'):
        # 尝试不同的可能键名
        y_pdp = pdp_results.get('average', pdp_results.get('individual'))[0]
        x_pdp = pdp_results.get('values', pdp_results.get('grid_values'))[0]
    else:
        # 如果返回的是元组 (average, values)
        y_pdp = pdp_results[0][0]
        x_pdp = pdp_results[1][0]

    # 保存数据
    df_pdp = pd.DataFrame({
        'Feature': feature,
        'Real_Value_X': x_pdp,
        'PDP_Response_Y': y_pdp
    })
    results.append(df_pdp)

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_pdp, y_pdp, marker='o', markersize=4, color='#1f77b4', label='PDP Trend')

    # 添加地毯图
    ax.vlines(raw_values, ymin=min(y_pdp), ymax=min(y_pdp) + (max(y_pdp) - min(y_pdp)) * 0.05,
              color='red', alpha=0.2, linewidth=0.5)

    ax.set_xlabel(f"{feature} (Real Values)")
    ax.set_ylabel("Partial Dependence")
    ax.set_title(f"PDP for {feature}")
    plt.tight_layout()

    fig_filename = os.path.join(save_fig_path, f"{clean_filename(feature)}_PDP_Real.png")
    fig.savefig(fig_filename, dpi=300)
    plt.close(fig)

# 保存 Excel
if results:
    all_results = pd.concat(results, ignore_index=True)
    excel_filename = os.path.join(save_data_path, "partial_dependence_real_values.xlsx")
    all_results.to_excel(excel_filename, index=False)
    print(f"\n全部完成！数据已保存至：\n{excel_filename}")