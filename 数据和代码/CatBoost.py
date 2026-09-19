import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool

# 读取数据集
data = pd.read_csv('分出的数据.csv', encoding='gbk')  # 请确保该文件存在
for col in data.columns:
    if any('%' in str(val) for val in data[col]):
        data[col] = data[col].str.replace('%', '').astype(float)

# 删除最后一列为0的行
data = data[data.iloc[:, -1] != 0]

# 提取自变量和因变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 指定分类特征的索引（根据实际数据调整）
cat_features = []

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 转换为CatBoost Pool格式
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# 设置参数搜索范围（缩小范围）
grid_params = {
    'iterations': [500, 1000],
    'learning_rate': [0.05, 0.1],
    'depth': [6, 8]
}

# 使用 CatBoost 内置的 grid_search 方法
model = CatBoostRegressor(loss_function='RMSE', cat_features=cat_features, random_seed=42, verbose=0)
grid_search_result = model.grid_search(grid_params, train_pool, cv=3, verbose=True)

# 使用最佳参数训练模型
best_params = grid_search_result['params']
best_model = CatBoostRegressor(**best_params, loss_function='RMSE', cat_features=cat_features, random_seed=42)
best_model.fit(train_pool, eval_set=test_pool, plot=False)

# 预测
y_pred_test = best_model.predict(test_pool)
y_pred_train = best_model.predict(train_pool)

# 评估
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
print("测试集 MSE: %.4f" % mean_squared_error(y_test, y_pred_test))
print("测试集 R^2: %.4f" % test_r2)
print("训练集 MSE: %.6f" % mean_squared_error(y_train, y_pred_train))
print("训练集 R^2: %.4f" % train_r2)

# 计算拟合直线的斜率和截距（测试集）
test_slope, test_intercept = np.polyfit(y_test, y_pred_test, 1)
# 计算拟合直线的斜率和截距（训练集）
train_slope, train_intercept = np.polyfit(y_train, y_pred_train, 1)

# 计算置信区间和预测区间
def get_confidence_prediction_bands(y_true, y_pred, confidence=0.95):
    n = len(y_true)
    se = np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 2))
    t_value = 1.96  # for 95% confidence interval

    pred_band = t_value * se * np.sqrt(1 + 1 / n)
    conf_band = t_value * se * np.sqrt(1 / n)

    return conf_band, pred_band

test_conf_band, test_pred_band = get_confidence_prediction_bands(y_test, y_pred_test)
train_conf_band, train_pred_band = get_confidence_prediction_bands(y_train, y_pred_train)

# 绘制测试集的拟合图像
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

plt.xlim(y_test.min() - 0.2, y_test.max())
plt.plot([y_test.min(), y_test.max()],
         [test_slope * y_test.min() + test_intercept, test_slope * y_test.max() + test_intercept],
         color='black', linestyle='dashdot', linewidth=1.5, alpha=0.6, label='Linear Fit')

plt.scatter(y_pred_test, y_test, color='#00FFFF', marker='o', alpha=0.9, label='Test Data Set', edgecolor='black', s=70, linewidth=0.5)

plt.fill_between([y_test.min(), y_test.max()],
                 [test_slope * y_test.min() + test_intercept - test_conf_band, test_slope * y_test.max() + test_intercept - test_conf_band],
                 [test_slope * y_test.min() + test_intercept + test_conf_band, test_slope * y_test.max() + test_intercept + test_conf_band],
                 color='#FFA500', alpha=0.5, label='95% Confidence Band')

plt.fill_between([y_test.min(), y_test.max()],
                 [test_slope * y_test.min() + test_intercept - test_pred_band, test_slope * y_test.max() + test_intercept - test_pred_band],
                 [test_slope * y_test.min() + test_intercept + test_pred_band, test_slope * y_test.max() + test_intercept + test_pred_band],
                 color='#90EE90', alpha=0.3, label='95% Prediction Band')

plt.xlabel('Predicted degradation rate', fontsize=32, fontweight='normal', color='black', fontname='Arial')
plt.ylabel('Actual degradation rate', fontsize=32, fontweight='normal', color='black', fontname='Arial')
plt.xticks(fontsize=30, fontweight='normal', fontname='Arial', color='black')
plt.yticks(fontsize=30, fontweight='normal', fontname='Arial', color='black')
plt.tick_params(axis='both', direction='in', length=3)
plt.legend(loc='upper left', frameon=False, prop={'weight': 'normal', 'size': 21, 'family': 'Arial'})
plt.text(0.60, 0.02, f'Fitting equation:\ny = {test_slope:.2f}x + {test_intercept:.2f}\nR$^2$: {test_r2:.2f}',
         fontsize=25, color='black', ha='left', va='bottom', fontname='Arial', fontweight='normal', transform=ax.transAxes)
plt.grid(False)
plt.tight_layout()
plt.savefig(r"D:\pycharm\shijie\photo\Cat Boost-test_fit-w.png", dpi=600)
# plt.show()

# 绘制训练集的拟合图像
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

plt.xlim(y_train.min() - 0.2, y_train.max())
plt.plot([y_train.min(), y_train.max()],
         [train_slope * y_train.min() + train_intercept, train_slope * y_train.max() + train_intercept],
         color='black', linestyle='dashdot', linewidth=1.5, alpha=0.6, label='Linear Fit')

plt.scatter(y_pred_train, y_train, color='#00BFFF', marker='o', alpha=0.9, label='Train Data Set', edgecolor='black', s=70, linewidth=0.5)

plt.fill_between([y_train.min(), y_train.max()],
                 [train_slope * y_train.min() + train_intercept - train_conf_band, train_slope * y_train.max() + train_intercept - train_conf_band],
                 [train_slope * y_train.min() + train_intercept + train_conf_band, train_slope * y_train.max() + train_intercept + train_conf_band],
                 color='#F08080', alpha=0.5, label='95% Confidence Band')

plt.fill_between([y_train.min(), y_train.max()],
                 [train_slope * y_train.min() + train_intercept - train_pred_band, train_slope * y_train.max() + train_intercept - train_pred_band],
                 [train_slope * y_train.min() + train_intercept + train_pred_band, train_slope * y_train.max() + train_pred_band],
                 color='#FFDAB9', alpha=0.3, label='95% Prediction Band')

plt.xlabel('Predicted degradation rate', fontsize=32, fontweight='normal', color='black', fontname='Arial')
plt.ylabel('Actual degradation rate', fontsize=32, fontweight='normal', color='black', fontname='Arial')
plt.xticks(fontsize=30, fontweight='normal', fontname='Arial', color='black')
plt.yticks(fontsize=30, fontweight='normal', fontname='Arial', color='black')
plt.tick_params(axis='both', direction='in', length=3)
plt.legend(loc='upper left', frameon=False, prop={'weight': 'normal', 'size': 22, 'family': 'Arial'})
plt.text(0.6, 0.02, f'Fitting equation:\ny = {train_slope:.2f}x + {train_intercept:.2f}\nR$^2$: {train_r2:.3f}',
         fontsize=25, color='black', ha='left', va='bottom', fontname='Arial', fontweight='normal', transform=ax.transAxes)
plt.grid(False)
plt.tight_layout()

plt.savefig(r"D:\pycharm\shijie\photo\Cat Boost-train_fit-w.png", dpi=600)
# plt.show()
