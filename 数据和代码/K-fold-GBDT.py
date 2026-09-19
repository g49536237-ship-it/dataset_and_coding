# =======================
# 0. 导入所需库
# =======================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# =======================
# 1. 读取数据
# =======================
data = pd.read_csv('3-2.csv', encoding='gbk')

# 处理百分号
for col in data.columns:
    if data[col].astype(str).str.contains('%').any():
        data[col] = data[col].str.replace('%', '').astype(float)

# 自变量 & 因变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# =======================
# 2. 划分训练集和测试集
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=34
)


# =======================
# 3. GBDT 参数搜索
# =======================
param_grid = {
    'n_estimators': [50, 70, 80],
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [3, 4, 5]
}

gbdt = GradientBoostingRegressor(random_state=20)

grid_search = GridSearchCV(
    estimator=gbdt,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("最佳参数：", grid_search.best_params_)

best_gbdt_model = grid_search.best_estimator_


# =======================
# 4. 训练集 & 测试集评估
# =======================
y_train_pred = best_gbdt_model.predict(X_train)
y_test_pred = best_gbdt_model.predict(X_test)

print(f"训练集 MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"训练集 R² : {r2_score(y_train, y_train_pred):.4f}")

print(f"测试集 MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"测试集 R² : {r2_score(y_test, y_test_pred):.4f}")


# =======================
# 5. 10 折交叉验证
# =======================
kfold = KFold(n_splits=10, shuffle=True, random_state=43)

cv_mse = -cross_val_score(
    best_gbdt_model, X, y,
    cv=kfold,
    scoring='neg_mean_squared_error'
)

cv_rmse = np.sqrt(cv_mse)

cv_r2 = cross_val_score(
    best_gbdt_model, X, y,
    cv=kfold,
    scoring='r2'
)

print("10折交叉验证 RMSE:", cv_rmse)
print("10折交叉验证 R²:", cv_r2)
print(f"平均 RMSE: {cv_rmse.mean():.4f}")
print(f"平均 R² : {cv_r2.mean():.4f}")


# =======================
# 6. RMSE 单独绘图
# =======================
plt.figure(figsize=(8, 6))

colors_rmse = sns.color_palette("PuBuGn", n_colors=kfold.get_n_splits())
sns.barplot(
    x=np.arange(1, kfold.get_n_splits() + 1),
    y=cv_rmse,
    palette=colors_rmse
)

plt.axhline(y=cv_rmse.mean(), color='lightsteelblue', linestyle='--')

# plt.xlabel('Fold', fontname='Arial', fontsize=28)
plt.ylabel('RMSE', fontname='Arial', fontsize=32)

plt.xticks(fontsize=30, fontname='Arial')
plt.yticks(fontsize=30, fontname='Arial')

ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax + (ymax - ymin) * 0.15)

plt.text(
    0.02, 0.97,
    f'Mean RMSE: {cv_rmse.mean():.2f}',
    transform=plt.gca().transAxes,
    fontsize=26,
    fontname='Arial',
    color='lightsteelblue',
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig(r"D:\pytharm\shijie\picture\K-fold-GBDT-RMSE.png", dpi=600)
plt.close()


# =======================
# 7. R² 单独绘图
# =======================
plt.figure(figsize=(8, 6))

colors_r2 = sns.color_palette("YlGnBu", n_colors=kfold.get_n_splits())
sns.barplot(
    x=np.arange(1, kfold.get_n_splits() + 1),
    y=cv_r2,
    palette=colors_r2
)

plt.axhline(y=cv_r2.mean(), color='skyblue', linestyle='--')

# plt.xlabel('Fold', fontname='Arial', fontsize=28)
plt.ylabel('R²', fontname='Arial', fontsize=32)

plt.xticks(fontsize=30, fontname='Arial')
plt.yticks(fontsize=30, fontname='Arial')

ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax + (ymax - ymin) * 0.15)

plt.text(
    0.02, 0.97,
    f'Mean R²: {cv_r2.mean():.2f}',
    transform=plt.gca().transAxes,
    fontsize=26,
    fontname='Arial',
    color='skyblue',
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig(r"D:\pytharm\shijie\picture\K-fold-GBDT-R2.png", dpi=600)
plt.close()
