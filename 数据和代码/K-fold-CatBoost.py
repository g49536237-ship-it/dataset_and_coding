# =======================
# 0. 导入库
# =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool


# =======================
# 1. 读取并预处理数据
# =======================
data = pd.read_csv('3-2.csv', encoding='gbk')

# 处理百分号
for col in data.columns:
    if data[col].astype(str).str.contains('%').any():
        data[col] = data[col].str.replace('%', '').astype(float)

# 删除因变量为 0 的样本
data = data[data.iloc[:, -1] != 0]

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# CatBoost 分类特征索引（如无可为空）
cat_features = []


# =======================
# 2. 划分训练 / 测试集
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)


# =======================
# 3. CatBoost 参数搜索
# =======================
grid_params = {
    'iterations': [500, 1000],
    'learning_rate': [0.05, 0.1],
    'depth': [6, 8]
}

base_model = CatBoostRegressor(
    loss_function='RMSE',
    cat_features=cat_features,
    random_seed=42,
    verbose=0
)

grid_search_result = base_model.grid_search(
    grid_params,
    train_pool,
    cv=3,
    verbose=True
)

best_params = grid_search_result['params']

best_model = CatBoostRegressor(
    **best_params,
    loss_function='RMSE',
    cat_features=cat_features,
    random_seed=42,
    verbose=0
)

best_model.fit(train_pool, eval_set=test_pool, plot=True)


# =======================
# 4. 训练集 & 测试集评估
# =======================
y_train_pred = best_model.predict(train_pool)
y_test_pred = best_model.predict(test_pool)

print("训练集 MSE: %.4f" % mean_squared_error(y_train, y_train_pred))
print("训练集 R² : %.4f" % r2_score(y_train, y_train_pred))

print("测试集 MSE: %.4f" % mean_squared_error(y_test, y_test_pred))
print("测试集 R² : %.4f" % r2_score(y_test, y_test_pred))


# =======================
# 5. 10 折交叉验证
# =======================
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

cv_mse = -cross_val_score(
    best_model, X, y,
    cv=kfold,
    scoring='neg_mean_squared_error'
)

cv_rmse = np.sqrt(cv_mse)

cv_r2 = cross_val_score(
    best_model, X, y,
    cv=kfold,
    scoring='r2'
)

print("10折交叉验证 RMSE:", cv_rmse)
print("10折交叉验证 R²:", cv_r2)
print(f"RMSE 均值: {cv_rmse.mean():.4f}")
print(f"R² 均值 : {cv_r2.mean():.4f}")


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
plt.savefig(r"D:\pytharm\shijie\picture\K-fold-CatBoost-RMSE.png", dpi=600)
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
plt.savefig(r"D:\pytharm\shijie\picture\K-fold-CatBoost-R2.png", dpi=600)
plt.close()
