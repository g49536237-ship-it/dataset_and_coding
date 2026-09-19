import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

# 读取数据集
#data = pd.read_csv('selected_features_2_1.csv')
data = pd.read_csv('3-2.csv', encoding='gbk')

# 提取自变量和因变量
X = data.iloc[:, :-1]  # 自变量，选择除最后一列以外的所有列
y = data.iloc[:, -1]   # 因变量，选择最后一列
#print(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)  #random_state=43       CNNTesting: random_state=18

# # 定义KNN回归模型
knn_model = KNeighborsRegressor(n_neighbors=3, weights='uniform', algorithm='auto', metric='manhattan')
"""
# 定义参数网格
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5],           # 可尝试不同的邻居数
    'weights': ['uniform', 'distance'],        # 权重方式
    'metric': ['euclidean', 'manhattan']       # 距离度量方式
}

# 创建 KNN 回归模型
knn_model = KNeighborsRegressor()

# 使用 GridSearchCV 进行参数搜索
grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid,
    cv=5,                    # 5 折交叉验证
    scoring='r2',            # 以 R^2 作为评分标准
    n_jobs=-1                # 使用所有可用的 CPU 核心
)
"""

# 定义参数网格
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5],           # 可尝试不同的邻居数
    'weights': ['uniform', 'distance'],        # 权重方式
    'metric': ['euclidean', 'manhattan']       # 距离度量方式
}

# 使用网格搜索来寻找最佳参数
grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数：", grid_search.best_params_)

# 获取最佳模型
best_gbdt_model = grid_search.best_estimator_
# 拟合模型
grid_search.fit(X_train, y_train)
# 拟合模型
knn_model.fit(X_train, y_train)

# 预测测试集数据
y_pred_test = knn_model.predict(X_test)
# 预测训练集数据
y_pred_train = knn_model.predict(X_train)

# 将 y_pred 转换为 DataFrame
#df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})

# 计算均方误差（MSE）
test_mse = mean_squared_error(y_test, y_pred_test)
print("测试集均方误差 (MSE): %.4f" % test_mse)

# 计算决定系数（R^2）
test_r2 = r2_score(y_test, y_pred_test)
print("测试集决定系数 (R^2): %.4f" % test_r2)

# 计算训练集的均方误差（MSE）
train_mse = mean_squared_error(y_train, y_pred_train)
print("训练集均方误差 (MSE): %.4f" % train_mse)

# 计算训练集的决定系数（R^2）
train_r2 = r2_score(y_train, y_pred_train)
print("训练集决定系数 (R^2): %.4f" % train_r2)


# """
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>画图部分>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# 计算拟合直线的斜率和截距（测试集）
test_slope, test_intercept = np.polyfit(y_test, y_pred_test, 1)
# 计算拟合直线的斜率和截距（训练集）
train_slope, train_intercept = np.polyfit(y_train, y_pred_train, 1)

# 计算置信区间和预测区间
def get_confidence_prediction_bands(y_true, y_pred, confidence=0.95):
    n = len(y_true)
    se = np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 2))
    t_value = 1.96  # for 95% confidence interval

    pred_band = t_value * se * np.sqrt(1 + 1/n)
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

# 设置横坐标的长度（范围）
plt.xlim(y_test.min() - 0.2, y_test.max())

plt.plot([y_test.min(), y_test.max()],
         [test_slope*y_test.min()+test_intercept, test_slope*y_test.max()+test_intercept],
         color='black',
         linestyle='dashdot',
         linewidth=1.5,
         alpha=0.6,
         label='Linear Fit')

plt.scatter(y_pred_test, y_test, color='#00FFFF', marker='o', alpha=0.9, label='Test Data Set', edgecolor='black', s=70, linewidth=0.5)

# 绘制95%置信区间
plt.fill_between([y_test.min(), y_test.max()],
                 [test_slope*y_test.min()+test_intercept-test_conf_band, test_slope*y_test.max()+test_intercept-test_conf_band],
                 [test_slope*y_test.min()+test_intercept+test_conf_band, test_slope*y_test.max()+test_intercept+test_conf_band],
                 color='#FFA500',
                 alpha=0.5,
                 label='95% Confidence Band')

# 绘制95%预测区间
plt.fill_between([y_test.min(), y_test.max()],
                 [test_slope*y_test.min()+test_intercept-test_pred_band, test_slope*y_test.max()+test_intercept-test_pred_band],
                 [test_slope*y_test.min()+test_intercept+test_pred_band, test_slope*y_test.max()+test_intercept+test_pred_band],
                 color='#90EE90',
                 alpha=0.3,
                 label='95% Prediction Band')

# plt.xlabel(r'Predicted $q_e$ (mg/g)', fontsize=20, fontweight='bold', color='black', fontname='Arial')
plt.xlabel('Predicted degradation rate', fontsize=32, fontweight='normal', color='black', fontname='Arial')  #设置横纵坐标标签
plt.ylabel('Actual degradation rate', fontsize=32, fontweight='normal', color='black', fontname='Arial')
plt.xticks(fontsize=30, fontweight='normal', fontname='Arial', color='black')   #设置横纵坐标数值标签     bold
plt.yticks(fontsize=30, fontweight='normal', fontname='Arial', color='black')
# 设置坐标轴短线的开口方向为向内
plt.tick_params(axis='both', direction='in', length=3)

# 调整图例的位置和字体
plt.legend(loc='upper left', frameon=False, prop={'weight': 'normal', 'size': 21, 'family': 'Arial'})

#添加文本
plt.text(0.60, 0.02, f'Fitting equation:\ny = {test_slope:.2f}x + {test_intercept:.2f}\nR$^2$: {test_r2:.2f}',
         fontsize=25, color='black', ha='left', va='bottom', fontname='Arial', fontweight='normal',transform=ax.transAxes)

# 去掉网格
plt.grid(False)

plt.tight_layout()

# 保存测试集拟合图像
plt.savefig(r"D:\pytharm\shijie\picture\KNN-test_fit.png", dpi=600)
# plt.savefig('C:/class/result/model-pre-result/123/CNN-test_fit.png', dpi=600)
# plt.show()

#####-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 绘制训练集的拟合图像
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

# 设置横坐标的长度（范围）
plt.xlim(y_train.min() - 0.2, y_train.max())

plt.plot([y_train.min(), y_train.max()],
         [train_slope*y_train.min()+train_intercept, train_slope*y_train.max()+train_intercept],
         color='black',
         linestyle='dashdot',
         linewidth=1.5,
         alpha=0.6,
         label='Linear Fit')

plt.scatter(y_pred_train, y_train, color='#00BFFF', marker='o', alpha=0.9, label='Train Data Set', edgecolor='black', s=70, linewidth=0.5)

# 绘制95%置信区间
plt.fill_between([y_train.min(), y_train.max()],
                 [train_slope*y_train.min()+train_intercept-train_conf_band, train_slope*y_train.max()+train_intercept-train_conf_band],
                 [train_slope*y_train.min()+train_intercept+train_conf_band, train_slope*y_train.max()+train_intercept+train_conf_band],
                 color='#F08080',
                 alpha=0.5,
                 label='95% Confidence Band')

# 绘制95%预测区间
plt.fill_between([y_train.min(), y_train.max()],
                 [train_slope*y_train.min()+train_intercept-train_pred_band, train_slope*y_train.max()+train_intercept-train_pred_band],
                 [train_slope*y_train.min()+train_intercept+train_pred_band, train_slope*y_train.max()+train_intercept+train_pred_band],
                 color='#FFDAB9',
                 alpha=0.3,
                 label='95% Prediction Band')

plt.xlabel('Predicted degradation rate', fontsize=32, fontweight='normal', color='black', fontname='Arial')  #设置横纵坐标标签
plt.ylabel('Actual degradation rate', fontsize=32, fontweight='normal', color='black', fontname='Arial')
#plt.title('Test Data: True vs. Predicted values', fontsize=22, fontweight='bold', color='black', fontname='Arial')
plt.xticks(fontsize=30, fontweight='normal', fontname='Arial', color='black')   #设置横纵坐标数值标签     bold
plt.yticks(fontsize=30, fontweight='normal', fontname='Arial', color='black')
# 设置坐标轴短线的开口方向为向内
plt.tick_params(axis='both', direction='in', length=3)

# 调整图例的位置和字体
plt.legend(loc='upper left', frameon=False, prop={'weight': 'normal', 'size': 22, 'family': 'Arial'})

# 添加文本
plt.text(0.6, 0.02, f'Fitting equation:\ny = {train_slope:.2f}x + {train_intercept:.2f}\nR$^2$: {train_r2:.2f}',
         fontsize=25, color='black', ha='left', va='bottom', fontname='Arial', fontweight='normal', transform=ax.transAxes)

plt.grid(False)
plt.tight_layout()

# 保存训练集拟合图像
plt.savefig(r"D:\pytharm\shijie\picture\KNN-train_fit.png", dpi=600)
#plt.savefig('C:/class/result/model-pre-result/123/CNN-train_fit.png', dpi=600)
# plt.show()

# """

