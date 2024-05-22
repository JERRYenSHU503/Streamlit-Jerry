import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件并创建DataFrame
df = pd.read_csv("survey_results_public.csv")

# 展示DataFrame的前几行数据
df.head()

# 选择感兴趣的列
df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]

# 重命名列为更容易理解的名称
df = df.rename({"ConvertedComp": "Salary"}, axis=1)

# 删除Salary列中包含缺失值的行
df = df[df["Salary"].notnull()]

# 显示DataFrame的信息，包括列的数据类型和缺失值数量
df.info()

# 删除所有包含缺失值的行
df = df.dropna()

# 仅保留“Employment”列为“Employed full-time”的行，并删除该列
df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)

# 进一步清理数据：删除其他不需要的行，并显示每个国家的数量
df['Country'].value_counts()

# 定义一个函数来将国家类别缩短为'Other'或原始类别，取决于类别数量是否超过给定的阈值
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

# 调用shorten_categories函数并更新Country列
country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)

# 展示更新后的Country列的数量分布
df.Country.value_counts()

# 创建箱线图展示Salary和Country之间的关系
fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

# 进一步清理数据：删除Salary不在指定范围内的行
df = df[df["Salary"] <= 250000]
df = df[df["Salary"] >= 10000]
df = df[df['Country'] != 'Other']

# 创建更新后的箱线图展示Salary和Country之间的关系
fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

# 清洗YearsCodePro列的数据，将字符串转换为数字
def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

# 显示EdLevel列的唯一值
df["EdLevel"].unique()

# 清洗YearsCodePro列的数据，将字符串转换为数字
def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

# 应用clean_experience函数将YearsCodePro列中的字符串数据转换为数字
df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

# 显示EdLevel列的唯一值
df["EdLevel"].unique()

# 清洗EdLevel列的数据，将不同的学历等级归类为统一的类别
def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

# 应用clean_education函数将EdLevel列中的学历数据进行清洗和分类
df['EdLevel'] = df['EdLevel'].apply(clean_education)

# 使用LabelEncoder将EdLevel列中的类别转换为数值
from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])

# 使用LabelEncoder将Country列中的国家名称转换为数值
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])

# 准备训练数据和目标数据
X = df.drop("Salary", axis=1)
y = df["Salary"]

# 使用线性回归模型进行训练
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y.values)

# 使用训练好的线性回归模型进行预测
y_pred = linear_reg.predict(X)

# 计算预测结果与真实结果之间的均方根误差
from sklearn.metrics import mean_squared_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))

# 输出均方根误差
print("${:,.02f}".format(error))

# 使用决策树回归模型进行训练
from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, y.values)

# 使用训练好的决策树回归模型进行预测
y_pred = dec_tree_reg.predict(X)

# 计算预测结果与真实结果之间的均方根误差
error = np.sqrt(mean_squared_error(y, y_pred))

# 输出决策树回归模型的均方根误差
print("${:,.02f}".format(error))
# 导入随机森林回归器
from sklearn.ensemble import RandomForestRegressor

# 创建随机森林回归器对象并进行训练
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X, y.values)

# 使用训练好的随机森林回归器进行预测
y_pred = random_forest_reg.predict(X)

# 计算预测结果与真实结果之间的均方根误差
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

# 导入网格搜索模块
from sklearn.model_selection import GridSearchCV

# 定义决策树回归器的最大深度参数范围
max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {"max_depth": max_depth}

# 创建决策树回归器对象
regressor = DecisionTreeRegressor(random_state=0)

# 使用网格搜索进行超参数调优
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)

# 获取最优的决策树回归器
regressor = gs.best_estimator_

# 使用最优的决策树回归器进行训练
regressor.fit(X, y.values)

# 使用最优的决策树回归器进行预测
y_pred = regressor.predict(X)

# 计算最优模型预测结果与真实结果之间的均方根误差
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

# 准备新数据进行预测
X = np.array([["United States", 'Master’s degree', 15]])

# 对新数据中的国家和学历进行编码转换
X[:, 0] = le_country.transform(X[:, 0])
X[:, 1] = le_education.transform(X[:, 1])
X = X.astype(float)

# 使用加载的最优决策树回归器进行预测
y_pred = regressor.predict(X)
y_pred

# 将模型和编码器保存到文件中
import pickle

data = {"model": regressor, "le_country": le_country, "le_education": le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

# 从文件中加载模型和编码器
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# 使用加载的最优决策树回归器进行新数据的预测
y_pred = regressor_loaded.predict(X)
y_pred
