# 导入必要的库
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# intro.py
import streamlit as st
def app():
 st.header('简介')
 st.write('这是简介页面的内容...')

# Streamlit页面配置
st.title('鸟类数据分析')

# 应用介绍，用markdown格式书写
st.markdown("""
欢迎加入我们的鸟类数据可视化教程！本章我们将专注于展示如何使用Python对鸟类数据进行动态展示。
从基础的散点图到复杂的2D直方图，我们将引导你通过代码和示例，轻松掌握数据可视化的核心技巧。
无论你是数据分析新手还是希望深化可视化技能的研究者，这里都有适合你的内容。让我们开始探索吧！
""")

# 加载数据的代码和描述
st.header('加载数据')
code_load_data = """
birds = pd.read_csv('https://static-1300131294.cos.ap-shanghai.myqcloud.com/data/birds.csv')
"""
st.code(code_load_data, language='python')
st.write('从在线URL中加载鸟类数据CSV文件。以下是数据框架的前几行：')

# 加载数据
birds = pd.read_csv('https://static-1300131294.cos.ap-shanghai.myqcloud.com/data/birds.csv')

# 展示数据框架的前几行
st.dataframe(birds.head())

# 展示和代码散点图
st.header('散点图')
scatter_code = """
fig, ax = plt.subplots(figsize=(12, 8))
birds.plot(kind='scatter', x='MaxLength', y='Order', ax=ax)
plt.title('每个订单的最大长度')
plt.ylabel('订单')
plt.xlabel('最大长度')
st.pyplot(fig)
"""
st.code(scatter_code, language='python')
st.write('展示散点图以显示鸟类订单的基本长度分布情况。')

# 绘制散点图
fig, ax = plt.subplots(figsize=(12, 8))
birds.plot(kind='scatter', x='MaxLength', y='Order', ax=ax)
plt.title('每个订单的最大长度')
plt.ylabel('订单')
plt.xlabel('最大长度')
st.pyplot(fig)

# 展示和代码直方图
st.header('直方图')
histogram_code = """
fig, ax = plt.subplots(figsize=(12, 12))
birds['MaxBodyMass'].plot(kind='hist', bins=10, ax=ax)
plt.title('MaxBodyMass分布')
st.pyplot(fig)
"""
st.code(histogram_code, language='python')
st.write('显示直方图以评估整个数据集中MaxBodyMass的分布情况。')

# 创建直方图
fig, ax = plt.subplots(figsize=(12, 12))
birds['MaxBodyMass'].plot(kind='hist', bins=10, ax=ax)
plt.title('MaxBodyMass分布')
st.pyplot(fig)

# 将bins改为30
st.header('增加直方图Bins')
st.markdown("""
正如观察所见，400多种鸟类的最大体质量大多在2000以下。通过增加`bins`参数至30以获得更深入的洞察。
""")
code_hist_bins_30 = """
birds['MaxBodyMass'].plot(kind='hist', bins=30, figsize=(12, 12))
plt.show()
"""
st.code(code_hist_bins_30, language='python')

# 使用bins=30绘图
fig, ax = plt.subplots(figsize=(12, 12))
birds['MaxBodyMass'].plot(kind='hist', bins=30, ax=ax)
plt.title('MaxBodyMass分布 - 30个Bins')
st.pyplot(fig)

# 筛选数据并将bins增至40
st.header('筛选数据的直方图')
st.markdown("""
筛选数据以仅获取体质量低于60的鸟类，并展示使用40个`bins`的分布情况。
""")
code_filtered_40_bins = """
filteredBirds = birds[(birds['MaxBodyMass'] > 1) & (birds['MaxBodyMass'] < 60)]
filteredBirds['MaxBodyMass'].plot(kind='hist', bins=40, figsize=(12, 12))
plt.show()
"""
st.code(code_filtered_40_bins, language='python')

# 应用筛选并绘图
filteredBirds = birds[(birds['MaxBodyMass'] > 1) & (birds['MaxBodyMass'] < 60)]
fig, ax = plt.subplots(figsize=(12, 12))
filteredBirds['MaxBodyMass'].plot(kind='hist', bins=40, ax=ax)
plt.title('筛选后的MaxBodyMass - 40个Bins')
st.pyplot(fig)

# 2D 直方图
st.header('2D 直方图')
st.markdown("""
创建一个2D直方图来比较`MaxBodyMass`和`MaxLength`之间的关系。这种可视化使用更亮的颜色来展示高度聚集的点。
""")
code_hist2d = """
x = filteredBirds['MaxBodyMass']
y = filteredBirds['MaxLength']
fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(x, y)
plt.show()
"""
st.code(code_hist2d, language='python')

# 绘制2D直方图
x = filteredBirds['MaxBodyMass']
y = filteredBirds['MaxLength']
fig, ax = plt.subplots(tight_layout=True)
hist = ax.hist2d(x, y)
plt.title('MaxBodyMass与MaxLength的2D直方图')
st.pyplot(fig)