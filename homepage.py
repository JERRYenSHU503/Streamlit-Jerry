#import streamlit as st

#st.set_page_config(
#    page_title="Learn AI more easily",
#    page_icon="👋",
#)

#st.title("An interactive and visual Machine Learning book")
#st.sidebar.success("Select a page above.")
import importlib
import streamlit as st

st.title("欢迎来到我们的教学平台！")

markdown_text = """


在这里，我们不会深入探讨机器学习的每个技术细节，也不会涵盖所有机器学习的章节。然而，我们精心准备的可视化内容将帮助您生动地理解数据处理与模型训练等过程。通过交互式的展示，您将更轻松地掌握复杂概念，并能够在实践中加深理解。

除此之外，我们还将向您展示机器学习在各领域的应用，让您了解这一强大技术在现实世界中的广泛应用场景。无论您是初学者还是经验丰富的专业人士，我们都相信这里的内容能够为您带来新的启发和认识。

赶快开始您的机器学习之旅吧！探索教学平台，探索应用，发现更多可能性！
### 课程大纲
- 1 训练第一个模型
- 2 数据处理与可视化
- 3.1 机器学习分类器简介
- 3.2 机器学习分类器演示
- 4.1 薪资数据集展示
- 4.2 薪资预测
- 5.1 图像识别
- 5.2 训练自己的模型
- 6.1 卷积神经网络入门与演示
- 6.2 卷积神经网络应用：数字识别
"""
# 用streamlit展示markdown文本
st.markdown(markdown_text)

