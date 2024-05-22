import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(layout="wide")
# 页面标题
st.title("CNN可视化展示")

st.markdown("[源网站](https://poloclub.github.io/cnn-explainer/)")
md_text="""
卷积神经网络(Convolutional Neural Network，CNN)是一种用于识别数据模式的神经网络，特别擅长解决分类问题，比如图像分类。它是神经网络的一种，神经网络是由多个神经元组成的，分布在各个层中，每个神经元都有自己可学习的权重和偏差。让我们来分解一下卷积神经网络的基本构建模块。

- 张量(Tensor)可以被看作是一个n维矩阵。在上面的卷积神经网络中，张量是三维的，除了输出层。
- 神经元(Neuron)可以被看作是一个函数，接受多个输入并产生单个输出。在上图中，神经元的输出被表示为红色→蓝色的激活图。
- 层(Layer)简单地是具有相同操作的神经元集合，包括相同的超参数。
- 核权重(Kernel Weights)和偏差(Biases)虽然对每个神经元是唯一的，但在训练阶段进行调整，使分类器能够适应所提供的问题和数据集。在可视化中，它们以黄色→绿色的发散色标来表示。您可以通过点击神经元或悬停在卷积弹性解释视图中的核/偏差上来查看具体值。
- 卷积神经网络传达一个可微的分数函数，在输出层的可视化中表示为类别分数。

"""
st.markdown(md_text)
# 使用iframe嵌入外部网页
url = "https://static-1300131294.cos.ap-shanghai.myqcloud.com/html/cnn-vis-3/index.html"
width = 1200
height = 800

components.iframe(url, width=width, height=height)

