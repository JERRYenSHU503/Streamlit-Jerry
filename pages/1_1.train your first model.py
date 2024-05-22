import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import streamlit as st
import matplotlib.pyplot as plt



# 加载MNIST手写数字数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
# 将图像从28x28的矩阵转换为784的向量（28*28=784），并进行归一化处理，使其值在0到1之间。
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255
# 将标签转换为One-Hot编码，用于多分类。
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 创建一个简单的神经网络模型
model = Sequential([
    Dense(128, activation="relu"),  # 第一层有128个节点的全连接层，使用ReLU激活函数
    Dense(10, activation="softmax") # 输出层有10个节点（数字0-9），使用softmax激活函数进行多分类
])

# 编译模型，指定优化器、损失函数和度量标准
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 自定义Streamlit回调，用于实时监控训练过程中的各项指标并在Streamlit中动态显示
class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(StreamlitCallback, self).__init__()
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        # 在每个epoch结束时执行，将损失和准确率记录下来
        self.loss.append(logs['loss'])
        self.accuracy.append(logs['accuracy'])
        self.val_loss.append(logs['val_loss'])
        self.val_accuracy.append(logs['val_accuracy'])
        
        # 在Streamlit中打印并显示损失和准确率的曲线图
        info_text = f"""
         **Epoch:** {epoch + 1}
        - **Loss:** {logs['loss']}
        - **Accuracy:** {logs['accuracy']}
        - **Val Loss:** {logs['val_loss']}
        - **Val Accuracy:** {logs['val_accuracy']}
        """

        st.info(info_text)
        #st.info(f"Epoch: {epoch + 1}, \n Loss: {logs['loss']}, \n Acc: {logs['accuracy']}, \n Val Loss: {logs['val_loss']}, \n Val Acc: {logs['val_accuracy']}")
        # 绘制损失和准确率的曲线图
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(self.loss, label='train_loss')
        ax[0].plot(self.val_loss, label='val_loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        
        ax[1].plot(self.accuracy, label='train_accuracy')
        ax[1].plot(self.val_accuracy, label='val_accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        
        st.pyplot(fig)

def print_summary(history):
    # 提取训练过程中的最终损失和准确率
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    st.write("""
    ## 训练总结
    模型训练已完成。以下是模型在训练过程中的表现总结。
    """)
    
    # 显示最终的损失和准确率
    st.write(f"- 最终训练损失: {final_loss:.4f}")
    st.write(f"- 最终训练准确率: {final_accuracy:.4f}")
    st.write(f"- 最终验证损失: {final_val_loss:.4f}")
    st.write(f"- 最终验证准确率: {final_val_accuracy:.4f}")

@st.cache_data
def intro_text():
    st.title('手写数字识别应用')
    st.write("""
    此应用使用深度学习模型来识别手写数字（0-9）。它基于著名的MNIST数据集，这是最常用于入门深度学习和计算机视觉的数据集之一。
    
    这个模型是一个简单的神经网络，包括输入层、一个隐藏层和输出层，能够从0到9的手写数字图像中识别数字。
    """)

    st.image('data/MNIST.png', caption='MNIST手写数据集', use_column_width=True)

    st.write("""
    ### 功能和特点
    - **数据集**: MNIST（Mixed National Institute of Standards and Technology database）是美国国家标准与技术研究院收集整理的大型手写数字数据集，包含60,000个训练图像和10,000个测试图像，每个图像都是28x28像素的灰度图。
    - **模型结构**: 一个输入层，一个具有ReLU激活的隐藏层，以及一个具有Softmax激活的输出层，用于进行10分类（数字0-9）。
    - **训练过程**: 通过该应用，您可以启动模型的训练过程，并实时查看损失和准确率的进展。
    
    如果您现在不知道这些概念也完全没关系，使用这个应用，即使没有深度学习或编程的复杂背景，您也可以轻松地了解和参与到构建和训练一个机器学习模型的过程中。
    """)
md_text="""
- **Epoch**: 这是指模型完整遍历整个训练数据集一次的过程。一个epoch包含多个批次(batch)的数据处理。在深度学习中，通常会设定模型训练多个epochs以优化模型参数。
- **Loss**: 损失(Loss)是衡量模型预测结果与真实标签之间差距的一个量化指标。在训练过程中，模型试图通过优化算法（如Adam）最小化这个损失值。
- **Accuracy**: 准确率(Accuracy)用于衡量模型分类正确的样本占总样本的比例。
- **Val Loss**: 验证损失(Validation Loss)，这是模型在验证集上计算得到的损失值。验证集是独立于训练集的一部分数据，用来评估模型的泛化能力，
- **Val Acc**: 验证准确率(Validation Accuracy)，表示模型在验证集上的分类准确率，是评估模型在未直接参与训练的数据上的表现如何的一个重要指标。
"""

# 在Streamlit中运行
def run_app():
    if st.button('开始训练模型'):
        callback = StreamlitCallback()
        history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[callback])
        st.markdown(md_text)
        print_summary(history)  # 调用打印总结信息的函数
    
if __name__ == '__main__':
    intro_text()
    run_app()