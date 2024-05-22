import streamlit as st
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

# 加载已经训练好的模型
model = tf.keras.applications.MobileNetV2(weights='imagenet')

st.title('TensorFlow图像识别')

st.markdown("""
在本章我们会展示TensorFlow的图像识别功能，您可以上传自己的图片，系统会使用**MobileNetV2**模型对图像进行分类和预测。\n
**MobileNetV2**是一个用于图像分类的预训练模型，它可以识别**ImageNet**数据集中的1000个类别。\n
**ImageNet**是一个大规模的图像数据集，其中包含超过1000万张图像，涵盖了各种各样的物体类别，例如动物、植物、交通工具、建筑等等。\n
[MobileNetV2官网](https://www.kaggle.com/models/google/mobilenet-v2)
""")
# 创建文件上传组件
uploaded_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'])

# 如果有上传的文件
if uploaded_file is not None:
    # 读取上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的图片', use_column_width=True)

    # 转换图片大小并预处理
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    # 使用模型进行预测
    prediction = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=3)[0]

    st.write('预测结果：')
    for (imagenet_id, label, score) in decoded_predictions:
        st.info(f"{label}: {score:.2f}%")

# 显示默认图片
else:
    default_image_paths = [
        "data/orange.jpeg",
        "data/dog.jpeg",
    ]
    st.write('图片示例：')
    for path in default_image_paths:
        image = Image.open(path)
        st.image(image, caption='', width=500)

        # 转换图片大小并预处理
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)

        # 使用模型进行预测
        prediction = model.predict(image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=3)[0]

        st.write('预测结果：')
        for (imagenet_id, label, score) in decoded_predictions:
            st.info(f"{label}: {score:.2f}%")