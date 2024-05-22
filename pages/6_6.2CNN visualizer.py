import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(layout="wide")
# 页面标题
st.title("卷积神经网络的交互式节点链接可视化")

st.markdown("[源网站](https://adamharley.com/nn_vis/cnn/3d.html)")
md_text="""
本页面展示了针对手写数字识别训练的神经网络\n
你能看见网络在您输入下的实际行为，实时观察网络的激活模式如何响应。\n
### 在左上角的方框内写个数字试试吧
"""
st.markdown(md_text)
# 使用iframe嵌入外部网页
url = "https://static-1300131294.cos.ap-shanghai.myqcloud.com/html/cnn-vis/cnn.html"

width = 1200 # 可以根据目标用户的平均屏幕宽度调整
height = 800   # 同样，根据内容调整高度

components.iframe(url, width=width, height=height)
