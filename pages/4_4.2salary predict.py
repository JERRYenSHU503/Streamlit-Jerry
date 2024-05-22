import streamlit as st
import pickle
import numpy as np

@st.cache_data
def load_model():
    with open('data/saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("使用决策树回归预测薪水")

    st.write("""现在，我们可以利用得到的数据预测薪水""")
    st.write("""### 输入信息进行预测""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "学士学位以下",
        "学士学位",
        "硕士学位",
        "博士及以上",
    )
    
    def convert(chinese_education):
     if chinese_education == "学士学位以下":
        return "Less than a Bachelors"
     elif chinese_education == "学士学位":
        return "Bachelor’s degree"
     elif chinese_education == "硕士学位":
        return "Master’s degree"
     elif chinese_education == "博士及以上":
        return "Post grad"
     else:
        return None

    country = st.selectbox("国家", countries)
    education = st.selectbox("教育水平", education)
    education_en = convert(education)
    expericence = st.slider("工作经验年限", 0, 50, 3)

    X = np.array([[country, education_en, expericence ]])
    X[:, 0] = le_country.transform(X[:,0])
    X[:, 1] = le_education.transform(X[:,1])
    X = X.astype(float)

    salary = regressor.predict(X)
    st.subheader(f"预计薪水为 ${salary[0]:.2f}")

show_predict_page()
