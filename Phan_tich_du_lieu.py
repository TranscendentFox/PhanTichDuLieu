import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Lịch sử dự đoán
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Model", "Likes", "Dislikes", "Predicted Views"])

# Giao diện người dùng
st.title("Dự đoán lượt xem dựa trên lượt thích và không thích")

# Bên trái: Chọn mô hình, lịch sử, phương trình hồi quy
with st.sidebar:
    model_option = st.selectbox("Chọn mô hình:", ["Mô hình 1: Chỉ lượt thích", "Mô hình 2: Lượt thích và không thích"])

    # Nút hiển thị phương trình
    if st.button("Hiển thị phương trình hồi quy"):
        try:
            with open('model.pkl', 'rb') as f:
                model1, model2 = pickle.load(f)
            if model_option == "Mô hình 1: Chỉ lượt thích":
                st.write(f"Phương trình hồi quy: Views = {model1.coef_[0]:.2f} * Likes + {model1.intercept_:.2f}")
            else:
                st.write(
                    f"Phương trình hồi quy: Views = {model2.coef_[0]:.2f} * Likes + {model2.coef_[1]:.2f} * Dislikes + {model2.intercept_:.2f}")
        except Exception as e:
            st.error("Lỗi khi tải mô hình.")

    # Nút hiển thị lịch sử
    if st.button("Hiển thị lịch sử"):
        st.write("Lịch sử dự đoán:")
        st.dataframe(st.session_state.history)

# Phần bên phải: Nhập dữ liệu và dự đoán
col1, col2 = st.columns(2)

with col2:
    likes = st.number_input("Nhập lượt thích:", min_value=0)
    dislikes = st.number_input("Nhập lượt không thích:", min_value=0)

    # Tải mô hình hồi quy
    try:
        with open('model.pkl', 'rb') as f:
            model1, model2 = pickle.load(f)
    except FileNotFoundError:
        # Tạo mô hình hồi quy nếu bị lỗi
        data = pd.DataFrame({
            'Likes': [100, 200, 300, 400, 500],
            'Dislikes': [10, 20, 30, 40, 50],
            'Views': [1000, 2000, 3000, 4000, 5000]
        })

        # Mô hình 1: 1 biến độc lập
        model1 = LinearRegression()
        X1 = data[['Likes']]
        model1.fit(X1, data['Views'])

        # Mô hình 2: 2 biến độc lập
        model2 = LinearRegression()
        X2 = data[['Likes', 'Dislikes']]
        model2.fit(X2, data['Views'])

        # Lưu cả hai mô hình
        with open('model.pkl', 'wb') as f:
            pickle.dump((model1, model2), f)

    # Dự đoán lượt xem
    if st.button("Dự đoán"):
        check = True
        if model_option == "Mô hình 1: Chỉ lượt thích":
            if likes <= 0:  # Kiểm tra
                st.error("Vui lòng nhập lượt thích hợp lệ.")
                check = False
            else:
                predicted_views = model1.predict(np.array([[likes]]))
                st.success(f"Lượt xem dự đoán: {predicted_views[0]:.2f}")

        else:  # Mô hình 2
            if likes <= 0 or dislikes <= 0:  # Kiểm tra lượt thích và không thích
                st.error("Vui lòng nhập lượt thích và không thích hợp lệ.")
                check = False
            else:
                predicted_views = model2.predict(np.array([[likes, dislikes]]))
                st.success(f"Lượt xem dự đoán: {predicted_views[0]:.2f}")

        # Lưu lịch sử
        if check:
            new_entry = pd.DataFrame({
                "Model": [model_option],
                "Likes": [likes],
                "Dislikes": [dislikes if model_option == "Mô hình 2: Lượt thích và không thích" else 0],
                "Predicted Views": [predicted_views[0]]
            })
            st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)