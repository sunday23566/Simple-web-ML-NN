import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(page_title="Video Game Success Predictor", layout="wide")
st.title("🎮 Video Game Success Predictor")
st.markdown("**ระบบทำนายความสำเร็จของวิดีโอเกม + ระดับเรตติ้ง ESRB**")

# ==================== Load Models & Encoders ====================
@st.cache_resource
def load_models():
    ensemble = joblib.load('models/ensemble_model.pkl')
    nn_model = tf.keras.models.load_model('models/nn_model.keras')
    sales_encoder = joblib.load('models/sales_encoder.pkl')
    sales_scaler = joblib.load('models/sales_scaler.pkl')
    esrb_console_encoder = joblib.load('models/esrb_console_encoder.pkl')
    esrb_label_encoder = joblib.load('models/esrb_label_encoder.pkl')
    return ensemble, nn_model, sales_encoder, sales_scaler, esrb_console_encoder, esrb_label_encoder

ensemble, nn_model, sales_encoder, sales_scaler, esrb_console_encoder, esrb_label_encoder = load_models()

# ==================== Tabs 4 หน้า ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 อธิบาย Ensemble ML",
    "📖 อธิบาย Neural Network",
    "🔍 ทดสอบ Ensemble ML",
    "🔍 ทดสอบ Neural Network"
])

# ===================== TAB 1: Explain Ensemble =====================
with tab1:
    st.header("Model 1: Ensemble Machine Learning (VotingClassifier)")
    st.markdown("""
    **แนวทางการพัฒนา**  
    ใช้ Dataset Video Game Sales (16,598 rows) ที่มี missing values ใน Year, Publisher, Genre  
    → Data Prep: impute median, fill “Unknown”, OneHotEncode (Platform, Genre, Publisher), StandardScaler  
    → Target: Hit (Global_Sales > 1 ล้าน) / Flop

    **ทฤษฎีอัลกอริทึม**  
    VotingClassifier (soft voting) รวม 3 โมเดลต่างประเภท:  
    - RandomForest (Bagging)  
    - XGBoost (Boosting)  
    - LogisticRegression (Linear)  

    **ขั้นตอนการพัฒนา**  
    1. โหลด + แสดง missing values  
    2. เตรียมข้อมูล + สร้าง Target  
    3. Train VotingClassifier  
    4. ประเมินผลด้วย Accuracy & F1-score  

    **แหล่งอ้างอิง**  
    - Kaggle: Video Game Sales Dataset  
    - scikit-learn VotingClassifier Documentation
    """)

# ===================== TAB 2: Explain Neural Network =====================
with tab2:
    st.header("Model 2: Neural Network (TensorFlow)")
    st.markdown("""
    **แนวทางการพัฒนา**  
    ใช้ Dataset Video Games Rating By ESRB (1,895 rows)  
    → Data Prep: OneHotEncode console + 34 binary content descriptors  

    **ทฤษฎีอัลกอริทึม**  
    Feedforward Neural Network  
    - Input → Dense(128, ReLU) + Dropout(0.3)  
    - Dense(64, ReLU) + Dropout(0.3)  
    - Output: Dense(5, softmax)  

    **ขั้นตอนการพัฒนา**  
    1. Encode categorical + binary features  
    2. Split train/test 80/20  
    3. Train 50 epochs + EarlyStopping  
    4. ใช้ sparse_categorical_crossentropy  

    **แหล่งอ้างอิง**  
    - Kaggle: Video Games Rating By ESRB  
    - TensorFlow Keras Documentation
    """)

# ===================== TAB 3: Test Ensemble =====================
with tab3:
    st.header("🔍 ทดสอบ Model 1: ทำนาย Hit / Flop")
    st.info("ใส่ข้อมูลเกมเพื่อทำนายว่าเกมจะเป็น **Hit** หรือ **Flop**")

    with st.form("ensemble_form"):
        col1, col2 = st.columns(2)
        with col1:
            platform = st.selectbox("Platform", sales_encoder.categories_[0])
            genre = st.selectbox("Genre", sales_encoder.categories_[1])
            publisher = st.selectbox("Publisher", sales_encoder.categories_[2])
            year = st.number_input("Year", value=2020, step=1)
        with col2:
            na_sales = st.number_input("NA Sales (ล้าน)", value=0.0, step=0.1)
            eu_sales = st.number_input("EU Sales (ล้าน)", value=0.0, step=0.1)
            jp_sales = st.number_input("JP Sales (ล้าน)", value=0.0, step=0.1)
            other_sales = st.number_input("Other Sales (ล้าน)", value=0.0, step=0.1)

        submitted = st.form_submit_button("🚀 Predict Hit / Flop")
        if submitted:
            # สร้าง input dataframe
            cat_input = pd.DataFrame([[platform, genre, publisher]], columns=['Platform', 'Genre', 'Publisher'])
            cat_encoded = sales_encoder.transform(cat_input)

            num_input = pd.DataFrame([[year, na_sales, eu_sales, jp_sales, other_sales]],
                                     columns=['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])

            full_input = pd.concat([num_input.reset_index(drop=True),
                                    pd.DataFrame(cat_encoded, columns=sales_encoder.get_feature_names_out())], axis=1)

            scaled_input = sales_scaler.transform(full_input)
            pred = ensemble.predict(scaled_input)[0]

            if pred == 1:
                st.success("🎉 ผลการทำนาย: **HIT** (ยอดขายเกิน 1 ล้าน)")
            else:
                st.error("📉 ผลการทำนาย: **FLOP** (ยอดขายต่ำกว่า 1 ล้าน)")

# ===================== TAB 4: Test Neural Network =====================
with tab4:
    st.header("🔍 ทดสอบ Model 2: ทำนาย ESRB Rating")
    st.info("เลือก console และเนื้อหาเกมเพื่อทำนายเรตติ้ง")

    # โหลด content columns
    esrb_clean = pd.read_csv('datasets/cleaned/cleaned_esrb.csv')
    content_cols = [col for col in esrb_clean.columns if not col.startswith('console_')]

    console_options = esrb_console_encoder.categories_[0]

    with st.form("nn_form"):
        console = st.selectbox("Console", console_options)

        st.subheader("เนื้อหาเกม (ติ๊กที่ตรงกับเกม)")
        cols = st.columns(4)
        checkboxes = {}
        for i, col in enumerate(content_cols):
            with cols[i % 4]:
                checkboxes[col] = st.checkbox(col.replace('_', ' ').title(), value=False)

        submitted = st.form_submit_button("🚀 Predict ESRB Rating")
        if submitted:
            # สร้าง input
            console_df = pd.DataFrame([[console]], columns=['console'])
            console_encoded = esrb_console_encoder.transform(console_df)

            content_input = pd.DataFrame([ [int(checkboxes[col]) for col in content_cols] ], columns=content_cols)

            full_input = pd.concat([pd.DataFrame(console_encoded, columns=esrb_console_encoder.get_feature_names_out()),
                                    content_input], axis=1)

            # Predict
            pred_prob = nn_model.predict(full_input, verbose=0)
            pred_class = np.argmax(pred_prob, axis=1)[0]
            rating = esrb_label_encoder.inverse_transform([pred_class])[0]

            st.success(f"📛 ผลการทำนาย: **{rating}**")

st.caption("Video Game Success Predictor | Project IS 2568")