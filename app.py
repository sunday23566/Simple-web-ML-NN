import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

st.markdown("""
<style>
    .big-metric {
        font-size: 28px !important;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background-color: #00C853 !important;
    }
    .flop-bar > div > div > div {
        background-color: #FF5252 !important;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Video Game Success Predictor", layout="wide", page_icon="🎮")
st.title("🎮 Video Game Success Predictor")
st.markdown("**ระบบทำนายความสำเร็จของเกม + ระดับเรตติ้ง ESRB**")

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

# ESRB Images
esrb_images = {
    "E": "https://www.esrb.org/wp-content/uploads/2019/05/E.svg",
    "ET": "https://www.esrb.org/wp-content/uploads/2019/05/E10plus.svg",
    "T": "https://www.esrb.org/wp-content/uploads/2019/05/T.svg",
    "M": "https://www.esrb.org/wp-content/uploads/2019/05/T.svg",
    "AO": "https://www.esrb.org/wp-content/uploads/2019/05/AO.svg"
}

# ==================== Tabs ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 อธิบาย Ensemble ML",
    "📖 อธิบาย Neural Network",
    "🔍 ทดสอบ Ensemble ML (Hit/Flop)",
    "🔍 ทดสอบ Neural Network (ESRB Rating)"
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
                
    **Dataset**: Video Games Rating By ESRB  
    **Data Prep**: OneHotEncode console + 34 content descriptors  
    **โครงสร้าง**: Dense(128, ReLU, Dropout) → Dense(64, ReLU, Dropout) → Softmax  
    **Target**: ESRB Rating (E, ET, T, M, AO)

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

# ===================== TAB 3: Ensemble + สวยงามมากขึ้น =====================
with tab3:
    st.header("🔍 ทดสอบ Ensemble ML – ทำนาย Hit / Flop")
    with st.form("ensemble_form"):
        col1, col2 = st.columns(2)
        with col1:
            platform = st.selectbox("Platform", sales_encoder.categories_[0])
            genre = st.selectbox("Genre", sales_encoder.categories_[1])
            publisher = st.selectbox("Publisher", sales_encoder.categories_[2])
            year = st.number_input("Year", value=2025, step=1)
        with col2:
            na = st.number_input("NA Sales (ล้าน)", value=0.5, step=0.1)
            eu = st.number_input("EU Sales (ล้าน)", value=0.3, step=0.1)
            jp = st.number_input("JP Sales (ล้าน)", value=0.1, step=0.1)
            other = st.number_input("Other Sales (ล้าน)", value=0.2, step=0.1)

        if st.form_submit_button("🚀 Predict Hit / Flop", type="primary"):
            cat_input = pd.DataFrame([[platform, genre, publisher]], columns=['Platform', 'Genre', 'Publisher'])
            cat_encoded = sales_encoder.transform(cat_input)
            num_input = pd.DataFrame([[year, na, eu, jp, other]],
                                     columns=['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
            full_input = pd.concat([num_input.reset_index(drop=True),
                                    pd.DataFrame(cat_encoded, columns=sales_encoder.get_feature_names_out())], axis=1)
            scaled = sales_scaler.transform(full_input)

            pred = ensemble.predict(scaled)[0]
            prob = ensemble.predict_proba(scaled)[0]
            hit_prob = prob[1] * 100
            flop_prob = prob[0] * 100

            # แสดงผล
            if pred == 1:
                st.success("🎉 **HIT** – เกมนี้จะขายดีเกิน 1 ล้าน copies!")
            else:
                st.error("📉 **FLOP** – เกมนี้ยอดขายน่าจะต่ำกว่า 1 ล้าน copies")

            # UI ความมั่นใจใหม่ (สวยมากขึ้น)
            st.subheader("📊 ความมั่นใจของการทำนาย")
            col_hit, col_flop = st.columns(2)
            with col_hit:
                st.metric(label="🎯 HIT", value=f"{hit_prob:.1f}%", delta=None)
                st.progress(hit_prob / 100)
            with col_flop:
                st.metric(label="📉 FLOP", value=f"{flop_prob:.1f}%", delta=None)
                st.progress(flop_prob / 100, key="flop-bar")

# ===================== TAB 4: NN + ความมั่นใจสวยขึ้น =====================
with tab4:
    st.header("🔍 ทดสอบ Neural Network – ทำนาย ESRB Rating")
    st.info("เลือก console และเนื้อหาเกม → ระบบจะทำนายเรตติ้ง + แสดงโลโก้ + ความมั่นใจ")

    esrb_clean = pd.read_csv('datasets/cleaned/cleaned_esrb.csv')
    content_cols = [col for col in esrb_clean.columns if not col.startswith('console_')]

    with st.form("nn_form"):
        console = st.selectbox("Console", esrb_console_encoder.categories_[0])

        st.subheader("เนื้อหาเกม (ติ๊กทุกอย่างที่ตรง)")
        cols = st.columns(4)
        checkboxes = {}
        for i, col in enumerate(content_cols):
            with cols[i % 4]:
                checkboxes[col] = st.checkbox(col.replace('_', ' ').title(), value=False)

        submitted = st.form_submit_button("🚀 Predict ESRB Rating", type="primary")

        if submitted:
            console_df = pd.DataFrame([[console]], columns=['console'])
            console_encoded = esrb_console_encoder.transform(console_df)
            content_input = pd.DataFrame([[int(checkboxes[col]) for col in content_cols]], columns=content_cols)
            full_input = pd.concat([pd.DataFrame(console_encoded, columns=esrb_console_encoder.get_feature_names_out()),
                                    content_input], axis=1)

            pred_prob = nn_model.predict(full_input, verbose=0)[0]
            pred_class = np.argmax(pred_prob)
            rating = esrb_label_encoder.inverse_transform([pred_class])[0]
            confidence = float(pred_prob[pred_class]) * 100

            # แสดงผลสวย
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.image(esrb_images.get(rating, ""), width=220)
            with col_b:
                st.success(f"**{rating}**")
                st.metric(label="📊 ความมั่นใจ", value=f"{confidence:.1f}%", delta=None)
                st.progress(confidence / 100)

                if rating == "E": st.write("เหมาะสำหรับทุกวัย")
                elif rating == "ET": st.write("เหมาะสำหรับอายุ 10 ปีขึ้นไป")
                elif rating == "T": st.write("เหมาะสำหรับอายุ 13 ปีขึ้นไป")
                elif rating == "M": st.write("เหมาะสำหรับอายุ 17 ปีขึ้นไป (Mature)")
                elif rating == "AO": st.write("สำหรับผู้ใหญ่เท่านั้น")

st.caption("Video Game Success Predictor | Project IS 2568 | Deployed on Streamlit Community Cloud")