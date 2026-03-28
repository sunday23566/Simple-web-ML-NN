import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

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
    "E": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/ESRB_Everyone_2013.svg/200px-ESRB_Everyone_2013.svg.png",
    "ET": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/ESRB_Everyone_10%2B_2013.svg/200px-ESRB_Everyone_10%2B_2013.svg.png",
    "T": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/ESRB_Teen_2013.svg/200px-ESRB_Teen_2013.svg.png",
    "M": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/ESRB_Mature_17%2B_2013.svg/200px-ESRB_Mature_17%2B_2013.svg.png",
    "AO": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/ESRB_Adults_Only_18%2B_2013.svg/200px-ESRB_Adults_Only_18%2B_2013.svg.png"
}

# ==================== Tabs ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 อธิบาย Ensemble ML",
    "📖 อธิบาย Neural Network",
    "🔍 ทดสอบ Ensemble ML (Hit/Flop)",
    "🔍 ทดสอบ Neural Network (ESRB Rating)"
])

# ===================== TAB 1 & 2 (Explain) =====================
with tab1:
    st.header("Model 1: Ensemble Machine Learning")
    st.markdown("""
    **Dataset**: Video Game Sales  
    **Data Prep**: Impute missing + OneHotEncode + StandardScaler  
    **โมเดล**: VotingClassifier (RandomForest + XGBoost + LogisticRegression)  
    **Target**: Hit / Flop
    """)

with tab2:
    st.header("Model 2: Neural Network (TensorFlow)")
    st.markdown("""
    **Dataset**: Video Games Rating By ESRB  
    **Data Prep**: OneHotEncode console + 34 content descriptors  
    **โครงสร้าง**: Dense(128, ReLU, Dropout) → Dense(64, ReLU, Dropout) → Softmax  
    **Target**: ESRB Rating (E, ET, T, M, AO)
    """)

# ===================== TAB 3: Ensemble Test + Confidence =====================
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
            prob = ensemble.predict_proba(scaled)[0]          # <-- เพิ่มส่วนนี้
            hit_prob = prob[1] * 100
            flop_prob = prob[0] * 100

            if pred == 1:
                st.success(f"🎉 **HIT** – เกมนี้จะขายดีเกิน 1 ล้าน copies!")
            else:
                st.error(f"📉 **FLOP** – เกมนี้ยอดขายน่าจะต่ำกว่า 1 ล้าน copies")

            # แสดง Confidence
            st.progress(hit_prob / 100)
            st.caption(f"**ความมั่นใจ** Hit: **{hit_prob:.1f}%** | Flop: **{flop_prob:.1f}%**")

# ===================== TAB 4: NN Test + Confidence (ปรับให้สวยขึ้น) =====================
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

            # แสดงผล
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.image(esrb_images.get(rating, ""), width=220)
            with col_b:
                st.success(f"**{rating}**")
                st.progress(confidence / 100)
                st.caption(f"**ความมั่นใจ** {confidence:.1f}%")

                if rating == "E": st.write("เหมาะสำหรับทุกวัย")
                elif rating == "ET": st.write("เหมาะสำหรับอายุ 10 ปีขึ้นไป")
                elif rating == "T": st.write("เหมาะสำหรับอายุ 13 ปีขึ้นไป")
                elif rating == "M": st.write("เหมาะสำหรับอายุ 17 ปีขึ้นไป (Mature)")
                elif rating == "AO": st.write("สำหรับผู้ใหญ่เท่านั้น")

st.caption("Video Game Success Predictor | Project IS 2568 | Deployed on Streamlit Community Cloud")