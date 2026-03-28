# Simple-web-ML-NN
Simple web ML&amp;NN in project intelligent

# Video Game Success Predictor - Project 2568

**ระบบทำนายความสำเร็จของวิดีโอเกม + ระดับเรตติ้ง ESRB**

### 1. Dataset ที่ใช้ (2 ชุด)
- **Dataset 1**: Video Game Sales (16,598 rows)  
  ที่มา: https://www.kaggle.com/datasets/gregorut/videogamesales  
  ความไม่สมบูรณ์: มี missing values ใน Year, Publisher, Genre  
  Target: Hit / Flop (Global_Sales > 1 ล้าน)

- **Dataset 2**: Video Games Rating By ESRB (1,895 rows)  
  ที่มา: https://www.kaggle.com/datasets/imohtn/video-games-rating-by-esrb  
  ความไม่สมบูรณ์: Categorical เยอะ (console + 34 content descriptors)  
  Target: ESRB Rating (E, ET, T, M, AO)

### 2. โมเดลที่พัฒนา
- **Model 1 (Machine Learning Ensemble)**  
  VotingClassifier (soft voting) ประกอบด้วย 3 โมเดล:  
  - RandomForestClassifier  
  - XGBClassifier  
  - LogisticRegression

- **Model 2 (Neural Network)**  
  TensorFlow Keras Sequential  
  - Input → Dense(128, ReLU, Dropout 0.3)  
  - Dense(64, ReLU, Dropout 0.3)  
  - Output: Dense(5, softmax)

### 3. Web Application
พัฒนาด้วย **Streamlit**  
มี 4 หน้าตามข้อกำหนด:
- หน้า 1: อธิบาย Ensemble ML (data prep + theory + steps + reference)
- หน้า 2: อธิบาย Neural Network (data prep + theory + steps + reference)
- หน้า 3: ทดสอบ Ensemble ML (Hit/Flop)
- หน้า 4: ทดสอบ Neural Network (ESRB Rating)

### 4. Deployment
- Deploy บน **Streamlit Community Cloud**  
- ลิงก์เว็บไซต์: https://simple-web-ml-nn-mz4vhghupu9vz9jdczji67.streamlit.app

### โครงสร้างโฟลเดอร์
- `datasets/` → cleaned
- `models/` → ensemble_model.pkl + nn_model.keras (ใช้ Git LFS)
- `app.py` + `requirements.txt`