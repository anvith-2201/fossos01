Osteoporosis Detection Using X-ray Images

📌 Project Overview

This project utilizes AI-powered deep learning to detect osteoporosis from X-ray images, offering a fast, cost-effective, and accessible alternative to traditional DXA scans. By leveraging a Convolutional Neural Network (CNN), the system classifies bone X-rays into Normal, Osteopenia, or Osteoporosis, enabling early diagnosis and intervention. The trained model is deployed in a Streamlit-based web application, allowing users to upload X-ray images and receive instant AI-based risk assessments.


📂 Dataset & Preprocessing
Dataset Categories:

1️⃣ Normal – Healthy bones.
2️⃣ Osteopenia – Early-stage bone loss, moderate risk.
3️⃣ Osteoporosis – Severe bone loss, high risk of fractures.

Preprocessing Steps:
✅ Resize images to 224x224 pixels.✅ Convert images to grayscale.✅ Normalize pixel values for better model efficiency.✅ Split data into Train (70%), Validation (15%), and Test (15%) sets.

🧠 AI Model - Convolutional Neural Network (CNN)
Model Architecture:
✅ 3 Convolutional Layers – Extracts bone structure features.✅ Max Pooling Layers – Reduces image dimensions while preserving key features.✅ Fully Connected Layers (FC) – Classifies images into three categories.✅ ReLU Activation & Dropout Layers – Prevents overfitting and enhances generalization.

Model Training & Performance:
Optimizer: Adam
Loss Function: CrossEntropyLoss
Training Accuracy: ~90%
Test Accuracy: ~85%

💻 Web Application (Streamlit)
How It Works:
1️⃣ User uploads an X-ray image.2️⃣ The AI model analyzes the image and classifies it into one of three categories.3️⃣ The result is displayed instantly with a color-coded risk indicator:
✅ Normal (Green Message)

⚠️ Osteopenia (Yellow Warning)

🚨 Osteoporosis (Red Alert, Doctor Consultation Recommended)

🚀 Installation & Running the Project
1️⃣ Install Dependencies
Ensure you have Python installed, then run:
pip install streamlit torch torchvision numpy opencv-python pillow

2️⃣ Run the Web App
streamlit run app.py

📈 Future Improvements
🔹 Use Pretrained Models (ResNet, EfficientNet) for better accuracy.🔹 Explainable AI (Grad-CAM) to highlight which part of the X-ray influenced the prediction.🔹 Cloud Deployment (AWS, Hugging Face, Google Cloud) for global accessibility.🔹 Mobile App Integration for portable X-ray analysis.

📌 Conclusion
This AI-powered osteoporosis detection system provides a fast, accurate, and cost-effective solution for early bone health screening. With future advancements, it has the potential to improve early diagnosis and patient care worldwide. 🚀

