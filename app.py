import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn as nn

# Set Streamlit page title
st.set_page_config(page_title="Osteoporosis X-ray Detection", layout="centered")

# App title & instructions
st.title("🦴 Osteoporosis X-ray Detection")
st.write("Upload an X-ray image to predict osteoporosis severity.")

# Load the trained model
class OsteoporosisCNN(nn.Module):
    def __init__(self):
        super(OsteoporosisCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 3)  # 3 classes: Normal, Osteopenia, Osteoporosis
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OsteoporosisCNN().to(device)
model.load_state_dict(torch.load("osteoporosis_model.pth", map_location=device))
model.eval()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# File uploader
uploaded_file = st.file_uploader("📤 Upload an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_names = ["Normal", "Osteopenia", "Osteoporosis"]
        result = class_names[predicted.item()]

    # Display the result
    if result == "Osteoporosis":
        st.error(f"⚠️ {result} detected. Please consult a doctor.")
    elif result == "Osteopenia":
        st.warning(f"⚠️ {result} detected. Early signs of osteoporosis.")
    else:
        st.success(f"✅ {result}. Your bones are healthy!")