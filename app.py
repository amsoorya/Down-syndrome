import streamlit as st
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

st.title("Fetal Down Syndrome Screening System")
st.caption("AI-assisted ultrasound analysis demo")


@st.cache_resource
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model, weights

model, weights = load_model()

# FIXED LINE
transform = weights.transforms()

uploaded = st.file_uploader("Upload image")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_cat = torch.topk(probs, 3)

    for i in range(3):
        label = weights.meta["categories"][top_cat[i]]
        st.write(f"{label}: {top_prob[i].item()*100:.2f}%")
