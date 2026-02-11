import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

st.set_page_config(page_title="Medical Image Classifier")

st.title("Pretrained Image Classifier (ResNet-18)")

# -----------------------
# Load pretrained model
# -----------------------
@st.cache_resource
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model, weights

model, weights = load_model()

# -----------------------
# Image preprocessing
# -----------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.meta["mean"],
        std=weights.meta["std"]
    ),
])

# -----------------------
# Upload image
# -----------------------
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image")

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top5_prob, top5_catid = torch.topk(probs, 5)

    st.subheader("Top Predictions")

    for i in range(5):
        label = weights.meta["categories"][top5_catid[i]]
        confidence = top5_prob[i].item() * 100
        st.write(f"{label}: {confidence:.2f}%")
