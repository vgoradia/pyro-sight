import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import cv2

st.set_page_config(page_title="PyroSight", page_icon="🔥", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .stAlert {
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stFileUploader {
        border-radius: 10px;
    }
    h3 {
        color: #ff6b35;
        border-bottom: 2px solid #ff6b35;
        padding-bottom: 0.3rem;
        margin-top: 2rem;
    }
    .stSpinner {
        color: #ff6b35;
    }
    .stInfo {
        background-color: #1e2130;
        border-left: 4px solid #ff6b35;
        border-radius: 8px;
        font-size: 1rem;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🔥 PyroSight")
st.write("CNN-powered wildfire detection system trained on satellite imagery. Upload an image for instant fire detection.")

@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id="vgoradia/PyroSight", filename="pyrosight_model.keras")
    model = tf.keras.models.load_model(model_path)
    _ = model(np.zeros((1, 150, 150, 3)))
    return model

model = load_model()

def generate_gradcam(model, img_array):
    img_tensor = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)
        loss = prediction[:, 0]
    
    grads = tape.gradient(loss, img_tensor)
    grads = tf.abs(grads)
    saliency = tf.reduce_max(grads, axis=-1)[0].numpy()
    
    saliency = saliency / (saliency.max() + 1e-8)
    saliency = np.uint8(255 * saliency)
    heatmap_color = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    
    original = np.uint8(255 * img_array[0])
    superimposed = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    
    return superimposed

import anthropic

def generate_explanation(fire_prob, no_fire_prob):
    import os
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    if fire_prob > 50:
        status = f"wildfire detected with {fire_prob:.1f}% confidence"
    else:
        status = f"no wildfire detected, {no_fire_prob:.1f}% confidence it is clear"
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""You are an AI wildfire detection assistant. A CNN model analyzed a satellite image and returned: {status}.

Write a 3-sentence plain English explanation that:
1. States what was detected and the confidence level
2. Explains what this means in practical terms for emergency responders
3. Gives a brief recommendation

Keep it professional, clear, and concise."""
        }]
    )
    return message.content[0].text

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    fire_prob = float(prediction) * 100
    no_fire_prob = 100 - fire_prob

    st.markdown("### Detection Results")

    if fire_prob > 50:
        st.error(f"🔥 Wildfire Detected — {fire_prob:.1f}% confidence")
    else:
        st.success(f"✅ No Wildfire Detected — {no_fire_prob:.1f}% confidence")

    fig = go.Figure(go.Bar(
        x=["No Fire", "Wildfire"],
        y=[no_fire_prob, fire_prob],
        marker_color=["#28a745", "#dc3545"]
    ))
    fig.update_layout(title="Detection Confidence", yaxis_title="%", yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Grad-CAM Heatmap")
    st.write("Red and yellow areas show where the model detected fire signatures.")
    gradcam_img = generate_gradcam(model, img_array)
    gradcam_pil = Image.fromarray(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB))
    st.image(gradcam_pil, caption="AI Attention Heatmap", width=400)

    st.markdown("### AI Analysis")
    with st.spinner("Generating your analysis..."):
        explanation = generate_explanation(fire_prob, no_fire_prob)
    st.info(explanation)

    st.caption("PyroSight uses a CNN trained on 30,000+ satellite images getting 97.6 percent accuracy.")
    
