# 🔥 PyroSight

A CNN-powered wildfire detection system trained on 30,000+ satellite images, achieving 97.6% test accuracy — built with Python, TensorFlow, OpenCV, Streamlit, and Claude API.

🔗 **Live Demo:** https://pyro-sight-gfptqzhfg6q7n96b7cqnaf.streamlit.app

🤗 **Model:** https://huggingface.co/vgoradia/PyroSight

---

## Overview

Wildfires are one of the most destructive climate disasters on the planet and early detection is everything. PyroSight uses a custom-trained CNN on real satellite imagery to detect wildfire signatures instantly, with visual attention heatmaps and AI-generated analysis to support faster decision-making.

---

## Features

- **97.6% Test Accuracy** — trained on 30,000+ labeled satellite images across wildfire and non-wildfire classes
- **Real-Time Detection** — upload any satellite image and get an instant classification with confidence score
- **Saliency Heatmap** — visual overlay showing which regions of the image triggered the detection
- **AI Analysis** — AI using Claude API generates a plain English explanation of the result with recommendations for emergency responders
- **Confidence Chart** — interactive bar chart showing fire vs no-fire probability

---

## Tech Stack

- Python
- TensorFlow & Keras (CNN training and inference)
- NumPy & Pillow (image processing)
- OpenCV (heatmap generation)
- Plotly (visualizations)
- Streamlit (web app)
- Anthropic Claude API (AI analysis)
- Hugging Face Hub (model hosting)

---

## Run Locally

```bash
git clone https://github.com/vgoradia/pyro-sight.git
cd pyro-sight
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

---

## Example Usage

Upload any satellite image from the test set or a real wildfire region and PyroSight will return a detection result, confidence score, attention heatmap, and AI-generated analysis in seconds.

---

## Project Structure
```
pyro-sight/
├── app.py              # Main Streamlit app
├── train.py            # CNN training script
├── requirements.txt    # Dependencies
└── README.md
```
---

Built by Veer Goradia
