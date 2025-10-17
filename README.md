# 🌙 Luma — Mental Health Support Chatbot

Luma is an **empathetic, text-based mental health assistant** built with **PyTorch**, **Transformers**, and **Streamlit**.  
It provides short, supportive responses for emotional wellbeing and crisis-awareness — not medical advice.

---

## 🚀 Live Deployment
[Open the Chatbot](https://huggingface.co/spaces/l-isaro/Luma)

---

## ✨ Features

- 🧠 **FLAN-T5–based model** fine-tuned on mental health conversations.  
- 🗂 **Local PyTorch inference** — no Hugging Face Inference API needed.  
- 🕊️ **Empathetic tag classification**: the model generates one tag (e.g., `anxious`, `sad`, `stressed`) and returns a short supportive reply.  
- ⚠️ **Crisis phrase detection**: automatic safe fallback message for harmful or suicidal intent.  
- 🎨 **Streamlit UI** with warm styling, bubble chat, and sidebar tools (grounding & breathing exercises).  
- ⚙️ **Offline-friendly** — can run entirely on your machine or GPU instance.

---


---

## 🚀 Quick Start

### 1. Clone and install dependencies
```bash
git clone https://github.com/<your-username>/luma-chatbot.git
cd luma-chatbot
pip install -r requirements.txt
```
### 2. Add your model

You can either:

- **Use a local folder** (e.g., `model/`) containing:
  - `config.json`
  - `pytorch_model.bin`
  - `tokenizer.json`
  - and any other tokenizer files.

**or**

- **Pull from Hugging Face Hub:**
  ```bash
  export MODEL_ID="l-isaro/luma-chatbot"

## 🧠 How It Works

1. The model is a **T5-style seq2seq generator** fine-tuned to output a single tag describing the user’s emotional state.  
2. **Luma** maps that tag to a predefined supportive message (with breathing or grounding exercises).  
3. **Crisis-related phrases** (e.g., `"I want to die"`, `"end my life"`) trigger an **immediate safe fallback** — the model is bypassed.

## ⚠️ Safety Note

**Luma is not a medical or crisis service.**  
All generated responses are for **supportive conversation only**.  
If you are in immediate danger, **contact your local emergency services** or a **crisis hotline**.

## 📈 Training Summary

| **Experiment #** | **Model** | **Learning Rate** | **Epochs** | **BLEU** | **ROUGE-1** | **ROUGE-L** | **Validation Perplexity** |
|------------------:|------------|------------------:|-----------:|----------:|-------------:|-------------:|---------------------------:|
| **1** | t5-small | 5e-5 | 3 | 0.00146 | 0.12250 | 0.09986 | 24.17127 |
| **2** | flan-t5-base | 5e-5 | 8 | 0.03464 | 0.22356 | 0.18989 | 4.11419 |
| **3** | flan-t5-base (expanded dataset) | 5e-5 | 8 | 0.15204 | 0.22512 | 0.18905 | 2.34787 |


