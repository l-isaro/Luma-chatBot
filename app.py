import os, json, random
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Tuple

# --- transformers (TF backend, no torch needed) ---
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model

# ---------------------------
# Paths
# ---------------------------
MODEL_PATH = r"model/conversationalchatbotmodel.h5"     # Keras model
CSV_PATH = r"data/mental_health_training.csv"           # 4 cols: question,answer,pattern,tag
INTENTS_JSON_PATH = r"data/intents.json"               # For class order and replies
CONF_THRESHOLD = 0.45
EMBEDDER_NAME = "bert-base-uncased"  # Matches notebook

# ---------------------------
# Page UI
# ---------------------------
st.set_page_config(page_title="Luma — your light in moments of darkness", page_icon="✨", layout="centered")
st.markdown("""
<style>
:root { --luma-bg:#0b0f14; --luma-text:#e6edf3; --luma-muted:#a0aec0; }
html, body, [class*="css"] {
  background: radial-gradient(1200px 600px at 0% 0%, rgba(167,139,250,0.12), transparent 60%),
              radial-gradient(1200px 600px at 100% 0%, rgba(255,209,102,0.10), transparent 60%),
              var(--luma-bg) !important;
}
.block-container { padding-top: 2rem !important; }
.luma-hero { background: linear-gradient(135deg, rgba(167,139,250,0.14), rgba(255,209,102,0.10));
  border: 1px solid rgba(255,255,255,0.06); padding: 18px; border-radius: 18px; margin-bottom: 12px; }
.luma-title { font-size: 2rem; font-weight: 800; color: var(--luma-text); }
.luma-tag { color: var(--luma-muted); font-size: 1rem; margin-top: .25rem; }
.small { font-size: .92rem; color: var(--luma-muted); }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 12px 0 8px 0; }
.chat-bubble { padding: 12px 16px; border-radius: 18px; margin: 6px 0; max-width: 90%; }
.user-bubble { background:#E3F2FD; color:#000; margin-left:auto; }
.bot-bubble { background:#7d7979; color:#000; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Loaders
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_tf_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found at {path}")
        raise FileNotFoundError(f"Model not found at {path}")
    return tf_load_model(path, compile=False)

@st.cache_resource(show_spinner=True)
def load_bert(name: str):
    try:
        from huggingface_hub import snapshot_download, HfApi
        st.write("Checking network access to Hugging Face Hub...")
        api = HfApi()
        model_info = api.model_info(name)
        st.write(f"Model info: {model_info.modelId}")
        st.write("Pre-downloading BERT model weights...")
        snapshot_download(repo_id=name, cache_dir=".cache", allow_patterns=["*.bin", "*.json", "*.h5"])
        st.write("Loading BERT tokenizer...")
        tok = AutoTokenizer.from_pretrained(name, use_fast=True, cache_dir=".cache")
        st.write("Loading BERT model...")
        mdl = TFAutoModel.from_pretrained(name, cache_dir=".cache")
        return tok, mdl
    except Exception as e:
        st.error(f"Failed to load BERT: {str(e)}")
        st.error("Possible issues: Network failure, incompatible transformers/tensorflow versions, or missing dependencies.")
        st.error("Check logs for details and ensure Hugging Face Hub is accessible.")
        raise

# Cache loads
try:
    st.write("Loading Keras model...")
    pipeline = load_tf_model(MODEL_PATH)
    st.write("Loading BERT tokenizer and model...")
    tokenizer, bert = load_bert(EMBEDDER_NAME)
    st.write("Loading labels and replies...")
    LABELS = load_labels("data/labels.json", INTENTS_JSON_PATH, CSV_PATH)
    REPLIES = load_replies(CSV_PATH, INTENTS_JSON_PATH)
    st.write("Initialization complete!")
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

@st.cache_resource(show_spinner=True)
def load_labels(labels_path: str, intents_json_path: str, csv_path: str) -> List[str]:
    # 1) Prefer explicit labels.json (exact training order)
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
            return labels

    # 2) Fall back to intents.json (first tag per block, in order)
    if os.path.exists(intents_json_path):
        try:
            data = json.load(open(intents_json_path, "r", encoding="utf-8"))
            if isinstance(data, dict) and "intents" in data:
                labels = []
                for item in data["intents"]:
                    tag = item.get("tag") or (item.get("tags") or [None])[0]
                    if tag:
                        labels.append(str(tag))
                if labels:
                    return labels
        except Exception as e:
            st.error(f"Failed to load intents.json: {e}")

    # 3) Last resort: use CSV tags (alphabetical)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "tag" in df.columns:
                return sorted(set(df["tag"].astype(str)))
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

    # Default list (may not match training!)
    st.warning("Using default labels, which may not match training!")
    return [
        "greeting", "anxious", "sad", "stressed", "depressed", "worthless", "scared", "friends",
        "not-talking", "panic-attack", "breathing-exercise", "grounding", "self-care", "motivation",
        "hope", "therapy", "relapse", "anger", "guilt-shame", "overwhelm", "suicide"
    ]

@st.cache_resource(show_spinner=True)
def load_replies(csv_path: str, intents_json_path: str) -> Dict[str, List[str]]:
    # 1) CSV answers grouped by tag
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if {"answer", "tag"}.issubset(df.columns):
                m: Dict[str, List[str]] = {}
                for t, grp in df.groupby("tag"):
                    answers = [a for a in grp["answer"].astype(str) if a.strip()]
                    if answers:
                        m[str(t)] = sorted(set(answers))
                if any(m.values()):
                    return m
        except Exception as e:
            st.error(f"Failed to load CSV replies: {e}")

    # 2) intents.json responses
    if os.path.exists(intents_json_path):
        try:
            data = json.load(open(intents_json_path, "r", encoding="utf-8"))
            m: Dict[str, List[str]] = {}
            for item in data.get("intents", []):
                tag = item.get("tag") or (item.get("tags") or [None])[0]
                resps = item.get("responses", [])
                if tag and resps:
                    lst = [r.strip() for r in resps if isinstance(r, str) and r.strip()]
                    if lst:
                        m.setdefault(str(tag), []).extend(lst)
            for k in list(m.keys()):
                m[k] = sorted(set(m[k]))
            if any(m.values()):
                return m
        except Exception as e:
            st.error(f"Failed to load intents.json replies: {e}")

    # 3) Short fallbacks
    return {
        "greeting": ["Hello there. How are you feeling right now?"],
        "anxious": ["That sounds exhausting. Would you like to try a short breathing exercise together?"],
        "sad": ["I’m sorry you’re feeling this way. You’re not alone here. What’s been hardest this week?"],
        "stressed": ["That’s a lot to hold. Let’s sort things into ‘can control’ and ‘can’t control’. What’s one small step for today?"],
        "depressed": ["You deserve care. We can take this one small step at a time."],
        "worthless": ["That’s a painful place to be. What would you tell a friend who felt this way?"],
        "scared": ["Thank you for trusting me. Would it help to name a few things you can see right now?"],
        "friends": ["Feeling disconnected hurts. Want to plan one small step to reach out this week?"],
        "not-talking": ["I respect your pace. I’ll be here when you’re ready."],
        "panic-attack": ["You might be riding a panic spike. You’re safe with me right now."],
        "breathing-exercise": ["Let’s do 4-4-6 breathing for a minute."],
        "grounding": ["Let’s ground with the 5-4-3-2-1 method together."],
        "self-care": ["Tiny self-care plan: hydrate, step outside 2 minutes, send a kind message to yourself."],
        "motivation": ["Motivation often follows action. What’s a 2-minute version you can try now?"],
        "hope": ["Recovery is possible and rarely linear. You’ve already taken a brave step."],
        "therapy": ["Therapy can offer a safe space. Would you like tips for finding providers?"],
        "relapse": ["Relapses happen and don’t erase progress. What early signs did you notice?"],
        "anger": ["Anger can point to a boundary or a need. What happened right before it rose?"],
        "guilt-shame": ["Let’s separate ‘made a mistake’ from ‘am a mistake’. What would a kinder voice say?"],
        "overwhelm": ["Let’s do a brain dump, then pick just one ‘must’ for today."],
        "suicide": [
            "I’m really sorry you’re in this much pain. You deserve support and safety.\n"
            "**If you’re in immediate danger, contact local emergency services now.**\n"
            "If you can, reach out to someone you trust or a local crisis hotline. I’m here with you."
        ],
    }

# Cache loads
try:
    pipeline = load_tf_model(MODEL_PATH)
    tokenizer, bert = load_bert(EMBEDDER_NAME)
    LABELS = load_labels("data/labels.json", INTENTS_JSON_PATH, CSV_PATH)
    REPLIES = load_replies(CSV_PATH, INTENTS_JSON_PATH)
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# ---------------------------
# Helpers
# ---------------------------
def breathing_steps():
    return "**4-4-6** breathing: Inhale 4 • Hold 4 • Exhale 6 — repeat 5–6 rounds."

def grounding_steps():
    return "**5-4-3-2-1** grounding: 5 see • 4 touch • 3 hear • 2 smell • 1 taste."

def crisis_message():
    return ("I’m really sorry you’re in this much pain. You deserve support and safety.\n\n"
            "**If you’re in immediate danger, contact local emergency services now.**\n\n"
            "If you can, reach out to someone you trust or a local crisis hotline. I’m here with you.")

def text_to_bert768(texts: List[str]) -> np.ndarray:
    try:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="tf")
        outputs = bert(**enc)
        cls = outputs.last_hidden_state[:, 0, :]
        return cls.numpy()
    except Exception as e:
        st.error(f"Error generating BERT embedding: {e}")
        return np.zeros((len(texts), 768))

def predict_intent_with_conf(text: str) -> Tuple[str, float]:
    try:
        x = text_to_bert768([text])  # Shape (1, 768)
        probs = pipeline.predict(x, verbose=0)  # Keras softmax output
        if isinstance(probs, list):
            probs = probs[0]
        probs = np.asarray(probs)
        if probs.ndim == 1:
            probs = np.expand_dims(probs, 0)
        idx = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        tag = LABELS[idx] if 0 <= idx < len(LABELS) else "greeting"
        return tag, conf
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "greeting", 0.0

def sample_reply(tag: str) -> str:
    cand = REPLIES.get(tag, [])
    return random.choice(cand) if cand else "I’m here with you. Tell me more about what’s going on."

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("## ✨ Luma tools")
    if st.button("60-sec breathing"):
        st.session_state.setdefault("assistant_push", []).append(breathing_steps())
    if st.button("Try grounding"):
        st.session_state.setdefault("assistant_push", []).append(grounding_steps())
    st.markdown("---")
    st.caption(f"Model: `{MODEL_PATH}` • Threshold: {CONF_THRESHOLD:.2f}")
    if not os.path.exists("data/labels.json"):
        st.caption("Tip: add data/labels.json for exact class order mapping.")

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="luma-hero">
  <div class="luma-title">Luma</div>
  <div class="luma-tag">Helping you find your light again.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Chat state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I’m Luma. What’s on your mind today?"}
    ]

for msg in st.session_state.pop("assistant_push", []):
    st.session_state.messages.append({"role": "assistant", "content": msg})

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(f'<div class="chat-bubble {"user-bubble" if m["role"]=="user" else "bot-bubble"}">{m["content"]}</div>', unsafe_allow_html=True)

user_msg = st.chat_input(placeholder="Type a message…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble user-bubble">{user_msg}</div>', unsafe_allow_html=True)

    # Crisis guard
    if re := __import__("re"):
        if re.search(r"(kill myself|end my life|don.?t want to be here|i want to die|can.?t go on)", user_msg, re.I):
            reply = crisis_message()
            tag, conf = "suicide", 1.0
        else:
            tag, conf = predict_intent_with_conf(user_msg)
            if conf < CONF_THRESHOLD and tag != "suicide":
                reply = ("Thanks for sharing that with me. "
                         "Would you like to focus on **anxiety**, **sadness**, or **stress**, "
                         "or try a **breathing**/**grounding** exercise?")
            else:
                reply = sample_reply(tag)
                if tag in ("panic-attack", "anxious"):
                    reply += "\n\n" + breathing_steps() + "\n\n" + grounding_steps()
                elif tag == "grounding":
                    reply += "\n\n" + grounding_steps()
                elif tag == "breathing-exercise":
                    reply += "\n\n" + breathing_steps()

    with st.chat_message("assistant"):
        st.markdown(f'<div class="chat-bubble bot-bubble">{reply}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(f'<div class="small">© {datetime.now().year} Luma • Supportive conversation and wellness techniques — not medical advice.</div>', unsafe_allow_html=True)