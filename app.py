import os
if os.environ.get("HOME", "/") in ("/", "", None):
    os.environ["HOME"] = "/tmp"

import json, random, time, re
import streamlit as st
from datetime import datetime
from typing import Dict, List, Tuple

MODEL_ID = os.environ.get("MODEL_ID", "l-isaro/luma-chatbot").lower()# local dir OR hub repo
HF_TOKEN = os.environ.get("HF_TOKEN")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.45"))

LABEL_MAP: Dict[str, str] = {}
try:
    if os.environ.get("LABEL_MAP"):
        LABEL_MAP = json.loads(os.environ["LABEL_MAP"])
except Exception:
    LABEL_MAP = {}

# =========================
#          UI 
# =========================
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
.luma-tag { font-size: 1rem; color: var(--luma-muted); margin-top: .25rem; }
.small { font-size: .92rem; color: var(--luma-muted); }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 12px 0 8px 0; }
.chat-bubble { padding: 12px 16px; border-radius: 18px; margin: 6px 0; max-width: 90%; }
.user-bubble { background:#E3F2FD; color:#000; margin-left:auto; }
.bot-bubble { background:#7d7979; color:#000; }
</style>
""", unsafe_allow_html=True)

# =========================
# Replies 
# =========================
DEFAULT_REPLIES: Dict[str, List[str]] = {
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
FALLBACK_REPLY = "I’m here with you. Tell me more about what’s going on."
TAGS = sorted(list(set(DEFAULT_REPLIES.keys()) | {"greeting"}))

def breathing_steps():
    return "**4-4-6** breathing: Inhale 4 • Hold 4 • Exhale 6 — repeat 5–6 rounds."
def grounding_steps():
    return "**5-4-3-2-1** grounding: 5 see • 4 touch • 3 hear • 2 smell • 1 taste."
def crisis_message():
    return ("I’m really sorry you’re in this much pain. You deserve support and safety.\n\n"
            "**If you’re in immediate danger, contact local emergency services now.**\n\n"
            "If you can, reach out to someone you trust or a local crisis hotline. I’m here with you.")

def map_label(raw: str) -> str:
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    tag = raw.replace("LABEL_", "").replace("_", "-").lower().strip()
    return tag or "greeting"
def sample_reply(tag: str) -> str:
    choices = DEFAULT_REPLIES.get(tag, [])
    return random.choice(choices) if choices else FALLBACK_REPLY

# =========================
# Load model 
# =========================
@st.cache_resource(show_spinner=True)
def load_generator():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN or None)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN or None)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",      # GPU if available
        torch_dtype="auto"
    )
    return pipe

gen = load_generator()

# =========================
# Inference (seq2seq → single tag)
# =========================
CLASSIFY_PROMPT_PREFIX = (
    "You are a supportive mental health assistant. "
    "Given the user's message, output exactly ONE tag from this list: "
    f"{', '.join(TAGS)}. Respond with only the tag.\n\nUser: "
)

def _normalize_tag(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"^tag\s*:\s*", "", t)  # handle "Tag: anxious"
    t = t.split()[0].strip(":,.;")
    alias = {"panic": "panic-attack", "breathing": "breathing-exercise", "ground": "grounding"}
    return alias.get(t, t)

def predict_tag(user_text: str) -> Tuple[str, float]:
    prompt = f"{CLASSIFY_PROMPT_PREFIX}{user_text}\nTag:"
    out = gen(prompt, max_new_tokens=6, do_sample=False, num_beams=1)
    raw = out[0]["generated_text"]
    tag = _normalize_tag(raw)
    if tag in TAGS:
        return tag, 1.0
    for t in TAGS:
        if tag and t.startswith(tag[:max(1, len(tag))]):
            return t, 0.7
    return "greeting", 0.0

# =========================
# Sidebar (unchanged look)
# =========================
with st.sidebar:
    st.markdown("## ✨ Luma tools")
    if st.button("60-sec breathing"):
        st.session_state.setdefault("assistant_push", []).append(breathing_steps())
    if st.button("Try grounding"):
        st.session_state.setdefault("assistant_push", []).append(grounding_steps())
    st.markdown("---")
    access_label = "Private (token set)" if HF_TOKEN else "Public / Local"
    st.caption(f"Model: `{MODEL_ID}` • Threshold: {CONF_THRESHOLD:.2f} • Access: {access_label}")
    if LABEL_MAP:
        st.caption(f"Label map active: {LABEL_MAP}")

# =========================
# Header (unchanged look)
# =========================
st.markdown("""
<div class="luma-hero">
  <div class="luma-title">Luma</div>
  <div class="luma-tag">Helping you find your light again.</div>
</div>
""", unsafe_allow_html=True)

# =========================
# Chat state (unchanged flow)
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I’m Luma. What’s on your mind today?"}
    ]

for msg in st.session_state.pop("assistant_push", []):
    st.session_state.messages.append({"role": "assistant", "content": msg})

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        cls = "user-bubble" if m["role"] == "user" else "bot-bubble"
        st.markdown(f'<div class="chat-bubble {cls}">{m["content"]}</div>', unsafe_allow_html=True)

# =========================
# User input + response
# =========================
user_msg = st.chat_input(placeholder="Type a message…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble user-bubble">{user_msg}</div>', unsafe_allow_html=True)

    if re.search(r"(kill myself|end my life|don.?t want to be here|i want to die|can.?t go on)", user_msg, re.I):
        tag, conf = "suicide", 1.0
        reply = crisis_message()
    else:
        tag, conf = predict_tag(user_msg)
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
