import streamlit as st
import torch
from PIL import Image
import numpy as np
import io, json, os, timm, requests

st.set_page_config(page_title="Skin Lesion AI (Local)", page_icon="SL", layout="centered", menu_items={"Get help": None, "Report a bug": None, "About": None})
def load_css(path):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("assets/styles.css")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NAMES = {"akiec":"Actinic keratosis / Bowen's (in-situ SCC)","bcc":"Basal cell carcinoma","bkl":"Benign keratosis","df":"Dermatofibroma","mel":"Melanoma","nv":"Melanocytic nevus (mole)","vasc":"Vascular lesion (angioma)"}
DESC = {"akiec":"Precancerous or in-situ squamous changes","bcc":"Common skin cancer, usually slow-growing","bkl":"Benign keratosis like seborrheic keratosis","df":"Benign fibrous nodule","mel":"Serious skin cancer, needs urgent care","nv":"Common benign mole","vasc":"Blood-vessel related lesion"}

@st.cache_resource
def load_model_and_classes():
    try:
        base="models"
        with open(os.path.join(base,"classes.json")) as f:
            classes=json.load(f)
        model_name="convnext_tiny"
        if os.path.exists(os.path.join(base,"model_name.txt")):
            model_name=open(os.path.join(base,"model_name.txt")).read().strip()
        img_size=224
        if os.path.exists(os.path.join(base,"img_size.txt")):
            img_size=int(open(os.path.join(base,"img_size.txt")).read().strip())
        m=timm.create_model(model_name, pretrained=False, num_classes=len(classes))
        m.load_state_dict(torch.load(os.path.join(base,"lesion_triage_mobilenet_v2.pt"), map_location="cpu"))
        m.eval().to(DEVICE)
        return m, classes, model_name, img_size, ""
    except Exception as e:
        return None, None, None, 224, "Model or classes not found: "+str(e)

def load_triage_defaults():
    p = os.path.join("models","triage.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p, "r"))
            return float(d.get("urgent", 0.9)), float(d.get("soon", 0.6))
        except:
            pass
    return 0.9, 0.6

def load_ovr_thresholds():
    p = os.path.join("models","ovr_thresholds.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p, "r"))
            return {k: float(v["thr"] if isinstance(v, dict) else v) for k,v in d.items()}
        except:
            pass
    return {}

def make_transform(img_size):
    import torchvision as tv
    return tv.transforms.Compose([tv.transforms.Resize((img_size,img_size)), tv.transforms.ToTensor(), tv.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

def suspicious_score(classes, probs):
    idx={c:i for i,c in enumerate(classes)}
    s=0.0
    if "mel" in idx: s+=probs[idx["mel"]]
    if "bcc" in idx: s+=0.6*probs[idx["bcc"]]
    if "akiec" in idx: s+=0.6*probs[idx["akiec"]]
    return float(s)

def triage(score, th_urgent, th_soon):
    if score>=th_urgent:
        return "Urgent evaluation (book dermatologist ASAP)",["Avoid sun exposure","Do not self-treat","Keep clear photos for your visit"]
    elif score>=th_soon:
        return "Timely visit (within 1-2 weeks)",["Monitor size/asymmetry/color changes","Keep a change timeline","Prepare for an appointment"]
    else:
        return "Watch and follow-up",["Re-photograph in 2-4 weeks","Use sun protection","Seek care sooner if it changes quickly"]

def have_ollama():
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=0.5)
        return r.status_code == 200
    except Exception:
        return False

def ollama_generate(model, prompt, temperature=0.3, num_ctx=2048, timeout=120):
    url = "http://127.0.0.1:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_ctx": num_ctx}}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

_LLM_OBJ = None
def have_llama_cpp(model_path):
    if not os.path.exists(model_path):
        return False
    try:
        import llama_cpp  # noqa: F401
        return True
    except Exception:
        return False

def llama_cpp_generate(model_path, prompt, temperature=0.3, n_ctx=2048, n_gpu_layers=-1, max_tokens=256):
    global _LLM_OBJ
    if _LLM_OBJ is None:
        from llama_cpp import Llama
        _LLM_OBJ = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, seed=0, logits_all=False)
    out = _LLM_OBJ.create_completion(prompt=prompt, temperature=temperature, max_tokens=max_tokens, stop=["</s>"])
    return out["choices"][0]["text"].strip()

def build_explain_prompt(top_pairs, advice_level, uncertain):
    tops = ", ".join([f"{NAMES.get(c,c)} {p*100:.0f}%" for c,p in top_pairs[:3]])
    note = "uncertainty present" if uncertain else "confident"
    return ("You are a careful medical information assistant. Explain the classification probabilities of a dermoscopy image to a layperson in clear English. Do not give a diagnosis or treatment. Keep it under 120 words, use 3-5 bullets. Include: what the scores mean in plain language, why images can be uncertain, when to seek care, and one tip to take a better follow-up photo. "
            f"Top predictions: {tops}. Triage advice: {advice_level}. Overall confidence: {note}.")

st.markdown(
    """
    <div class="app-hero">
      <div class="pill">Local • Private</div>
      <div class="title">Skin Lesion AI</div>
      <div class="subtitle">On-device triage (non-diagnostic)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

model, CLASSES, MODEL_NAME, IMG_SIZE, load_msg = load_model_and_classes()
if load_msg:
    st.error(load_msg)
    st.info("Train your model first, which will create models/lesion_triage_mobilenet_v2.pt and metadata files.")

file = st.file_uploader("Upload a close, in-focus photo of a single lesion", type=["jpg","jpeg","png"])

u_def, s_def = load_triage_defaults()
if s_def >= u_def:
    s_def = max(0.0, u_def - 0.01)
th_urgent = float(u_def)
th_soon = float(s_def)

if file and model is not None and CLASSES is not None:
    with st.expander("Result", expanded=True):
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Preview", use_container_width=True)
        t = make_transform(IMG_SIZE)
        x = t(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        ovr_thr = load_ovr_thresholds()
        pairs = [(CLASSES[i], float(probs[i])) for i in range(len(CLASSES))]
        pairs.sort(key=lambda z: z[1], reverse=True)
        candidates = [(c, p) for c, p in pairs if ovr_thr.get(c, 0.0) <= p]
        pred_code = candidates[0][0] if candidates else pairs[0][0]
        pred_label = NAMES.get(pred_code, pred_code)
        s = suspicious_score(CLASSES, probs)
        advice_level, tips = triage(s, th_urgent, th_soon)
        badge_cls = "ok"
        if advice_level.startswith("Urgent"):
            badge_cls = "danger"
        elif advice_level.startswith("Timely"):
            badge_cls = "warn"
        st.markdown(f'<div class="badge {badge_cls}">{advice_level}</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        rows = []
        for code, p in pairs[:5]:
            name = NAMES.get(code, code)
            pct = f"{p*100:.0f}%"
            rows.append(f"<div class='result-row'><div class='name'>{name}</div><div class='meter'><div class='meter-fill' style='--p:{p:.4f}'></div></div><div class='pct'>{pct}</div></div>")
        st.markdown("".join(rows), unsafe_allow_html=True)
        p_nv = dict(pairs).get("nv", 0.0); p_bkl = dict(pairs).get("bkl", 0.0)
        if abs(p_nv - p_bkl) < 0.12 and max(p_nv, p_bkl) > 0.25:
            st.info("Benign look-alikes: could be Melanocytic nevus or Benign keratosis.")
        top1 = pairs[0][1]; margin = pairs[0][1] - (pairs[1][1] if len(pairs) > 1 else 0.0)
        uncertain_flag = (top1 < 0.45) or (margin < 0.10)
        prompt = build_explain_prompt(pairs, advice_level, uncertain_flag)
        text = None
        try:
            with st.spinner("Generating explanation..."):
                if have_ollama():
                    model_name = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:3b-instruct")
                    text = ollama_generate(model_name, prompt)
                else:
                    gguf_path = os.environ.get("LOCAL_LLM_GGUF", "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf")
                    if have_llama_cpp(gguf_path):
                        text = llama_cpp_generate(gguf_path, prompt)
        except Exception as e:
            st.error(f"LLM error: {e}")
        if text:
            st.markdown(text)
        else:
            st.info("Local LLM not found. Install Ollama and run `ollama pull qwen2.5:3b-instruct`, or place a GGUF at models/llm/ and set env LOCAL_LLM_GGUF.")

st.markdown("---")
st.caption("This tool is not a diagnosis and does not replace professional medical advice.")