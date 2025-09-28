import os, json, argparse, numpy as np, pandas as pd
import torch, torchvision as tv
from PIL import Image
import timm
from sklearn.metrics import recall_score

def load_model(base, num_classes):
    mname = open(os.path.join(base,"model_name.txt")).read().strip()
    m = timm.create_model(mname, pretrained=False, num_classes=num_classes)
    sd = torch.load(os.path.join(base,"lesion_triage_mobilenet_v2.pt"), map_location="cpu")
    m.load_state_dict(sd, strict=True)
    return m, mname

def make_tf(img_size):
    return tv.transforms.Compose([
        tv.transforms.Resize((img_size,img_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def suspicious_score(classes, probs):
    idx = {c:i for i,c in enumerate(classes)}
    s = np.zeros(probs.shape[0], dtype=np.float32)
    if "mel" in idx: s += probs[:, idx["mel"]]
    if "bcc" in idx: s += 0.6*probs[:, idx["bcc"]]
    if "akiec" in idx: s += 0.6*probs[:, idx["akiec"]]
    return s

def pick_threshold(y_true, scores, target_sens):
    order = np.argsort(-scores)
    y = y_true[order]
    s = scores[order]
    tp = np.cumsum(y==1)
    pos = (y==1).sum()
    sens = tp / np.maximum(1, pos)
    k = np.argmax(sens >= target_sens) if np.any(sens >= target_sens) else len(sens)-1
    thr = s[k] if len(s)>0 else 1.0
    return float(thr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/val.csv")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--sens_urgent", type=float, default=0.95)
    ap.add_argument("--sens_soon", type=float, default=0.85)
    ap.add_argument("--out", default="models/triage.json")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    with open(os.path.join(args.models_dir,"classes.json")) as f:
        classes = json.load(f)
    img_size = 224
    p = os.path.join(args.models_dir,"img_size.txt")
    if os.path.exists(p):
        try: img_size = int(open(p).read().strip())
        except: pass
    tf = make_tf(img_size)
    df = pd.read_csv(args.csv)
    items = [(r[0], int(r[1])) for r in df.values if os.path.exists(r[0])]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m,_ = load_model(args.models_dir, len(classes))
    m.eval().to(device)
    probs = []
    ys = []
    for i in range(0, len(items), args.batch):
        batch = items[i:i+args.batch]
        xs = []
        yb = []
        for pth,lab in batch:
            x = tf(Image.open(pth).convert("RGB"))
            xs.append(x); yb.append(lab)
        x = torch.stack(xs).to(device)
        with torch.no_grad():
            o = m(x)
            pr = torch.softmax(o, dim=1).cpu().numpy()
        probs.append(pr); ys.extend(yb)
    probs = np.concatenate(probs) if probs else np.zeros((0,len(classes)))
    ys = np.array(ys)
    s = suspicious_score(classes, probs)
    y_mel = (ys == classes.index("mel")).astype(np.int32) if "mel" in classes else np.zeros_like(ys)
    urgent_thr = pick_threshold(y_mel, s, args.sens_urgent) if y_mel.sum()>0 else 0.9
    y_hic = np.isin(ys, [classes.index(c) for c in ["mel","bcc","akiec"] if c in classes]).astype(np.int32)
    soon_thr = pick_threshold(y_hic, s, args.sens_soon) if y_hic.sum()>0 else 0.6
    urgent_thr = max(urgent_thr, soon_thr + 1e-3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"urgent": urgent_thr, "soon": soon_thr}, f)
    print({"urgent": urgent_thr, "soon": soon_thr, "n_val": int(len(ys))})
if __name__ == "__main__":
    main()
