import os, json, argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import torchvision as tv
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm import tqdm
import timm
from timm.data import create_transform
from timm.utils import ModelEmaV2

class CsvImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        p = self.df.iloc[idx, 0]
        y = int(self.df.iloc[idx, 1])
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y

def build_loaders(csv_dir, batch, workers, img_size):
    with open(os.path.join(csv_dir, "classes.json")) as f:
        classes = json.load(f)
    train_t = create_transform(input_size=img_size, is_training=True, auto_augment="rand-m9-mstd0.5-inc1", interpolation="bicubic")
    val_t = create_transform(input_size=img_size, is_training=False, interpolation="bicubic")
    tr_ds = CsvImageDataset(os.path.join(csv_dir,"train.csv"), transform=train_t)
    va_ds = CsvImageDataset(os.path.join(csv_dir,"val.csv"), transform=val_t)
    te_ds = CsvImageDataset(os.path.join(csv_dir,"test.csv"), transform=val_t)
    labels = tr_ds.df["label"].values
    class_count = np.bincount(labels, minlength=len(classes))
    class_weight = 1.0 / np.maximum(class_count, 1)
    samples_weight = class_weight[labels]
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    dl_tr = dict(batch_size=batch, sampler=sampler, num_workers=workers, pin_memory=True)
    dl_ev = dict(batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    if workers>0:
        dl_tr.update(dict(persistent_workers=True, prefetch_factor=4))
        dl_ev.update(dict(persistent_workers=True, prefetch_factor=4))
    tr = DataLoader(tr_ds, **dl_tr)
    va = DataLoader(va_ds, **dl_ev)
    te = DataLoader(te_ds, **dl_ev)
    return tr, va, te, classes, torch.tensor(class_weight, dtype=torch.float32)

def one_hot(y, num):
    return F.one_hot(y.view(-1).long(), num_classes=num).float()

def soft_ce(logits, targets):
    logp = F.log_softmax(logits, dim=1)
    return -(targets * logp).sum(dim=1).mean()

def do_mixup(x, y_oh, alpha):
    if alpha <= 0: return x, y_oh
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[idx]
    y = lam * y_oh + (1 - lam) * y_oh[idx]
    return x, y

def do_cutmix(x, y_oh, alpha):
    if alpha <= 0: return x, y_oh
    lam = np.random.beta(alpha, alpha)
    b, c, h, w = x.size()
    cx = np.random.randint(w); cy = np.random.randint(h)
    rw = int(w * np.sqrt(1 - lam)); rh = int(h * np.sqrt(1 - lam))
    x1 = np.clip(cx - rw // 2, 0, w); y1 = np.clip(cy - rh // 2, 0, h)
    x2 = np.clip(cx + rw // 2, 0, w); y2 = np.clip(cy + rh // 2, 0, h)
    idx = torch.randperm(b, device=x.device)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    y = lam * y_oh + (1 - lam) * y_oh[idx]
    return x, y

def evaluate(model, loader, device, tta=False):
    model.eval()
    ys = []; ps = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            o1 = model(x)
            out = o1
            if tta:
                o2 = model(torch.flip(x, dims=[3]))
                out = (o1 + o2) * 0.5
            p = out.argmax(1)
            ys.append(y.cpu().numpy())
            ps.append(p.cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    acc = (y_true==y_pred).mean() if len(y_true)>0 else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") if len(y_true)>0 else 0.0
    bacc = balanced_accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0
    return acc, macro_f1, bacc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out", default="models/lesion_triage_mobilenet_v2.pt")
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=0.2)
    ap.add_argument("--smoothing", type=float, default=0.1)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    tr, va, te, classes, class_weight = build_loaders(args.csv_dir, args.batch, args.workers, args.img_size)
    model = timm.create_model(args.model, pretrained=True, num_classes=len(classes), drop_path_rate=0.1)
    model = model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = ModelEmaV2(model, decay=0.999)
    scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))
    crit_ce = torch.nn.CrossEntropyLoss(weight=class_weight.to(device), label_smoothing=args.smoothing)
    best_f1 = -1.0
    warm = max(3, args.epochs//20)
    for epoch in range(1, args.epochs+1):
        model.train()
        if epoch <= warm:
            for g in opt.param_groups:
                g["lr"] = args.lr * epoch / warm
        pbar = tqdm(tr, desc=f"epoch {epoch}/{args.epochs}")
        for x,y in pbar:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True).view(-1).long()
            use_mix = (args.mixup>0) or (args.cutmix>0)
            if use_mix:
                y_oh = one_hot(y, len(classes))
                if np.random.rand() < 0.5:
                    x, y_soft = do_mixup(x, y_oh, args.mixup)
                else:
                    x, y_soft = do_cutmix(x, y_oh, args.cutmix)
                with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                    out = model(x)
                    loss = soft_ce(out, y_soft)
            else:
                with torch.amp.autocast('cuda', enabled=(device=="cuda")):
                    out = model(x)
                    loss = crit_ce(out, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            pbar.set_postfix(loss=float(loss.item()))
        sch.step()
        va_acc, va_f1, va_bacc = evaluate(ema.module, va, device, tta=True)
        if va_f1 > best_f1:
            best_f1 = va_f1
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(ema.module.state_dict(), args.out)
            with open(os.path.join(os.path.dirname(args.out), "model_name.txt"), "w") as f:
                f.write(args.model.strip())
            with open(os.path.join(os.path.dirname(args.out), "img_size.txt"), "w") as f:
                f.write(str(args.img_size))
            with open(os.path.join(os.path.dirname(args.out), "classes.json"), "w") as f:
                json.dump(classes, f)
    te_acc, te_f1, te_bacc = evaluate(ema.module, te, device, tta=True)
    print({"val_best_macro_f1": float(best_f1), "test_acc": float(te_acc), "test_macro_f1": float(te_f1), "test_bal_acc": float(te_bacc)})

if __name__ == "__main__":
    main()
