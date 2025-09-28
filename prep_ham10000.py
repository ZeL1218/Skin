import os
import argparse
import pandas as pd
import json
from sklearn.model_selection import train_test_split
LABELS = ["akiec","bcc","bkl","df","mel","nv","vasc"]
MAP = {k:i for i,k in enumerate(LABELS)}
def find_path(base, img_id):
    name = img_id + ".jpg"
    p1 = os.path.join(base, "HAM10000_images_part_1", name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(base, "HAM10000_images_part_2", name)
    if os.path.exists(p2): return p2
    p3 = os.path.join(base, "images", name)
    if os.path.exists(p3): return p3
    return None
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    args = ap.parse_args()
    meta_path = os.path.join(args.data_dir, "HAM10000_metadata.csv")
    df = pd.read_csv(meta_path)
    df = df[df["dx"].isin(LABELS)].copy()
    df["path"] = df["image_id"].apply(lambda x: find_path(args.data_dir, x))
    df = df.dropna(subset=["path"])
    df["label"] = df["dx"].map(MAP)
    train_df, tmp = train_test_split(df, test_size=args.val_ratio+args.test_ratio, random_state=args.seed, stratify=df["label"])
    rel = args.test_ratio/(args.val_ratio+args.test_ratio) if (args.val_ratio+args.test_ratio)>0 else 0.5
    val_df, test_df = train_test_split(tmp, test_size=rel, random_state=args.seed, stratify=tmp["label"])
    for name, d in [("train.csv", train_df), ("val.csv", val_df), ("test.csv", test_df)]:
        d[["path","label"]].to_csv(os.path.join(args.data_dir, name), index=False)
    with open(os.path.join(args.data_dir, "classes.json"), "w") as f:
        json.dump(LABELS, f)
if __name__ == "__main__":
    main()
