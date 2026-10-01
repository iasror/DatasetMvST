# ==============================================================================
# === MVST+CVA — LR SWEEP (R2-P7: fairness of hyperparameter selection) =========
# === Basis: mvst_cva_ablation.py, HANYA konfigurasi "cva_default":            ===
# ===   lambda = 0.5, tau = 0.1, exclude_same_class = True, proj_dim = 64      ===
# === Arsitektur, dataset, seed protocol, dan urutan pemanggilan RNG identik   ===
# === dengan script ablasi, sehingga baris lr = 1e-4 seharusnya MEREPRODUKSI    ===
# === hasil MVST+CVA di manuskrip (cek reproduktifitas).                        ===
# === Tambahan:                                                                  ===
# ===   - LEARNING_RATES = [5e-5, 1e-4, 2e-4, 5e-4] sebagai loop terluar        ===
# ===     (sama dengan sweep ST-GCN)                                             ===
# ===   - kolom 'lr' di setiap baris hasil                                       ===
# ===   - resume per (lr, seed): generator DataLoader dipakai berurutan lintas   ===
# ===     fold dalam satu seed, jadi seed yang belum lengkap diulang dari fold 1 ===
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
import os


# ==============================================================================
# === SEED CONFIGURATION (identik) ==============================================
# ==============================================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==============================================================================
# === DATASET (identik dengan mvst_cva_ablation.py) =============================
# ==============================================================================

class MultiViewLandmarkDataset(Dataset):
    def __init__(self, root_dir, max_frames=150, transform=None):
        self.root_dir = Path(root_dir)
        self.max_frames = max_frames
        self.transform = transform
        file_pairs = self._find_file_pairs()
        if not file_pairs:
            raise FileNotFoundError(f"Tidak ada pasangan file CSV lengkap di {root_dir}.")
        self.metadata = pd.DataFrame(file_pairs)
        self.class_to_idx = {"backhand": 0, "backhandNewbie": 1,
                             "forehand": 2, "forehandNewbie": 3}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.metadata["label"] = self.metadata["class"].map(self.class_to_idx)
        print(f"Ditemukan {len(self.metadata)} sampel pasangan data yang lengkap.")
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.data_cache = []
        print("\nMemproses dan memuat seluruh data ke memori (RAM)...")
        for idx in tqdm(range(len(self.metadata)), desc="Caching Dataset"):
            item_meta = self.metadata.iloc[idx]
            landmarks_back, mask_back = self._process_single_csv(item_meta["back_path"])
            landmarks_side, mask_side = self._process_single_csv(item_meta["side_path"])
            self.data_cache.append((landmarks_back, mask_back, landmarks_side, mask_side,
                                    torch.tensor(item_meta["label"], dtype=torch.long)))

    def _find_file_pairs(self):
        file_pairs_dict = {}
        print("Memindai pasangan file...")
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for csv_file in class_dir.glob("*.csv"):
                file_name = csv_file.name
                base_name = file_name.replace("_Back.csv", "").replace("_Side.csv", "")
                try:
                    subject_id = base_name.split("_")[1][0]
                except IndexError:
                    subject_id = base_name.split("_")[0][0]
                sample_key = f"{class_name}/{base_name}"
                if sample_key not in file_pairs_dict:
                    file_pairs_dict[sample_key] = {"class": class_name, "subject_id": subject_id}
                if "_Back.csv" in file_name:
                    file_pairs_dict[sample_key]["back_path"] = str(csv_file)
                elif "_Side.csv" in file_name:
                    file_pairs_dict[sample_key]["side_path"] = str(csv_file)
        return [p for _, p in file_pairs_dict.items()
                if "back_path" in p and "side_path" in p]

    def __len__(self):
        return len(self.metadata)

    def _process_single_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        n_landmarks, n_coords = 33, 3
        all_frames_data = np.zeros((self.max_frames, n_landmarks, n_coords), dtype=np.float32)
        frame_numbers = sorted(df["frame_number"].unique())
        n_frames = len(frame_numbers)
        for i, frame_num in enumerate(frame_numbers):
            if i >= self.max_frames:
                break
            frame_data = df[df["frame_number"] == frame_num][["x", "y", "z"]].values
            if len(frame_data) == n_landmarks:
                hip_center = (frame_data[self.LEFT_HIP] + frame_data[self.RIGHT_HIP]) * 0.5
                normalized = frame_data - hip_center
                max_dist = np.max(np.linalg.norm(normalized, axis=1))
                if max_dist > 1e-6:
                    normalized /= max_dist
                all_frames_data[i] = normalized
        all_frames_data = np.nan_to_num(all_frames_data, nan=0.0)
        valid_len = min(n_frames, self.max_frames)
        mask = torch.tensor([1]*valid_len + [0]*max(0, self.max_frames - valid_len),
                            dtype=torch.bool)
        landmarks_tensor = torch.tensor(all_frames_data, dtype=torch.float32)
        if self.transform:
            landmarks_tensor = self.transform(landmarks_tensor)
        return landmarks_tensor, mask

    def __getitem__(self, idx):
        return self.data_cache[idx]


# ==============================================================================
# === MODEL (identik dengan mvst_cva_ablation.py) ===============================
# ==============================================================================

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, num_landmarks, num_frames, d_model, nhead,
                 num_encoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(3, d_model)
        self.spatial_pos_encoder = nn.Parameter(torch.zeros(1, num_landmarks, d_model))
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.spatial_transformer_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=num_encoder_layers)
        self.temporal_pos_encoder = nn.Parameter(torch.zeros(1, num_frames, d_model))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.temporal_transformer_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=num_encoder_layers)

    def forward(self, x, mask):
        B, T, L, _ = x.shape
        x = self.input_embedding(x)
        x_spatial = x.view(B*T, L, self.d_model) + self.spatial_pos_encoder
        spatial_output = self.spatial_transformer_encoder(x_spatial)
        frame_features = spatial_output.mean(dim=1)
        x_temporal = frame_features.view(B, T, self.d_model)
        x_temporal = x_temporal + self.temporal_pos_encoder[:, :T, :]
        return self.temporal_transformer_encoder(x_temporal, src_key_padding_mask=~mask)


class ProjectionHead(nn.Module):
    def __init__(self, d_model, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                 nn.Linear(d_model, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def cross_view_infonce(z_back, z_side, labels=None,
                       temperature=0.1, exclude_same_class_negatives=False):
    B = z_back.size(0)
    logits = (z_back @ z_side.t()) / temperature
    targets = torch.arange(B, device=z_back.device)
    if exclude_same_class_negatives and labels is not None and B > 1:
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        off_diag = ~torch.eye(B, dtype=torch.bool, device=z_back.device)
        logits = logits.masked_fill(same_class & off_diag, float("-inf"))
    loss_b2s = F.cross_entropy(logits, targets)
    loss_s2b = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_b2s + loss_s2b)


class MultiViewSpatioTemporalTransformer(nn.Module):
    def __init__(self, num_landmarks, num_frames, num_classes,
                 d_model, nhead, num_encoder_layers, dim_feedforward,
                 dropout, view_mode="concat", cva_proj_dim=64):
        super().__init__()
        self.view_mode = view_mode.lower()
        assert self.view_mode in ["back", "side", "concat", "concat_cva"]
        self.use_cva = (self.view_mode == "concat_cva")
        self.encoder = SpatioTemporalEncoder(
            num_landmarks=num_landmarks, num_frames=num_frames, d_model=d_model,
            nhead=nhead, num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout)
        classifier_in_dim = d_model * 2 if self.view_mode in ["concat", "concat_cva"] else d_model
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(classifier_in_dim), nn.Linear(classifier_in_dim, d_model),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes))
        if self.use_cva:
            self.projection_head = ProjectionHead(d_model, cva_proj_dim)

    def forward(self, x_back, mask_back, x_side, mask_side, return_features=False):
        if self.view_mode == "back":
            return self.classifier_head(self.encoder(x_back, mask_back).mean(dim=1))
        if self.view_mode == "side":
            return self.classifier_head(self.encoder(x_side, mask_side).mean(dim=1))
        feature_back = self.encoder(x_back, mask_back).mean(dim=1)
        feature_side = self.encoder(x_side, mask_side).mean(dim=1)
        logits = self.classifier_head(torch.cat([feature_back, feature_side], dim=1))
        if return_features:
            return logits, feature_back, feature_side
        return logits


# ==============================================================================
# === TRAINING & EVALUATION (identik) ===========================================
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    cva_lambda=0.0, cva_temperature=0.1, cva_exclude_same_class=False):
    model.train()
    total_loss, total_ce, total_cva = 0.0, 0.0, 0.0
    for data in dataloader:
        lb, mb, ls, ms, labels = [d.to(device) for d in data]
        optimizer.zero_grad()
        if model.use_cva:
            logits, fb, fs = model(lb, mb, ls, ms, return_features=True)
            ce_loss = criterion(logits, labels)
            zb = model.projection_head(fb)
            zs = model.projection_head(fs)
            cva_loss = cross_view_infonce(zb, zs, labels=labels,
                                          temperature=cva_temperature,
                                          exclude_same_class_negatives=cva_exclude_same_class)
            loss = ce_loss + cva_lambda * cva_loss
            total_ce += ce_loss.item()
            total_cva += cva_loss.item()
        else:
            logits = model(lb, mb, ls, ms)
            loss = criterion(logits, labels)
            total_ce += loss.item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    n = len(dataloader)
    return total_loss/n, total_ce/n, total_cva/n


def evaluate(model, dataloader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in dataloader:
            lb, mb, ls, ms, labels = [d.to(device) for d in data]
            logits = model(lb, mb, ls, ms)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    if return_predictions:
        return total_loss / len(dataloader), acc, f1, all_labels, all_preds
    return total_loss / len(dataloader), acc, f1


# ==============================================================================
# === MAIN — LR SWEEP ===========================================================
# ==============================================================================

def run(root_dir="Dataset_AQA_CSV", learning_rates=(5e-5, 1e-4, 2e-4, 5e-4),
        seeds=(42, 7, 123, 777, 2024), num_epochs=40, batch_size=8,
        out_prefix="mvst_cva_lr_sweep"):

    NUM_FRAMES, NUM_LANDMARKS, NUM_CLASSES = 150, 33, 4
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS = 128, 4, 2
    DIM_FEEDFORWARD, DROPOUT, CVA_PROJ_DIM = 256, 0.1, 64
    CFG_LAMBDA, CFG_TAU, CFG_EXCL = 0.5, 0.1, True       # = "cva_default"
    VIEW_MODE = "concat_cva"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_csv = f"{out_prefix}_results.csv"

    print("\n" + "=" * 70)
    print("MVST+CVA — LR SWEEP (R2-P7)")
    print(f"Device : {DEVICE} | Seeds : {list(seeds)} | LR : {list(learning_rates)}")
    print(f"CVA    : lambda={CFG_LAMBDA}, tau={CFG_TAU}, excl_same_class={CFG_EXCL}")
    print("=" * 70 + "\n")

    full_dataset = MultiViewLandmarkDataset(root_dir, max_frames=NUM_FRAMES)
    paired_folds = list(zip(["A", "B", "C", "H"], ["E", "F", "G", "S"]))

    # ---- Resume per (lr, seed): hanya blok seed yang lengkap 4 fold yang dipakai ----
    all_results, done = [], set()
    if Path(results_csv).exists():
        prev = pd.read_csv(results_csv)
        counts = prev.groupby(["lr", "seed"]).size()
        complete = {k for k, v in counts.items() if v == len(paired_folds)}
        prev = prev[[(r.lr, r.seed) in complete for r in prev.itertuples()]]
        all_results = prev.to_dict("records")
        done = {(float(lr), int(s)) for lr, s in complete}
        print(f"Resume: {len(done)} blok (lr, seed) lengkap dilewati.")

    for lr in learning_rates:
        print("\n" + "%" * 70)
        print(f"LEARNING RATE: {lr}")
        print("%" * 70)

        for seed in seeds:
            if (float(lr), int(seed)) in done:
                continue

            # --- urutan RNG identik dengan mvst_cva_ablation.py ---
            set_global_seed(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            for fold, (val_trained, val_novice) in enumerate(paired_folds, 1):
                set_global_seed(seed + fold)
                val_subjects = [val_trained, val_novice]
                print(f"\n{'='*70}\nLR {lr} | SEED {seed} | FOLD {fold}/4 | "
                      f"Val: {val_trained} & {val_novice}\n{'='*70}")

                md = full_dataset.metadata
                train_idx = md[~md["subject_id"].isin(val_subjects)].index.tolist()
                val_idx = md[md["subject_id"].isin(val_subjects)].index.tolist()

                train_loader = DataLoader(Subset(full_dataset, train_idx),
                                          batch_size=batch_size, shuffle=True, num_workers=0,
                                          worker_init_fn=seed_worker, generator=g)
                val_loader = DataLoader(Subset(full_dataset, val_idx),
                                        batch_size=batch_size, shuffle=False, num_workers=0,
                                        worker_init_fn=seed_worker, generator=g)

                model = MultiViewSpatioTemporalTransformer(
                    num_landmarks=NUM_LANDMARKS, num_frames=NUM_FRAMES,
                    num_classes=NUM_CLASSES, d_model=D_MODEL, nhead=NHEAD,
                    num_encoder_layers=NUM_ENCODER_LAYERS,
                    dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
                    view_mode=VIEW_MODE, cva_proj_dim=CVA_PROJ_DIM).to(DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr)

                best_val_acc, best_val_f1, best_epoch = 0.0, 0.0, 0
                final_acc, final_f1 = 0.0, 0.0
                for epoch in range(1, num_epochs + 1):
                    train_loss, ce_l, cva_l = train_one_epoch(
                        model, train_loader, criterion, optimizer, DEVICE,
                        cva_lambda=CFG_LAMBDA, cva_temperature=CFG_TAU,
                        cva_exclude_same_class=CFG_EXCL)
                    _, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE,
                                                        return_predictions=True)
                    if val_f1 > best_val_f1:
                        best_val_acc, best_val_f1, best_epoch = val_acc, val_f1, epoch
                    final_acc, final_f1 = val_acc, val_f1
                    print(f"Ep[{epoch:02d}/{num_epochs}] TrLoss:{train_loss:.4f}"
                          f"(CE:{ce_l:.4f} | CVA:{cva_l:.4f}) | "
                          f"ValAcc:{val_acc:.4f} ValF1:{val_f1:.4f}")

                print(f"-> Best: Acc={best_val_acc:.4f} F1={best_val_f1:.4f} Ep={best_epoch} | "
                      f"Final: Acc={final_acc:.4f} F1={final_f1:.4f}")

                all_results.append({
                    "model": "mvst_cva", "lr": lr, "seed": seed,
                    "fold_seed": seed + fold, "fold": fold,
                    "val_subjects": f"{val_trained} & {val_novice}",
                    "best_epoch": best_epoch,
                    "accuracy": best_val_acc, "f1_score": best_val_f1,
                    "final_epoch_accuracy": final_acc, "final_epoch_f1": final_f1,
                    "cva_lambda": CFG_LAMBDA, "cva_tau": CFG_TAU, "excl_same_class": CFG_EXCL,
                })
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            # simpan setelah setiap blok (lr, seed) selesai
            pd.DataFrame(all_results).to_csv(results_csv, index=False)

        lr_res = pd.DataFrame([r for r in all_results if float(r["lr"]) == float(lr)])
        if len(lr_res):
            print(f"\nAGREGAT LR {lr} ({len(lr_res)} obs): "
                  f"Best {lr_res['accuracy'].mean():.4f} ± {lr_res['accuracy'].std(ddof=1):.4f} | "
                  f"Final {lr_res['final_epoch_accuracy'].mean():.4f} ± "
                  f"{lr_res['final_epoch_accuracy'].std(ddof=1):.4f}")

    df = pd.DataFrame(all_results)
    summary = df.groupby("lr").agg(
        n_obs=("accuracy", "count"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", lambda x: x.std(ddof=1)),
        f1_mean=("f1_score", "mean"), f1_std=("f1_score", lambda x: x.std(ddof=1)),
        final_acc_mean=("final_epoch_accuracy", "mean"),
        final_acc_std=("final_epoch_accuracy", lambda x: x.std(ddof=1)),
        final_f1_mean=("final_epoch_f1", "mean"),
        final_f1_std=("final_epoch_f1", lambda x: x.std(ddof=1)),
    ).reset_index()
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("RINGKASAN LR SWEEP MVST+CVA (mean ± std, ddof=1, n=20 per LR)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nFile hasil:")
    print(f"1. {results_csv}   (fold-level, semua LR)")
    print(f"2. {out_prefix}_summary.csv   (ringkasan per LR)")
    print("Cek: baris lr=1e-4 seharusnya identik dengan hasil MVST+CVA (cva_default) di manuskrip.")
    return df


if __name__ == "__main__":
    run()