# ==============================================================================
# === MVST — LEARNING CURVE: Concat vs CVA default (R2-P11) =====================
# === Basis: mvst_cva_lr_sweep.py (terbukti mereproduksi Tabel 3 persis) dan    ===
# ===        mvst_cva_ablation.py (Tabel 7).                                     ===
# === Arsitektur, dataset, seed protocol, dan URUTAN RNG identik, lr = 1e-4.     ===
# ===                                                                            ===
# === Yang DITAMBAHKAN (tanpa mengubah hasil training):                         ===
# ===   1. Log per epoch: train loss total/CE/CVA, val loss, val acc, val F1.   ===
# ===   2. Diagnostik representasi pada pass validasi yang SAMA (tidak ada      ===
# ===      iterasi DataLoader tambahan, karena val loader ikut memakai          ===
# ===      generator g; iterasi tambahan akan mengubah hasil):                  ===
# ===        - align   : rata-rata ||f_back - f_side||^2 (ter-normalisasi L2),   ===
# ===                    pasangan view dari instans yang sama (Wang & Isola 2020)===
# ===        - unif    : log E[exp(-2||x - y||^2)] atas semua pasangan embedding ===
# ===        - pos_cos : cosine rata-rata back vs side, instans yang sama        ===
# ===        - neg_cos : cosine rata-rata back vs side, instans berbeda          ===
# ===      Diagnostik dihitung pada fitur encoder ter-pool (sebelum projection  ===
# ===      head), sehingga concat dan CVA dibandingkan di ruang yang sama.       ===
# ===   3. Verifikasi otomatis: akurasi final-epoch dibandingkan dengan         ===
# ===      cva_ablation_5seed_results.csv (Tabel 7). Harus selisih 0.0.          ===
# ===   4. Resume per (config, seed): generator g dipakai berurutan lintas fold ===
# ===      dalam satu seed, jadi seed yang belum lengkap diulang dari fold 1.    ===
# ===                                                                            ===
# === Output:                                                                    ===
# ===   mvst_cva_learning_curve_epochs.csv  (1 baris per config-seed-fold-epoch) ===
# ===   mvst_cva_learning_curve_folds.csv   (1 baris per config-seed-fold)       ===
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

# ------------------------------------------------------------------------------
# Konfigurasi yang dijalankan: (nama, view_mode, lambda, tau, excl_same_class)
# Nama harus sama dengan kolom 'view_mode' di cva_ablation_5seed_results.csv
# agar verifikasi otomatis berjalan.
# ------------------------------------------------------------------------------
CONFIGS = [
    ("concat",      "concat",     0.0, 0.1, True),
    ("cva_default", "concat_cva", 0.5, 0.1, True),
    # ("cva_lambda_100", "concat_cva", 1.0, 0.1, True),   # opsional (S2), +50% waktu
]
REFERENCE_CSV = "cva_ablation_5seed_results.csv"   # hasil Tabel 7, untuk verifikasi


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
# === TRAINING (identik) ========================================================
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


# ==============================================================================
# === EVALUASI + DIAGNOSTIK (satu pass validasi, tanpa iterasi tambahan) ========
# ==============================================================================

def representation_diagnostics(fb, fs):
    """fb, fs: (N, d) fitur encoder ter-pool dari back/side view, baris i = instans i."""
    nb, ns = F.normalize(fb, dim=1), F.normalize(fs, dim=1)
    N = nb.size(0)
    align = ((nb - ns) ** 2).sum(dim=1).mean().item()
    pos_cos = (nb * ns).sum(dim=1).mean().item()
    sim = nb @ ns.t()
    off = ~torch.eye(N, dtype=torch.bool, device=nb.device)
    neg_cos = sim[off].mean().item() if N > 1 else float("nan")
    z = torch.cat([nb, ns], dim=0)
    sq = torch.pdist(z, p=2).pow(2)
    unif = torch.log(torch.exp(-2.0 * sq).mean()).item()
    return dict(align=align, unif=unif, pos_cos=pos_cos, neg_cos=neg_cos)


def evaluate_with_diag(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, feats_b, feats_s = [], [], [], []
    with torch.no_grad():
        for data in dataloader:
            lb, mb, ls, ms, labels = [d.to(device) for d in data]
            logits, fb, fs = model(lb, mb, ls, ms, return_features=True)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            feats_b.append(fb.cpu())
            feats_s.append(fs.cpu())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    diag = representation_diagnostics(torch.cat(feats_b), torch.cat(feats_s))
    return total_loss / len(dataloader), acc, f1, diag


# ==============================================================================
# === MAIN ======================================================================
# ==============================================================================

def run(root_dir="Dataset_AQA_CSV", configs=CONFIGS, lr=1e-4,
        seeds=(42, 7, 123, 777, 2024), num_epochs=40, batch_size=8,
        out_prefix="mvst_cva_learning_curve"):

    NUM_FRAMES, NUM_LANDMARKS, NUM_CLASSES = 150, 33, 4
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS = 128, 4, 2
    DIM_FEEDFORWARD, DROPOUT, CVA_PROJ_DIM = 256, 0.1, 64

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs_csv = f"{out_prefix}_epochs.csv"
    folds_csv = f"{out_prefix}_folds.csv"

    print("\n" + "=" * 70)
    print("MVST — LEARNING CURVE Concat vs CVA (R2-P11)")
    print(f"Device : {DEVICE} | Seeds : {list(seeds)} | lr : {lr} | Epochs : {num_epochs}")
    print(f"Configs: {[c[0] for c in configs]}")
    print("=" * 70 + "\n")

    full_dataset = MultiViewLandmarkDataset(root_dir, max_frames=NUM_FRAMES)
    paired_folds = list(zip(["A", "B", "C", "H"], ["E", "F", "G", "S"]))

    # ---- Resume per (config, seed) ----
    epoch_rows, fold_rows, done = [], [], set()
    if Path(epochs_csv).exists() and Path(folds_csv).exists():
        prev_f = pd.read_csv(folds_csv)
        counts = prev_f.groupby(["config", "seed"]).size()
        complete = {k for k, v in counts.items() if v == len(paired_folds)}
        keep = lambda df: df[[(r.config, r.seed) in complete for r in df.itertuples()]]
        fold_rows = keep(prev_f).to_dict("records")
        epoch_rows = keep(pd.read_csv(epochs_csv)).to_dict("records")
        done = {(c, int(s)) for c, s in complete}
        print(f"Resume: {len(done)} blok (config, seed) lengkap dilewati.")

    for cfg_name, view_mode, lam, tau, excl in configs:
        print("\n" + "%" * 70 + f"\nCONFIG: {cfg_name} (view_mode={view_mode}, "
              f"lambda={lam}, tau={tau}, excl={excl})\n" + "%" * 70)

        for seed in seeds:
            if (cfg_name, int(seed)) in done:
                continue

            # --- urutan RNG identik dengan mvst_cva_ablation.py ---
            set_global_seed(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            for fold, (val_trained, val_novice) in enumerate(paired_folds, 1):
                set_global_seed(seed + fold)
                val_subjects = [val_trained, val_novice]
                print(f"\n{'='*70}\n{cfg_name} | SEED {seed} | FOLD {fold}/4 | "
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
                    view_mode=view_mode, cva_proj_dim=CVA_PROJ_DIM).to(DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr)

                best_acc, best_f1, best_ep = 0.0, 0.0, 0
                final_acc, final_f1 = 0.0, 0.0
                for epoch in range(1, num_epochs + 1):
                    tr_loss, ce_l, cva_l = train_one_epoch(
                        model, train_loader, criterion, optimizer, DEVICE,
                        cva_lambda=lam, cva_temperature=tau, cva_exclude_same_class=excl)
                    va_loss, va_acc, va_f1, diag = evaluate_with_diag(
                        model, val_loader, criterion, DEVICE)
                    if va_f1 > best_f1:
                        best_acc, best_f1, best_ep = va_acc, va_f1, epoch
                    final_acc, final_f1 = va_acc, va_f1

                    epoch_rows.append({
                        "config": cfg_name, "seed": seed, "fold": fold, "epoch": epoch,
                        "train_loss": tr_loss, "train_ce": ce_l, "train_cva": cva_l,
                        "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
                        **{f"val_{k}": v for k, v in diag.items()},
                    })
                    print(f"Ep[{epoch:02d}/{num_epochs}] TrLoss:{tr_loss:.4f}"
                          f"(CE:{ce_l:.4f}|CVA:{cva_l:.4f}) | VaLoss:{va_loss:.4f} "
                          f"VaAcc:{va_acc:.4f} VaF1:{va_f1:.4f} | "
                          f"align:{diag['align']:.3f} unif:{diag['unif']:.3f}")

                print(f"-> Best: Acc={best_acc:.4f} F1={best_f1:.4f} Ep={best_ep} | "
                      f"Final: Acc={final_acc:.4f} F1={final_f1:.4f}")
                fold_rows.append({
                    "config": cfg_name, "seed": seed, "fold": fold,
                    "val_subjects": f"{val_trained} & {val_novice}",
                    "best_epoch": best_ep, "accuracy": best_acc, "f1_score": best_f1,
                    "final_epoch_accuracy": final_acc, "final_epoch_f1": final_f1,
                    "cva_lambda": lam, "cva_tau": tau, "excl_same_class": excl,
                })
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            # simpan setelah setiap blok (config, seed) selesai
            pd.DataFrame(epoch_rows).to_csv(epochs_csv, index=False)
            pd.DataFrame(fold_rows).to_csv(folds_csv, index=False)

    folds = pd.DataFrame(fold_rows)
    epochs = pd.DataFrame(epoch_rows)

    # ---- Verifikasi terhadap Tabel 7 ----
    print("\n" + "=" * 70 + "\nVERIFIKASI terhadap " + REFERENCE_CSV + "\n" + "=" * 70)
    if Path(REFERENCE_CSV).exists():
        ref = pd.read_csv(REFERENCE_CSV).rename(columns={"view_mode": "config"})
        m = folds.merge(ref[["config", "seed", "fold", "final_epoch_accuracy", "accuracy"]],
                        on=["config", "seed", "fold"], suffixes=("", "_ref"))
        for cfg, d in m.groupby("config"):
            dmax = (d["final_epoch_accuracy"] - d["final_epoch_accuracy_ref"]).abs().max()
            bmax = (d["accuracy"] - d["accuracy_ref"]).abs().max()
            status = "OK (identik)" if dmax < 1e-9 and bmax < 1e-9 else "BERBEDA - JANGAN DIPAKAI"
            print(f"{cfg:14s}: n={len(d)} | max|diff| final={dmax:.6f} best={bmax:.6f} -> {status}")
    else:
        print(f"File {REFERENCE_CSV} tidak ditemukan; verifikasi dilewati. "
              f"Taruh file Tabel 7 di folder yang sama lalu jalankan ulang (resume akan melewati training).")

    # ---- Ringkasan singkat ----
    print("\n" + "=" * 70 + "\nRINGKASAN (final epoch, mean ± SD ddof=1, n=20)\n" + "=" * 70)
    print(folds.groupby("config")["final_epoch_accuracy"].agg(
        mean="mean", sd=lambda x: x.std(ddof=1)).to_string())
    last10 = epochs[epochs["epoch"] > num_epochs - 10]
    fluct = last10.groupby(["config", "seed", "fold"])["val_acc"].std(ddof=1).reset_index()
    print("\nFluktuasi antar-epoch (SD val_acc epoch 31-40 dalam satu run), rata-rata per config:")
    print(fluct.groupby("config")["val_acc"].mean().to_string())
    print(f"\nFile hasil:\n1. {epochs_csv}\n2. {folds_csv}")
    return epochs, folds


if __name__ == "__main__":
    run()
