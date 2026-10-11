# ==============================================================================
# === ST-GCN MULTI-VIEW BASELINE — LR SWEEP (P2: baseline under-tuning check) ===
# === Arsitektur identik dengan baseline original (TIDAK diubah) ================
# === Protokol identik dengan mvst_stgcn5seed-1.py / MVST-CVA:
# ===   - SEEDS = [42, 7, 123, 777, 2024]
# ===   - per-fold seed = seed + fold
# ===   - 4-Fold Stratified Paired Subject-Independent Validation
# ===   - best-epoch (val F1) DAN final-epoch dilaporkan
# === Tambahan untuk P2:
# ===   - LEARNING_RATES = [5e-5, 1e-4, 2e-4, 5e-4] sebagai loop terluar
# ===   - kolom 'lr' ditambahkan ke setiap baris hasil
# ===   - tujuan: menunjukkan lr=1e-4 near-optimal, bukan under-tuned
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
import os
import time

try:
    from thop import profile
except ImportError:
    print("Warning: 'thop' belum terinstall. 'pip install thop' untuk FLOPs.")
    profile = None


# ==============================================================================
# === SWEEP PROTOCOL ============================================================
# ==============================================================================

SEEDS = [42, 7, 123, 777, 2024]
LEARNING_RATES = [5e-5, 1e-4, 2e-4, 5e-4]  # 1e-4 = nilai existing di manuskrip


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
# === DATASET (identik dengan versi MVST / mvst_stgcn5seed-1.py) ================
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
        self.class_to_idx = {
            "backhand": 0, "backhandNewbie": 1,
            "forehand": 2, "forehandNewbie": 3
        }
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
            self.data_cache.append((
                landmarks_back, mask_back, landmarks_side, mask_side,
                torch.tensor(item_meta["label"], dtype=torch.long)
            ))

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
        mask = torch.tensor(
            [1]*min(n_frames, self.max_frames) + [0]*max(0, self.max_frames - n_frames),
            dtype=torch.bool
        )
        landmarks_tensor = torch.tensor(all_frames_data, dtype=torch.float32)
        if self.transform:
            landmarks_tensor = self.transform(landmarks_tensor)
        return landmarks_tensor, mask

    def __getitem__(self, idx):
        return self.data_cache[idx]


# ==============================================================================
# === ARSITEKTUR ST-GCN (TIDAK DIUBAH dari baseline original) ===================
# ==============================================================================

class AdaptiveGraph(nn.Module):
    def __init__(self, num_node):
        super().__init__()
        self.A = nn.Parameter(torch.ones(num_node, num_node) / num_node)


class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1),
                                       padding=(4, 0), stride=(stride, 1))
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, A):
        res = self.residual(x)
        x = self.spatial_conv(x)
        B, C, T, V = x.shape
        x = x.reshape(B, C * T, V)
        x = torch.matmul(x, A)
        x = x.reshape(B, C, T, V)
        x = self.temporal_conv(x)
        x = x + res
        return self.relu(x)


class STGCN_MultiView(nn.Module):
    def __init__(self, num_class=4, num_node=33, in_channels=3, hidden_dim=64):
        super().__init__()
        self.graph = AdaptiveGraph(num_node)

        self.stgcn_back = nn.Sequential(
            STGCN_Block(in_channels, hidden_dim),
            STGCN_Block(hidden_dim, hidden_dim)
        )
        self.stgcn_side = nn.Sequential(
            STGCN_Block(in_channels, hidden_dim),
            STGCN_Block(hidden_dim, hidden_dim)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_class)
        )

    def forward(self, x_back, mask_back, x_side, mask_side):
        x_b = x_back.permute(0, 3, 1, 2)
        x_s = x_side.permute(0, 3, 1, 2)

        out_b = self.stgcn_back[0](x_b, self.graph.A)
        out_b = self.stgcn_back[1](out_b, self.graph.A)

        out_s = self.stgcn_side[0](x_s, self.graph.A)
        out_s = self.stgcn_side[1](out_s, self.graph.A)

        feat_b = out_b.mean(dim=(2, 3))
        feat_s = out_s.mean(dim=(2, 3))

        feat_fused = torch.cat([feat_b, feat_s], dim=1)
        return self.fc(feat_fused)


# ==============================================================================
# === MODEL COMPLEXITY (dihitung sekali saja, tidak tergantung LR) ==============
# ==============================================================================

def calculate_model_complexity(model, device, num_frames=150, num_landmarks=33):
    if profile is None:
        return {"params_m": None, "flops_g": None, "inference_ms": None, "fps": None}
    try:
        model.eval()
        dxb = torch.randn(1, num_frames, num_landmarks, 3).to(device)
        dmb = torch.ones(1, num_frames, dtype=torch.bool).to(device)
        dxs = torch.randn(1, num_frames, num_landmarks, 3).to(device)
        dms = torch.ones(1, num_frames, dtype=torch.bool).to(device)

        macs, params = profile(model, inputs=(dxb, dmb, dxs, dms), verbose=False)
        flops = macs * 2

        with torch.no_grad():
            for _ in range(10):
                _ = model(dxb, dmb, dxs, dms)

        times = []
        with torch.no_grad():
            for _ in range(100):
                if device.type == "cuda":
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    _ = model(dxb, dmb, dxs, dms)
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
                else:
                    t0 = time.time()
                    _ = model(dxb, dmb, dxs, dms)
                    times.append((time.time() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        return {"params_m": params/1e6, "flops_g": flops/1e9,
                "inference_ms": avg_ms, "fps": 1000.0/avg_ms}
    except Exception as ex:
        print(f"Gagal menghitung kompleksitas: {ex}")
        return {"params_m": None, "flops_g": None, "inference_ms": None, "fps": None}


# ==============================================================================
# === TRAINING & EVALUATION =====================================================
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in dataloader:
        lb, mb, ls, ms, labels = [d.to(device) for d in data]
        optimizer.zero_grad()
        outputs = model(lb, mb, ls, ms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in dataloader:
            lb, mb, ls, ms, labels = [d.to(device) for d in data]
            outputs = model(lb, mb, ls, ms)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ==============================================================================
# === MAIN — LR SWEEP (loop LR di luar, loop seed x fold di dalam) =============
# ==============================================================================

if __name__ == "__main__":

    ROOT_DIR      = "Dataset_AQA_CSV"
    NUM_EPOCHS    = 40
    BATCH_SIZE    = 8
    NUM_FRAMES    = 150
    NUM_CLASSES   = 4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("ST-GCN MULTI-VIEW BASELINE — LR SWEEP (P2)")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Seeds         : {SEEDS}")
    print(f"Learning rates: {LEARNING_RATES}")
    print("=" * 70 + "\n")

    # Dataset dimuat SEKALI (deterministik, tidak tergantung seed/LR)
    full_dataset = MultiViewLandmarkDataset(ROOT_DIR, max_frames=NUM_FRAMES)

    TRAINED_SUBJECTS = ["A", "B", "C", "H"]
    NOVICE_SUBJECTS  = ["E", "F", "G", "S"]
    paired_folds = list(zip(TRAINED_SUBJECTS, NOVICE_SUBJECTS))
    print(f"\nPasangan fold: {paired_folds}")

    # Kompleksitas dihitung sekali — tidak berubah lintas LR (arsitektur sama)
    set_global_seed(SEEDS[0])
    temp_model = STGCN_MultiView(num_class=NUM_CLASSES, num_node=33,
                                 in_channels=3, hidden_dim=64).to(DEVICE)
    complexity = calculate_model_complexity(temp_model, DEVICE, NUM_FRAMES, 33)
    complexity["model"] = "stgcn_multiview"
    print("\nKompleksitas Model ST-GCN (identik untuk semua LR):")
    print(f"Parameters (M) : {complexity['params_m']}")
    print(f"FLOPs (G)      : {complexity['flops_g']}")
    del temp_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    all_results = []

    # ============================ LOOP LEARNING RATE ============================
    for lr in LEARNING_RATES:

        print("\n" + "%" * 70)
        print(f"LEARNING RATE: {lr}")
        print("%" * 70)

        # ============================ LOOP SEED ============================
        for seed in SEEDS:

            print("\n" + "#" * 70)
            print(f"LR {lr} | SEED: {seed}")
            print("#" * 70)

            set_global_seed(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            # ============================ LOOP FOLD ============================
            for fold, (val_trained, val_novice) in enumerate(paired_folds, 1):

                set_global_seed(seed + fold)

                val_subjects = [val_trained, val_novice]

                print("\n" + "=" * 70)
                print(f"LR {lr} | SEED {seed} | FOLD {fold}/4 | "
                      f"Validasi: {val_trained} (Pakar) & {val_novice} (Pemula)")
                print(f"[SEED] fold seed = {seed + fold}")
                print("=" * 70)

                train_idx = full_dataset.metadata[
                    ~full_dataset.metadata["subject_id"].isin(val_subjects)
                ].index.tolist()
                val_idx = full_dataset.metadata[
                    full_dataset.metadata["subject_id"].isin(val_subjects)
                ].index.tolist()

                train_loader = DataLoader(
                    Subset(full_dataset, train_idx),
                    batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=0, worker_init_fn=seed_worker, generator=g
                )
                val_loader = DataLoader(
                    Subset(full_dataset, val_idx),
                    batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=0, worker_init_fn=seed_worker, generator=g
                )

                model = STGCN_MultiView(num_class=NUM_CLASSES, num_node=33,
                                        in_channels=3, hidden_dim=64).to(DEVICE)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr)

                best_val_acc, best_val_f1, best_epoch = 0.0, 0.0, 0
                final_acc, final_f1 = 0.0, 0.0

                for epoch in range(1, NUM_EPOCHS + 1):
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
                    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, DEVICE)

                    if val_f1 > best_val_f1:
                        best_val_acc, best_val_f1 = val_acc, val_f1
                        best_epoch = epoch

                    final_acc, final_f1 = val_acc, val_f1

                    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
                          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

                print(f"\nHasil fold (LR {lr}, SEED {seed}, FOLD {fold}):")
                print(f"Best Epoch: {best_epoch} | "
                      f"Acc: {best_val_acc:.4f} | F1: {best_val_f1:.4f}")
                print(f"[Final-epoch] Acc: {final_acc:.4f} | F1: {final_f1:.4f}")

                all_results.append({
                    "model": "stgcn_multiview",
                    "lr": lr,
                    "seed": seed, "fold_seed": seed + fold,
                    "fold": fold,
                    "val_subjects": f"{val_trained} & {val_novice}",
                    "best_epoch": best_epoch,
                    "accuracy": best_val_acc, "f1_score": best_val_f1,
                    "final_epoch_accuracy": final_acc, "final_epoch_f1": final_f1
                })

                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            # Summary per (lr, seed)
            seed_res = [r for r in all_results if r["seed"] == seed and r["lr"] == lr]
            accs = [r["accuracy"] for r in seed_res]
            f1s  = [r["f1_score"] for r in seed_res]
            print("\n" + "-" * 70)
            print(f"SUMMARY LR {lr} | SEED {seed}: "
                  f"Acc {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
                  f"F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

        # Summary per LR (setelah semua seed untuk LR ini selesai)
        lr_res = [r for r in all_results if r["lr"] == lr]
        accs = [r["accuracy"] for r in lr_res]
        f1s  = [r["f1_score"] for r in lr_res]
        final_accs = [r["final_epoch_accuracy"] for r in lr_res]
        final_f1s  = [r["final_epoch_f1"] for r in lr_res]
        print("\n" + "*" * 70)
        print(f"AGREGAT LR {lr} ({len(lr_res)} observasi = 5 seed x 4 fold)")
        print(f"[Best-epoch ] Acc: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f} | "
              f"F1: {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}")
        print(f"[Final-epoch] Acc: {np.mean(final_accs):.4f} ± {np.std(final_accs, ddof=1):.4f} | "
              f"F1: {np.mean(final_f1s):.4f} ± {np.std(final_f1s, ddof=1):.4f}")
        print("*" * 70)

    # ===================== SIMPAN & AGREGAT FINAL =====================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("stgcn_lr_sweep_results.csv", index=False)

    per_lr = results_df.groupby("lr").agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", lambda x: x.std(ddof=1)),
        f1_mean=("f1_score", "mean"),  f1_std=("f1_score", lambda x: x.std(ddof=1)),
        final_acc_mean=("final_epoch_accuracy", "mean"),
        final_acc_std=("final_epoch_accuracy", lambda x: x.std(ddof=1)),
        final_f1_mean=("final_epoch_f1", "mean"),
        final_f1_std=("final_epoch_f1", lambda x: x.std(ddof=1)),
    ).reset_index()
    per_lr.to_csv("stgcn_lr_sweep_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("RINGKASAN LR SWEEP (mean ± std, ddof=1, n=20 per LR)")
    print("=" * 70)
    print(per_lr.to_string(index=False))

    print("\nFile hasil:")
    print("1. stgcn_lr_sweep_results.csv   (fold-level, semua LR, untuk uji statistik lanjutan)")
    print("2. stgcn_lr_sweep_summary.csv   (ringkasan per LR, siap dimasukkan ke response letter)")