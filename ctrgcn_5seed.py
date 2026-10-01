# ==============================================================================
# === CTR-GCN (Channel-wise Topology Refinement GCN) — ICCV 2021 ==============
# === Adapted for MediaPipe 33-landmark format, multi-view (back + side) =======
# === Protokol identik: 5-seed × 4-fold, best-epoch + final-epoch =============
# === Referensi: Chen et al. "Channel-wise Topology Refinement Graph Convolution
# ===            for Skeleton-Based Action Recognition." ICCV 2021.
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
import time

try:
    from thop import profile
except ImportError:
    print("Warning: 'thop' belum terinstall. 'pip install thop' untuk FLOPs.")
    profile = None


# ==============================================================================
# === SEED PROTOCOL (identik dengan MVST) ======================================
# ==============================================================================

SEEDS = [42, 7, 123, 777, 2024]

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
# === MEDIAPIPE 33-LANDMARK GRAPH TOPOLOGY =====================================
# Edges mengikuti koneksi anatomis MediaPipe Pose:
# https://google.github.io/mediapipe/solutions/pose.html
# ==============================================================================

MEDIAPIPE_EDGES = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Upper body
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Lower body
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

NUM_NODES = 33


def build_adjacency_matrix(num_nodes: int, edges: list,
                            strategy: str = "spatial") -> torch.Tensor:
    """
    Build normalized adjacency matrix from edge list.
    strategy='spatial': D^{-1/2} A D^{-1/2} normalization.
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Add self-loops
    A += np.eye(num_nodes, dtype=np.float32)
    # Degree-normalize
    D = np.array(A.sum(axis=1), dtype=np.float32)
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]
    return torch.tensor(A_norm, dtype=torch.float32)


# ==============================================================================
# === CTR-GCN COMPONENTS =======================================================
# ==============================================================================

class ChannelWiseTopologyRefinement(nn.Module):
    """
    Channel-wise Topology Refinement (CTR) module — core contribution of
    CTR-GCN. For each output channel, learns a channel-specific adjacency
    refinement tensor on top of the shared base topology.

    Original paper: Chen et al., ICCV 2021.
    """
    def __init__(self, in_channels: int, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        # Channel-specific topology: learn a (C, V, V) refinement
        self.channel_topology = nn.Parameter(
            torch.zeros(in_channels, num_nodes, num_nodes)
        )
        nn.init.normal_(self.channel_topology, std=0.01)

    def forward(self, x: torch.Tensor,
                A_base: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, C, T, V)
        A_base : (V, V)  — shared base adjacency
        Returns: (B, C, T, V) aggregated features
        """
        B, C, T, V = x.shape
        # Refined adjacency per channel: (C, V, V) = base + channel-specific
        A_refined = A_base.unsqueeze(0) + self.channel_topology  # (C, V, V)
        A_refined = F.softmax(A_refined, dim=-1)

        # Efficient einsum: for each channel, aggregate over neighbors
        # x: (B, C, T, V) → reshape to (B*T, C, V) for batch-matmul
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * T, C, V)
        # A_refined: (C, V, V)
        # out[bt, c, v] = sum_u A_refined[c, v, u] * x_flat[bt, c, u]
        # einsum: 'bcv, cvu -> bcu' → result (B*T, C, V)
        out = torch.einsum('bcv,cvu->bcu', x_flat, A_refined)
        out = out.view(B, T, C, V).permute(0, 2, 1, 3).contiguous()
        return out


class CTRGCNBlock(nn.Module):
    """
    One CTR-GCN block: spatial GCN with channel-wise topology refinement +
    temporal convolution with residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_nodes: int, stride: int = 1,
                 temporal_kernel: int = 9, dropout: float = 0.0):
        super().__init__()

        self.ctr = ChannelWiseTopologyRefinement(in_channels, num_nodes)

        # Pointwise projection after graph aggregation
        self.gcn_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        # Temporal convolution: (1, K) with padding to keep time dim
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel, 1),
                      padding=(pad, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor,
                A_base: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, V)"""
        res = self.residual(x)
        # CTR spatial aggregation
        out = self.ctr(x, A_base)               # (B, C, T, V)
        out = self.gcn_proj(out)                 # (B, C_out, T, V)
        # Temporal convolution
        out = self.relu(self.tcn(out) + res)
        out = self.dropout(out)
        return out


# ==============================================================================
# === CTR-GCN MULTI-VIEW MODEL =================================================
# ==============================================================================

class CTRGCN_MultiView(nn.Module):
    """
    Multi-view CTR-GCN:
    - Two independent CTR-GCN branches (back-view, side-view) sharing
      the same base graph topology but with independent learned parameters.
    - Late fusion via concatenation (consistent with MVST baseline).

    Architecture depth set to be comparable with MVST in parameter count.
    """
    def __init__(self, num_classes: int = 4, num_nodes: int = 33,
                 in_channels: int = 3, base_channels: int = 64,
                 num_blocks: int = 4, dropout: float = 0.1):
        super().__init__()

        self.num_nodes = num_nodes

        # Shared base adjacency (fixed, not learned — CTR modules add
        # channel-specific refinements on top)
        A = build_adjacency_matrix(num_nodes, MEDIAPIPE_EDGES)
        self.register_buffer('A_base', A)

        # Build block channels: [base, base, 2*base, 2*base]
        channels = [in_channels] + [base_channels] * (num_blocks // 2) + \
                   [base_channels * 2] * (num_blocks - num_blocks // 2)

        # Back-view branch
        self.back_blocks = nn.ModuleList([
            CTRGCNBlock(channels[i], channels[i + 1],
                        num_nodes=num_nodes, dropout=dropout)
            for i in range(num_blocks)
        ])

        # Side-view branch (independent parameters, same architecture)
        self.side_blocks = nn.ModuleList([
            CTRGCNBlock(channels[i], channels[i + 1],
                        num_nodes=num_nodes, dropout=dropout)
            for i in range(num_blocks)
        ])

        out_ch = channels[-1]  # 2 * base_channels

        # Classifier after late fusion (concat back + side)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_ch * 2),
            nn.Linear(out_ch * 2, out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_ch, num_classes),
        )

    def _encode_view(self, x: torch.Tensor, mask: torch.Tensor,
                     blocks: nn.ModuleList) -> torch.Tensor:
        """
        Encode one view through CTR-GCN blocks.
        x    : (B, T, V, C)  — dataset format
        mask : (B, T)
        Returns: (B, out_channels)
        """
        # Reshape to (B, C, T, V) — Conv2d format
        B, T, V, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()   # (B, C, T, V)

        for block in blocks:
            x = block(x, self.A_base)             # (B, C', T, V)

        # Global average pooling over time (masked) and nodes
        # Apply mask: zero out padded frames
        mask_float = mask.float().unsqueeze(1).unsqueeze(-1)  # (B,1,T,1)
        x = x * mask_float

        valid_len = mask.sum(dim=1, keepdim=True).float()     # (B, 1)
        valid_len = valid_len.clamp(min=1.0)

        # Mean over T (masked) then mean over V
        feat = x.sum(dim=2) / valid_len.unsqueeze(-1)         # (B, C', V)
        feat = feat.mean(dim=2)                               # (B, C')
        return feat

    def forward(self, x_back, mask_back, x_side, mask_side):
        feat_back = self._encode_view(x_back, mask_back, self.back_blocks)
        feat_side = self._encode_view(x_side, mask_side, self.side_blocks)
        fused = torch.cat([feat_back, feat_side], dim=1)
        return self.classifier(fused)


# ==============================================================================
# === DATASET (identik dengan MVST) ============================================
# ==============================================================================

class MultiViewLandmarkDataset(Dataset):
    def __init__(self, root_dir, max_frames=150):
        self.root_dir = Path(root_dir)
        self.max_frames = max_frames
        file_pairs = self._find_file_pairs()

        if not file_pairs:
            raise FileNotFoundError(
                f"Tidak ada pasangan CSV di {root_dir}.")

        self.metadata = pd.DataFrame(file_pairs)
        self.class_to_idx = {
            "backhand": 0, "backhandNewbie": 1,
            "forehand": 2, "forehandNewbie": 3
        }
        self.metadata["label"] = self.metadata["class"].map(self.class_to_idx)
        print(f"Ditemukan {len(self.metadata)} sampel paired.")

        self.LEFT_HIP  = 23
        self.RIGHT_HIP = 24

        self.data_cache = []
        print("Memuat data ke RAM...")
        for idx in tqdm(range(len(self.metadata)), desc="Caching"):
            m = self.metadata.iloc[idx]
            lb, mb = self._process_csv(m["back_path"])
            ls, ms = self._process_csv(m["side_path"])
            self.data_cache.append((
                lb, mb, ls, ms,
                torch.tensor(m["label"], dtype=torch.long)
            ))

    def _find_file_pairs(self):
        d = {}
        print("Memindai pasangan file...")
        for cls_dir in self.root_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            for csv_f in cls_dir.glob("*.csv"):
                name = csv_f.name
                base = name.replace("_Back.csv", "").replace("_Side.csv", "")
                try:
                    subj = base.split("_")[1][0]
                except IndexError:
                    subj = base.split("_")[0][0]
                key = f"{cls_dir.name}/{base}"
                if key not in d:
                    d[key] = {"class": cls_dir.name, "subject_id": subj}
                if "_Back.csv" in name:
                    d[key]["back_path"] = str(csv_f)
                elif "_Side.csv" in name:
                    d[key]["side_path"] = str(csv_f)
        return [v for v in d.values()
                if "back_path" in v and "side_path" in v]

    def __len__(self):
        return len(self.metadata)

    def _process_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        data = np.zeros((self.max_frames, 33, 3), dtype=np.float32)
        frames = sorted(df["frame_number"].unique())
        n = len(frames)

        for i, fn in enumerate(frames):
            if i >= self.max_frames:
                break
            fd = df[df["frame_number"] == fn][["x", "y", "z"]].values
            if len(fd) == 33:
                hip_c = (fd[self.LEFT_HIP] + fd[self.RIGHT_HIP]) * 0.5
                nd = fd - hip_c
                md = np.max(np.linalg.norm(nd, axis=1))
                if md > 1e-6:
                    nd /= md
                data[i] = nd

        data = np.nan_to_num(data, nan=0.0)
        valid = min(n, self.max_frames)
        mask = torch.tensor(
            [1]*valid + [0]*max(0, self.max_frames - valid),
            dtype=torch.bool
        )
        return torch.tensor(data, dtype=torch.float32), mask

    def __getitem__(self, idx):
        return self.data_cache[idx]


# ==============================================================================
# === MODEL COMPLEXITY ==========================================================
# ==============================================================================

def calculate_complexity(model, device, num_frames=150, num_nodes=33):
    if profile is None:
        return {"params_m": None, "flops_g": None, "fps": None}
    try:
        model.eval()
        dxb = torch.randn(1, num_frames, num_nodes, 3).to(device)
        dmb = torch.ones(1, num_frames, dtype=torch.bool).to(device)
        dxs = torch.randn(1, num_frames, num_nodes, 3).to(device)
        dms = torch.ones(1, num_frames, dtype=torch.bool).to(device)

        macs, params = profile(
            model, inputs=(dxb, dmb, dxs, dms), verbose=False)
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
                    s.record(); _ = model(dxb, dmb, dxs, dms); e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
                else:
                    t0 = time.time()
                    _ = model(dxb, dmb, dxs, dms)
                    times.append((time.time() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        return {"params_m": params/1e6,
                "flops_g": flops/1e9,
                "fps": 1000.0/avg_ms}
    except Exception as ex:
        print(f"Kompleksitas gagal: {ex}")
        return {"params_m": None, "flops_g": None, "fps": None}


# ==============================================================================
# === TRAINING & EVALUATION =====================================================
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for lb, mb, ls, ms, labels in loader:
        lb, mb, ls, ms, labels = (x.to(device) for x in
                                   [lb, mb, ls, ms, labels])
        optimizer.zero_grad()
        loss = criterion(model(lb, mb, ls, ms), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total, preds_all, labels_all = 0.0, [], []
    with torch.no_grad():
        for lb, mb, ls, ms, labels in loader:
            lb, mb, ls, ms, labels = (x.to(device) for x in
                                       [lb, mb, ls, ms, labels])
            out  = model(lb, mb, ls, ms)
            total += criterion(out, labels).item()
            preds_all.extend(torch.argmax(out, 1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total / len(loader), acc, f1


# ==============================================================================
# === MAIN ======================================================================
# ==============================================================================

if __name__ == "__main__":

    ROOT_DIR     = "Dataset_AQA_CSV"
    NUM_EPOCHS   = 40
    BATCH_SIZE   = 8
    LR           = 1e-4
    NUM_FRAMES   = 150
    NUM_CLASSES  = 4
    BASE_CH      = 64    # baseline channel width — tuned for ~0.35-0.45M params
    NUM_BLOCKS   = 4     # 4 CTR-GCN blocks
    DROPOUT      = 0.1

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("CTR-GCN MULTI-VIEW — 5-SEED HEAD-TO-HEAD PROTOCOL")
    print("Chen et al., Channel-wise Topology Refinement GCN, ICCV 2021")
    print("=" * 70)
    print(f"Device       : {DEVICE}")
    print(f"Seeds        : {SEEDS}")
    print(f"Architecture : {NUM_BLOCKS} CTR-GCN blocks, base_ch={BASE_CH}")
    print(f"Nodes        : {NUM_NODES} (MediaPipe 33-landmark graph)")
    print(f"Fusion       : Late concat (back + side), identik dengan ST-GCN/MVST")
    print("=" * 70 + "\n")

    full_dataset = MultiViewLandmarkDataset(ROOT_DIR, max_frames=NUM_FRAMES)

    # Model complexity (satu kali, di seed pertama)
    set_global_seed(SEEDS[0])
    tmp = CTRGCN_MultiView(NUM_CLASSES, NUM_NODES, 3,
                           BASE_CH, NUM_BLOCKS, DROPOUT).to(DEVICE)
    cmplx = calculate_complexity(tmp, DEVICE, NUM_FRAMES, NUM_NODES)
    print(f"\nKompleksitas CTR-GCN Multi-View:")
    print(f"  Params (M)  : {cmplx['params_m']}")
    print(f"  FLOPs (G)   : {cmplx['flops_g']}")
    print(f"  FPS         : {cmplx['fps']}")
    del tmp
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    TRAINED = ["A", "B", "C", "H"]
    NOVICE  = ["E", "F", "G", "S"]
    paired_folds = list(zip(TRAINED, NOVICE))

    all_results = []

    # ============================ LOOP SEED ============================
    for seed in SEEDS:

        print("\n" + "#" * 70)
        print(f"SEED: {seed}")
        print("#" * 70)

        set_global_seed(seed)
        g = torch.Generator(); g.manual_seed(seed)

        for fold, (val_t, val_n) in enumerate(paired_folds, 1):

            set_global_seed(seed + fold)

            val_subj = [val_t, val_n]
            print(f"\n{'='*70}")
            print(f"SEED {seed} | FOLD {fold}/4 | "
                  f"Val: {val_t} (Pakar) & {val_n} (Pemula)")
            print(f"[SEED] fold seed = {seed + fold}")
            print("="*70)

            tr_idx = full_dataset.metadata[
                ~full_dataset.metadata["subject_id"].isin(val_subj)
            ].index.tolist()
            va_idx = full_dataset.metadata[
                full_dataset.metadata["subject_id"].isin(val_subj)
            ].index.tolist()

            tr_loader = DataLoader(
                Subset(full_dataset, tr_idx),
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, worker_init_fn=seed_worker, generator=g)
            va_loader = DataLoader(
                Subset(full_dataset, va_idx),
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, worker_init_fn=seed_worker, generator=g)

            model = CTRGCN_MultiView(
                NUM_CLASSES, NUM_NODES, 3,
                BASE_CH, NUM_BLOCKS, DROPOUT).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=LR)

            best_acc, best_f1, best_ep = 0.0, 0.0, 0
            final_acc, final_f1 = 0.0, 0.0

            for epoch in range(1, NUM_EPOCHS + 1):
                tr_loss = train_one_epoch(
                    model, tr_loader, criterion, optimizer, DEVICE)
                _, va_acc, va_f1 = evaluate(
                    model, va_loader, criterion, DEVICE)

                if va_f1 > best_f1:
                    best_acc, best_f1, best_ep = va_acc, va_f1, epoch

                final_acc, final_f1 = va_acc, va_f1

                print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
                      f"TrLoss:{tr_loss:.4f} | "
                      f"ValAcc:{va_acc:.4f} | ValF1:{va_f1:.4f}")

            print(f"\n→ Best: Acc={best_acc:.4f} F1={best_f1:.4f} "
                  f"Ep={best_ep} | "
                  f"Final: Acc={final_acc:.4f} F1={final_f1:.4f}")

            all_results.append({
                "model": "ctrgcn_multiview",
                "seed": seed, "fold_seed": seed + fold, "fold": fold,
                "val_subjects": f"{val_t} & {val_n}",
                "best_epoch": best_ep,
                "accuracy": best_acc, "f1_score": best_f1,
                "final_epoch_accuracy": final_acc,
                "final_epoch_f1": final_f1,
            })

            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        # Per-seed summary
        seed_r = [r for r in all_results if r["seed"] == seed]
        accs = [r["accuracy"] for r in seed_r]
        f1s  = [r["f1_score"] for r in seed_r]
        print(f"\n--- SUMMARY SEED {seed}: "
              f"Acc {np.mean(accs):.4f}±{np.std(accs):.4f} | "
              f"F1 {np.mean(f1s):.4f}±{np.std(f1s):.4f} ---")

    # ========================= SIMPAN & AGREGAT ========================
    df = pd.DataFrame(all_results)
    df.to_csv("ctrgcn_5seed_4fold_results.csv", index=False)

    per_seed = df.groupby("seed").agg(
        acc_mean=("accuracy","mean"), acc_std=("accuracy","std"),
        f1_mean=("f1_score","mean"),  f1_std=("f1_score","std"),
        final_acc_mean=("final_epoch_accuracy","mean"),
        final_f1_mean=("final_epoch_f1","mean"),
    ).reset_index()
    per_seed.to_csv("ctrgcn_5seed_per_seed_summary.csv", index=False)

    pd.DataFrame([{**cmplx, "model": "ctrgcn_multiview"}]).to_csv(
        "ctrgcn_complexity.csv", index=False)

    print("\n" + "=" * 70)
    print("PER-SEED SUMMARY: CTR-GCN MULTI-VIEW")
    print("=" * 70)
    print(per_seed.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"AGREGAT FINAL CTR-GCN ({len(SEEDS)} seeds × 4 folds = {len(df)} obs)")
    print("=" * 70)
    print(f"[Best-epoch ] Acc: {df['accuracy'].mean():.4f} ± "
          f"{df['accuracy'].std():.4f} | "
          f"F1: {df['f1_score'].mean():.4f} ± "
          f"{df['f1_score'].std():.4f}")
    print(f"[Final-epoch] Acc: {df['final_epoch_accuracy'].mean():.4f} ± "
          f"{df['final_epoch_accuracy'].std():.4f} | "
          f"F1: {df['final_epoch_f1'].mean():.4f} ± "
          f"{df['final_epoch_f1'].std():.4f}")

    print("\nFile hasil:")
    print("1. ctrgcn_5seed_4fold_results.csv      (fold-level, uji statistik)")
    print("2. ctrgcn_5seed_per_seed_summary.csv")
    print("3. ctrgcn_complexity.csv")