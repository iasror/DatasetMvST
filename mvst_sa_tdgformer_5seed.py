# ==============================================================================
# === SA-TDGFormer (Two-stream spatio-temporal GCN-Transformer) — MULTI-VIEW ===
# === Referensi: D. Chen, M. Chen, P. Wu, M. Wu, T. Zhang, C. Li,            ===
# ===   "Two-stream spatio-temporal GCN-transformer networks for skeleton-   ===
# ===   based action recognition", Scientific Reports 15, 4982 (2025),       ===
# ===   doi: 10.1038/s41598-025-87752-8. Tidak ada kode publik; reimplementasi===
# ===   dari deskripsi paper (Fig. 1, Fig. 2, Eq. AGCN/TDCN/ST-TT, Sec. Late ===
# ===   fusion).                                                             ===
# ===                                                                        ===
# === Mengikuti paper:                                                       ===
# ===   - Gnet: 9 blok [AGCN -> TDCN]. AGCN: A_tilde (kerangka fisik) + L    ===
# ===     (dipelajari penuh) + E (per-sampel, bergantung data), dengan       ===
# ===     residual. Spatial GCN = Conv -> BN -> ReLU (inset Fig. 1).         ===
# ===   - TDCN (Fig. 2): cabang paralel [1x1 -> 3x1 dilasi D -> BN] untuk    ===
# ===     D = 1,2,3,4 + cabang [1x1 -> 3x1 max-pool -> BN], concat,          ===
# ===     + residual 1x1, ReLU.                                              ===
# ===   - Tnet: 9 layer; tiap layer = Spatial Transformer (antar-joint per   ===
# ===     frame) -> Temporal Transformer (per joint antar-frame).            ===
# ===   - Late fusion: O = alpha*G(V) + beta*T(V); kedua stream DILATIH      ===
# ===     TERPISAH, skor digabung setelah training.                          ===
# ===   - Input joint-only (konfigurasi yang juga dilaporkan paper, Tabel 4).===
# ===                                                                        ===
# === Keputusan yang dikonfirmasi user (2026-09-23), karena paper tidak     ===
# === menyebut / untuk konsistensi protokol:                                 ===
# ===   - Channel GCN: 64 di semua 9 blok (opsi 1a).                         ===
# ===   - Stream dilatih terpisah (opsi 2b).                                 ===
# ===   - Tanpa label smoothing (opsi 3a), sama dengan baseline lain.        ===
# ===   - alpha = beta = 0.5 (nilai tidak disebut di paper); fusion pada     ===
# ===     probabilitas softmax.                                              ===
# ===   - Tnet: d = 128, 4 head, FFN 256 (paper tidak menyebut).             ===
# ===   - Optimizer & jadwal sama dengan protokol manuskrip: AdamW 1e-4,     ===
# ===     40 epoch, batch 8, 5 seed x 4 fold subject-independent.            ===
# ===   - Multi-view: cabang independen per view (tidak berbagi bobot),      ===
# ===     fitur back & side di-concat sebelum classifier tiap stream.        ===
# ===   - Pooling: masked mean atas frame valid, mean atas joint.            ===
# ===   - Best-epoch fused = epoch dengan macro-F1 fused tertinggi (indeks   ===
# ===     epoch yang sama untuk kedua stream); final = epoch 40.            ===
# ==============================================================================

import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

try:
    from thop import profile
except ImportError:
    print("Warning: 'thop' belum terinstall. 'pip install thop' untuk FLOPs versi thop.")
    profile = None

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    FlopCounterMode = None


# ==============================================================================
# === SEED PROTOCOL (identik dengan script ST-GCN/MVST) =========================
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
# === DATASET (identik dengan versi ST-GCN/MVST) ================================
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
# === KERANGKA MEDIAPIPE POSE (33 landmark) =====================================
# ==============================================================================
# Daftar koneksi = mediapipe.solutions.pose.POSE_CONNECTIONS (35 edge).
# CATATAN: pastikan daftar ini sama dengan yang dipakai di script CTR-GCN.

MEDIAPIPE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def build_normalized_adjacency(num_node=33, edges=MEDIAPIPE_EDGES):
    """A_tilde = D^-1/2 (A + I) D^-1/2 dari kerangka fisik."""
    A = torch.zeros(num_node, num_node)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(num_node)
    d_inv_sqrt = A.sum(dim=1).pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


# ==============================================================================
# === STREAM 1: Gnet = 9 x [AGCN -> TDCN] =======================================
# ==============================================================================

class AGCNUnit(nn.Module):
    """H' = ReLU( BN( W * (H x (A_tilde + L + E)) ) + residual )."""
    def __init__(self, in_channels, out_channels, A_norm, inter_ratio=4):
        super().__init__()
        num_node = A_norm.shape[0]
        self.register_buffer("A", A_norm.clone())
        self.L = nn.Parameter(torch.zeros(num_node, num_node))     # dipelajari penuh
        self.inter = max(out_channels // inter_ratio, 4)
        self.theta = nn.Conv2d(in_channels, self.inter, kernel_size=1)  # untuk E
        self.phi = nn.Conv2d(in_channels, self.inter, kernel_size=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        th = self.theta(x).permute(0, 3, 1, 2).reshape(N, V, self.inter * T)
        ph = self.phi(x).reshape(N, self.inter * T, V)
        E = torch.softmax(torch.bmm(th, ph) / (self.inter * T), dim=-1)  # (N, V, V)
        A_eff = self.A.unsqueeze(0) + self.L.unsqueeze(0) + E           # (N, V, V)
        xa = torch.einsum("nctv,nvw->nctw", x, A_eff)
        y = self.bn(self.conv(xa))
        return self.relu(y + self.residual(x))


class TDCN(nn.Module):
    """Fig. 2: cabang [1x1 -> 3x1 dilasi D -> BN] untuk D in dilations,
    + cabang [1x1 -> 3x1 max-pool -> BN]; concat; + residual 1x1; ReLU."""
    def __init__(self, channels, dilations=(1, 2, 3, 4)):
        super().__init__()
        n_branch = len(dilations) + 1
        bc = channels // n_branch
        last_bc = channels - bc * len(dilations)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, bc, kernel_size=1),
                nn.Conv2d(bc, bc, kernel_size=(3, 1), padding=(d, 0), dilation=(d, 1)),
                nn.BatchNorm2d(bc),
            ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(channels, last_bc, kernel_size=1),
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(last_bc),
        ))
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.relu(out + self.residual(x))


class GnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A_norm):
        super().__init__()
        self.agcn = AGCNUnit(in_channels, out_channels, A_norm)
        self.tdcn = TDCN(out_channels)

    def forward(self, x):
        return self.tdcn(self.agcn(x))


class GnetView(nn.Module):
    """Satu view: data BN -> 9 blok -> masked mean (T) & mean (V)."""
    def __init__(self, in_channels=3, hidden=64, num_node=33, n_blocks=9):
        super().__init__()
        A_norm = build_normalized_adjacency(num_node)
        self.data_bn = nn.BatchNorm1d(in_channels * num_node)
        blocks = [GnetBlock(in_channels, hidden, A_norm)]
        for _ in range(n_blocks - 1):
            blocks.append(GnetBlock(hidden, hidden, A_norm))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, mask):
        # x: (N, T, V, C) ; mask: (N, T) bool
        N, T, V, C = x.shape
        h = x.permute(0, 3, 2, 1).reshape(N, C * V, T)       # (N, C*V, T)
        h = self.data_bn(h)
        h = h.reshape(N, C, V, T).permute(0, 1, 3, 2)        # (N, C, T, V)
        for blk in self.blocks:
            h = blk(h)
        m = mask.float().unsqueeze(1).unsqueeze(-1)           # (N, 1, T, 1)
        h = (h * m).sum(dim=2) / m.sum(dim=2).clamp(min=1e-6)  # (N, C, V)
        return h.mean(dim=-1)                                 # (N, C)


class Gnet_MultiView(nn.Module):
    def __init__(self, num_class=4, num_node=33, in_channels=3, hidden=64, n_blocks=9):
        super().__init__()
        self.back = GnetView(in_channels, hidden, num_node, n_blocks)
        self.side = GnetView(in_channels, hidden, num_node, n_blocks)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden * 2, num_class))

    def forward(self, x_back, mask_back, x_side, mask_side):
        f = torch.cat([self.back(x_back, mask_back), self.side(x_side, mask_side)], dim=1)
        return self.classifier(f)


# ==============================================================================
# === STREAM 2: Tnet = 9 x [Spatial Transformer -> Temporal Transformer] ========
# ==============================================================================

class STTLayer(nn.Module):
    """Satu layer: H' = TT(ST(H))."""
    def __init__(self, d_model=128, n_heads=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.spatial = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.temporal = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)

    def forward(self, h, key_padding_mask):
        # h: (N, T, V, d) ; key_padding_mask: (N*V, T), True = abaikan
        N, T, V, d = h.shape
        h = self.spatial(h.reshape(N * T, V, d)).reshape(N, T, V, d)
        h = h.permute(0, 2, 1, 3).reshape(N * V, T, d)
        h = self.temporal(h, src_key_padding_mask=key_padding_mask)
        return h.reshape(N, V, T, d).permute(0, 2, 1, 3)


class TnetView(nn.Module):
    def __init__(self, in_channels=3, d_model=128, n_heads=4, dim_ff=256,
                 n_layers=9, num_node=33, max_frames=150, grad_checkpoint=False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.embed = nn.Linear(in_channels, d_model)
        self.pos_spatial = nn.Parameter(torch.zeros(1, 1, num_node, d_model))
        self.pos_temporal = nn.Parameter(torch.zeros(1, max_frames, 1, d_model))
        nn.init.trunc_normal_(self.pos_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_temporal, std=0.02)
        self.layers = nn.ModuleList(
            [STTLayer(d_model, n_heads, dim_ff) for _ in range(n_layers)])

    def forward(self, x, mask):
        # x: (N, T, V, 3) ; mask: (N, T) bool, True = frame valid
        N, T, V, _ = x.shape
        h = self.embed(x) + self.pos_spatial + self.pos_temporal[:, :T]
        kpm = (~mask).unsqueeze(1).expand(N, V, T).reshape(N * V, T)
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(layer, h, kpm, use_reentrant=False)
            else:
                h = layer(h, kpm)
        m = mask.float().unsqueeze(-1).unsqueeze(-1)          # (N, T, 1, 1)
        h = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)  # (N, V, d)
        return h.mean(dim=1)                                  # (N, d)


class Tnet_MultiView(nn.Module):
    def __init__(self, num_class=4, num_node=33, in_channels=3, d_model=128,
                 n_heads=4, dim_ff=256, n_layers=9, max_frames=150, grad_checkpoint=False):
        super().__init__()
        self.back = TnetView(in_channels, d_model, n_heads, dim_ff, n_layers, num_node,
                             max_frames, grad_checkpoint)
        self.side = TnetView(in_channels, d_model, n_heads, dim_ff, n_layers, num_node,
                             max_frames, grad_checkpoint)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_model * 2, num_class))

    def forward(self, x_back, mask_back, x_side, mask_side):
        f = torch.cat([self.back(x_back, mask_back), self.side(x_side, mask_side)], dim=1)
        return self.classifier(f)


# ==============================================================================
# === Pembungkus inferensi (hanya untuk mengukur kompleksitas gabungan) =========
# ==============================================================================

class SATDGFormer_Inference(nn.Module):
    def __init__(self, gnet, tnet, alpha=0.5, beta=0.5):
        super().__init__()
        self.gnet, self.tnet = gnet, tnet
        self.alpha, self.beta = alpha, beta

    def forward(self, xb, mb, xs, ms):
        pg = torch.softmax(self.gnet(xb, mb, xs, ms), dim=1)
        pt = torch.softmax(self.tnet(xb, mb, xs, ms), dim=1)
        return self.alpha * pg + self.beta * pt


def calculate_model_complexity(model, device, num_frames=150, num_landmarks=33):
    """Mengembalikan:
    - params_m_thop / flops_g_thop : cara ukur yang sama dengan baseline lain (thop).
    - params_m_true                : sum(p.numel()) semua parameter.
    - flops_g_torch                : torch.utils.flop_counter (menghitung juga
                                     matmul attention, bmm, einsum).
    - inference_ms / fps           : batch 1, rata-rata 100 forward."""
    out = {"params_m_thop": None, "flops_g_thop": None,
           "params_m_true": sum(p.numel() for p in model.parameters()) / 1e6,
           "flops_g_torch": None, "inference_ms": None, "fps": None}
    model.eval()
    dxb = torch.randn(1, num_frames, num_landmarks, 3).to(device)
    dmb = torch.ones(1, num_frames, dtype=torch.bool).to(device)
    dxs = torch.randn(1, num_frames, num_landmarks, 3).to(device)
    dms = torch.ones(1, num_frames, dtype=torch.bool).to(device)
    inputs = (dxb, dmb, dxs, dms)

    if FlopCounterMode is not None:
        try:
            import copy
            # Mode train (tanpa gradien) supaya nn.TransformerEncoderLayer tidak memakai
            # "fast path" inferensi yang tidak terbaca oleh FlopCounterMode.
            m_fc = copy.deepcopy(model).train()
            fc = FlopCounterMode(display=False)
            with fc, torch.no_grad():
                m_fc(*inputs)
            out["flops_g_torch"] = fc.get_total_flops() / 1e9
            del m_fc
        except Exception as ex:
            print(f"FlopCounterMode gagal: {ex}")

    if profile is not None:
        try:
            import copy
            m_copy = copy.deepcopy(model)   # thop menambah buffer ke modul
            macs, params = profile(m_copy, inputs=inputs, verbose=False)
            out["params_m_thop"] = params / 1e6
            out["flops_g_thop"] = macs * 2 / 1e9
            del m_copy
        except Exception as ex:
            print(f"thop gagal: {ex}")

    try:
        with torch.no_grad():
            for _ in range(10):
                model(*inputs)
        times = []
        with torch.no_grad():
            for _ in range(100):
                if device.type == "cuda":
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    model(*inputs)
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
                else:
                    t0 = time.time()
                    model(*inputs)
                    times.append((time.time() - t0) * 1000)
        out["inference_ms"] = sum(times) / len(times)
        out["fps"] = 1000.0 / out["inference_ms"]
    except Exception as ex:
        print(f"Pengukuran waktu gagal: {ex}")
    return out


# ==============================================================================
# === TRAINING & EVALUATION =====================================================
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in dataloader:
        lb, mb, ls, ms, labels = [d.to(device) for d in data]
        optimizer.zero_grad()
        loss = criterion(model(lb, mb, ls, ms), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def predict_probs(model, dataloader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for data in dataloader:
            lb, mb, ls, ms, y = [d.to(device) for d in data]
            probs.append(torch.softmax(model(lb, mb, ls, ms), dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics_from_probs(probs, labels):
    preds = probs.argmax(axis=1)
    return (accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro", zero_division=0))


def best_and_final(per_epoch_probs, labels):
    """Best = epoch dengan macro-F1 tertinggi (kemunculan pertama, sama dengan
    aturan baseline lain: update hanya jika F1 > sebelumnya). Final = epoch terakhir."""
    best = (0.0, 0.0, 0)  # acc, f1, epoch
    for ep, p in enumerate(per_epoch_probs, 1):
        acc, f1 = metrics_from_probs(p, labels)
        if f1 > best[1]:
            best = (acc, f1, ep)
    final_acc, final_f1 = metrics_from_probs(per_epoch_probs[-1], labels)
    return best, (final_acc, final_f1)


def train_stream(stream_name, model_fn, full_dataset, train_idx, val_idx,
                 fold_seed, num_epochs, batch_size, lr, device):
    """Latih satu stream dari nol; kembalikan list probabilitas val per epoch."""
    set_global_seed(fold_seed)
    g = torch.Generator()
    g.manual_seed(fold_seed)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=0)
    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    per_epoch_probs, labels = [], None
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        probs, labels = predict_probs(model, val_loader, device)
        per_epoch_probs.append(probs)
        acc, f1 = metrics_from_probs(probs, labels)
        print(f"  [{stream_name}] Epoch [{epoch:02d}/{num_epochs}] "
              f"Train Loss: {tr_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return per_epoch_probs, labels


# ==============================================================================
# === MAIN ======================================================================
# ==============================================================================

def run(root_dir="Dataset_AQA_CSV", num_epochs=40, batch_size=8, lr=1e-4,
        num_frames=150, num_classes=4, seeds=SEEDS, alpha=0.5, beta=0.5,
        out_prefix="sa_tdgformer", probs_dir="sa_tdgformer_val_probs",
        grad_checkpoint=False):
    # grad_checkpoint=True: hemat memori GPU untuk Tnet (hitung ulang aktivasi saat
    # backward). Hasil tidak berubah, hanya lebih lambat. Pakai jika muncul CUDA OOM.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_csv = f"{out_prefix}_5seed_4fold_results.csv"
    Path(probs_dir).mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("SA-TDGFormer MULTI-VIEW (Gnet + Tnet, dilatih terpisah) — 5-SEED PROTOCOL")
    print("=" * 70)
    print(f"Device : {device}")
    print(f"Seeds  : {seeds}")
    print(f"Fusion : {alpha} * softmax(Gnet) + {beta} * softmax(Tnet)")
    print("=" * 70 + "\n")

    full_dataset = MultiViewLandmarkDataset(root_dir, max_frames=num_frames)

    gnet_fn = lambda: Gnet_MultiView(num_class=num_classes, num_node=33, in_channels=3,
                                     hidden=64, n_blocks=9)
    tnet_fn = lambda: Tnet_MultiView(num_class=num_classes, num_node=33, in_channels=3,
                                     d_model=128, n_heads=4, dim_ff=256, n_layers=9,
                                     max_frames=num_frames, grad_checkpoint=grad_checkpoint)

    # ---- Kompleksitas (Gnet, Tnet, gabungan); dilewati saat resume ----
    complexity_csv = f"{out_prefix}_model_complexity.csv"
    if Path(complexity_csv).exists():
        print(f"Kompleksitas sudah ada di {complexity_csv}, tidak dihitung ulang.")
    else:
        set_global_seed(seeds[0])
        rows = []
        g_tmp, t_tmp = gnet_fn().to(device), tnet_fn().to(device)
        for name, mdl in [("gnet", g_tmp), ("tnet", t_tmp),
                          ("sa_tdgformer_fused", SATDGFormer_Inference(g_tmp, t_tmp, alpha, beta))]:
            c = calculate_model_complexity(mdl, device, num_frames, 33)
            c["model"] = name
            rows.append(c)
            print(f"Kompleksitas {name}: {c}")
        pd.DataFrame(rows).to_csv(complexity_csv, index=False)
        del g_tmp, t_tmp
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Resume: lewati seed-fold yang sudah selesai ----
    done = set()
    all_results = []
    if Path(results_csv).exists():
        prev = pd.read_csv(results_csv)
        all_results = prev.to_dict("records")
        done = {(int(r["seed"]), int(r["fold"])) for r in all_results}
        print(f"Resume: {len(done)} seed-fold sudah selesai, dilewati.")

    trained_subjects = ["A", "B", "C", "H"]
    novice_subjects = ["E", "F", "G", "S"]
    paired_folds = list(zip(trained_subjects, novice_subjects))

    for seed in seeds:
        for fold, (val_trained, val_novice) in enumerate(paired_folds, 1):
            if (seed, fold) in done:
                continue
            fold_seed = seed + fold
            val_subjects = [val_trained, val_novice]
            print("\n" + "=" * 70)
            print(f"SEED {seed} | FOLD {fold}/4 | Validasi: {val_trained} & {val_novice} "
                  f"| fold seed = {fold_seed}")
            print("=" * 70)

            md = full_dataset.metadata
            train_idx = md[~md["subject_id"].isin(val_subjects)].index.tolist()
            val_idx = md[md["subject_id"].isin(val_subjects)].index.tolist()

            g_probs, labels_g = train_stream("Gnet", gnet_fn, full_dataset, train_idx, val_idx,
                                             fold_seed, num_epochs, batch_size, lr, device)
            t_probs, labels_t = train_stream("Tnet", tnet_fn, full_dataset, train_idx, val_idx,
                                             fold_seed, num_epochs, batch_size, lr, device)
            assert np.array_equal(labels_g, labels_t), "Urutan label val tidak sama antar stream"
            labels = labels_g

            fused = [alpha * pg + beta * pt for pg, pt in zip(g_probs, t_probs)]
            (b_acc, b_f1, b_ep), (f_acc, f_f1) = best_and_final(fused, labels)
            (gb_acc, gb_f1, gb_ep), (gf_acc, gf_f1) = best_and_final(g_probs, labels)
            (tb_acc, tb_f1, tb_ep), (tf_acc, tf_f1) = best_and_final(t_probs, labels)

            np.savez_compressed(Path(probs_dir) / f"seed{seed}_fold{fold}.npz",
                                gnet=np.stack(g_probs), tnet=np.stack(t_probs), labels=labels)

            print(f"\n[Fused] Best Epoch {b_ep} | Acc {b_acc:.4f} | F1 {b_f1:.4f} "
                  f"|| Final Acc {f_acc:.4f} | F1 {f_f1:.4f}")
            print(f"[Gnet ] Final Acc {gf_acc:.4f} | [Tnet ] Final Acc {tf_acc:.4f}")

            all_results.append({
                "model": "sa_tdgformer_multiview",
                "seed": seed, "fold_seed": fold_seed, "fold": fold,
                "val_subjects": f"{val_trained} & {val_novice}",
                "best_epoch": b_ep, "accuracy": b_acc, "f1_score": b_f1,
                "final_epoch_accuracy": f_acc, "final_epoch_f1": f_f1,
                "gnet_best_epoch": gb_ep, "gnet_accuracy": gb_acc, "gnet_f1_score": gb_f1,
                "gnet_final_epoch_accuracy": gf_acc, "gnet_final_epoch_f1": gf_f1,
                "tnet_best_epoch": tb_ep, "tnet_accuracy": tb_acc, "tnet_f1_score": tb_f1,
                "tnet_final_epoch_accuracy": tf_acc, "tnet_final_epoch_f1": tf_f1,
            })
            pd.DataFrame(all_results).to_csv(results_csv, index=False)  # simpan tiap fold

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print(f"AGREGAT SA-TDGFormer ({len(df)} observasi seed-fold)")
    print("=" * 70)
    for tag, pre in [("Fused", ""), ("Gnet ", "gnet_"), ("Tnet ", "tnet_")]:
        print(f"[{tag}] Best Acc {df[pre+'accuracy'].mean():.4f} ± {df[pre+'accuracy'].std():.4f} | "
              f"Final Acc {df[pre+'final_epoch_accuracy'].mean():.4f} ± "
              f"{df[pre+'final_epoch_accuracy'].std():.4f}")
    print("\nFile hasil:")
    print(f"1. {results_csv}")
    print(f"2. {out_prefix}_model_complexity.csv")
    print(f"3. {probs_dir}/seed*_fold*.npz  (probabilitas val per epoch, per stream)")
    return df


if __name__ == "__main__":
    run()