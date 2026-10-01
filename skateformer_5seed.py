# ==============================================================================
# === SkateFormer (ECCV 2024) — Adapted for MediaPipe 33-Landmark ==============
# === Do & Kim, "Skeletal-Temporal Transformer for Human Action Recognition"
# === ECCV 2024, pp.401-420.
# === Multi-view (back + side), late fusion concat — identik dengan baseline lain
# === Protokol: 5-seed × 4-fold, best-epoch + final-epoch
# ==============================================================================
#
# LAPORAN PERUBAHAN vs paper asli:
# 1. 33 MediaPipe nodes (bukan 25 NTU/COCO), partisi anatomis dari dokumentasi resmi
# 2. 4 skate-type = 4 partition combinations (joint-local, joint-distant,
#    frame-local, frame-distant) sesuai paper
# 3. Tidak ada augmentasi, tidak ada pretrained weights — identik protokol baseline
# 4. Multi-view: dua branch independen + late concat fusion
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
import random, os, time, math

try:
    from thop import profile
except ImportError:
    print("Warning: 'thop' belum terinstall.")
    profile = None


# ==============================================================================
# === SEED PROTOCOL =============================================================
# ==============================================================================

SEEDS = [42, 7, 123, 777, 2024]

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws); random.seed(ws)


# ==============================================================================
# === MEDIAPIPE 33-LANDMARK ANATOMICAL PARTITIONS ==============================
# Berdasarkan: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
#
# 0  nose          1  l.eye(i)      2  l.eye         3  l.eye(o)
# 4  r.eye(i)      5  r.eye         6  r.eye(o)      7  l.ear
# 8  r.ear         9  mouth(l)     10  mouth(r)
# 11 l.shoulder   12 r.shoulder   13 l.elbow        14 r.elbow
# 15 l.wrist      16 r.wrist      17 l.pinky        18 r.pinky
# 19 l.index      20 r.index      21 l.thumb        22 r.thumb
# 23 l.hip        24 r.hip        25 l.knee         26 r.knee
# 27 l.ankle      28 r.ankle      29 l.heel         30 r.heel
# 31 l.foot.idx   32 r.foot.idx
# ==============================================================================

NUM_NODES = 33

# ── Neighboring joint partitions (Skate-Type 1 & 2) ──────────────────────────
# G1: Face/Head  — landmarks yang bergerak bersama secara anatomis
# G2: Upper Arm  — bahu dan siku (kunci untuk analisis swing)
# G3: Hand+Torso — pergelangan, jari, pinggul
# G4: Lower Body — lutut, pergelangan kaki, telapak kaki
G1 = list(range(0, 11))                    # [0-10]  face & head (11 joints)
G2 = [11, 12, 13, 14]                      # [11-14] shoulders & elbows (4)
G3 = [15,16,17,18,19,20,21,22,23,24]       # [15-24] wrists, fingers, hips (10)
G4 = [25,26,27,28,29,30,31,32]             # [25-32] knees, ankles, feet (8)

# ── Distant joint pairs (Skate-Type 2) ───────────────────────────────────────
# "Distant" = biomekanik distant (bukan tetangga anatomis langsung)
# Untuk tennis: kombinasi lintas grup yang relevan
DISTANT_GROUPS = [
    G1 + G4,                        # face+head ↔ lower body (full body tension)
    G2 + G4,                        # shoulders/elbows ↔ legs (stroke power chain)
    G1 + G2,                        # head ↔ shoulders (gaze+shoulder alignment)
    G3 + G4,                        # hands/torso ↔ legs (hip-wrist chain)
]

ALL_JOINT_PARTITIONS = [G1, G2, G3, G4]   # Skate-Type 1
ALL_DISTANT_PARTITIONS = DISTANT_GROUPS    # Skate-Type 2

# ── Edge list (untuk positional encoding berbasis graph) ─────────────────────
MEDIAPIPE_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]


# ==============================================================================
# === SKATE-MSA: PARTITION-SPECIFIC MULTI-HEAD SELF-ATTENTION ==================
# ==============================================================================

class SkateAttention(nn.Module):
    """
    Partition-specific attention untuk satu Skate-Type.
    Melakukan self-attention HANYA di dalam partisi (subset joint × subset frame),
    bukan ke seluruh (33 × T). Ini adalah efisiensi kunci SkateFormer.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) where N = subset of (joint × time) tokens"""
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)


class SkateMSA(nn.Module):
    """
    4 Skate-Type attention dalam satu blok:
      Type 1: neighboring joints × local frames (window W_t)
      Type 2: distant joints × local frames
      Type 3: all joints × global frames (CLS-style temporal global)
      Type 4: neighboring joints × global temporal summary
    Output di-aggregate per token via learned weighted sum.
    """
    def __init__(self, d_model: int, n_heads: int = 4,
                 dropout: float = 0.1, local_window: int = 16):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.local_w  = local_window

        # 4 separate attention heads, one per Skate-Type
        self.attn1 = SkateAttention(d_model, n_heads, dropout)  # neigh.j × local.t
        self.attn2 = SkateAttention(d_model, n_heads, dropout)  # distant.j × local.t
        self.attn3 = SkateAttention(d_model, n_heads, dropout)  # all.j × global.t
        self.attn4 = SkateAttention(d_model, n_heads, dropout)  # neigh.j × global.t

        # Learnable mix weight
        self.mix = nn.Parameter(torch.ones(4) / 4.0)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                joint_partition: list,
                distant_partition: list) -> torch.Tensor:
        """
        x : (B, T, V, D)
        joint_partition   : list of joint indices (neighboring group)
        distant_partition : list of joint indices (distant group)
        Returns: (B, T, V, D)
        """
        B, T, V, D = x.shape
        W = min(self.local_w, T)

        # Helper: gather subset of joints
        def gather(tens, idx):
            return tens[:, :, idx, :]  # (B, T, |idx|, D)

        out = torch.zeros_like(x)
        w = F.softmax(self.mix, dim=0)

        # ── Type 1: neighboring joints × local time windows ──────────────────
        xj = gather(x, joint_partition)                # (B, T, Jn, D)
        # chunk time into local windows
        n_wins = max(1, T // W)
        chunks = xj.split(W, dim=1)                   # list of (B, W', Jn, D)
        out1 = torch.zeros_like(xj)
        start = 0
        for ch in chunks:
            b, w_len, jn, d = ch.shape
            tokens = ch.reshape(b, w_len * jn, d)
            attended = self.attn1(tokens).reshape(b, w_len, jn, d)
            out1[:, start:start+w_len, :, :] = attended
            start += w_len
        out[:, :, joint_partition, :] += w[0] * out1

        # ── Type 2: distant joints × local time windows ───────────────────────
        xd = gather(x, distant_partition)              # (B, T, Jd, D)
        out2 = torch.zeros_like(xd)
        start = 0
        for ch in xd.split(W, dim=1):
            b, w_len, jd, d = ch.shape
            tokens = ch.reshape(b, w_len * jd, d)
            attended = self.attn2(tokens).reshape(b, w_len, jd, d)
            out2[:, start:start+w_len, :, :] = attended
            start += w_len
        out[:, :, distant_partition, :] += w[1] * out2

        # ── Type 3: all joints × global temporal summary ──────────────────────
        # Memory-efficient: use temporal mean as global context (B, V, D),
        # then run attention over V joints only — avoids T*V tokens
        x_t_mean = x.mean(dim=1)                        # (B, V, D)
        out3 = self.attn3(x_t_mean)                     # (B, V, D)
        # Broadcast back to (B, T, V, D)
        out += w[2] * out3.unsqueeze(1).expand(-1, T, -1, -1)

        # ── Type 4: neighboring joints × global temporal ──────────────────────
        # Use temporal mean of neighbor joints → attention over Jn tokens only
        xjg_mean = x_t_mean[:, joint_partition, :]      # (B, Jn, D)
        out4 = self.attn4(xjg_mean)                     # (B, Jn, D)
        # Broadcast to (B, T, Jn, D)
        out[:, :, joint_partition, :] += \
            w[3] * out4.unsqueeze(1).expand(-1, T, -1, -1)

        return self.dropout(self.norm(out + x))  # residual + LayerNorm


class SkateFormerBlock(nn.Module):
    """Single SkateFormer block: SkateMSA + FFN + residuals."""
    def __init__(self, d_model: int, n_heads: int = 4,
                 ff_dim: int = 256, dropout: float = 0.1,
                 local_window: int = 16):
        super().__init__()
        self.skate_msa = SkateMSA(d_model, n_heads, dropout, local_window)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                joint_partition: list,
                distant_partition: list) -> torch.Tensor:
        x = self.skate_msa(x, joint_partition, distant_partition)
        x = x + self.ffn(x)
        return x


# ==============================================================================
# === SKATEFORMER MULTI-VIEW ====================================================
# ==============================================================================

class SkateFormer_MultiView(nn.Module):
    """
    SkateFormer adapted for MediaPipe 33 landmarks, multi-view late concat.
    Two independent branches (back/side), fused via concatenation.
    """
    def __init__(self, num_classes: int = 4,
                 in_channels: int = 3,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_blocks: int = 3,
                 ff_dim: int = 128,
                 dropout: float = 0.1,
                 local_window: int = 16):
        super().__init__()
        self.d_model      = d_model
        self.n_blocks     = n_blocks
        self.local_window = local_window

        # ── Back-view encoder ──────────────────────────────────────────────────
        self.back_embed  = nn.Linear(in_channels, d_model)
        self.back_pos_v  = nn.Parameter(torch.zeros(1, 1, NUM_NODES, d_model))
        self.back_pos_t  = nn.Parameter(torch.zeros(1, 150, 1, d_model))
        self.back_blocks = nn.ModuleList([
            SkateFormerBlock(d_model, n_heads, ff_dim, dropout, local_window)
            for _ in range(n_blocks)
        ])
        self.back_norm = nn.LayerNorm(d_model)

        # ── Side-view encoder (independent parameters) ────────────────────────
        self.side_embed  = nn.Linear(in_channels, d_model)
        self.side_pos_v  = nn.Parameter(torch.zeros(1, 1, NUM_NODES, d_model))
        self.side_pos_t  = nn.Parameter(torch.zeros(1, 150, 1, d_model))
        self.side_blocks = nn.ModuleList([
            SkateFormerBlock(d_model, n_heads, ff_dim, dropout, local_window)
            for _ in range(n_blocks)
        ])
        self.side_norm = nn.LayerNorm(d_model)

        # ── Classifier (late concat fusion) ───────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Store partition indices
        # Rotate through 4 joint groups so each block sees a different group
        self.joint_parts   = ALL_JOINT_PARTITIONS
        self.distant_parts = ALL_DISTANT_PARTITIONS

    def _encode(self, x: torch.Tensor, mask: torch.Tensor,
                embed: nn.Module, pos_v, pos_t,
                blocks: nn.ModuleList,
                norm: nn.Module) -> torch.Tensor:
        """
        x    : (B, T, V, C)
        mask : (B, T)
        Returns: (B, D)
        """
        B, T, V, C = x.shape

        # Embed + positional encoding
        h = embed(x) + pos_v + pos_t[:, :T, :, :]   # (B, T, V, D)

        # SkateFormer blocks — rotate partition groups
        for i, block in enumerate(blocks):
            j_idx = i % len(self.joint_parts)
            d_idx = i % len(self.distant_parts)
            h = block(h,
                      self.joint_parts[j_idx],
                      self.distant_parts[d_idx])

        h = norm(h)   # (B, T, V, D)

        # Masked temporal mean pooling → then joint mean
        mask_f = mask.float().unsqueeze(-1).unsqueeze(-1)   # (B,T,1,1)
        h = h * mask_f
        valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        feat = h.sum(dim=1) / valid.unsqueeze(-1)           # (B, V, D)
        feat = feat.mean(dim=1)                              # (B, D)
        return feat

    def forward(self, x_back, mask_back, x_side, mask_side):
        fb = self._encode(x_back, mask_back,
                          self.back_embed, self.back_pos_v, self.back_pos_t,
                          self.back_blocks, self.back_norm)
        fs = self._encode(x_side, mask_side,
                          self.side_embed, self.side_pos_v, self.side_pos_t,
                          self.side_blocks, self.side_norm)
        return self.classifier(torch.cat([fb, fs], dim=1))


# ==============================================================================
# === DATASET (identik) =========================================================
# ==============================================================================

class MultiViewLandmarkDataset(Dataset):
    def __init__(self, root_dir, max_frames=150):
        self.root_dir  = Path(root_dir)
        self.max_frames= max_frames
        file_pairs = self._find_file_pairs()
        if not file_pairs:
            raise FileNotFoundError(f"Tidak ada paired CSV di {root_dir}.")

        self.metadata = pd.DataFrame(file_pairs)
        self.class_to_idx = {
            "backhand":0,"backhandNewbie":1,"forehand":2,"forehandNewbie":3}
        self.metadata["label"] = self.metadata["class"].map(self.class_to_idx)
        print(f"Ditemukan {len(self.metadata)} sampel paired.")

        self.L_HIP, self.R_HIP = 23, 24
        self.data_cache = []
        print("Memuat data ke RAM...")
        for idx in tqdm(range(len(self.metadata)), desc="Caching"):
            m = self.metadata.iloc[idx]
            lb, mb = self._process_csv(m["back_path"])
            ls, ms = self._process_csv(m["side_path"])
            self.data_cache.append((lb,mb,ls,ms,
                                    torch.tensor(m["label"],dtype=torch.long)))

    def _find_file_pairs(self):
        d = {}
        print("Memindai pasangan file...")
        for cls_dir in self.root_dir.iterdir():
            if not cls_dir.is_dir(): continue
            for csv_f in cls_dir.glob("*.csv"):
                name = csv_f.name
                base = name.replace("_Back.csv","").replace("_Side.csv","")
                try: subj = base.split("_")[1][0]
                except: subj = base.split("_")[0][0]
                key = f"{cls_dir.name}/{base}"
                if key not in d:
                    d[key] = {"class":cls_dir.name,"subject_id":subj}
                if "_Back.csv" in name: d[key]["back_path"] = str(csv_f)
                elif "_Side.csv" in name: d[key]["side_path"] = str(csv_f)
        return [v for v in d.values() if "back_path" in v and "side_path" in v]

    def __len__(self): return len(self.metadata)

    def _process_csv(self, path):
        df = pd.read_csv(path)
        data = np.zeros((self.max_frames, 33, 3), dtype=np.float32)
        frames = sorted(df["frame_number"].unique())
        n = len(frames)
        for i, fn in enumerate(frames):
            if i >= self.max_frames: break
            fd = df[df["frame_number"]==fn][["x","y","z"]].values
            if len(fd) == 33:
                hc = (fd[self.L_HIP]+fd[self.R_HIP])*0.5
                nd = fd - hc
                md = np.max(np.linalg.norm(nd, axis=1))
                if md > 1e-6: nd /= md
                data[i] = nd
        data = np.nan_to_num(data, nan=0.0)
        valid = min(n, self.max_frames)
        mask = torch.tensor([1]*valid+[0]*max(0,self.max_frames-valid),
                            dtype=torch.bool)
        return torch.tensor(data, dtype=torch.float32), mask

    def __getitem__(self, idx): return self.data_cache[idx]


# ==============================================================================
# === COMPLEXITY, TRAINING, EVAL ================================================
# ==============================================================================

def calculate_complexity(model, device, num_frames=150, num_nodes=33):
    if profile is None:
        return {"params_m":None,"flops_g":None,"fps":None}
    try:
        model.eval()
        dxb = torch.randn(1,num_frames,num_nodes,3).to(device)
        dmb = torch.ones(1,num_frames,dtype=torch.bool).to(device)
        dxs = torch.randn(1,num_frames,num_nodes,3).to(device)
        dms = torch.ones(1,num_frames,dtype=torch.bool).to(device)
        macs, params = profile(model, inputs=(dxb,dmb,dxs,dms), verbose=False)
        flops = macs*2
        with torch.no_grad():
            for _ in range(10): _ = model(dxb,dmb,dxs,dms)
        times = []
        with torch.no_grad():
            for _ in range(100):
                if device.type=="cuda":
                    s=torch.cuda.Event(enable_timing=True)
                    e=torch.cuda.Event(enable_timing=True)
                    s.record(); _ = model(dxb,dmb,dxs,dms); e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
                else:
                    t0=time.time(); _ = model(dxb,dmb,dxs,dms)
                    times.append((time.time()-t0)*1000)
        avg_ms = sum(times)/len(times)
        return {"params_m":params/1e6,"flops_g":flops/1e9,"fps":1000/avg_ms}
    except Exception as ex:
        print(f"Kompleksitas gagal: {ex}")
        return {"params_m":None,"flops_g":None,"fps":None}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train(); total=0.0
    for lb,mb,ls,ms,labels in loader:
        lb,mb,ls,ms,labels = (x.to(device) for x in [lb,mb,ls,ms,labels])
        optimizer.zero_grad()
        loss = criterion(model(lb,mb,ls,ms), labels)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total/len(loader)


def evaluate(model, loader, criterion, device):
    model.eval(); total=0.0; preds_all=[]; labels_all=[]
    with torch.no_grad():
        for lb,mb,ls,ms,labels in loader:
            lb,mb,ls,ms,labels = (x.to(device) for x in [lb,mb,ls,ms,labels])
            out = model(lb,mb,ls,ms)
            total += criterion(out, labels).item()
            preds_all.extend(torch.argmax(out,1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total/len(loader), acc, f1


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
    D_MODEL      = 64        # embedding dim (lightweight untuk dataset kecil)
    N_HEADS      = 4
    N_BLOCKS     = 3         # 3 SkateFormer blocks
    FF_DIM       = 128
    DROPOUT      = 0.1
    LOCAL_WINDOW = 16        # local temporal window (T/~9)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("SkateFormer MULTI-VIEW — 5-SEED HEAD-TO-HEAD PROTOCOL")
    print("Do & Kim, Skeletal-Temporal Transformer, ECCV 2024")
    print("=" * 70)
    print(f"Device       : {DEVICE}")
    print(f"Seeds        : {SEEDS}")
    print(f"Architecture : {N_BLOCKS} blocks, d_model={D_MODEL}, "
          f"local_window={LOCAL_WINDOW}")
    print(f"Nodes        : {NUM_NODES} (MediaPipe, 4 anatomical partitions)")
    print(f"Partitions   : G1_face{G1[:3]}.. G2_upper{G2} "
          f"G3_hand+torso{G3[:3]}.. G4_lower{G4[:3]}..")
    print(f"Fusion       : Late concat (back+side), identik dengan baseline lain")
    print("=" * 70 + "\n")

    full_dataset = MultiViewLandmarkDataset(ROOT_DIR, max_frames=NUM_FRAMES)

    # Kompleksitas
    set_global_seed(SEEDS[0])
    tmp = SkateFormer_MultiView(
        NUM_CLASSES, 3, D_MODEL, N_HEADS, N_BLOCKS,
        FF_DIM, DROPOUT, LOCAL_WINDOW).to(DEVICE)
    cmplx = calculate_complexity(tmp, DEVICE, NUM_FRAMES, NUM_NODES)
    print(f"\nKompleksitas SkateFormer Multi-View:")
    print(f"  Params (M) : {cmplx['params_m']}")
    print(f"  FLOPs (G)  : {cmplx['flops_g']}")
    print(f"  FPS        : {cmplx['fps']}")
    del tmp
    if DEVICE.type=="cuda": torch.cuda.empty_cache()

    TRAINED = ["A","B","C","H"]; NOVICE = ["E","F","G","S"]
    paired_folds = list(zip(TRAINED, NOVICE))

    all_results = []

    for seed in SEEDS:
        print("\n" + "#"*70)
        print(f"SEED: {seed}")
        print("#"*70)

        set_global_seed(seed)
        g = torch.Generator(); g.manual_seed(seed)

        for fold, (val_t, val_n) in enumerate(paired_folds, 1):
            set_global_seed(seed + fold)
            val_subj = [val_t, val_n]

            print(f"\n{'='*70}")
            print(f"SEED {seed} | FOLD {fold}/4 | "
                  f"Val: {val_t} (Pakar) & {val_n} (Pemula)")
            print(f"[SEED] fold seed = {seed+fold}")
            print("="*70)

            tr_idx = full_dataset.metadata[
                ~full_dataset.metadata["subject_id"].isin(val_subj)
            ].index.tolist()
            va_idx = full_dataset.metadata[
                full_dataset.metadata["subject_id"].isin(val_subj)
            ].index.tolist()

            tr_loader = DataLoader(
                Subset(full_dataset, tr_idx), batch_size=BATCH_SIZE,
                shuffle=True, num_workers=0,
                worker_init_fn=seed_worker, generator=g)
            va_loader = DataLoader(
                Subset(full_dataset, va_idx), batch_size=BATCH_SIZE,
                shuffle=False, num_workers=0,
                worker_init_fn=seed_worker, generator=g)

            model = SkateFormer_MultiView(
                NUM_CLASSES, 3, D_MODEL, N_HEADS, N_BLOCKS,
                FF_DIM, DROPOUT, LOCAL_WINDOW).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=LR)

            best_acc, best_f1, best_ep = 0.0, 0.0, 0
            final_acc, final_f1 = 0.0, 0.0

            for epoch in range(1, NUM_EPOCHS+1):
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
                "model": "skateformer_multiview",
                "seed": seed, "fold_seed": seed+fold, "fold": fold,
                "val_subjects": f"{val_t} & {val_n}",
                "best_epoch": best_ep,
                "accuracy": best_acc, "f1_score": best_f1,
                "final_epoch_accuracy": final_acc, "final_epoch_f1": final_f1,
            })

            del model
            if DEVICE.type=="cuda": torch.cuda.empty_cache()

        seed_r = [r for r in all_results if r["seed"]==seed]
        accs = [r["accuracy"] for r in seed_r]
        f1s  = [r["f1_score"] for r in seed_r]
        print(f"\n--- SUMMARY SEED {seed}: "
              f"Acc {np.mean(accs):.4f}±{np.std(accs):.4f} | "
              f"F1 {np.mean(f1s):.4f}±{np.std(f1s):.4f} ---")

    # ── SAVE ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv("skateformer_5seed_4fold_results.csv", index=False)

    per_seed = df.groupby("seed").agg(
        acc_mean=("accuracy","mean"),    acc_std=("accuracy","std"),
        f1_mean=("f1_score","mean"),     f1_std=("f1_score","std"),
        final_acc_mean=("final_epoch_accuracy","mean"),
        final_f1_mean=("final_epoch_f1","mean"),
    ).reset_index()
    per_seed.to_csv("skateformer_5seed_per_seed_summary.csv", index=False)

    pd.DataFrame([{**cmplx,"model":"skateformer_multiview"}]).to_csv(
        "skateformer_complexity.csv", index=False)

    print("\n" + "="*70)
    print("PER-SEED SUMMARY: SkateFormer MULTI-VIEW")
    print("="*70)
    print(per_seed.to_string(index=False))

    print("\n" + "="*70)
    print(f"AGREGAT FINAL SkateFormer ({len(SEEDS)} seeds×4 folds={len(df)} obs)")
    print("="*70)
    print(f"[Best-epoch ] Acc: {df['accuracy'].mean():.4f} ± "
          f"{df['accuracy'].std():.4f} | "
          f"F1: {df['f1_score'].mean():.4f} ± "
          f"{df['f1_score'].std():.4f}")
    print(f"[Final-epoch] Acc: {df['final_epoch_accuracy'].mean():.4f} ± "
          f"{df['final_epoch_accuracy'].std():.4f} | "
          f"F1: {df['final_epoch_f1'].mean():.4f} ± "
          f"{df['final_epoch_f1'].std():.4f}")

    print("\nFile hasil:")
    print("1. skateformer_5seed_4fold_results.csv")
    print("2. skateformer_5seed_per_seed_summary.csv")
    print("3. skateformer_complexity.csv")