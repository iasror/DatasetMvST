# ==============================================================================
# === LR SWEEP BASELINE: CTR-GCN / SkateFormer (R2-P7) ==========================
# === Pemakaian:                                                                ===
# ===     python baseline_lr_sweep.py ctrgcn                                    ===
# ===     python baseline_lr_sweep.py skateformer                               ===
# === Script ini MENGIMPOR model, dataset, train_one_epoch, dan evaluate dari   ===
# === script asli (ctrgcn_5seed.py / skateformer_5seed.py) sehingga arsitektur  ===
# === dan training identik dengan Tabel 3. Taruh di folder yang sama.           ===
# === Urutan RNG identik dengan main loop script asli:                          ===
# ===   set_global_seed(seed); g = Generator(seed); per fold set_global_seed(   ===
# ===   seed + fold); train & val DataLoader memakai generator g yang sama.     ===
# ===   -> baris lr = 1e-4 seharusnya mereproduksi hasil Tabel 3.               ===
# === LR = 5e-5, 1e-4, 2e-4, 5e-4 (sama dengan sweep ST-GCN dan MVST+CVA).      ===
# === Resume per (lr, seed): seed yang belum lengkap 4 fold diulang dari fold 1.===
# ==============================================================================

import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

METHODS = {
    # nama     : (modul script asli, fungsi pembuat model sesuai argumen di script asli)
    "ctrgcn": ("ctrgcn_5seed",
               lambda mod: mod.CTRGCN_MultiView(4, mod.NUM_NODES, 3, 64, 4, 0.1)),
    "skateformer": ("skateformer_5seed",
                    lambda mod: mod.SkateFormer_MultiView(4, 3, 64, 4, 3, 128, 0.1, 16)),
}


def run(method, root_dir="Dataset_AQA_CSV", learning_rates=(5e-5, 1e-4, 2e-4, 5e-4),
        seeds=(42, 7, 123, 777, 2024), num_epochs=40, batch_size=8, num_frames=150):

    module_name, build_model = METHODS[method]
    try:
        mod = importlib.import_module(module_name)
    except ImportError as ex:
        raise SystemExit(f"Tidak bisa mengimpor '{module_name}.py'. Pastikan file script asli "
                         f"ada di folder yang sama dengan script ini. Detail: {ex}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_csv = f"{method}_lr_sweep_results.csv"

    print("\n" + "=" * 70)
    print(f"LR SWEEP — {method.upper()} (R2-P7)")
    print(f"Device : {device} | Seeds : {list(seeds)} | LR : {list(learning_rates)}")
    print("=" * 70 + "\n")

    full_dataset = mod.MultiViewLandmarkDataset(root_dir, max_frames=num_frames)
    paired_folds = list(zip(["A", "B", "C", "H"], ["E", "F", "G", "S"]))

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
        print("\n" + "%" * 70 + f"\nLEARNING RATE: {lr}\n" + "%" * 70)
        for seed in seeds:
            if (float(lr), int(seed)) in done:
                continue
            mod.set_global_seed(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            for fold, (val_t, val_n) in enumerate(paired_folds, 1):
                mod.set_global_seed(seed + fold)
                val_subj = [val_t, val_n]
                print(f"\n{'='*70}\n{method} | LR {lr} | SEED {seed} | FOLD {fold}/4 | "
                      f"Val: {val_t} & {val_n}\n{'='*70}")

                md = full_dataset.metadata
                tr_idx = md[~md["subject_id"].isin(val_subj)].index.tolist()
                va_idx = md[md["subject_id"].isin(val_subj)].index.tolist()
                tr_loader = DataLoader(Subset(full_dataset, tr_idx), batch_size=batch_size,
                                       shuffle=True, num_workers=0,
                                       worker_init_fn=mod.seed_worker, generator=g)
                va_loader = DataLoader(Subset(full_dataset, va_idx), batch_size=batch_size,
                                       shuffle=False, num_workers=0,
                                       worker_init_fn=mod.seed_worker, generator=g)

                model = build_model(mod).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr)

                best_acc, best_f1, best_ep = 0.0, 0.0, 0
                final_acc, final_f1 = 0.0, 0.0
                for epoch in range(1, num_epochs + 1):
                    tr_loss = mod.train_one_epoch(model, tr_loader, criterion, optimizer, device)
                    _, va_acc, va_f1 = mod.evaluate(model, va_loader, criterion, device)
                    if va_f1 > best_f1:
                        best_acc, best_f1, best_ep = va_acc, va_f1, epoch
                    final_acc, final_f1 = va_acc, va_f1
                    print(f"Epoch [{epoch:02d}/{num_epochs}] TrLoss:{tr_loss:.4f} | "
                          f"ValAcc:{va_acc:.4f} | ValF1:{va_f1:.4f}")

                print(f"-> Best: Acc={best_acc:.4f} F1={best_f1:.4f} Ep={best_ep} | "
                      f"Final: Acc={final_acc:.4f} F1={final_f1:.4f}")
                all_results.append({
                    "model": method, "lr": lr, "seed": seed, "fold_seed": seed + fold,
                    "fold": fold, "val_subjects": f"{val_t} & {val_n}", "best_epoch": best_ep,
                    "accuracy": best_acc, "f1_score": best_f1,
                    "final_epoch_accuracy": final_acc, "final_epoch_f1": final_f1,
                })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            pd.DataFrame(all_results).to_csv(results_csv, index=False)

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
    summary.to_csv(f"{method}_lr_sweep_summary.csv", index=False)
    print("\n" + "=" * 70 + f"\nRINGKASAN LR SWEEP {method.upper()} (ddof=1)\n" + "=" * 70)
    print(summary.to_string(index=False))
    print(f"\nFile hasil: {results_csv}, {method}_lr_sweep_summary.csv")
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in METHODS:
        raise SystemExit("Pemakaian: python baseline_lr_sweep.py [ctrgcn|skateformer]")
    run(sys.argv[1])