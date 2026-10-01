# ==============================================================================
# === LR SWEEP — SA-TDGFormer Multi-View (R2-P7) ================================
# === Mengimpor seluruh komponen dari mvst_sa_tdgformer_5seed.py (taruh di      ===
# === folder yang sama). Gnet dan Tnet tetap dilatih TERPISAH dengan LR yang    ===
# === sama, lalu skor softmax digabung (0.5 / 0.5), identik dengan Tabel 3.     ===
# === Default LR = 5e-5, 2e-4, 5e-4: hasil 1e-4 sudah ada di                    ===
# ===   sa_tdgformer_5seed_4fold_results.csv (tiap stream dilatih dari seed     ===
# ===   fold sendiri, jadi hasil 1e-4 tidak bergantung pada LR lain).           ===
# ===   Tambahkan 1e-4 di LEARNING_RATES bila ingin uji reproduktifitas.        ===
# === Resume per (lr, seed, fold). Opsi grad_checkpoint=True bila CUDA OOM.     ===
# ==============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import mvst_sa_tdgformer_5seed as sa

LEARNING_RATES = (5e-5, 2e-4, 5e-4)


def run(root_dir="Dataset_AQA_CSV", learning_rates=LEARNING_RATES,
        seeds=(42, 7, 123, 777, 2024), num_epochs=40, batch_size=8, num_frames=150,
        alpha=0.5, beta=0.5, grad_checkpoint=False, out_prefix="sa_tdgformer_lr_sweep"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_csv = f"{out_prefix}_results.csv"

    print("\n" + "=" * 70)
    print("LR SWEEP — SA-TDGFormer MULTI-VIEW (Gnet + Tnet terpisah) (R2-P7)")
    print(f"Device : {device} | Seeds : {list(seeds)} | LR : {list(learning_rates)}")
    print("=" * 70 + "\n")

    full_dataset = sa.MultiViewLandmarkDataset(root_dir, max_frames=num_frames)
    gnet_fn = lambda: sa.Gnet_MultiView(num_class=4, num_node=33, in_channels=3,
                                        hidden=64, n_blocks=9)
    tnet_fn = lambda: sa.Tnet_MultiView(num_class=4, num_node=33, in_channels=3,
                                        d_model=128, n_heads=4, dim_ff=256, n_layers=9,
                                        max_frames=num_frames, grad_checkpoint=grad_checkpoint)

    all_results, done = [], set()
    if Path(results_csv).exists():
        prev = pd.read_csv(results_csv)
        all_results = prev.to_dict("records")
        done = {(float(r["lr"]), int(r["seed"]), int(r["fold"])) for r in all_results}
        print(f"Resume: {len(done)} (lr, seed, fold) sudah selesai, dilewati.")

    paired_folds = list(zip(["A", "B", "C", "H"], ["E", "F", "G", "S"]))
    for lr in learning_rates:
        print("\n" + "%" * 70 + f"\nLEARNING RATE: {lr}\n" + "%" * 70)
        for seed in seeds:
            for fold, (val_t, val_n) in enumerate(paired_folds, 1):
                if (float(lr), int(seed), fold) in done:
                    continue
                fold_seed = seed + fold
                print(f"\n{'='*70}\nLR {lr} | SEED {seed} | FOLD {fold}/4 | "
                      f"Val: {val_t} & {val_n} | fold seed = {fold_seed}\n{'='*70}")
                md = full_dataset.metadata
                val_subj = [val_t, val_n]
                train_idx = md[~md["subject_id"].isin(val_subj)].index.tolist()
                val_idx = md[md["subject_id"].isin(val_subj)].index.tolist()

                g_probs, lab_g = sa.train_stream("Gnet", gnet_fn, full_dataset, train_idx, val_idx,
                                                 fold_seed, num_epochs, batch_size, lr, device)
                t_probs, lab_t = sa.train_stream("Tnet", tnet_fn, full_dataset, train_idx, val_idx,
                                                 fold_seed, num_epochs, batch_size, lr, device)
                assert np.array_equal(lab_g, lab_t)
                fused = [alpha * pg + beta * pt for pg, pt in zip(g_probs, t_probs)]
                (b_acc, b_f1, b_ep), (f_acc, f_f1) = sa.best_and_final(fused, lab_g)
                (_, _, _), (gf_acc, gf_f1) = sa.best_and_final(g_probs, lab_g)
                (_, _, _), (tf_acc, tf_f1) = sa.best_and_final(t_probs, lab_g)

                print(f"[Fused] Best Ep {b_ep} Acc {b_acc:.4f} F1 {b_f1:.4f} || "
                      f"Final Acc {f_acc:.4f} F1 {f_f1:.4f}")
                all_results.append({
                    "model": "sa_tdgformer_multiview", "lr": lr, "seed": seed,
                    "fold_seed": fold_seed, "fold": fold, "val_subjects": f"{val_t} & {val_n}",
                    "best_epoch": b_ep, "accuracy": b_acc, "f1_score": b_f1,
                    "final_epoch_accuracy": f_acc, "final_epoch_f1": f_f1,
                    "gnet_final_epoch_accuracy": gf_acc, "tnet_final_epoch_accuracy": tf_acc,
                })
                pd.DataFrame(all_results).to_csv(results_csv, index=False)

    df = pd.DataFrame(all_results)
    summary = df.groupby("lr").agg(
        n_obs=("accuracy", "count"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", lambda x: x.std(ddof=1)),
        final_acc_mean=("final_epoch_accuracy", "mean"),
        final_acc_std=("final_epoch_accuracy", lambda x: x.std(ddof=1)),
        final_f1_mean=("final_epoch_f1", "mean"),
        final_f1_std=("final_epoch_f1", lambda x: x.std(ddof=1)),
    ).reset_index()
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    print("\n" + "=" * 70 + "\nRINGKASAN LR SWEEP SA-TDGFormer (ddof=1)\n" + "=" * 70)
    print(summary.to_string(index=False))
    print(f"\nFile hasil: {results_csv}, {out_prefix}_summary.csv")
    return df


if __name__ == "__main__":
    run()