# MVST: Multi-View Spatio-Temporal Transformer for Skill-Aware Tennis Stroke Recognition

Code and anonymized skeleton data for the paper:

> I. Asror, M. Abdurohman, B. Erfianto, and A. Rizal, "MVST: A Multi-View Spatio-Temporal Transformer for Skill-Aware Tennis Stroke Recognition from Skeleton Sequences", *International Journal of Intelligent Engineering and Systems* (under review).

## Repository structure

```
├── checkpoints/             # trained MVST+CVA models (see below)
├── Dataset_AQA_CSV/        # anonymized skeleton data
│   ├── backhand/           # backhand, trained
│   ├── backhandNewbie/     # backhand, novice
│   ├── forehand/           # forehand, trained
│   └── forehandNewbie/     # forehand, novice
├── *.py                    # training and evaluation scripts
├── *.csv                   # per-run results (one row per seed and fold)
└── requirements.txt
```

`Newbie` in the folder names corresponds to *novice* in the paper.

## Dataset

- 8 subjects: trained (A, B, C, H) and novice (E, F, G, S).
- 96 paired stroke instances, each with a back view and a side view (192 CSV files).
- Each CSV contains the 33 MediaPipe Pose landmarks per frame (`x`, `y` image-normalized; `z` relative depth).
- File naming: `<Stroke>_<Subject><Repetition>_<View>.csv`, e.g. `Backhand_A1_Back.csv`.
- No videos, images, or personal identities are included. All participants gave written informed consent for the release of their anonymized skeleton data.

## Usage

```bash
pip install -r requirements.txt
python "mvst cva lr sweep.py"
```

Run the scripts from the repository root. Each script writes one row per seed and fold to a CSV file.

## Evaluation protocol

- 4 subject-independent folds, each holding out one trained and one novice subject: (A,E), (B,F), (C,G), (H,S).
- 5 seeds: 42, 7, 123, 777, 2024. The per-fold seed is seed + fold.
- AdamW, lr = 1e-4, batch size 8, 40 epochs.
- The final-epoch result is the primary endpoint. The best-epoch result (highest macro F1 on the held-out subjects) is an optimistic reference only.

In the code, `val_subjects` / `val_loader` refer to the held-out subjects of each fold. There is no separate validation set.

## Checkpoints

`checkpoints/` contains the 20 MVST+CVA models reported in the paper (lr = 1e-4, 5 seeds × 4 folds), saved at the final epoch (epoch 40), which is the primary endpoint. File name: `mvst_cva_lr0.0001_seed<seed>_fold<fold>.pt`.

The models were generated with `python "mvst cva lr sweep.py" --ckpt`. The per-run results of this run (`mvst_cva_ckpt_results.csv`) are identical to those reported in the paper (final-epoch accuracy 83.54% ± 15.14%).

Each file stores `state_dict`, `config` (model hyperparameters), `seed`, `fold`, `val_subjects`, `final_epoch_accuracy` and `final_epoch_f1`:

```python
import importlib.util, torch
spec = importlib.util.spec_from_file_location("mvst", "mvst cva lr sweep.py")
mvst = importlib.util.module_from_spec(spec); spec.loader.exec_module(mvst)

ckpt = torch.load("checkpoints/mvst_cva_lr0.0001_seed42_fold1.pt", map_location="cpu")
model = mvst.MultiViewSpatioTemporalTransformer(**ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

Notes:
- Each model was trained on six subjects and should be evaluated only on the two held-out subjects of its fold (`val_subjects`). Evaluating it on other subjects would include its training data.
- The `state_dict` also contains the CVA projection head, which is used only during training and is not needed for inference.
- The checkpoints are provided for verification and further research. They were trained on eight subjects from a single recording session and are not intended as a general-purpose skill classifier.

## License

- Code: MIT
- Data: CC BY-NC 4.0 (non-commercial research use only)

## Contact

Ibnu Asror, Telkom University, iasror@telkomuniversity.ac.id
