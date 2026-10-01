# MVST: Multi-View Spatio-Temporal Transformer for Skill-Aware Tennis Stroke Recognition

Code and anonymized skeleton data for the paper:

> I. Asror, M. Abdurohman, B. Erfianto, and A. Rizal, "MVST: A Multi-View Spatio-Temporal Transformer for Skill-Aware Tennis Stroke Recognition from Skeleton Sequences", *International Journal of Intelligent Engineering and Systems* (under review).

## Repository structure

```
├── Dataset_AQA_CSV/        # anonymized skeleton data
│   ├── backhand/           # backhand, trained
│   ├── backhandNewbie/     # backhand, novice
│   ├── forehand/           # forehand, trained
│   └── forehandNewbie/     # forehand, novice
├── *.py                    # training and evaluation scripts
├── results/                # per-run results (CSV)
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
python mvst_cva_lr_sweep.py
```

Run the scripts from the repository root. Each script writes one row per seed and fold to a CSV file.

## Evaluation protocol

- 4 subject-independent folds, each holding out one trained and one novice subject: (A,E), (B,F), (C,G), (H,S).
- 5 seeds: 42, 7, 123, 777, 2024. The per-fold seed is seed + fold.
- AdamW, lr = 1e-4, batch size 8, 40 epochs.
- The final-epoch result is the primary endpoint. The best-epoch result (highest macro F1 on the held-out subjects) is an optimistic reference only.

In the code, `val_subjects` / `val_loader` refer to the held-out subjects of each fold. There is no separate validation set.

## License

- Code: MIT
- Data: CC BY-NC 4.0 (non-commercial research use only)

## Contact

Ibnu Asror, Telkom University, iasror@telkomuniversity.ac.id
