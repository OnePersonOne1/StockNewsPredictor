"""
train.py — 학습 loop (Phase 2)

- Optimizer : AdamW(lr=2e-5, weight_decay=0.01)
- Scheduler : linear warmup (warmup_ratio=0.1)
- Loss      : L = mean_h CE( ŷ^(h), y^(h) )   (4 horizon 평균)
- Best val macro-F1 (horizon 평균) 기준 체크포인트 저장 → BEST_CKPT
- per-epoch: train/val loss, val accuracy & macro-F1 (horizon별)

사용:
  python phase2/train.py                  # full 학습 (config.EPOCHS)
  python phase2/train.py --epochs 1       # 1 epoch 시험 (loss 감소 확인)
  python phase2/train.py --smoke          # 초소형 1 step 검증 (CPU 가능)
  python phase2/train.py --batch-size 4   # OOM fallback
"""
from __future__ import annotations
import sys
import pathlib
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import (  # noqa: E402
    DATASET_FINAL, ENCODER_NAME, HORIZONS, SEED,
    LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO, EPOCHS, BATCH_SIZE,
    MAX_HEADLINES, MAX_LENGTH, FREEZE_ENCODER, GRAD_CLIP,
    BEST_CKPT, CHECKPOINT_DIR,
)
from phase2.dataset import make_splits  # noqa: E402
from phase2.model import HeadlineAttentionModel  # noqa: E402


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(tokenized, labels, device):
    tk = {k: v.to(device) for k, v in tokenized.items()}
    lb = {k: v.to(device) for k, v in labels.items()}
    return tk, lb


def compute_loss(out, labels, criterion):
    losses = [criterion(out["logits"][f"h{h}"], labels[f"h{h}"]) for h in HORIZONS]
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """val/test 평가 → (avg_loss, per-horizon {acc, macro_f1}, 평균 macro_f1)."""
    model.eval()
    tot_loss, n_batch = 0.0, 0
    preds = {h: [] for h in HORIZONS}
    trues = {h: [] for h in HORIZONS}
    for tokenized, labels, _ in loader:
        tk, lb = to_device(tokenized, labels, device)
        out = model(**tk)
        tot_loss += float(compute_loss(out, lb, criterion)); n_batch += 1
        for h in HORIZONS:
            preds[h].append(out["logits"][f"h{h}"].argmax(-1).cpu().numpy())
            trues[h].append(lb[f"h{h}"].cpu().numpy())
    per_h = {}
    f1s = []
    from phase1.config import CLASS_IDX
    for h in HORIZONS:
        p = np.concatenate(preds[h]); t = np.concatenate(trues[h])
        m = t != -100                       # NaN(ignore) 제외
        p, t = p[m], t[m]
        acc = accuracy_score(t, p) if len(t) else float("nan")
        mf1 = f1_score(t, p, average="macro", labels=CLASS_IDX, zero_division=0)
        per_h[h] = {"acc": acc, "macro_f1": mf1}
        f1s.append(mf1)
    return tot_loss / max(n_batch, 1), per_h, float(np.mean(f1s))


def train(args):
    seed_everything(getattr(args, "seed", SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | encoder: {ENCODER_NAME}")

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)

    train_ds, val_ds, test_ds, sigma = make_splits(
        DATASET_FINAL, tokenizer, args.max_headlines, MAX_LENGTH)
    print(f"sigma_h(train): { {k: round(v,4) for k,v in sigma.items()} }")
    print(f"splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if args.smoke:  # 초소형: 각 split 앞 4개, 1 epoch
        train_ds = Subset(train_ds, range(min(4, len(train_ds))))
        val_ds = Subset(val_ds, range(min(4, len(val_ds))))
        args.epochs = 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = HeadlineAttentionModel(ENCODER_NAME, HORIZONS,
                                   freeze_encoder=FREEZE_ENCODER).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    best_f1 = -1.0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # 학습 로그 파일(설정별 고유 이름; 모든 per-epoch 기록 보존)
    from phase1.config import RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logf = RESULTS_DIR / (f"trainlog_mh{args.max_headlines}_bs{args.batch_size}"
                          f"_ep{args.epochs}_seed{args.seed}.txt")
    log_lines = [f"# encoder={ENCODER_NAME} lr={LEARNING_RATE} wd={WEIGHT_DECAY} "
                 f"warmup={WARMUP_RATIO} epochs={args.epochs} batch={args.batch_size} "
                 f"mh={args.max_headlines} max_len={MAX_LENGTH} seed={args.seed} "
                 f"freeze={FREEZE_ENCODER} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"]

    def _log(s):
        print(s); log_lines.append(s)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for tokenized, labels, _ in train_loader:
            tk, lb = to_device(tokenized, labels, device)
            optimizer.zero_grad()
            out = model(**tk)
            loss = compute_loss(out, lb, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step(); scheduler.step()
            running += float(loss); n += 1
        train_loss = running / max(n, 1)

        val_loss, per_h, val_f1 = evaluate(model, val_loader, criterion, device)
        _log(f"[epoch {epoch}] train_loss={train_loss:.4f} "
             f"val_loss={val_loss:.4f} val_macroF1(avg)={val_f1:.4f} | " +
             " ".join(f"h{h}={per_h[h]['macro_f1']:.3f}" for h in HORIZONS))

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model_state": model.state_dict(),
                        "sigma": sigma, "epoch": epoch, "val_macro_f1": val_f1,
                        "config": {"encoder": ENCODER_NAME,
                                   "max_headlines": args.max_headlines,
                                   "max_length": MAX_LENGTH}},
                       BEST_CKPT)
            _log(f"   ✓ best 갱신 (val_macroF1={best_f1:.4f}, epoch {epoch})")

    _log(f"학습 종료. best val macro-F1={best_f1:.4f}")
    logf.write_text("\n".join(log_lines), encoding="utf-8")
    print("로그 저장:", logf)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-headlines", type=int, default=MAX_HEADLINES)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED,
                   help="시드(학습 변동성 점검용). 기본=config.SEED")
    p.add_argument("--smoke", action="store_true",
                   help="초소형 1-step 검증 (CPU 가능)")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
