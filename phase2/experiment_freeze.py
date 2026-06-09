"""
experiment_freeze.py — RoBERTa encoder freeze + 헤드라인 확대 실험 (Phase 2 진단)

주 모델(fine-tune, MAX_HEADLINES=30)이 test 에서 단일 클래스로 붕괴하는 원인이
'406행으로 1.1억 파라미터 fine-tune' 이라는 가설을 검증한다:
  - encoder 를 동결(freeze)하여 사전학습 [CLS] 특징을 그대로 쓰고,
    작은 query/index_emb/head 만 학습 → 저분산 구성.
  - MAX_HEADLINES 30→N(기본 100) 으로 늘려 TF-IDF(하루 전체 헤드라인) 와
    정보량 비대칭을 줄인다.
  - encoder 가 고정이므로 헤드라인 [CLS] 임베딩을 **한 번만 계산해 캐시** →
    head 학습은 수초. (encoder=klue/roberta-base, ε, split 은 불변)

산출(주 결과 파일은 건드리지 않음):
  results/roberta_freeze_metrics.csv   — freeze 모델 8셀 acc/macro-F1 + 예측클래스수
  results/roberta_freeze_compare.md    — TF-IDF / RoBERTa(주) / RoBERTa(freeze) 비교
  data/checkpoints/freeze_mh{N}.pt
"""
from __future__ import annotations
import sys
import pathlib
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import (  # noqa: E402
    DATASET_FINAL, ENCODER_NAME, HORIZONS, INDEX_NAMES, MAX_LENGTH, SEED,
    WEIGHT_DECAY, WARMUP_RATIO, CHECKPOINT_DIR, RESULTS_DIR,
)
from phase2.dataset import make_splits  # noqa: E402
from phase2.model import HeadlineAttentionModel  # noqa: E402
from phase2.train import seed_everything, compute_loss  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402


@torch.no_grad()
def precompute(ds, encoder, device, max_headlines):
    """동결 encoder 로 각 행의 헤드라인 [CLS] 임베딩을 1회 계산해 캐시."""
    encoder.eval()
    E, MASK, IDX = [], [], []
    Y = {h: [] for h in HORIZONS}
    META = []
    for i in range(len(ds)):
        tk, lb, meta = ds[i]
        ii = tk["input_ids"].to(device)          # [M, L]
        am = tk["attention_mask"].to(device)
        cls = encoder(input_ids=ii, attention_mask=am).last_hidden_state[:, 0]
        E.append(cls.cpu())                      # [M, d]
        MASK.append(tk["headline_mask"])
        IDX.append(tk["index_id"])
        for h in HORIZONS:
            Y[h].append(lb[f"h{h}"])
        META.append(meta)
    out = {
        "E": torch.stack(E),                     # [N, M, d]
        "mask": torch.stack(MASK),               # [N, M]
        "idx": torch.stack(IDX),                 # [N]
        "Y": {h: torch.stack(Y[h]) for h in HORIZONS},
        "meta": META,
    }
    return out


def _batches(n, bs, shuffle, generator=None):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        yield order[s:s + bs]


@torch.no_grad()
def _eval_cached(model, cache, device):
    model.eval()
    preds = {h: [] for h in HORIZONS}
    idx_names = [m["index_name"] for m in cache["meta"]]
    E = cache["E"].to(device); mask = cache["mask"].to(device); idx = cache["idx"].to(device)
    out = model.pool_and_classify(E, mask, idx)
    for h in HORIZONS:
        preds[h] = out["logits"][f"h{h}"].argmax(-1).cpu().numpy()
    return preds, idx_names, out["attention"]


def _macro_f1_avg(cache, preds):
    f1s = []
    for h in HORIZONS:
        t = cache["Y"][h].numpy()
        f1s.append(f1_score(t, preds[h], average="macro", labels=[0, 1, 2],
                            zero_division=0))
    return float(np.mean(f1s))


def train_head(train_c, val_c, device, args):
    seed_everything(SEED)
    model = HeadlineAttentionModel(ENCODER_NAME, HORIZONS, freeze_encoder=True).to(device)
    # 학습 대상: query / index_emb / heads (encoder 제외)
    params = [p for n, p in model.named_parameters()
              if not n.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    Etr = train_c["E"].to(device); Mtr = train_c["mask"].to(device)
    Itr = train_c["idx"].to(device)
    Ytr = {h: train_c["Y"][h].to(device) for h in HORIZONS}
    n = Etr.size(0)
    gen = torch.Generator().manual_seed(SEED)

    total_steps = max(1, (n + args.batch_size - 1) // args.batch_size * args.epochs)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bidx in _batches(n, args.batch_size, shuffle=True, generator=gen):
            e = Etr[bidx]; mk = Mtr[bidx]; ix = Itr[bidx]
            out = model.pool_and_classify(e, mk, ix)
            lb = {f"h{h}": Ytr[h][bidx] for h in HORIZONS}
            loss = compute_loss(out, lb, criterion)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step()

        vpred, _, _ = _eval_cached(model, val_c, device)
        vf1 = _macro_f1_avg(val_c, vpred)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % max(1, args.epochs // 6) == 0 or epoch == 1:
            print(f"[epoch {epoch:>2}] train_loss={float(loss):.4f} "
                  f"val_macroF1(avg)={vf1:.4f} (best {best_f1:.4f})")

    model.load_state_dict(best_state)
    return model, best_f1


def evaluate_test(model, test_c, device):
    preds, idx_names, _ = _eval_cached(model, test_c, device)
    records = []
    for index_name in INDEX_NAMES:
        sel = [j for j, nm in enumerate(idx_names) if nm == index_name]
        for h in HORIZONS:
            t = test_c["Y"][h].numpy()[sel]
            p = preds[h][sel]
            acc = accuracy_score(t, p)
            mf1 = f1_score(t, p, average="macro", labels=[0, 1, 2], zero_division=0)
            n_pred_cls = len(np.unique(p))
            records.append({"index": index_name, "horizon": h, "n_test": len(t),
                            "accuracy": acc, "macro_f1": mf1,
                            "n_pred_classes": n_pred_cls})
    return pd.DataFrame.from_records(records)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | freeze encoder | MAX_HEADLINES={args.max_headlines} "
          f"| lr={args.lr} epochs={args.epochs}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    train_ds, val_ds, test_ds, _ = make_splits(
        DATASET_FINAL, tokenizer, args.max_headlines, MAX_LENGTH)

    # 동결 encoder 로 임베딩 캐시 (1회)
    enc = HeadlineAttentionModel(ENCODER_NAME, HORIZONS, freeze_encoder=True).encoder.to(device)
    print("임베딩 캐시 계산 중...")
    train_c = precompute(train_ds, enc, device, args.max_headlines)
    val_c = precompute(val_ds, enc, device, args.max_headlines)
    test_c = precompute(test_ds, enc, device, args.max_headlines)
    del enc; torch.cuda.empty_cache()

    model, best_val = train_head(train_c, val_c, device, args)
    print(f"\nbest val macro-F1(avg) = {best_val:.4f}")

    ckpt = CHECKPOINT_DIR / f"freeze_mh{args.max_headlines}.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "config": {"encoder": ENCODER_NAME, "freeze": True,
                           "max_headlines": args.max_headlines}}, ckpt)

    res = evaluate_test(model, test_c, device)
    _report(res, args, ckpt)
    return res


def _report(res, args, ckpt):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "roberta_freeze_metrics.csv",
               index=False, encoding="utf-8-sig")

    print("=" * 70)
    print(f"RoBERTa(freeze, MAX_HEADLINES={args.max_headlines}) — test")
    print("=" * 70)
    print("\n[macro-F1]")
    print(res.pivot(index="index", columns="horizon", values="macro_f1").round(3).to_string())
    print("\n[예측에 등장한 클래스 수]  (1=단일클래스 붕괴)")
    print(res.pivot(index="index", columns="horizon", values="n_pred_classes").to_string())

    # 비교표: TF-IDF / RoBERTa(주) / RoBERTa(freeze)
    def _load_f1(fname, label):
        p = RESULTS_DIR / fname
        if not p.exists():
            return None
        d = pd.read_csv(p)[["index", "horizon", "macro_f1"]].copy()
        d["method"] = label
        return d
    frames = [_load_f1("baseline_metrics.csv", "TF-IDF"),
              _load_f1("test_metrics.csv", "RoBERTa(ft,mh30)"),
              res[["index", "horizon", "macro_f1"]].assign(
                  method=f"RoBERTa(freeze,mh{args.max_headlines})")]
    long = pd.concat([f for f in frames if f is not None], ignore_index=True)
    order = ["TF-IDF", "RoBERTa(ft,mh30)", f"RoBERTa(freeze,mh{args.max_headlines})"]
    wide = long.pivot_table(index=["index", "method"], columns="horizon",
                            values="macro_f1").reindex(order, level="method")

    md = ["# RoBERTa freeze 실험 — test macro-F1 비교", "",
          "| index | method | " + " | ".join(f"h={h}" for h in HORIZONS) + " |",
          "|---|---|" + "---|" * len(HORIZONS)]
    for idx in INDEX_NAMES:
        for m in order:
            try:
                vals = " | ".join(f"{wide.loc[(idx, m), h]:.3f}" for h in HORIZONS)
            except KeyError:
                continue
            md.append(f"| {idx} | {m} | {vals} |")
    md += ["", "주: 주 비교는 h=1,5 (h21/h252 는 test 단일클래스 붕괴로 신뢰도 낮음).",
           "ft=fine-tune, mh=MAX_HEADLINES."]
    (RESULTS_DIR / "roberta_freeze_compare.md").write_text("\n".join(md), encoding="utf-8")

    print("\n[비교]")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(wide.to_string())
    print("\n저장:", RESULTS_DIR / "roberta_freeze_metrics.csv")
    print("저장:", RESULTS_DIR / "roberta_freeze_compare.md")
    print("저장:", ckpt)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--max-headlines", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())
