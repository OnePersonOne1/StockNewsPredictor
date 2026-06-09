"""
model.py — HeadlineAttentionModel (Phase 2 주 모델)

구조:
  e_i   = Encoder(headline_i)[CLS]                  ∈ R^d
  α_i   = softmax_i( q · e_i )   (패딩 헤드라인은 -inf 로 제외)
  d_t   = Σ_i α_i e_i            (learnable query attention pooling)
  d_t  += index_emb(index_id)    (KOSPI/KOSDAQ 구분)
  ŷ^(h) = softmax( W_h d_t + b_h )  for h ∈ horizons

- Encoder: klue/roberta-base, 기본 fine-tuning (freeze_encoder=False)
- forward 는 horizon별 logits dict 와 attention weights(해석용) 를 반환
"""
from __future__ import annotations
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# 패키지/스크립트 양쪽에서 import 가능하도록
try:
    from phase1.config import ENCODER_NAME, HORIZONS, N_CLASSES, INDEX_NAMES
except ImportError:  # pragma: no cover
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from phase1.config import ENCODER_NAME, HORIZONS, N_CLASSES, INDEX_NAMES


class HeadlineAttentionModel(nn.Module):
    def __init__(self, encoder_name: str = ENCODER_NAME,
                 horizons=tuple(HORIZONS), n_classes: int = N_CLASSES,
                 freeze_encoder: bool = False,
                 n_indices: int = len(INDEX_NAMES)):
        super().__init__()
        self.horizons = list(horizons)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        d = self.encoder.config.hidden_size

        # learnable query vector q
        self.query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        # 지수 구분용 임베딩 (d_t 에 더해 KOSPI/KOSDAQ 정보 주입)
        self.index_emb = nn.Embedding(n_indices, d)
        nn.init.normal_(self.index_emb.weight, std=0.02)
        # horizon별 분류 head
        self.heads = nn.ModuleDict(
            {f"h{h}": nn.Linear(d, n_classes) for h in self.horizons}
        )

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, headline_mask, index_id):
        """
        input_ids/attention_mask : [B, M, L]
        headline_mask            : [B, M]  (True=실제, False=패딩)
        index_id                 : [B]
        반환: {'logits': {h: [B, n_classes]}, 'attention': [B, M]}
        """
        B, M, L = input_ids.shape
        flat_ids = input_ids.reshape(B * M, L)
        flat_att = attention_mask.reshape(B * M, L)

        # frozen 이면 encoder 출력은 학습 파라미터에 대해 상수 → no_grad 로
        # activation 저장을 피해 메모리 절약(헤드라인 수 ↑ 가능). 수학적으로 동일.
        enc_ctx = torch.no_grad() if self.freeze_encoder else nullcontext()
        with enc_ctx:
            enc = self.encoder(input_ids=flat_ids, attention_mask=flat_att)
            cls = enc.last_hidden_state[:, 0]      # [B*M, d]  ([CLS]/<s>)
        e = cls.view(B, M, cls.size(-1))            # [B, M, d]
        return self.pool_and_classify(e, headline_mask, index_id)

    def pool_and_classify(self, e, headline_mask, index_id):
        """encoder 출력 e=[B,M,d] 로부터 attention pooling + horizon head.
        frozen encoder 실험에서 캐시된 임베딩을 재사용하기 위해 분리."""
        # attention score: q · e_i
        scores = torch.einsum("bmd,d->bm", e, self.query)  # [B, M]
        scores = scores.masked_fill(~headline_mask, float("-inf"))
        # 전부 패딩인 행 방지(이론상 발생 안 함): NaN softmax 가드
        all_pad = (~headline_mask).all(dim=1, keepdim=True)
        scores = scores.masked_fill(all_pad & ~headline_mask, 0.0)
        alpha = F.softmax(scores, dim=1)            # [B, M]

        d_t = torch.einsum("bm,bmd->bd", alpha, e)  # [B, d]
        d_t = d_t + self.index_emb(index_id)        # 지수 정보 주입

        logits = {f"h{h}": self.heads[f"h{h}"](d_t) for h in self.horizons}
        return {"logits": logits, "attention": alpha}


if __name__ == "__main__":
    # forward pass 검증: dummy input 으로 출력 shape 확인 (인코더 다운로드 필요)
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from phase1.config import MAX_HEADLINES, MAX_LENGTH

    torch.manual_seed(42)
    model = HeadlineAttentionModel()
    model.eval()

    B, M, L = 2, MAX_HEADLINES, MAX_LENGTH
    vocab = model.encoder.config.vocab_size
    input_ids = torch.randint(0, vocab, (B, M, L))
    attention_mask = torch.ones(B, M, L, dtype=torch.long)
    headline_mask = torch.ones(B, M, dtype=torch.bool)
    headline_mask[1, 10:] = False              # 두번째 샘플은 10개만 실제
    index_id = torch.tensor([0, 1])

    with torch.no_grad():
        out = model(input_ids, attention_mask, headline_mask, index_id)
    for h, lg in out["logits"].items():
        print(f"logits[{h}]:", tuple(lg.shape))
    a = out["attention"]
    print("attention:", tuple(a.shape),
          "| sample1 sum=%.4f" % float(a[0].sum()),
          "| sample2 패딩부 합=%.6f (0 이어야 함)" % float(a[1, 10:].sum()))
    print("smoke OK")
