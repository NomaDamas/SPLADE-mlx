# SPLADE-mlx 프로젝트 계획

Apple Silicon(MLX)에 특화된 SPLADE / V-SPLADE 추론 구현이 없다는 문제를 해결한다.
전체 흐름: **Mac에서 기존(PyTorch) 모델 속도 측정 → MLX 포팅 → 포팅된 모델 속도·품질 측정 및 보고서 작성.**

- 작업 머신: Apple M4 Max (12P+4E, 64GB), macOS, mlx 0.32.0, uv 사용
- 참고 오픈소스: [Blaizzy/mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — BERT/XLM-R/ModernBERT 등 dense 임베딩만 지원하고 **MLM head + SPLADE sparse head는 미지원** → 우리가 채울 빈자리가 명확함. 코드 구조(`models/bert.py`, `convert`, `load` API)는 그대로 벤치마킹할 것.

---

## 1. 배경 정리

### SPLADE란
BERT 계열 인코더의 **MLM head(vocab 30,522차원 logits)** 위에 아래 변환을 얹어 sparse vector를 만드는 모델:

```
w_j = max_{i in tokens} log(1 + ReLU(logit_{ij}))   (j = vocab index)
```

- attention mask 적용 후 시퀀스 축으로 max-pooling → 문서/쿼리가 30,522차원 희소 벡터가 됨
- 검색 점수 = 쿼리·문서 sparse vector 내적 (inverted index 호환)

### V-SPLADE란
SIGIR'22 *An Efficiency Study for SPLADE Models* (Lassance & Clinchant)의 **efficiency Level V** 구성.
핵심 특징은 **쿼리/문서 인코더 분리(비대칭)**:

- `naver/efficient-splade-V-large-query` (쿼리 인코더)
- `naver/efficient-splade-V-large-doc` (문서 인코더)
- MS MARCO dev 38.8 MRR@10, 인퍼런스 지연 45.3ms (논문 기준)
- Level VI-BT는 쿼리 인코더가 BERT-tiny라 쿼리 인코딩 0.7ms — stretch goal로 좋음

### 포팅의 실체
= MLX로 `BertForMaskedLM`(인코더 + MLM prediction head, tied embeddings) 구현
+ SPLADE 활성화/풀링 + HF safetensors → MLX 가중치 변환 + tokenizers 연동.
mlx-embeddings의 `bert.py`는 인코더까지만 있으므로 MLM head 부분이 신규 작업.

---

## 2. 대상 모델 매트릭스

| 우선순위 | 모델 | 아키텍처 | 비고 |
|---|---|---|---|
| **P0** | `naver/splade-cocondenser-ensembledistil` | BERT-base + MLM | SPLADE++ 대표 모델, 대칭(쿼리=문서), ungated |
| **P0** | `naver/efficient-splade-V-large-{query,doc}` | **DistilBERT** ×2 | **V-SPLADE**, 비대칭 페어. (구현 중 확인: BERT가 아니라 DistilBERT 아키텍처) |
| P1 | `naver/splade-v3-distilbert` | DistilBERT + MLM | DistilBERT 지원은 P0에서 이미 구현됨. `splade-v3` 본체는 HF gated(약관 동의) |
| P1 | `naver/efficient-splade-VI-BT-large-{query,doc}` | BERT-tiny(쿼리) + BERT-base(문서) | 쿼리 인코딩 초고속 데모용 |
| 대안 | `prithivida/Splade_PP_en_v1` | BERT-base + MLM | Apache-2.0. naver 모델은 CC BY-NC-SA(비상업) → 라이선스 이슈 시 대체재 |

> **라이선스 주의**: naver SPLADE 계열은 CC BY-NC-SA 4.0(비상업). 연구/벤치마크는 문제없으나, 변환 가중치 재배포·상업 사용 시 prithivida 계열로 전환하거나 별도 검토 필요.

---

## 3. 단계별 계획

### Phase 0 — 스캐폴딩 (0.5일)

- `uv init` + **Python 3.12 고정** (시스템 3.14는 torch/transformers 휠 호환 리스크 → venv에서 3.12)
- 의존성: `mlx`, `torch`, `transformers`, `tokenizers`, `safetensors`, `numpy`, `datasets`(BEIR 로딩), `pytest`
- 레포 구조:

```
SPLADE-mlx/
├── pyproject.toml
├── splade_mlx/
│   ├── __init__.py           # load() 공개 API
│   ├── models/
│   │   ├── bert.py           # BertModel + MLM head + SpladeHead
│   │   └── distilbert.py     # P1
│   └── convert.py            # HF safetensors → MLX 가중치 (키 sanitize, dtype 캐스팅)
├── bench/
│   ├── workloads.py          # 공용 워크로드 정의 (torch/mlx 동일 입력 보장)
│   ├── bench_torch.py        # Phase 1
│   ├── bench_mlx.py          # Phase 4
│   └── report.py             # 결과 JSON → 표/보고서
├── tests/
│   └── test_parity.py        # torch ↔ mlx 수치 일치 테스트
├── results/                  # 벤치 결과 JSON (커밋함)
└── data/                     # 데이터셋 캐시 (gitignore)
```

**완료 조건**: `uv run python -c "import mlx.core, torch, transformers"` 성공.

### Phase 1 — Mac 위 기존 모델 베이스라인 측정 (1일)

측정 대상 백엔드: **PyTorch CPU / PyTorch MPS** (둘 다 측정해야 "MLX가 뭘 이겼는지" 명확해짐).

**워크로드** (torch/mlx 완전 동일 입력, 고정 seed):
- 쿼리 인코딩: MS MARCO dev 쿼리 샘플, seq_len ~32, batch {1, 8, 32}
- 문서 인코딩: BEIR NFCorpus 문서, seq_len {128, 256}, batch {1, 8, 32, 64}

**메트릭**:
- latency p50/p95 (warmup 10회 후 50회 측정, MPS는 `torch.mps.synchronize()` 필수)
- throughput (docs/s, tokens/s)
- peak memory (RSS + MPS allocated)
- 토크나이즈 시간 분리 측정 (encoder-only 시간과 end-to-end 둘 다)

**추가 산출물 — parity용 레퍼런스**: 고정 입력 32건에 대해 torch fp32 로짓·sparse vector를 `.npz`로 저장 → Phase 3 테스트의 정답지.

**완료 조건**: `results/baseline_{cpu,mps}.json` 생성 + 레퍼런스 `.npz` 저장.

### Phase 2 — MLX 포팅 (2~3일, 핵심)

1. **BERT 인코더 + MLM head** (`models/bert.py`)
   - embeddings(word/position/token_type + LayerNorm) → encoder layers → MLM transform(dense+gelu+LayerNorm) → decoder(word embedding tied + bias)
   - `SpladeHead`: `log1p(relu(logits))` → attention mask 적용 → max-pool. `mx.log1p`, `mx.maximum` 사용
2. **가중치 변환** (`convert.py`)
   - HF hub에서 safetensors 다운로드 → 키 매핑/sanitize → `mx.save_safetensors`
   - dtype: fp32(파리티 검증용) / **bf16(기본)** 저장 옵션
   - V-SPLADE는 query/doc 두 체크포인트를 하나의 비대칭 페어 API로 로드: `SpladePair.encode_query() / .encode_doc()`
3. **토크나이저**: HF `tokenizers` 그대로 사용 (torch 경로와 100% 동일 토큰 보장 → 파리티 변수 제거)
4. **공개 API** (mlx-embeddings 스타일):
   ```python
   from splade_mlx import load
   model, tokenizer = load("naver/splade-cocondenser-ensembledistil")
   sparse = model.encode(["hello world"])   # (B, 30522) or top-k (indices, values)
   ```
5. **양자화(후반)**: `mx.quantize` 8bit/4bit 옵션. 단, MLM decoder는 embedding tied라 양자화 시 품질 영향 큼 → Phase 3 품질 게이트 통과한 것만 채택
6. P1: DistilBERT(v3-distilbert), BERT-tiny(VI-BT query) 아키텍처 추가

**완료 조건**: P0 두 모델이 MLX에서 로드·인코딩 되고 출력 shape/희소도(활성 term 수)가 상식선.

### Phase 3 — 정합성(parity) 검증 (1일)

속도 보고서의 전제는 "같은 모델"이라는 증명. 3단계 게이트:

1. **로짓 파리티**: 고정 입력 32건, MLX fp32 vs torch fp32 레퍼런스 → `max |Δ| < 1e-3`
2. **Sparse vector 파리티**: top-k(k=64) term index 일치율 ≥ 99%, 활성 term weight cosine ≥ 0.9999
3. **검색 품질 파리티**: BEIR **NFCorpus + SciFact** (작아서 로컬로 충분) 전체 인코딩 → brute-force dot-product 랭킹 → **nDCG@10이 torch 결과와 ±0.002 이내**
   - bf16/양자화 버전도 같은 게이트로 품질 열화 측정 → 보고서에 dtype별 품질 표 포함

**완료 조건**: `pytest tests/` 통과 + dtype별 nDCG 표 확보.

### Phase 4 — MLX 벤치마크 & 성능 보고 (1일)

- Phase 1과 **동일한 harness·워크로드**로 MLX 측정 (bf16 기본, fp32/4bit/8bit 추가)
- MLX 특유 주의점: `mx.eval()`로 lazy evaluation 강제 후 시간 측정, warmup으로 커널 컴파일 비용 분리, `mx.compile` 적용 전/후 비교
- **REPORT.md** 작성:
  - 모델 × 백엔드(CPU/MPS/MLX) × batch × seq_len 지연/처리량 표 + speedup 배수
  - 메모리 비교, dtype별 품질(nDCG@10) vs 속도 트레이드오프
  - V-SPLADE 비대칭 구성의 쿼리 인코딩 지연 하이라이트 (온디바이스 검색 시나리오 소구점)
  - 재현 방법(커맨드 한 줄) 명시

**완료 조건**: REPORT.md + `results/*.json` 완비, 모든 수치가 실측 기반.

### Phase 5 — Stretch (선택)

- mlx-embeddings에 SPLADE 아키텍처 upstream PR (GPL v3 프로젝트임에 유의)
- 변환 가중치 HF 업로드 (`mlx-community` 스타일) — **naver 라이선스 검토 후**
- VI-BT(BERT-tiny 쿼리 인코더)로 "0.x ms 쿼리 인코딩 on M4" 데모
- top-k sparse 출력 + 간단한 inverted index로 end-to-end 로컬 검색 데모

---

## 4. 리스크 & 사전 결정

| 리스크 | 대응 |
|---|---|
| Python 3.14에서 torch 휠 문제 | uv로 3.12 고정 |
| `naver/splade-v3` gated 모델 | HF 약관 동의 필요 → P1로 미룸. P0는 ungated 모델만 |
| naver 라이선스 CC BY-NC-SA | 벤치마크·연구는 OK. 재배포는 Phase 5에서 별도 판단, 대안 prithivida(Apache) |
| bf16에서 `log1p(relu(·))` 수치 드리프트 | fp32 파리티 먼저 통과 → bf16은 품질 게이트(nDCG)로 판단 |
| MPS 벤치 불공정 시비 | synchronize 철저, warmup 동일, 토크나이즈 분리 측정, 방법론을 REPORT에 명시 |
| MLM decoder 양자화 품질 붕괴 | 양자화는 옵션 기능. 게이트 실패 시 embedding/decoder만 제외하는 mixed 양자화 |

## 5. 타임라인 요약

| Phase | 내용 | 예상 |
|---|---|---|
| 0 | 스캐폴딩 | 0.5일 |
| 1 | PyTorch 베이스라인 측정 + 레퍼런스 저장 | 1일 |
| 2 | MLX 포팅 (BERT+MLM+SPLADE, convert, API) | 2~3일 |
| 3 | 파리티 검증 (로짓/벡터/nDCG) | 1일 |
| 4 | MLX 벤치 + REPORT.md | 1일 |
| 5 | Stretch (upstream PR, HF 업로드, 데모) | 선택 |

**총 5.5~6.5일 (stretch 제외).**

---

## 참고 자료

- V-SPLADE (Efficient SPLADE Level V): https://huggingface.co/naver/efficient-splade-V-large-doc , https://huggingface.co/naver/efficient-splade-V-large-query
- 논문: *An Efficiency Study for SPLADE Models* — https://reneuir.org/assets/slides/ReNeuIR2022-efficient-splade.pdf
- SPLADE 원 저장소: https://github.com/naver/splade
- SPLADE v2 논문: https://arxiv.org/abs/2109.10086
- mlx-embeddings: https://github.com/Blaizzy/mlx-embeddings
