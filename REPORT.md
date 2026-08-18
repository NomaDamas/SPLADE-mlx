# SPLADE / V-SPLADE on Apple Silicon: PyTorch vs MLX 성능 보고서

**결론: MLX 포팅은 torch 최속 구성(MPS fp16) 대비 1.3~4.0배 빠르고, fp32 기준 검색 품질은
torch와 소수점 끝자리까지 동일하다.** 특히 V-SPLADE 쿼리 인코딩은 M4 Max에서
**1.6~2.1ms/query** 로, 온디바이스 실시간 검색에 충분한 수준이다.

- 날짜: 2026-08-18
- 머신: Apple M4 Max (12P+4E, 64GB), macOS 26.5.1
- 프레임워크: torch 2.13.0 (CPU/MPS) vs mlx 0.32.1
- 모델: `naver/splade-cocondenser-ensembledistil` (SPLADE++, BERT-base, 대칭),
  `naver/efficient-splade-V-large-{query,doc}` (**V-SPLADE**, DistilBERT, 비대칭 페어)
- 원시 데이터: `results/*.json`, 로그: `results/{baseline,mlx}_run.log`

## 1. 방법론

- 두 백엔드가 **동일한 입력**(BEIR NFCorpus 실제 텍스트, 동일 토크나이저, `padding="max_length"`)을
  인코딩. 워크로드: 쿼리 L32 × batch {1,8,32}, 문서 L128/L256 × batch {1,8,32,64}
- 지연시간 = 인코더 forward + SPLADE pooling. **토크나이즈 제외** (별도 측정: ~0.1ms/쿼리, JSON에 기록)
- warmup 3회 후 적응형 반복(12~50회, 최대 8초), MPS는 `torch.mps.synchronize()`,
  MLX는 `mx.eval()`로 lazy evaluation 강제. 측정 간 프로세스 병행 없음(순차 실행)
- 재현: 아래 §6 커맨드 한 줄씩

## 2. 정합성 (같은 모델임의 증명)

**수치 파리티 — `uv run pytest tests/` 9/9 통과** (3모델 × 3게이트):

| 게이트 | 기준 | 결과 |
|---|---|---|
| MLM 로짓 (fp32) | max \|Δ\| < 1e-3 | 통과 (3모델 모두) |
| Sparse vector cosine | ≥ 0.9999 | 통과 |
| top-64 term 일치율 | ≥ 99% | 통과 |

**검색 품질 nDCG@10** (BEIR 전체 코퍼스 인코딩 → brute-force dot product, 게이트 ±0.002 vs torch fp32):

| dataset | model | torch fp32 | mlx fp32 | mlx bf16 | mlx q8 | mlx q4 |
|---|---|---|---|---|---|---|
| NFCorpus | splade-cocondenser | 0.3480 | **0.3480 (±0.0000)** | +0.0001 | +0.0001 | −0.0025 |
| NFCorpus | efficient-splade-V | 0.3359 | **0.3359 (±0.0000)** | +0.0009 | +0.0006 | −0.0017 |
| SciFact | splade-cocondenser | 0.7024 | **0.7024 (±0.0000)** | +0.0004 | +0.0014 | −0.0002 |
| SciFact | efficient-splade-V | 0.6823 | **0.6823 (±0.0000)** | +0.0008 | +0.0001 | +0.0038 |

- **mlx-float32은 torch와 nDCG가 부동소수점 끝자리까지 동일** — 랭킹이 완전히 일치
- bf16/q8: 전 구간 ±0.0014 이내 → 게이트 통과
- q4: NFCorpus cocondenser에서 −0.0025로 게이트를 근소하게 벗어남 → "품질 트레이드오프 옵션"으로만 제공
- 절대값이 공개된 BEIR 수치(NFCorpus ~0.35, SciFact ~0.70)와 일치 → 파이프라인 자체의 타당성 교차 확인

## 3. 지연시간 / 처리량

(전체 표는 `results/comparison_tables.md`. 아래 speedup은 **torch 최속 구성인 MPS fp16 대비** best-MLX —
양자화 구성 포함이므로 표시된 열보다 빠른 구성이 기준일 수 있음)

### splade-cocondenser-ensembledistil (BERT-base, 110M)

| workload | torch cpu fp32 | torch mps fp16 | mlx fp32 | mlx bf16 | mlx bf16+compile | **speedup** |
|---|---|---|---|---|---|---|
| query-L32-B1 | 14.92 ms | 6.77 ms | 3.96 ms | 3.32 ms | 3.12 ms | **2.70x** |
| query-L32-B32 | 108.80 ms | 57.21 ms | 22.64 ms | 20.07 ms | 19.42 ms | **2.95x** |
| doc-L256-B1 | 40.41 ms | 11.35 ms | 7.96 ms | 7.28 ms | 7.27 ms | **1.56x** |
| doc-L256-B32 | 822.96 ms | 197.83 ms | 176.96 ms | 147.03 ms | 151.65 ms | **1.35x** |
| doc-L256-B64 | 1600.12 ms | 393.32 ms | 358.86 ms | 294.81 ms | 294.41 ms | **1.34x** |

### efficient-splade-V-large-query (V-SPLADE 쿼리 인코더, DistilBERT)

| workload | torch cpu fp32 | torch mps fp16 | mlx bf16+compile | mlx q8 | **speedup** |
|---|---|---|---|---|---|
| query-L32-B1 | 8.94 ms | 4.65 ms | 2.06 ms | **1.57 ms** | **2.96x** |
| query-L32-B8 | 23.10 ms | 14.94 ms | 4.17 ms | 4.82 ms | **3.59x** |
| query-L32-B32 | 68.69 ms | 49.41 ms | 12.22 ms | 16.65 ms | **4.04x** |

### efficient-splade-V-large-doc (V-SPLADE 문서 인코더, DistilBERT)

| workload | torch cpu fp32 | torch mps fp16 | mlx bf16 | mlx bf16+compile | **speedup** |
|---|---|---|---|---|---|
| doc-L128-B32 | 267.09 ms | 85.89 ms | 46.07 ms | 44.68 ms | **1.92x** |
| doc-L256-B1 | 24.52 ms | 7.46 ms | 4.27 ms | 4.23 ms | **1.77x** |
| doc-L256-B64 | 1032.40 ms | 273.49 ms | 186.08 ms | 179.66 ms | **1.52x** |

**처리량 하이라이트** (V-SPLADE, bf16+compile): 쿼리 **2,618 q/s** (L32-B32, 12.22ms/32),
단건 쿼리는 q8에서 **1.57ms** (637 q/s), 문서 인코딩 **716 docs/s** (L128-B32, 44.68ms/32).

## 4. 메모리

측정 기준이 다름에 유의: torch는 **프로세스 RSS(모델 로드 후)**, MLX는 **MLX active memory(가중치)**.

| 모델 | torch mps fp16 (RSS) | mlx bf16 (active) | mlx q4 (active) |
|---|---|---|---|
| splade-cocondenser | 621 MB | 266 MB | 85 MB |
| efficient-splade-V (per encoder) | ~795 MB | 181 MB | 58 MB |

q4 양자화 시 V-SPLADE 인코더 하나가 **58MB** — 쿼리+문서 페어를 합쳐도 120MB 이내.

## 5. 관찰

1. **작은 배치·짧은 시퀀스일수록 MLX 우위가 큼** (쿼리 B1~B32에서 2.7~4.0x). MPS 백엔드의
   커널 디스패치 오버헤드가 작업이 작을수록 상대적으로 커지기 때문. 검색 서비스의 실제
   병목인 "쿼리 1건 지연"에서 이득이 가장 크다는 뜻.
2. **bf16이 sweet spot**: fp32 대비 ~15-20% 빠르고 품질 열화 없음(±0.0014).
3. **`mx.compile` 효과는 소폭**(0~6%): 이미 커널이 큰 GEMM 위주라 그래프 오버헤드가 작음.
4. **q8/q4는 batch=1 전용 최적화**: B1에선 최속(1.57ms)이지만 배치가 커지면
   dequant 오버헤드로 bf16보다 느려짐. compute-bound 구간에서는 비양자화가 유리.
5. torch CPU fp32 대비로는 **4.5~8.9x**.
6. V-SPLADE의 설계 의도(쿼리 인코더 경량화)가 MLX에서 그대로 재현됨:
   쿼리 인코딩이 cocondenser 대비 ~1.6x 빠름 (DistilBERT 6층 vs BERT 12층).

## 6. 재현

```bash
uv sync
uv run python -m bench.save_reference                      # 파리티 레퍼런스 (torch fp32)
uv run python -m bench.bench_torch --backend cpu --dtype fp32
uv run python -m bench.bench_torch --backend mps --dtype fp32
uv run python -m bench.bench_torch --backend mps --dtype fp16
uv run pytest tests/                                       # 수치 파리티 9게이트
uv run python -m bench.eval_beir                           # nDCG@10 품질 파리티
uv run python -m bench.bench_mlx --dtype float32           # (bfloat16 / --compile / --quantize-bits 8|4)
uv run python -m bench.report                              # 비교 표 생성
```

MLX 모델 사용 예:

```python
from splade_mlx import load, load_pair

model, tok = load("naver/splade-cocondenser-ensembledistil", dtype="bfloat16")
pair = load_pair()          # V-SPLADE 비대칭 쿼리/문서 페어
q = pair.encode_query(["what causes vitamin d deficiency"])   # (1, 30522) sparse
```

## 7. 한계 및 후속 과제

- naver SPLADE 가중치는 CC BY-NC-SA 4.0(비상업). 변환 가중치 재배포 시 라이선스 검토 필요
  (Apache-2.0 대안: `prithivida/Splade_PP_en_v1`)
- `naver/splade-v3` 는 HF gated — DistilBERT 지원은 이미 완료라 약관 동의만 하면 즉시 포팅 가능
- Stretch: mlx-embeddings upstream PR, VI-BT(BERT-tiny 쿼리 인코더, sub-ms 쿼리 인코딩) 데모,
  top-k sparse 출력 + inverted index 로컬 검색 데모
