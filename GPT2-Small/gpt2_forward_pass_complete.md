# GPT-2 Small Complete Forward Pass 상세 분석

## 목차
1. [Input Processing](#1-input-processing)
   - 1.1 [Tokenization (BPE)](#11-tokenization-bpe)
   - 1.2 [Token Embedding](#12-token-embedding)
   - 1.3 [Position Embedding](#13-position-embedding)
   - 1.4 [Embedding Addition](#14-embedding-addition)
2. [Transformer Blocks (×12)](#2-transformer-blocks)
   - 2.1 [Layer Normalization 1](#21-layer-normalization-1)
   - 2.2 [Multi-Head Self-Attention](#22-multi-head-self-attention)
   - 2.3 [Residual Connection 1](#23-residual-connection-1)
   - 2.4 [Layer Normalization 2](#24-layer-normalization-2)
   - 2.5 [Feed-Forward Network (MLP)](#25-feed-forward-network-mlp)
   - 2.6 [Residual Connection 2](#26-residual-connection-2)
3. [Output Processing](#3-output-processing)
   - 3.1 [Final Layer Normalization](#31-final-layer-normalization)
   - 3.2 [Language Model Head](#32-language-model-head)
   - 3.3 [Softmax & Sampling](#33-softmax--sampling)
4. [전체 파이프라인 종합](#4-전체-파이프라인-종합)

---

## 표기법 (Notation)

본 문서에서 사용하는 변수 정의:

| 변수 | 의미 | GPT-2 Small 값 |
|------|------|----------------|
| **B** | Batch size (동시 처리하는 시퀀스 수) | 예시에서 1 |
| **L** | Sequence length (토큰 개수) | 최대 1024 |
| **V** | Vocabulary size (어휘 크기) | 50,257 |
| **d** 또는 **d_model** | Hidden dimension (은닉층 차원) | 768 |
| **H** 또는 **num_heads** | Number of attention heads (어텐션 헤드 수) | 12 |
| **d_k** | Head dimension (헤드당 차원) = d / H | 64 |
| **d_ff** | FFN hidden dimension (MLP 중간층 차원) | 3,072 (= 4 × d) |
| **N** | Number of layers (레이어 수) | 12 |

**Shape 표기 규칙:**
- `[B, L]` → Batch × Sequence length (2D 텐서)
- `[B, L, d]` → Batch × Sequence × Hidden (3D 텐서)
- `[B, H, L, L]` → Batch × Heads × Query positions × Key positions

---

## 모델 개요

```
GPT-2 Small Specifications:
├─ Total Parameters: 124,439,808 (124M)
├─ Hidden Size (d_model): 768
├─ Number of Layers: 12
├─ Attention Heads: 12
├─ Head Dimension (d_k): 64
├─ FFN Hidden Size: 3,072 (4 × 768)
├─ Vocabulary Size: 50,257
├─ Max Sequence Length: 1,024
└─ Activation: GELU
```

---

## 1. Input Processing

### 1.1 Tokenization (BPE)

**개요:** Byte Pair Encoding으로 텍스트를 토큰 ID 시퀀스로 변환

**입력/출력:**
```
Input:  "Hello world" (텍스트 문자열)
Output: [15496, 995] (토큰 ID 배열)
```

**처리 과정:**

**Step 1: UTF-8 바이트로 분해**
```python
text = "Hello world"
bytes_list = text.encode('utf-8')
# [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
#  H   e    l    l    o   공백  w    o    r    l    d
```

**Step 2: 바이트 → 유니코드 문자 매핑**

GPT-2는 256개 바이트를 출력 가능한 유니코드 문자로 1:1 매핑합니다:
```python
# GPT-2의 byte_encoder (encoder.json에 정의됨)
byte_encoder = {
    32: 'Ġ',    # 공백(space)은 'Ġ'로 표현
    72: 'H',
    101: 'e',
    108: 'l',
    111: 'o',
    119: 'w',
    114: 'r',
    100: 'd',
    ...  # 총 256개 매핑
}

initial_tokens = ['H', 'e', 'l', 'l', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']
```

**Step 3: BPE (Byte Pair Encoding) Merge (핵심 알고리즘)**

GPT-2는 **50,000개의 merge 규칙**을 학습해둔 파일(`merges.txt`)을 가지고 있습니다:
```
# merges.txt (우선순위 순서로 정렬됨)
Line 1: e r      → "er" 병합
Line 2: H e      → "He" 병합
Line 3: l l      → "ll" 병합
...
Line 1234: He ll  → "Hell" 병합
Line 5678: Hell o → "Hello" 병합
Line 9012: Ġ w    → "Ġw" 병합
Line 9876: Ġw orld → "Ġworld" 병합
```

**병합 과정 (반복):**
```
초기:     ['H', 'e', 'l', 'l', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']

반복 1:   모든 인접 쌍 중 merges.txt에서 가장 높은 우선순위 찾기
          ('H','e') → 'He' 발견!
결과:     ['He', 'l', 'l', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']

반복 2:   ('l','l') → 'll' 발견!
결과:     ['He', 'll', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']

반복 3:   ('o','r') → 'or' 발견!
결과:     ['He', 'll', 'o', 'Ġ', 'w', 'or', 'l', 'd']

반복 4:   ('or','l') → 'orl' 발견!
결과:     ['He', 'll', 'o', 'Ġ', 'w', 'orl', 'd']

반복 5:   ('orl','d') → 'orld' 발견!
결과:     ['He', 'll', 'o', 'Ġ', 'w', 'orld']

반복 6:   ('He','ll') → 'Hell' 발견!
결과:     ['Hell', 'o', 'Ġ', 'w', 'orld']

반복 7:   ('Hell','o') → 'Hello' 발견!
결과:     ['Hello', 'Ġ', 'w', 'orld']

반복 8:   ('Ġ','w') → 'Ġw' 발견!
결과:     ['Hello', 'Ġw', 'orld']

반복 9:   ('Ġw','orld') → 'Ġworld' 발견!
결과:     ['Hello', 'Ġworld']

반복 10:  더 이상 병합 가능한 쌍 없음 → 종료
최종:     ['Hello', 'Ġworld']
```

**Step 4: Vocabulary Lookup**

GPT-2는 **vocab.json** 파일에 50,257개 토큰의 문자열→ID 매핑을 저장:
```python
# vocab.json (실제 GPT-2 vocabulary)
vocab = {
    "Hello": 15496,
    "Ġworld": 995,
    "the": 1169,
    "Ġthe": 262,
    ...  # 총 50,257개 토큰
}

# 최종 변환
final_tokens = ['Hello', 'Ġworld']
token_ids = [vocab['Hello'], vocab['Ġworld']]
         # = [15496, 995]
```

**전체 흐름 요약:**
```
"Hello world"
      ↓ UTF-8 encode
[72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
      ↓ byte_encoder 매핑
['H', 'e', 'l', 'l', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']
      ↓ BPE merge (merges.txt 규칙 적용)
['Hello', 'Ġworld']
      ↓ vocab.json lookup
[15496, 995]
```

**핵심 파일:**
- `encoder.json` (또는 `vocab.json`): 토큰 문자열 ↔ ID 매핑 (50,257개)
- `vocab.bpe` (또는 `merges.txt`): BPE 병합 규칙 (50,000개)

**참고:** `Ġ`는 "이 토큰 앞에 공백이 있음"을 의미합니다. 따라서 `Ġworld`는 " world"를 나타냅니다.

**수학적 표현:**
```
T: Σ* → ℕ^L
T(text) = [t₁, t₂, ..., t_L]
where tᵢ ∈ {0, 1, ..., 50256}
```

**하드웨어 분석:**

```
CPU Implementation:
├─ UTF-8 encoding: ~50-100 cycles
├─ BPE merging: ~5,000-10,000 cycles per word
│  └─ Hash table lookups (merge rules)
│  └─ String concatenation
├─ Vocabulary lookup: ~100-200 cycles per token
└─ Total: ~2-4 μs for "Hello world"

GPU Implementation:
├─ NOT suitable (sequential algorithm)
├─ String operations non-parallelizable
└─ Overhead >> computation time

NPU Implementation:
├─ NOT applicable (no matrix operations)
└─ CPU preprocessing required

Recommendation: Always use CPU for tokenization
```

---

### 1.2 Token Embedding

**개요:** 토큰 ID를 768차원 벡터로 변환 (Lookup table)

**입력/출력:**
```
Input:  [B, L] = [1, 2] (token IDs)
Output: [B, L, 768] = [1, 2, 768] (embeddings)
```

**수학적 정의:**
```
E ∈ ℝ^(50257 × 768)  (Embedding matrix)
h_token[b, l] = E[x[b, l]]

Parameters: 50,257 × 768 = 38,597,376 (31% of total)
Memory (FP32): 154 MB
```

**처리 과정:**
```python
# Embedding matrix (learned parameters)
E = torch.randn(50257, 768)  # [vocab_size, d_model]

# Input token IDs
token_ids = torch.tensor([[15496, 995]])  # [B, L] = [1, 2]

# Embedding lookup (indexing operation)
token_embeddings = E[token_ids]  # [1, 2, 768]

# Equivalent to:
token_embeddings[0, 0] = E[15496]  # [768] - "Hello"
token_embeddings[0, 1] = E[995]    # [768] - "world"
```

**구체적 예시:**
```
Token ID 15496 ("Hello"):
E[15496] = [0.234, -0.891, 0.456, ..., -0.123]  # 768 values

Token ID 995 ("world"):
E[995] = [-0.567, 0.123, -0.789, ..., 0.456]

Output shape: [1, 2, 768]
[
  [  # Batch 0
    [0.234, -0.891, ..., -0.123],  # Token 0: "Hello"
    [-0.567, 0.123, ..., 0.456]    # Token 1: "world"
  ]
]
```

**하드웨어 연산 분석:**

```
Operation Type: Indexed Memory Read

CPU (Intel Xeon):
├─ Per token: Random memory access to 3 KB
├─ Cache behavior: Poor (random access pattern)
├─ L3 cache hit: ~2,160 cycles (~0.9 μs @ 2.5 GHz)
├─ DRAM miss: ~12,000 cycles (~5 μs)
├─ For L=1024: ~1 ms (DRAM-bound)
└─ Effective bandwidth: ~10 GB/s (20% of peak)

GPU (NVIDIA A100):
├─ Parallel lookup: All 1024 tokens simultaneously
├─ Memory coalescing: Difficult (random token IDs)
├─ L2 cache: 40 MB (stores ~13K token embeddings)
├─ HBM bandwidth: 900 GB/s → ~3 MB / 900 GB/s = 3.3 μs
├─ Actual: ~10-15 μs (cache misses)
└─ Speedup: 100× vs CPU

NPU (Apple Neural Engine):
├─ DMA-based lookup (multiple channels)
├─ On-chip cache: Top-K frequent tokens (K~2000)
│  └─ Covers ~90% of token usage (Zipf distribution)
├─ Cache hit: ~10-20 cycles (SRAM)
├─ Cache miss: ~500-1000 cycles (DRAM)
├─ For L=1024: ~30-50 μs
├─ Quantization (INT8): 4× faster, 4× less memory
└─ Power: 0.5 W (vs 50 W GPU) - 100× more efficient

Memory Bandwidth (L=1024):
├─ Data size: 1024 × 768 × 4 bytes = 3 MB
├─ CPU: 3 MB / 10 GB/s = 300 μs
├─ GPU: 3 MB / 900 GB/s = 3.3 μs
└─ NPU (cached): 3 MB / 100 GB/s = 30 μs
```

---

### 1.3 Position Embedding

**개요:** 각 토큰의 위치 정보를 768차원 벡터로 인코딩

**왜 Position Embedding이 필요한가?**

Transformer는 RNN과 달리 **순서 정보가 없습니다**. 입력을 병렬로 처리하기 때문에 "이 토큰이 몇 번째에 있는지" 알 수 없습니다:

```
Token Embedding만 사용하면:
"The cat sat on the mat" 에서
- 첫 번째 "the" = E["the"] = [0.1, 0.2, ...]
- 두 번째 "the" = E["the"] = [0.1, 0.2, ...]  ← 완전히 동일!

Position Embedding을 더하면:
- 첫 번째 "the" = E["the"] + P[0] = [0.1, 0.2, ...] + [0.01, 0.02, ...] = [0.11, 0.22, ...]
- 두 번째 "the" = E["the"] + P[4] = [0.1, 0.2, ...] + [0.05, 0.06, ...] = [0.15, 0.26, ...]
                                                                          ↑ 다른 값!
```

```
역할 분리:
┌─────────────────────────────────────────────┐
│ Token Embedding:    "무엇"인지 (의미 정보)    │
│ Position Embedding: "어디"에 있는지 (위치 정보)│
│ ─────────────────────────────────────────── │
│ 합계:               의미 + 위치 = 완전한 표현  │
└─────────────────────────────────────────────┘
```

**Position Embedding Lookup 방식:**

Position Embedding은 **토큰 내용과 무관**하게, 오직 **위치 인덱스**로만 결정됩니다:

```
Position Embedding lookup:

토큰이 뭐든 상관없이:
- 0번째 위치 → P[0] (768개 값)
- 1번째 위치 → P[1] (768개 값)
- 2번째 위치 → P[2] (768개 값)
- n번째 위치 → P[n] (768개 값)

예시:
"Hello world"     → positions [0, 1] → P[0], P[1]
"The cat"         → positions [0, 1] → P[0], P[1]  ← 같은 P[0], P[1]!
"I love you so"   → positions [0, 1, 2, 3] → P[0], P[1], P[2], P[3]
```

```
Token Embedding vs Position Embedding 비교:
┌─────────────────────────────────────────────────────────────────┐
│ Token Embedding:    검색 키 = token_id (토큰 내용)               │
│   "Hello" → E[15496], "world" → E[995]                         │
│                                                                 │
│ Position Embedding: 검색 키 = position (위치 인덱스)             │
│   위치 0 → P[0], 위치 1 → P[1] (토큰이 뭔지 안 봄!)              │
└─────────────────────────────────────────────────────────────────┘
```

**Position Embedding 값의 특성:**

```
P[0], P[1], ..., P[1023]의 768개 값들:

✗ 고정된 공식 (sin/cos) → GPT-2는 아님 (Original Transformer는 사용)
✗ 랜덤 값
✓ 학습된 파라미터 (Training 중 최적화됨)

모델이 "위치 0은 이런 패턴, 위치 1은 저런 패턴"을
스스로 학습해서 저장해둔 값 (각 위치의 고유한 "지문")
```

**입력/출력:**
```
Input:  position indices [0, 1, 2, ..., L-1]
Output: [B, L, 768] position embeddings
```

**수학적 정의:**
```
P ∈ ℝ^(1024 × 768)  (Position embedding matrix)
h_pos[b, l] = P[l]

Parameters: 1,024 × 768 = 786,432
Memory (FP32): 3.1 MB
```

**처리 과정:**
```python
# Position embedding matrix (learned parameters)
P = torch.randn(1024, 768)  # [max_seq_len, d_model]

# Generate position indices
seq_len = 2
position_ids = torch.arange(seq_len)  # [0, 1]

# Embedding lookup
position_embeddings = P[position_ids]  # [2, 768]

# Add batch dimension
position_embeddings = position_embeddings.unsqueeze(0)  # [1, 2, 768]
```

**GPT-2 vs Original Transformer:**
```
GPT-2: Learned Position Embeddings
├─ P is a learned parameter matrix
├─ More flexible, data-driven
└─ Better performance (empirically)

Original Transformer: Sinusoidal Position Encoding
├─ Fixed mathematical formula:
│   PE(pos, 2i) = sin(pos / 10000^(2i/d))
│   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
├─ No parameters to learn
└─ Generalizes to longer sequences
```

**하드웨어 연산 분석:**

```
Operation Type: Sequential Indexed Memory Read

CPU:
├─ Access pattern: SEQUENTIAL (0, 1, 2, ...)
├─ Hardware prefetcher: Highly effective
├─ Cache hit rate: ~95%
├─ Position 0: ~2,000 cycles (cold start)
├─ Position 1+: ~200-300 cycles (prefetched)
├─ For L=1024: ~309K cycles = 124 μs @ 2.5 GHz
└─ 8× faster than token embedding!

GPU:
├─ Sequential access: Perfect coalescing
├─ Memory bandwidth: 3 MB / 900 GB/s = 3.3 μs
├─ Same as token embedding (memory-bound)
└─ But more cache-friendly

NPU:
├─ IDEAL for NPU: Small matrix fits in SRAM
├─ Entire P matrix: 3.1 MB → Preload to SRAM
├─ Sequential DMA burst read: ~5 μs
├─ Zero DRAM access after initial load
├─ Quantization (INT8): 0.78 MB → ~2 μs
└─ Power: ~0.5 W

Performance Comparison (L=1024):
┌──────────┬────────────┬─────────────┬──────────┐
│ Hardware │ Time       │ Power       │ Energy   │
├──────────┼────────────┼─────────────┼──────────┤
│ CPU      │ 124 μs     │ 3 W         │ 0.37 mJ  │
│ GPU      │ 3.3 μs     │ 50 W        │ 0.17 mJ  │
│ NPU      │ 5 μs       │ 0.5 W       │ 0.0025 mJ│
│ NPU(INT8)│ 2 μs       │ 0.5 W       │ 0.001 mJ │
└──────────┴────────────┴─────────────┴──────────┘

Key Insight: Position embedding is hardware-friendly!
- Sequential access (perfect for prefetching/DMA)
- Small matrix (fits in cache/SRAM)
- NPU is 68× more energy-efficient than GPU
```

---

### 1.4 Embedding Addition

**개요:** Token embedding과 Position embedding을 element-wise로 더함

**입력/출력:**
```
Input:  token_emb [B, L, 768] + position_emb [B, L, 768]
Output: combined_emb [B, L, 768]
```

**수학적 표현:**
```
h₀ = h_token + h_pos
h₀[b, l, d] = E[x[b, l], d] + P[l, d]

Operations: B × L × d additions
          = 1 × 2 × 768 = 1,536 FP32 additions
```

**처리 과정:**
```python
# Token embeddings: [1, 2, 768]
token_emb = E[token_ids]

# Position embeddings: [1, 2, 768]
position_emb = P[position_ids]

# Element-wise addition
h0 = token_emb + position_emb  # [1, 2, 768]
```

**구체적 예시 (Hello world):**

```
Token Embedding (E에서 lookup):
┌─────────────────────────────────────────────────────────┐
│ E[15496] = [0.12, -0.34, 0.56, ..., 0.78]  ← "Hello"   │  768개 값
│ E[995]   = [0.23, 0.45, -0.67, ..., 0.89]  ← "Ġworld"  │  768개 값
└─────────────────────────────────────────────────────────┘
Shape: [1, 2, 768]

Position Embedding (P에서 lookup):
┌─────────────────────────────────────────────────────────┐
│ P[0] = [0.01, 0.02, -0.03, ..., 0.04]  ← 위치 0        │  768개 값
│ P[1] = [0.05, -0.06, 0.07, ..., 0.08]  ← 위치 1        │  768개 값
└─────────────────────────────────────────────────────────┘
Shape: [1, 2, 768]

Element-wise 덧셈:
┌─────────────────────────────────────────────────────────┐
│ h₀[0] = E[15496] + P[0]                                 │
│       = [0.12+0.01, -0.34+0.02, 0.56-0.03, ...]         │
│       = [0.13, -0.32, 0.53, ...]  ← "Hello at pos 0"   │
│                                                         │
│ h₀[1] = E[995] + P[1]                                   │
│       = [0.23+0.05, 0.45-0.06, -0.67+0.07, ...]         │
│       = [0.28, 0.39, -0.60, ...]  ← "Ġworld at pos 1"  │
└─────────────────────────────────────────────────────────┘
Shape: [1, 2, 768]  (변하지 않음!)
```

**시각화:**
```
Position:     0           1
           ┌─────┐     ┌─────┐
Token:     │Hello│     │world│
           └──┬──┘     └──┬──┘
              │            │
              ▼            ▼
Token Emb: [768 dim]   [768 dim]   ← E[token_id]에서 lookup
              +            +
Pos Emb:   [768 dim]   [768 dim]   ← P[position]에서 lookup
              │            │
              ▼            ▼
Result h₀: [768 dim]   [768 dim]   ← Transformer Block으로 전달
```

**핵심:** Shape은 변하지 않고, 각 위치의 768차원 벡터에 위치 정보가 "섞여" 들어갑니다.

**하드웨어 연산 분석:**

```
Operation Type: Element-wise Vector Addition (SIMD)

CPU:
├─ Scalar: 1,536 × 4 cycles = 6,144 cycles (2.5 μs)
├─ SSE (4-wide): 384 vector ops × 4 cycles = 1,536 cycles (0.6 μs)
├─ AVX (8-wide): 192 vector ops × 4 cycles = 768 cycles (0.3 μs)
├─ AVX-512 (16-wide): 96 vector ops × 2 cycles = 192 cycles (0.08 μs)
└─ Compiler auto-vectorizes (usually achieves AVX-512)

GPU:
├─ Launch 786,432 threads (one per element)
├─ Massive parallelism: ~5 μs (includes kernel launch)
├─ Memory coalescing: Perfect (sequential access)
└─ Compute-bound (addition is cheap)

NPU:
├─ Vector Processing Units (VPUs)
├─ Wide SIMD: 64-128 elements/cycle
├─ Standalone: ~12 μs @ 1 GHz
├─ Fused with embedding lookup: ~0 μs (FREE!)
│  └─ Pipeline: Fetch token_emb → Fetch pos_emb → Add (overlapped)
└─ Operation fusion eliminates memory writes

Performance (B=1, L=1024):
┌────────────┬──────────┬────────────┬──────────┐
│ Hardware   │ Time     │ Bottleneck │ Notes    │
├────────────┼──────────┼────────────┼──────────┤
│ CPU(scalar)│ 1.24 ms  │ Compute    │ Slow     │
│ CPU(AVX512)│ 39 μs    │ Memory     │ Good     │
│ GPU        │ 5 μs     │ Launch OH  │ Fast     │
│ NPU(fused) │ ~0 μs    │ N/A        │ FREE!    │
└────────────┴──────────┴────────────┴──────────┘

Key Optimization: Operator Fusion
Without fusion: Embedding → Memory → Add → Memory
With fusion:    Embedding → Add (in registers) → Next layer
Saves 2× memory bandwidth!
```

---

## 2. Transformer Blocks (×12)

GPT-2 Small has 12 identical Transformer blocks. Each block processes the input through:
1. Layer Normalization
2. Multi-Head Self-Attention
3. Residual Connection
4. Layer Normalization
5. Feed-Forward Network
6. Residual Connection

**Block Input/Output:**
```
Input:  h[b, l, 768] (from previous block or embedding layer)
Output: h[b, l, 768] (to next block or final layer norm)
```

**Single Block Parameters:**
- Layer Norm 1: 2 × 768 = 1,536
- Attention: 4 × (768 × 768) = 2,359,296
- Layer Norm 2: 2 × 768 = 1,536
- MLP: 2 × 768 × 3,072 + 3,072 + 768 = 4,722,432
- **Total per block: 7,084,800**
- **All 12 blocks: 85,017,600 (68% of total parameters)**

---

### 2.1 Layer Normalization 1

**개요:** 입력을 평균 0, 분산 1로 정규화 (Attention 전 적용)

**입력/출력:**
```
Input:  h [B, L, 768]
Output: h_norm [B, L, 768]
```

**수학적 정의:**
```
μ = mean(h, dim=-1)  # [B, L]
σ² = var(h, dim=-1)   # [B, L]

h_norm = γ ⊙ (h - μ) / √(σ² + ε) + β

where:
- γ (gain): [768] learnable
- β (bias): [768] learnable
- ε = 1e-5 (numerical stability)
- ⊙ : element-wise multiplication
```

**처리 과정:**
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model=768, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: [B, L, d_model]
        mean = x.mean(dim=-1, keepdim=True)  # [B, L, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, L, 1]

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # [B, L, d_model]

        # Scale and shift
        out = self.gamma * x_norm + self.beta  # [B, L, d_model]

        return out
```

**상세 연산 (B=1, L=1024, d=768):**
```
Step 1: Compute mean
  sum = Σ(h[l, d]) for d in 0..767
  mean = sum / 768
  Operations: 768 additions + 1 division = 769 ops per token
  Total: 1024 × 769 = 786,176 ops

Step 2: Compute variance
  var = Σ((h[l, d] - mean)²) / 768
  Operations: 768 subtractions + 768 multiplications + 768 additions + 1 division
            = 2,305 ops per token
  Total: 1024 × 2,305 = 2,360,320 ops

Step 3: Normalize
  x_norm = (h - mean) / sqrt(var + ε)
  Operations: 768 subtractions + 1 sqrt + 768 divisions
            = 1,537 ops per token
  Total: 1024 × 1,537 = 1,573,888 ops

Step 4: Scale and shift
  out = gamma * x_norm + beta
  Operations: 768 multiplications + 768 additions = 1,536 ops per token
  Total: 1024 × 1,536 = 1,572,864 ops

Total FLOPs: 786,176 + 2,360,320 + 1,573,888 + 1,572,864 ≈ 6.3M FLOPs
```

**하드웨어 연산 분석:**

```
Operation Type: Reduction + Element-wise operations

CPU:
├─ Reduction (mean, var): Sequential across d dimension
│  └─ Cannot vectorize reduction easily
├─ Element-wise ops: Vectorizable (AVX-512)
├─ Cycles: ~2,000-3,000 per token
├─ Total (L=1024): ~2-3M cycles = 1 ms @ 2.5 GHz
└─ Bottleneck: Reduction operations

GPU:
├─ Parallel reduction: Warp-level primitives (__shfl_down)
├─ Element-wise ops: Massively parallel
├─ Kernel fusion: Combine all operations
├─ Time: ~50-100 μs
└─ Memory-bound (low arithmetic intensity)

NPU:
├─ Vector reduction units
├─ Tree-based reduction in systolic array
├─ Fused with attention (no intermediate write)
├─ Time: ~30-50 μs
└─ Power-efficient (0.2 W)

Memory Bandwidth (L=1024):
├─ Read: 1024 × 768 × 4 = 3 MB
├─ Write: 1024 × 768 × 4 = 3 MB
├─ Total: 6 MB
├─ Arithmetic Intensity: 6.3M FLOPs / 6 MB = 1.05 FLOPs/byte
└─ Memory-bound on all hardware!

Optimization: Fuse with next operation (avoid write-read)
```

**√(var + ε) 하드웨어 구현:**

LayerNorm의 정규화 단계에서 `1/√(var + ε)` 연산이 필요합니다. sqrt는 비선형 연산이므로 하드웨어에서 특별한 처리가 필요합니다:

```
왜 sqrt가 필요한가:
├─ 정규화: x_norm = (x - μ) / √(σ² + ε)
├─ 비선형 연산 → 단순 곱/덧셈으로 불가
└─ 하드웨어 구현이 필요

rsqrt (Reciprocal Square Root) 접근:
├─ 1/√x를 먼저 계산 (rsqrt)
├─ x · rsqrt(x) = √x (필요시)
├─ rsqrt가 더 효율적인 이유:
│  └─ 나눗셈을 곱셈으로 변환
│  └─ (x - μ) × rsqrt(var) = 정규화
└─ GPU/NPU 모두 rsqrt 명령어 내장

LUT + Newton-Raphson (가장 흔한 구현):
┌─────────────────────────────────────────────────────────┐
│ Step 1: LUT로 초기값 추정                                │
│   - 상위 7-8 비트로 테이블 인덱싱                        │
│   - 128-256 엔트리 LUT                                  │
│   - 초기 정확도: ~8 비트                                 │
│                                                         │
│ Step 2: Newton-Raphson 반복 (1-2회)                     │
│   y_new = y × (1.5 - 0.5 × x × y²)                     │
│   - 1회 반복: ~16 비트 정확도                            │
│   - 2회 반복: ~32 비트 정확도 (FP32 충분)                │
│   - 곱셈만 사용 (나눗셈 없음!)                           │
└─────────────────────────────────────────────────────────┘

Non-uniform LUT 설계:
├─ Uniform spacing은 비효율적
│  └─ √x는 작은 x에서 변화가 급격
│  └─ 큰 x에서는 완만 → 해상도 낭비
├─ Log-domain 처리:
│  └─ FP32: 지수부(8비트) + 가수부(23비트) 분리
│  └─ 지수부: 2로 나눔 (right shift)
│  └─ 가수부: LUT로 처리
├─ Piecewise Linear:
│  └─ 구간별 기울기 + offset 저장
│  └─ LUT 크기 절약
└─ 작은 값에 촘촘하게 (Non-linear spacing)

NPU/TPU 최적화:
┌─────────────────────────────────────────────────────────┐
│ BF16 (Brain Float 16):                                  │
│   - 가수부 7비트만 → LUT 128 엔트리로 충분              │
│   - NR 반복 없이 LUT만으로 정확도 확보                  │
│   - 면적/전력 절약                                      │
│                                                         │
│ CORDIC 알고리즘:                                        │
│   - Shift + Add만으로 sqrt 계산                         │
│   - LUT 불필요 (면적 최소화)                            │
│   - 반복 횟수로 정확도 조절                             │
│   - 저전력 엣지 디바이스에 적합                         │
│                                                         │
│ LayerNorm 특화 최적화:                                  │
│   - Variance는 항상 양수                                │
│   - 범위가 예측 가능 (학습된 분포)                      │
│   - LUT 범위를 좁게 설계 가능                           │
└─────────────────────────────────────────────────────────┘

하드웨어별 구현:
┌──────────┬────────────────────────────────────────────┐
│ Hardware │ sqrt/rsqrt Implementation                  │
├──────────┼────────────────────────────────────────────┤
│ Intel CPU│ RSQRTSS (SSE) + 1 NR iteration            │
│          │ ~5 cycles + ~10 cycles = 15 cycles        │
├──────────┼────────────────────────────────────────────┤
│ NVIDIA   │ SFU (Special Function Unit)               │
│ GPU      │ rsqrtf(): ~8 cycles, full precision       │
│          │ __frsqrt_rn(): ~2 cycles, fast approx     │
├──────────┼────────────────────────────────────────────┤
│ Apple    │ Vector rsqrt in Neural Engine             │
│ NPU      │ Fused with LayerNorm pipeline             │
│          │ ~1 cycle amortized (pipelined)            │
├──────────┼────────────────────────────────────────────┤
│ TPU      │ BF16 optimized LUT                        │
│          │ Fused bfloat16 rsqrt                      │
└──────────┴────────────────────────────────────────────┘
```

---

### 2.2 Multi-Head Self-Attention

**개요:** 시퀀스 내 모든 위치 간의 관계를 학습하는 핵심 메커니즘

**입력/출력:**
```
Input:  h_norm [B, L, 768]
Output: attn_out [B, L, 768]
```

**수학적 정의:**
```
MultiHeadAttention(X) = Concat(head₁, ..., head₁₂) @ W_o

where each head:
  Q = X @ W_q  # Query
  K = X @ W_k  # Key
  V = X @ W_v  # Value

  head_i = Attention(Q_i, K_i, V_i)
         = softmax(Q_i @ K_i^T / √d_k) @ V_i

Parameters:
- W_q, W_k, W_v, W_o: each [768, 768]
- Total: 4 × 768 × 768 = 2,359,296
```

**Query, Key, Value의 역할:**
```
Q, K, V 각각의 의미:
┌─────────┬─────────────────────────────────────────────────────┐
│ Query   │ "나는 무엇을 찾고 있나?" - 검색 의도               │
│ (Q)     │ position i에서 "어떤 정보가 필요한지" 표현         │
├─────────┼─────────────────────────────────────────────────────┤
│ Key     │ "나는 어떤 특성을 가졌나?" - 검색 대상의 특성      │
│ (K)     │ position j가 "어떤 맥락에서 찾아져야 하는지" 표현  │
├─────────┼─────────────────────────────────────────────────────┤
│ Value   │ "내가 전달할 실제 정보" - 실제 내용                │
│ (V)     │ position j가 "실제로 전달할 의미 정보"             │
└─────────┴─────────────────────────────────────────────────────┘

Attention 연산 흐름:
┌─────────────────────────────────────────────────────────────┐
│ 1. Q @ K^T = attention score (상관관계/유사도 점수)        │
│    → "Query i가 Key j와 얼마나 관련있나?"                  │
│                                                             │
│ 2. softmax(score) = attention weight (확률 분포)           │
│    → "Query i가 각 Key j를 얼마나 attend할까?"             │
│                                                             │
│ 3. weight @ V = output (가중 합)                           │
│    → "관련도에 따라 Value들을 섞어서 출력"                 │
└─────────────────────────────────────────────────────────────┘
```

**검색 엔진 비유:**
```
Google 검색과 비교:
├─ Query  = 검색어 ("맛있는 파스타 레시피")
├─ Key    = 문서의 제목/태그 (검색 대상 특성)
├─ Value  = 문서의 실제 내용 (반환할 정보)
└─ 결과   = Query와 Key가 유사한 문서들의 Value를 가중 합

Self-Attention:
├─ Query  = position i의 "내가 뭘 찾고 싶은지" 벡터
├─ Key    = position j의 "내가 어떤 특성인지" 벡터
├─ Value  = position j의 "내가 전달할 정보" 벡터
└─ 결과   = Q·K 유사도로 V들을 가중 합
```

**왜 K와 V를 분리하는가?**
```
K = V 동일하다면:
├─ "유사도 계산용 특성" = "전달할 정보"
└─ 유연성 부족

K ≠ V 분리하면:
├─ K: "이 토큰이 어떤 맥락에서 검색되어야 하나" (검색 특성)
├─ V: "이 토큰이 실제로 전달할 의미는 무엇인가" (정보)
└─ 더 풍부한 표현력

예시 - "은행" 토큰:
├─ K: "금융 관련 문맥에서 찾아지도록" 학습될 수 있음
├─ V: "금융 기관의 의미 정보" 전달
└─ 또는 문맥에 따라 K가 "강가" 관련으로 학습되어
   다른 V 정보를 전달할 수 있음
```

**각 Head는 독립적인 Wq, Wk, Wv를 가지는가?**

```
개념적으로: 각 head별 독립 행렬

  Head 1:  Wq₁ [768, 64],  Wk₁ [768, 64],  Wv₁ [768, 64]
  Head 2:  Wq₂ [768, 64],  Wk₂ [768, 64],  Wv₂ [768, 64]
  ...
  Head 12: Wq₁₂ [768, 64], Wk₁₂ [768, 64], Wv₁₂ [768, 64]

실제 구현: 하나의 큰 행렬로 합쳐서 저장 (연산 효율)

  W_q [768, 768] = [Wq₁ | Wq₂ | ... | Wq₁₂]  ← column-wise concat
  W_k [768, 768] = [Wk₁ | Wk₂ | ... | Wk₁₂]
  W_v [768, 768] = [Wv₁ | Wv₂ | ... | Wv₁₂]

  X @ W_q → [L, 768] → view(L, 12, 64) → 12개 head로 split
  → GEMM을 한 번만 실행 (효율적)
  → 수학적으로 12번 따로 하는 것과 동일
```

**파라미터 수는 Single-Head와 동일:**
```
Single-Head:  Wq [768, 768]       = 589,824 params
Multi-Head:   12 × Wq [768, 64]   = 12 × 49,152 = 589,824 params (동일!)
```

**그럼 왜 Multi-Head가 더 좋은가?**
```
Single-Head Attention (d_k=768):
├─ 하나의 attention pattern
└─ 한 가지 관계만 포착

Multi-Head Attention (12 heads, d_k=64):
├─ Head 1:  문법적 의존 관계 ("주어 ↔ 동사")
├─ Head 2:  의미적 유사성 ("동의어들")
├─ Head 3:  위치적 패턴 ("인접 토큰")
├─ ...
├─ Head 12: 또 다른 패턴
└─ → 12가지 관계를 동시에 병렬 포착 → concat → W_o로 통합

핵심: 파라미터 수는 같지만 표현력이 훨씬 풍부함
├─ 각 head가 768차원을 64차원 부분공간(subspace)으로 투영
├─ 각 subspace에서 서로 다른 관계 특화 학습
└─ 최종적으로 다양한 관계 정보를 한 벡터로 통합
```

---

**처리 과정:**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=12):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, _ = x.shape

        # 1. Linear projections
        Q = self.W_q(x)  # [B, L, 768]
        K = self.W_k(x)  # [B, L, 768]
        V = self.W_v(x)  # [B, L, 768]

        # 2. Reshape for multi-head
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, 12, L, 64]
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, 12, L, 64]
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, 12, L, 64]

        # 3. Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)  # [B, 12, L, L]
        attn_weights = F.softmax(scores, dim=-1)  # [B, 12, L, L]
        attn_output = attn_weights @ V  # [B, 12, L, 64]

        # 4. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, 12, 64]
        attn_output = attn_output.view(B, L, self.d_model)  # [B, L, 768]

        # 5. Output projection
        output = self.W_o(attn_output)  # [B, L, 768]

        return output
```

**상세 연산 분해 (B=1, L=1024, d=768, H=12, d_k=64):**

**Step 1: QKV Projections (3× Matrix Multiply)**
```
Q = X @ W_q  where X: [1024, 768], W_q: [768, 768]

FLOPs per projection: 2 × 1024 × 768 × 768 = 1,207,959,552 (1.2B)
Total for Q, K, V: 3 × 1.2B = 3.6B FLOPs

Memory:
├─ Read X: 1024 × 768 × 4 = 3 MB (reused 3×)
├─ Read W_q, W_k, W_v: 3 × 768 × 768 × 4 = 7 MB
├─ Write Q, K, V: 3 × 1024 × 768 × 4 = 9 MB
└─ Total: 3 + 7 + 9 = 19 MB
```

**Step 2: Attention Scores (QK^T)**
```
Scores = Q @ K^T  where Q: [12, 1024, 64], K: [12, 1024, 64]
Result: [12, 1024, 1024]

FLOPs: 12 × 1024 × 1024 × 64 × 2 = 1,610,612,736 (1.6B)

Memory (critical!):
├─ Read Q, K: 2 × 12 × 1024 × 64 × 4 = 6 MB
├─ Write Scores: 12 × 1024 × 1024 × 4 = 48 MB (!!)
└─ Attention matrix is HUGE: O(L²) problem
```

**Step 3: Softmax**

**인덱스 설명:**
```
scores shape: [B, H, L, L] = [1, 12, 1024, 1024]
                 ↑  ↑   ↑    ↑
                 B  h   i    j
               dim0 dim1 dim2 dim3

h = head index     (0~11, 총 12개 헤드)
i = query position (0~L-1, "내가 어디서 보고 있나")
j = key position   (0~L-1, "어디를 보고 있나")

scores[b, h, i, :] → 헤드 h에서, 위치 i의 토큰이 모든 위치(j=0~L-1)를 얼마나 attend하는지
```

**Softmax 적용 차원 (dim=-1, 즉 j):**
```python
attn_weights = F.softmax(scores, dim=-1)  # PyTorch 코드
                               ↑
                          마지막 차원 (j, key position)
```

```
왜 마지막 차원(j)에 softmax를 적용하는가?

┌─────────────────────────────────────────────────────────┐
│ scores[b, h, i, :] = query i가 모든 key j에 대한 점수   │
│                              ↓ softmax (dim=-1)         │
│ attn[b, h, i, :]   = query i가 각 key j를 attend할 확률 │
│                      (합 = 1.0)                         │
└─────────────────────────────────────────────────────────┘

의미:
├─ i (dim=2): "나는 position i에서 보고 있다" (query)
├─ j (dim=3): "position j들을 얼마나 볼까" (keys)
└─ softmax(dim=-1): j들에 대한 확률 분포 생성

차원별 softmax 적용 여부:
┌──────┬─────────┬────────────────────┬──────────┐
│ dim  │ 인덱스  │ 역할               │ softmax? │
├──────┼─────────┼────────────────────┼──────────┤
│ 0    │ B       │ batch              │ ❌       │
│ 1    │ H       │ head               │ ❌       │
│ 2    │ i       │ query position     │ ❌       │
│ 3    │ j       │ key position       │ ✅       │
└──────┴─────────┴────────────────────┴──────────┘

❌ 만약 dim=2 (i)에 softmax를 적용했다면?
   scores[b, h, :, j] → "key j가 어떤 query i에게 attend 받을까?"
   → attention의 본래 의미가 아님! (역방향)

✅ dim=-1 (j)에 softmax 적용:
   scores[b, h, i, :] → "query i가 어떤 key j를 attend할까?"
   → 이것이 attention의 정의
```

**Softmax 상세 연산:**
```
scores[h, i, :] = [s₀, s₁, s₂, ..., s₁₀₂₃]  ← 1024개 값 (raw attention scores)
                         ↓ softmax
attn_weights[h, i, :] = [a₀, a₁, a₂, ..., a₁₀₂₃]  ← 합이 1인 확률 분포
```

```python
def softmax(scores):
    """
    scores: [1024] - 한 query position의 모든 key에 대한 점수
    """
    # Step 1: Numerical Stability (max subtraction)
    # exp(큰 값) → overflow 방지
    max_score = max(scores)           # 1023 comparisons
    scores = scores - max_score       # 1024 subtractions

    # Step 2: Exponential
    exp_scores = exp(scores)          # 1024 exp operations

    # Step 3: Sum
    sum_exp = sum(exp_scores)         # 1023 additions

    # Step 4: Normalize (divide)
    attn_weights = exp_scores / sum_exp  # 1024 divisions

    return attn_weights
```

**연산 카운트 (정확한 버전):**
```
Per softmax (1024 elements):
┌─────────────────────┬────────────┬─────────────────────┐
│ Operation           │ Count      │ Notes               │
├─────────────────────┼────────────┼─────────────────────┤
│ Find max            │ 1,023      │ Comparisons         │
│ Subtract max        │ 1,024      │ Subtractions        │
│ Exponential         │ 1,024      │ exp() - expensive!  │
│ Sum                 │ 1,023      │ Additions           │
│ Divide              │ 1,024      │ Divisions           │
├─────────────────────┼────────────┼─────────────────────┤
│ Total               │ 5,118      │ Per softmax call    │
└─────────────────────┴────────────┴─────────────────────┘

Total softmax calls: H × L = 12 × 1024 = 12,288
Total operations: 12,288 × 5,118 = 62.9M ops
```

**Causal Mask와 Softmax:**
```
GPT-2는 causal (autoregressive) → 미래 토큰 못 봄

Position i=2인 경우 (3번째 토큰):
scores[h, 2, :] = [s₀, s₁, s₂, -∞, -∞, -∞, ..., -∞]
                   ↑   ↑   ↑   ↑
                  볼수있음   masked (future tokens)

softmax 후:
attn[h, 2, :] = [0.3, 0.5, 0.2, 0, 0, 0, ..., 0]
                 ↑              ↑
              합 = 1.0      exp(-∞) = 0

실제 연산량 (삼각형 구조):
├─ Position 0: softmax over 1 element
├─ Position 1: softmax over 2 elements
├─ Position i: softmax over (i+1) elements
└─ 평균: L/2 elements per softmax
   → 이론적으로 ~절반 연산 절약 가능 (sparse 구현시)
```

**시각화 예시:**
```
Head h=0, Query position i=3 ("world"가 attend하는 패턴):

     j=0     j=1    j=2    j=3    j=4    j=5  ...
    "Hello"  ","    " "   "world" [MASK] [MASK]
      ↓       ↓      ↓      ↓       ↓      ↓
     0.1     0.3   0.05   0.55     0      0    ← attn_weights[0, 3, :]
      ↑                     ↑
 약간 attention        자기 자신에게 많이

softmax 보장:
├─ 모든 값 ≥ 0
├─ 합 = 1.0 (확률 분포)
└─ masked position = 0 (exp(-∞) = 0)
```

**exp() 하드웨어 구현:**
```
exp()는 softmax에서 가장 비싼 연산:
┌─────────────────────────────────────────────────────────┐
│ CPU:  ~15-20 cycles per exp (SIMD: vexpps)             │
│ GPU:  SFU (Special Function Unit), ~8 cycles           │
│ NPU:  LUT + polynomial approximation, ~2-4 cycles      │
│                                                         │
│ 최적화: exp(x) = 2^(x/ln2) = 2^(x × 1.4427)            │
│         2^x는 bit manipulation으로 빠르게 근사 가능      │
│         (IEEE 754 지수부 직접 조작)                     │
└─────────────────────────────────────────────────────────┘

하드웨어별 exp 구현:
┌──────────┬────────────────────────────────────────────┐
│ Hardware │ exp() Implementation                       │
├──────────┼────────────────────────────────────────────┤
│ Intel CPU│ vexpps (AVX-512): ~4 cycles/8 elements    │
│          │ 또는 LUT + Taylor series                   │
├──────────┼────────────────────────────────────────────┤
│ NVIDIA   │ SFU: __expf() ~8 cycles                   │
│ GPU      │ Full precision: expf() ~20 cycles         │
├──────────┼────────────────────────────────────────────┤
│ NPU      │ Piecewise polynomial (degree 2-3)         │
│          │ 또는 small LUT + linear interpolation     │
└──────────┴────────────────────────────────────────────┘
```

**Online Softmax (Flash Attention 핵심):**
```
기존 방식 (메모리 비효율):
┌─────────────────────────────────────────────────────────┐
│ 1. scores 전체 계산 → 48 MB 저장                        │
│ 2. max 찾기 (전체 스캔)                                 │
│ 3. exp 계산 → 48 MB 저장                               │
│ 4. sum 계산 (전체 스캔)                                 │
│ 5. divide → 48 MB 저장                                 │
│ 메모리: O(L²) = 48 MB                                   │
└─────────────────────────────────────────────────────────┘

Online Softmax (streaming):
┌─────────────────────────────────────────────────────────┐
│ max, sum을 streaming으로 동시 계산:                     │
│                                                         │
│ m = -∞, d = 0  (running max, running denominator)      │
│ for each score s:                                       │
│     m_new = max(m, s)                                   │
│     d_new = d × exp(m - m_new) + exp(s - m_new)        │
│     m, d = m_new, d_new                                │
│                                                         │
│ 메모리: O(1) - max, sum 두 개만 유지!                   │
│ Flash Attention이 이 방식 사용                          │
│ → 48 MB attention matrix를 SRAM에 안 써도 됨           │
└─────────────────────────────────────────────────────────┘
```

**Memory:**
```
├─ Read scores: 48 MB
├─ Write attn_weights: 48 MB
└─ Total: 96 MB (read + write)

Flash Attention 사용시:
├─ Tiled computation in SRAM
├─ Never materialize full 48 MB matrix
└─ ~10× memory bandwidth 절약
```

**Step 4: Attention × Value**
```
Output = attn_weights @ V  where attn: [12, 1024, 1024], V: [12, 1024, 64]
Result: [12, 1024, 64]

FLOPs: 12 × 1024 × 1024 × 64 × 2 = 1.6B

Memory:
├─ Read attn_weights: 48 MB
├─ Read V: 3 MB
├─ Write output: 3 MB
└─ Total: 54 MB
```

**Step 5: Output Projection**
```
Output = concat(heads) @ W_o
where concat: [1024, 768], W_o: [768, 768]

FLOPs: 2 × 1024 × 768 × 768 = 1.2B

Memory:
├─ Read concat: 3 MB
├─ Read W_o: 2.3 MB
├─ Write output: 3 MB
└─ Total: 8.3 MB
```

**Attention 총계 (B=1, L=1024):**
```
Total FLOPs: 3.6B + 1.6B + 0.04B + 1.6B + 1.2B = 8.04B FLOPs
Total Memory: 19 + 54 + 96 + 54 + 8.3 = 231 MB

Dominant cost:
├─ Compute: QK^T and attn@V (1.6B each)
├─ Memory: Attention matrix (48 MB × 2 read/write)
└─ Bottleneck: O(L²) memory for attention matrix!
```

**하드웨어 연산 분석:**

```
CPU (Intel Xeon, 2.5 GHz):
├─ QKV projections: ~30 ms (GEMM)
├─ Attention scores: ~15 ms (GEMM)
├─ Softmax: ~5 ms (reduction)
├─ Attn × V: ~15 ms (GEMM)
├─ Output projection: ~10 ms (GEMM)
├─ Total: ~75 ms per attention layer
└─ All 12 layers: ~900 ms

GPU (NVIDIA A100):
├─ QKV projections: ~100 μs (Tensor Cores, FP16)
├─ Attention scores: ~50 μs (GEMM)
├─ Softmax: ~20 μs (warp-level primitives)
├─ Attn × V: ~50 μs (GEMM)
├─ Output projection: ~30 μs (Tensor Cores)
├─ Total: ~250 μs per layer
├─ All 12 layers: ~3 ms
├─ Speedup: 300× vs CPU
└─ Memory bandwidth: 231 MB / 2 TB/s = 115 μs (memory not bottleneck)

GPU Optimizations:
├─ Tensor Cores: FP16 matmul at 312 TFLOPS
│  └─ QKV: 3.6B FLOPs / 312 TFLOPS = 11.5 μs (theoretical)
├─ Flash Attention: Eliminates O(L²) memory write
│  └─ Reduces 48 MB attention matrix to tiles
│  └─ 2-4× faster, enables longer sequences
└─ Fused softmax: Single kernel (no intermediate write)

NPU (Apple Neural Engine):
├─ Systolic array for GEMM: Efficient
├─ But: Attention matrix doesn't fit in SRAM (48 MB > 16 MB)
├─ Strategy: Tile attention (compute in chunks)
├─ QKV projections: ~500 μs (INT8)
├─ Attention: ~2-3 ms (tiled, DRAM access)
├─ Total per layer: ~3-4 ms
├─ All 12 layers: ~40 ms
└─ Trade-off: Slower than GPU, but 50× more power-efficient

Flash Attention (Critical Optimization):
┌──────────────────────────────────────────────────────┐
│ Standard Attention:                                  │
│ 1. Compute full S = Q @ K^T → 48 MB write to HBM    │
│ 2. Softmax(S) → 48 MB read + 48 MB write            │
│ 3. attn @ V → 48 MB read                            │
│ Total HBM traffic: 192 MB                            │
│                                                      │
│ Flash Attention:                                     │
│ 1. Tile Q, K, V into blocks (fit in SRAM)          │
│ 2. Compute attention block-by-block in SRAM         │
│ 3. Never materialize full attention matrix          │
│ Total HBM traffic: 9 MB (Q, K, V read + output)    │
│ Reduction: 21× less memory traffic!                 │
└──────────────────────────────────────────────────────┘
```

**Causal Masking (GPT-2):**
```
GPT-2 uses causal masking (prevents attending to future tokens)

Attention mask:
┌────────────────────────┐
│ 1  0  0  0  0  0  0  0 │  ← Token 0 can only see token 0
│ 1  1  0  0  0  0  0  0 │  ← Token 1 can see tokens 0, 1
│ 1  1  1  0  0  0  0  0 │
│ 1  1  1  1  0  0  0  0 │
│ 1  1  1  1  1  0  0  0 │
│ 1  1  1  1  1  1  0  0 │
│ 1  1  1  1  1  1  1  0 │
│ 1  1  1  1  1  1  1  1 │  ← Last token can see all
└────────────────────────┘

Implementation:
scores = Q @ K^T  # [B, H, L, L]
scores = scores.masked_fill(mask == 0, -inf)
attn = softmax(scores)  # Masked positions → 0 after softmax

Effect on computation:
- Still compute full O(L²) matrix
- But effectively ~half are masked (triangular)
- Sparse attention could optimize this (future work)
```

---

### 2.3 Residual Connection 1

**개요:** Attention 출력을 원래 입력에 더함 (gradient flow 개선)

**입력/출력:**
```
Input:  h_original [B, L, 768], attn_out [B, L, 768]
Output: h [B, L, 768]
```

**수학적 표현:**
```
h = h_original + attn_out

Operations: B × L × d additions
          = 1 × 1024 × 768 = 786,432 additions
```

**처리 과정:**
```python
# Before attention
h_original = h.clone()  # Save input

# Attention
h_norm = layer_norm1(h)
attn_out = attention(h_norm)

# Residual connection
h = h_original + attn_out  # Element-wise addition
```

**하드웨어 분석:**
```
Same as Embedding Addition (section 1.4)

CPU (AVX-512): ~40 μs
GPU: ~5 μs
NPU (fused): ~0 μs (combined with attention output write)
```

---

### 2.4 Layer Normalization 2

**개요:** MLP 전에 적용되는 정규화 (Layer Norm 1과 동일)

**입력/출력:**
```
Input:  h [B, L, 768]
Output: h_norm [B, L, 768]
```

Same as Layer Normalization 1 (section 2.1)

---

### 2.5 Feed-Forward Network (MLP)

**개요:** 2-layer fully connected network with GELU activation

**입력/출력:**
```
Input:  h_norm [B, L, 768]
Output: mlp_out [B, L, 768]
```

**수학적 정의:**
```
MLP(x) = W₂ @ GELU(W₁ @ x + b₁) + b₂

where:
- W₁: [768, 3072] (expansion)
- b₁: [3072]
- GELU: Gaussian Error Linear Unit activation
- W₂: [3072, 768] (projection)
- b₂: [768]

Parameters: 768×3072 + 3072 + 3072×768 + 768 = 4,722,432
```

**처리 과정:**
```python
class MLP(nn.Module):
    def __init__(self, d_model=768, d_ff=3072):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: [B, L, d_model]

        # Expansion
        hidden = self.fc1(x)  # [B, L, 3072]

        # GELU activation
        hidden = self.gelu(hidden)  # [B, L, 3072]

        # Projection
        output = self.fc2(hidden)  # [B, L, d_model]

        return output
```

**상세 연산 (B=1, L=1024):**

**Step 1: Linear 1 (Expansion)**
```
hidden = X @ W₁ + b₁
where X: [1024, 768], W₁: [768, 3072]

FLOPs: 2 × 1024 × 768 × 3072 = 4,831,838,208 (4.8B)

Memory:
├─ Read X: 1024 × 768 × 4 = 3 MB
├─ Read W₁: 768 × 3072 × 4 = 9.4 MB
├─ Write hidden: 1024 × 3072 × 4 = 12.6 MB
└─ Total: 25 MB
```

**Step 2: GELU Activation**
```
GELU(x) = x · Φ(x) = x · 0.5 · (1 + erf(x/√2))

Approximation (faster):
GELU(x) ≈ x · σ(1.702x)
or
GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])

Operations per element:
├─ Multiply: ~3-5
├─ Add: ~2-3
├─ Nonlinear (tanh/erf): ~10-20 (approximated)
└─ Total: ~15-28 ops

Total operations: 1024 × 3072 × 20 ≈ 63M ops

Memory:
├─ Read hidden: 12.6 MB
├─ Write hidden: 12.6 MB (in-place possible)
└─ Total: 25 MB
```

**Step 3: Linear 2 (Projection)**
```
output = hidden @ W₂ + b₂
where hidden: [1024, 3072], W₂: [3072, 768]

FLOPs: 2 × 1024 × 3072 × 768 = 4,831,838,208 (4.8B)

Memory:
├─ Read hidden: 12.6 MB
├─ Read W₂: 3072 × 768 × 4 = 9.4 MB
├─ Write output: 3 MB
└─ Total: 25 MB
```

**MLP 총계:**
```
Total FLOPs: 4.8B + 0.06B + 4.8B = 9.66B FLOPs
Total Memory: 25 + 25 + 25 = 75 MB

Note: MLP is MORE compute than Attention!
- MLP: 9.66B FLOPs
- Attention: 8.04B FLOPs
```

**하드웨어 연산 분석:**

```
CPU (Intel Xeon):
├─ FC1: ~40 ms (GEMM)
├─ GELU: ~5 ms (element-wise)
├─ FC2: ~40 ms (GEMM)
├─ Total: ~85 ms per MLP layer
└─ All 12 layers: ~1 second

GPU (NVIDIA A100):
├─ FC1: ~30 μs (Tensor Cores, FP16)
│  └─ 4.8B FLOPs / 312 TFLOPS = 15.4 μs (theoretical)
├─ GELU: ~10 μs (fused with FC1)
├─ FC2: ~30 μs (Tensor Cores)
├─ Total: ~70 μs per layer (with fusion)
├─ All 12 layers: ~840 μs
└─ Speedup: ~1,200× vs CPU

GPU Optimizations:
├─ Tensor Core utilization: Near 100%
│  └─ Large matrix multiplies (ideal for Tensor Cores)
├─ Fused FC1 + GELU: Single kernel
│  └─ Eliminates 12.6 MB intermediate write
├─ High arithmetic intensity:
│  └─ 9.66B FLOPs / 75 MB = 129 FLOPs/byte
│  └─ Compute-bound (good for GPU!)
└─ Memory bandwidth: 75 MB / 2 TB/s = 37.5 μs

NPU (Apple Neural Engine):
├─ Systolic array ideal for GEMM
├─ INT8 quantization:
│  └─ Weights: 9.4 + 9.4 = 18.8 MB → 4.7 MB (4×)
│  └─ Activations: INT8 or FP16
├─ FC1: ~300 μs (INT8)
├─ GELU: ~50 μs (vector ALU)
├─ FC2: ~300 μs (INT8)
├─ Total: ~650 μs per layer
├─ All 12 layers: ~8 ms
├─ Power: 0.5 W (vs 50 W GPU)
└─ Energy: 4 mJ (vs 40 mJ GPU = 10× more efficient)

Comparison:
┌──────────┬────────────┬──────────┬──────────┬────────────┐
│ Hardware │ Per Layer  │ 12 Layers│ Power    │ Energy     │
├──────────┼────────────┼──────────┼──────────┼────────────┤
│ CPU      │ 85 ms      │ 1020 ms  │ 3 W      │ 3,060 mJ   │
│ GPU      │ 70 μs      │ 840 μs   │ 50 W     │ 42 mJ      │
│ NPU(INT8)│ 650 μs     │ 7.8 ms   │ 0.5 W    │ 3.9 mJ     │
└──────────┴────────────┴──────────┴──────────┴────────────┘
```

**GELU vs ReLU:**
```
GELU (GPT-2):
├─ Smoother activation (differentiable everywhere)
├─ Better gradient flow
├─ Slightly better performance (empirically)
└─ More expensive to compute (~20 ops vs 2 for ReLU)

ReLU (older models):
├─ Simple: max(0, x)
├─ Fast: 2 ops (compare + conditional move)
└─ But: Dead neurons problem

Performance impact:
- GELU adds ~5-10% compute vs ReLU
- But training converges better
- Worth the cost!
```

---

### 2.6 Residual Connection 2

**개요:** MLP 출력을 원래 입력에 더함

**입력/출력:**
```
Input:  h_original [B, L, 768], mlp_out [B, L, 768]
Output: h [B, L, 768]
```

**수학적 표현:**
```
h = h_original + mlp_out
```

Same as Residual Connection 1 (section 2.3)

**Transformer Block 완료:**
```python
def transformer_block(x):
    # Pre-norm architecture (GPT-2)

    # Attention branch
    h = x
    h = layer_norm1(h)
    h = attention(h)
    x = x + h  # Residual 1

    # MLP branch
    h = x
    h = layer_norm2(h)
    h = mlp(h)
    x = x + h  # Residual 2

    return x
```

**Single Block Summary:**
```
Parameters: 7,084,800
FLOPs: 8.04B (Attention) + 9.66B (MLP) = 17.7B
Memory: 231 MB (Attention) + 75 MB (MLP) = 306 MB
Time (GPU): 250 μs (Attn) + 70 μs (MLP) = 320 μs

All 12 blocks:
├─ Parameters: 85,017,600 (68% of model)
├─ FLOPs: 212B (93% of total FLOPs)
├─ Time (GPU): 3.84 ms
└─ Dominant cost of GPT-2!
```

---

## 3. Output Processing

After 12 Transformer blocks, the output goes through final processing:

---

### 3.1 Final Layer Normalization

**개요:** 마지막 Transformer block 출력 정규화

**입력/출력:**
```
Input:  h [B, L, 768] (from last Transformer block)
Output: h_norm [B, L, 768]
```

Same as Layer Normalization (section 2.1)

```python
h_final = layer_norm_final(h)  # [B, L, 768]
```

**Parameters:** 1,536 (γ, β)
**FLOPs:** 6.3M
**Time (GPU):** ~50 μs

---

### 3.2 Language Model Head

**개요:** 768차원 벡터를 vocabulary 크기 (50,257)의 logits으로 변환

**입력/출력:**
```
Input:  h_norm [B, L, 768]
Output: logits [B, L, 50257]
```

**수학적 정의:**
```
logits = h_norm @ W_lm

where:
- W_lm: [768, 50257] (unembedding matrix)
- W_lm = E^T (weight tying with token embedding)
- No additional parameters!
```

---

**오퍼레이션 상세: 단순한 행렬 곱셈**

```
h_norm [L, 768]  @  W_lm [768, 50257]  =  logits [L, 50257]
       ↑                    ↑                      ↑
  Transformer의           가중치 행렬            각 토큰의
  최종 출력 벡터         (50,257개 열)          raw score
```

**직관: 각 열(column)이 토큰 하나를 의미**

`W_lm [768, 50257]`을 열 기준으로 쪼개면:
```
W_lm = [ w_"the" | w_"Hello" | w_"world" | w_"," | ... ]
          [768]      [768]       [768]      [768]
```

행렬 곱셈은 실질적으로 **50,257번의 내적(dot product)**:
```
logit["the"]   = h_norm · w_"the"    → h와 "the" 벡터의 유사도
logit["Hello"] = h_norm · w_"Hello"  → h와 "Hello" 벡터의 유사도
logit["world"] = h_norm · w_"world"  → h와 "world" 벡터의 유사도
...

높은 logit = "Transformer가 다음으로 이 토큰을 예측함"
```

---

**Weight Tying: W_lm = E^T (전치행렬 공유)**

```
Token Embedding:  E     [50257, 768]  (token_id → 768차원 벡터)
LM Head:          W_lm  [768, 50257]  = E^T  ← 같은 행렬의 전치!
```

추가 파라미터 0개. 같은 행렬을 양방향으로 활용:
```
입력 방향 (Embedding):
  token_id 15496 → E[15496] = [0.23, -0.89, ...]  "Hello의 의미 벡터"

출력 방향 (LM Head):
  h_norm · E[15496] = "h가 Hello의 의미와 얼마나 닮았나?"

→ h_norm이 "Hello" embedding과 방향이 비슷하면
  → logit["Hello"]가 높아짐
  → 다음 토큰으로 "Hello"를 예측
```

즉, **"토큰 → 벡터"** 변환 행렬의 역방향이 **"벡터 → 토큰"** 변환입니다.

---

**전체 흐름 예시: "Hello world" 다음 토큰 예측**

```
Transformer 12블록 통과 후 마지막 위치의 hidden state:
h_norm[-1] = [-0.12, 0.45, ..., 0.33]  ← "world" 이후 문맥 요약 (768차원)

LM Head 적용 (내적 50,257번):
logit["!"]     = h_norm[-1] · E["!"]     =  8.3  ← 높음
logit[","]     = h_norm[-1] · E[","]     =  7.1
logit["."]     = h_norm[-1] · E["."]     =  6.8
logit["the"]   = h_norm[-1] · E["the"]   =  3.2
logit["Hello"] = h_norm[-1] · E["Hello"] =  0.1  ← 낮음
...

→ softmax → 확률 분포 → 샘플링 → "!" 선택
```

---

**학습 시 vs 추론 시: 사용 position 차이**

```
학습 시 (Teacher Forcing):
  position 0의 logit → position 1 토큰 예측  ↗ L개 예측을
  position 1의 logit → position 2 토큰 예측    동시에 계산
  ...                                           (병렬, 효율적)
  position L-1의 logit → position L 토큰 예측 ↗

추론 시 (Autoregressive Generation):
  마지막 position의 logit만 사용 → 다음 토큰 예측
  └─ 나머지 position은 계산 불필요
  └─ FLOPs: 78.9B → 77M (1000× 절약)
```

**처리 과정:**
```python
class GPT2LMHead(nn.Module):
    def __init__(self, d_model=768, vocab_size=50257):
        super().__init__()
        # Weight tying: share with token embedding
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h):
        # h: [B, L, d_model]
        logits = self.lm_head(h)  # [B, L, vocab_size]
        return logits
```

**상세 연산 (B=1, L=1024):**
```
logits = h @ W_lm
where h: [1024, 768], W_lm: [768, 50257]

FLOPs: 2 × 1024 × 768 × 50,257 = 78,875,893,760 (78.9B FLOPs!)

Memory:
├─ Read h: 1024 × 768 × 4 = 3 MB
├─ Read W_lm: 768 × 50,257 × 4 = 154 MB
├─ Write logits: 1024 × 50,257 × 4 = 206 MB
└─ Total: 363 MB (huge!)

Note: This is the LARGEST memory operation in GPT-2!
```

**Optimization: Only compute last token**
```
During generation (autoregressive):
- Only need logits for last token (to predict next)
- Input: h[-1] [1, 768]
- Output: logits [1, 50257]

FLOPs: 2 × 1 × 768 × 50,257 = 77,194,752 (77M, 1000× less!)

Memory:
├─ Read h: 768 × 4 = 3 KB
├─ Read W_lm: 154 MB (still need full matrix)
├─ Write logits: 50,257 × 4 = 201 KB
└─ Total: 154 MB (mostly weight matrix)

This is why generation is efficient!
```

**하드웨어 연산 분석:**

```
Full Sequence (Training, L=1024):

CPU:
├─ FLOPs: 78.9B
├─ Time: 78.9B / (100 GFLOPS × 8 cores) = ~100 ms
└─ Memory-bound (154 MB weight read)

GPU (NVIDIA A100):
├─ FLOPs: 78.9B / 312 TFLOPS = 253 μs (theoretical)
├─ Memory: 363 MB / 2 TB/s = 181 μs
├─ Actual: ~300-500 μs (memory-bound)
└─ Bottleneck: Large vocabulary size

NPU:
├─ Challenges: 154 MB weights >> 16 MB SRAM
├─ Must tile or keep weights in DRAM
├─ Time: ~5-10 ms (DRAM-bound)
└─ Not ideal for large vocabulary

Single Token (Generation, L=1):

CPU:
├─ FLOPs: 77M
├─ Time: 77M / 100 GFLOPS = 770 μs
└─ Still memory-bound (154 MB weight)

GPU:
├─ FLOPs: 77M / 312 TFLOPS = 0.25 μs
├─ Memory: 154 MB / 2 TB/s = 77 μs
├─ Actual: ~100-150 μs
└─ Memory-bound (weight matrix read)

Optimization: Adaptive Softmax
├─ Partition vocabulary into clusters
├─ Frequent tokens: Small matrix
├─ Rare tokens: Larger matrix (computed only if needed)
├─ Reduces average computation by 5-10×
└─ Used in some large language models
```

---

### 3.3 Softmax & Sampling

**개요:** Logits을 확률 분포로 변환하고 다음 토큰 샘플링

**입력/출력:**
```
LM Head 출력:    logits [B, L, vocab_size]  ← 모든 position의 logits
                        ↓ logits[:, -1, :]  ← 마지막 position만 슬라이싱
Sampling 입력:   logits [B, vocab_size]
Output:          next_token_id [B]

학습 시: [B, L, V] 그대로 → 모든 position에 cross-entropy loss 계산
추론 시: [:, -1, :] 슬라이싱 → [B, V] → 샘플링 (다음 토큰 1개 생성)
```

**왜 마지막 position인가:**
```python
logits = lm_head(h_norm)        # [B, L, 50257] — 전체 출력
logits_last = logits[:, -1, :]  # [B, 50257]    — 마지막 position만 추출
next_token = sample(logits_last)

# position -1 (마지막)의 logit = "지금까지 본 모든 토큰을 고려한 후
#                                  다음에 올 토큰의 확률 분포"
# position 0의 logit는 첫 번째 토큰만 보고 예측한 것 → 추론 시 불필요
```

**수학적 정의:**
```
Softmax:
P(token_i) = exp(logit_i / temperature) / Σⱼ exp(logit_j / temperature)

where:
- temperature: controls randomness (default 1.0)
  - temperature → 0: deterministic (argmax)
  - temperature → ∞: uniform random

Sampling strategies:
1. Greedy: argmax(logits)
2. Temperature sampling: sample from P
3. Top-k: sample from top k tokens
4. Top-p (nucleus): sample from smallest set with cumulative prob p
```

**처리 과정:**
```python
def generate_next_token(logits, temperature=1.0, top_k=50, top_p=0.9):
    """
    logits: [vocab_size] logits for next token
    """
    # Apply temperature
    logits = logits / temperature  # [vocab_size]

    # Top-k filtering
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits[top_k_indices] = top_k_logits

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

    # Softmax
    probs = F.softmax(logits, dim=-1)  # [vocab_size]

    # Sample
    next_token = torch.multinomial(probs, num_samples=1)  # [1]

    return next_token
```

**상세 연산:**

**Step 1: Temperature Scaling**
```
logits = logits / temperature

Operations: 50,257 divisions
Time: ~10-20 μs (CPU), ~5 μs (GPU)
```

**Step 2: Top-k Selection**
```
Find top k=50 logits

Algorithm: Partial sort or heap
Operations: O(V log k) where V=50,257
Time: ~100-200 μs (CPU), ~20 μs (GPU)
```

**Step 3: Softmax**
```
For k=50 tokens (after top-k):

Operations:
├─ exp(x): 50 exponentials
├─ sum: 50 additions
├─ divide: 50 divisions
└─ Total: ~150 ops

Time: ~1-2 μs
```

**Step 4: Multinomial Sampling**
```
Sample from categorical distribution

Operations:
├─ Cumulative sum: 50 additions
├─ Random number: 1
├─ Binary search: log₂(50) ≈ 6 comparisons
└─ Total: ~57 ops

Time: <1 μs
```

**Total Sampling Time:**
```
CPU: ~200 μs
GPU: ~50 μs

Note: Negligible compared to model forward pass!
- Forward pass: 5-10 ms
- Sampling: 0.05 ms
- Sampling is <1% of total time
```

**Sampling Strategy Comparison:**

```
Greedy (argmax):
├─ Output: Most likely token
├─ Deterministic
├─ Fast: O(V) scan
├─ Problem: Repetitive, boring text
└─ Use case: Deterministic tasks (code, math)

Temperature Sampling:
├─ Output: Sample from full distribution
├─ Stochastic
├─ Temperature control:
│  ├─ T=0.1: Nearly deterministic, focused
│  ├─ T=1.0: Balanced
│  └─ T=2.0: Very random, creative
└─ Use case: General text generation

Top-k (k=50):
├─ Output: Sample from top 50 tokens
├─ Prevents sampling very unlikely tokens
├─ k=1: Greedy
├─ k=V: Full sampling
└─ Use case: Balanced creativity

Top-p (nucleus, p=0.9):
├─ Output: Sample from smallest set with 90% probability
├─ Adaptive: k varies by context
├─ More robust than fixed top-k
└─ Use case: High-quality generation (GPT-3 default)

Comparison (example distribution):
Token    Prob    Greedy  Temp   Top-k  Top-p
"the"    0.30    ✓       30%    ✓      ✓
"a"      0.25            25%    ✓      ✓
"that"   0.20            20%    ✓      ✓
"this"   0.10            10%    ✓      ✓
"my"     0.08            8%     ✓      ✓
...      0.07            7%     ...    ✗ (cumsum > 0.9)
```

---

## 4. 전체 파이프라인 종합

### 4.1 Complete Forward Pass

```python
class GPT2Small(nn.Module):
    def __init__(self):
        super().__init__()
        # Input processing
        self.token_embedding = nn.Embedding(50257, 768)
        self.position_embedding = nn.Embedding(1024, 768)

        # 12 Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=768, num_heads=12, d_ff=3072)
            for _ in range(12)
        ])

        # Output processing
        self.ln_final = LayerNorm(768)
        self.lm_head = nn.Linear(768, 50257, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        """
        input_ids: [B, L] token IDs
        returns: logits [B, L, vocab_size]
        """
        B, L = input_ids.shape

        # 1. Input processing
        token_emb = self.token_embedding(input_ids)  # [B, L, 768]
        position_ids = torch.arange(L, device=input_ids.device)
        position_emb = self.position_embedding(position_ids)  # [L, 768]
        h = token_emb + position_emb  # [B, L, 768]

        # 2. Transformer blocks
        for block in self.blocks:
            h = block(h)  # [B, L, 768]

        # 3. Output processing
        h = self.ln_final(h)  # [B, L, 768]
        logits = self.lm_head(h)  # [B, L, 50257]

        return logits
```

### 4.2 Performance Summary

**Parameters (Total: 124,439,808):**
```
┌──────────────────────┬─────────────┬────────────┐
│ Component            │ Parameters  │ Percentage │
├──────────────────────┼─────────────┼────────────┤
│ Token Embedding      │ 38,597,376  │ 31.0%      │
│ Position Embedding   │ 786,432     │ 0.6%       │
│ 12× Transformer:     │             │            │
│   - LayerNorm (×24)  │ 36,864      │ 0.03%      │
│   - Attention (×12)  │ 28,311,552  │ 22.7%      │
│   - MLP (×12)        │ 56,669,184  │ 45.5%      │
│ Final LayerNorm      │ 1,536       │ 0.001%     │
│ LM Head              │ 0 (tied)    │ 0%         │
├──────────────────────┼─────────────┼────────────┤
│ Total                │ 124,439,808 │ 100%       │
└──────────────────────┴─────────────┴────────────┘

Key insight: MLP dominates parameters (45.5%)
```

**FLOPs (Forward Pass, B=1, L=1024):**
```
┌──────────────────────┬─────────────┬────────────┐
│ Component            │ FLOPs       │ Percentage │
├──────────────────────┼─────────────┼────────────┤
│ Input Embeddings     │ ~0          │ 0%         │
│ 12× Transformer:     │             │            │
│   - LayerNorm        │ 151M        │ 0.1%       │
│   - Attention        │ 96.5B       │ 43.7%      │
│   - MLP              │ 115.9B      │ 52.5%      │
│ Final LayerNorm      │ 6.3M        │ 0.003%     │
│ LM Head (full seq)   │ 78.9B       │ 35.7%      │
├──────────────────────┼─────────────┼────────────┤
│ Total (w/o LM Head)  │ 212.6B      │ 96.4%      │
│ Total (with LM Head) │ 291.5B      │ 100%       │
└──────────────────────┴─────────────┴────────────┘

Note: During generation (single token):
- LM Head: 77M FLOPs (1000× less)
- Total: 212.7B FLOPs
```

**Memory (Inference, B=1, L=1024, FP32):**
```
Parameters:
├─ Weights: 124M × 4 bytes = 497 MB
└─ (FP16: 249 MB, INT8: 124 MB)

Activations (peak):
├─ Embeddings: 2 × 3 MB = 6 MB
├─ Per Transformer block: ~25 MB
│  └─ Attention matrix: 48 MB (dominant!)
├─ 12 blocks (with reuse): ~80 MB
├─ LM Head output: 206 MB
└─ Total: ~300 MB

KV Cache (for generation):
├─ Per layer: 2 × 1024 × 768 × 4 = 6 MB
├─ 12 layers: 72 MB
└─ Grows linearly with sequence length

Total Inference Memory:
├─ Weights: 497 MB
├─ Activations: 300 MB
├─ KV Cache: 72 MB
└─ Total: ~870 MB (~1 GB)
```

**Latency Breakdown (B=1, L=1024):**

```
CPU (Intel Xeon, 32 cores):
┌──────────────────────┬─────────────┐
│ Component            │ Time        │
├──────────────────────┼─────────────┤
│ Tokenization         │ 1 ms        │
│ Input Embeddings     │ 1.5 ms      │
│ 12× Transformer      │ 2,000 ms    │
│   - LayerNorm (×24)  │ 24 ms       │
│   - Attention (×12)  │ 900 ms      │
│   - MLP (×12)        │ 1,020 ms    │
│ Final LayerNorm      │ 1 ms        │
│ LM Head (full)       │ 100 ms      │
├──────────────────────┼─────────────┤
│ Total                │ ~2.1 s      │
└──────────────────────┴─────────────┘

GPU (NVIDIA A100, FP16 + Tensor Cores):
┌──────────────────────┬─────────────┐
│ Component            │ Time        │
├──────────────────────┼─────────────┤
│ Tokenization (CPU)   │ 1 ms        │
│ Input Embeddings     │ 15 μs       │
│ 12× Transformer      │ 4 ms        │
│   - LayerNorm (×24)  │ 1.2 ms      │
│   - Attention (×12)  │ 1.5 ms      │
│   - MLP (×12)        │ 0.9 ms      │
│ Final LayerNorm      │ 50 μs       │
│ LM Head (full)       │ 500 μs      │
├──────────────────────┼─────────────┤
│ Total                │ ~5.6 ms     │
└──────────────────────┴─────────────┘
Speedup: 375× vs CPU

NPU (Apple A17, INT8):
┌──────────────────────┬─────────────┐
│ Component            │ Time        │
├──────────────────────┼─────────────┤
│ Tokenization (CPU)   │ 1 ms        │
│ Input Embeddings     │ 50 μs       │
│ 12× Transformer      │ 45 ms       │
│   - Attention (×12)  │ 25 ms       │
│   - MLP (×12)        │ 18 ms       │
│ Final LayerNorm      │ 50 μs       │
│ LM Head (full)       │ 8 ms        │
├──────────────────────┼─────────────┤
│ Total                │ ~54 ms      │
│ Power                │ 0.8 W       │
│ Energy               │ 43.2 mJ     │
└──────────────────────┴─────────────┘
GPU Energy: 280 mJ (6.5× more than NPU)
```

**Generation (Autoregressive, 100 tokens):**
```
GPU (with KV Cache):
├─ First token (prompt L=1024): 5.6 ms
├─ Token 2-100 (L=1 each): 99 × 0.3 ms = 29.7 ms
│  └─ KV Cache eliminates re-computation
│  └─ LM Head: 77M FLOPs (fast)
├─ Total: 35.3 ms
└─ Throughput: 100 tokens / 35.3 ms = 2,832 tokens/sec

Without KV Cache:
├─ Token 1 (L=1024): 5.6 ms
├─ Token 2 (L=1025): 5.6 ms (recompute all)
├─ Token 3 (L=1026): 5.6 ms
├─ ...
├─ Token 100 (L=1123): 5.6 ms
├─ Total: 100 × 5.6 ms = 560 ms
└─ Throughput: 100 / 560 ms = 179 tokens/sec

KV Cache Speedup: 15.8× for generation!
```

### 4.3 Bottleneck Analysis

```
Memory Bottlenecks:
┌─────────────────────┬─────────────┬───────────────┐
│ Operation           │ Memory      │ Mitigation    │
├─────────────────────┼─────────────┼───────────────┤
│ Attention Matrix    │ 48 MB O(L²) │ Flash Attn    │
│ LM Head Output      │ 206 MB      │ Compute last  │
│ KV Cache            │ 72 MB       │ MQA/GQA       │
│ Embedding Matrix    │ 154 MB      │ Quantization  │
└─────────────────────┴─────────────┴───────────────┘

Compute Bottlenecks:
┌─────────────────────┬─────────────┬───────────────┐
│ Operation           │ FLOPs       │ Mitigation    │
├─────────────────────┼─────────────┼───────────────┤
│ MLP                 │ 115.9B      │ Sparsity      │
│ Attention           │ 96.5B       │ Sparse Attn   │
│ LM Head             │ 78.9B       │ Adaptive SM   │
└─────────────────────┴─────────────┴───────────────┘

Hardware Utilization:
┌──────────┬─────────────┬─────────────────────────┐
│ Hardware │ Bottleneck  │ Solution                │
├──────────┼─────────────┼─────────────────────────┤
│ CPU      │ Compute     │ Quantization, pruning   │
│ GPU      │ Memory BW   │ Fusion, Flash Attention │
│ NPU      │ SRAM size   │ Tiling, quantization    │
└──────────┴─────────────┴─────────────────────────┘
```

### 4.4 최적화 전략 요약

```
Training:
├─ Mixed Precision (FP16): 2× speedup
├─ Gradient Checkpointing: 3× memory reduction
├─ Distributed Training: Near-linear scaling
└─ Flash Attention: 2-4× faster, enables longer sequences

Inference:
├─ KV Cache: 15× speedup for generation
├─ Quantization (INT8): 4× less memory, 2-4× faster
├─ Flash Attention: 2-4× faster
├─ Operator Fusion: 20-30% speedup
├─ Batch Processing: Amortize fixed costs
└─ Sparse Attention: For very long sequences

Mobile/Edge:
├─ INT8 Quantization: Mandatory
├─ Pruning: 30-50% reduction with minimal loss
├─ Distillation: Smaller student model
└─ NPU Acceleration: 10× energy efficiency
```

---

## 5. End-to-End Example

**Input:** "Hello world"
**Task:** Generate next token

```
Step-by-step execution:

1. Tokenization (CPU):
   "Hello world" → [15496, 995]
   Time: ~2 μs

2. Input Processing (GPU):
   - Token Embedding: [15496, 995] → [2, 768]
   - Position Embedding: [0, 1] → [2, 768]
   - Addition: → [2, 768]
   Time: ~20 μs

3. Transformer Block 1:
   - LayerNorm: [2, 768] → [2, 768]
   - Attention: [2, 768] → [2, 768]
   - Residual: [2, 768] + [2, 768] → [2, 768]
   - LayerNorm: [2, 768] → [2, 768]
   - MLP: [2, 768] → [2, 768]
   - Residual: [2, 768] + [2, 768] → [2, 768]
   Time: ~300 μs

4. Transformer Blocks 2-12:
   Same as above
   Time: 11 × 300 μs = 3.3 ms

5. Final LayerNorm:
   [2, 768] → [2, 768]
   Time: ~50 μs

6. LM Head (last token only):
   [1, 768] → [1, 50257]
   Time: ~150 μs

7. Softmax + Sampling (CPU/GPU):
   [50257] → token_id (e.g., 318 for ",")
   Time: ~50 μs

Total: ~4 ms on GPU A100
Output: Token 318 (",")
Next input: "Hello world,"
```

---

## Conclusion

GPT-2 Small은 간단해 보이지만 복잡한 연산 파이프라인을 가지고 있습니다:

**핵심 특징:**
- **124M parameters**: 68% in MLPs, 23% in Attention
- **212B FLOPs**: MLP (53%), Attention (44%)
- **O(L²) complexity**: Attention matrix is bottleneck
- **Inference: 5-50 ms**: Depending on hardware (GPU vs NPU vs CPU)

**하드웨어 최적화의 중요성:**
- GPU: 375× faster than CPU (with Tensor Cores)
- NPU: 6× more energy-efficient than GPU
- KV Cache: 15× speedup for generation
- Flash Attention: 2-4× faster, enables 8K+ contexts

**Future Directions:**
- Longer contexts (Flash Attention, sparse attention)
- Efficiency (quantization, pruning, distillation)
- Specialized hardware (Groq LPU, Cerebras, TPU v5)
