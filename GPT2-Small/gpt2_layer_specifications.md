# GPT-2 Small Layer-by-Layer Specifications

## 모델 전체 개요
- **모델**: GPT-2 Small
- **총 파라미터**: 124M (124,439,808)
- **Hidden Size (d)**: 768
- **Attention Heads**: 12
- **Layers**: 12
- **Vocabulary Size**: 50,257
- **Max Sequence Length**: 1024

---

## 레이어별 상세 스펙

### 0. Input Processing (Tokenization)

| Field | Value |
|-------|-------|
| **Layer Index** | 0 |
| **Layer Name** | BPE Tokenization |
| **Processing Type** | String Processing (CPU-only) |
| **Input Dimension** | Text (variable length) |
| **Output Dimension** | [B, L] (token IDs) |
| **Mathematical Expression** | `T: Σ* → ℕ^L` (BPE algorithm) |
| **Parameters** | 0 (vocabulary: 50,257 entries) |
| **Required Ops** | ~5,000-10,000 CPU cycles per word |
| **Input Memory BW** | ~10-100 bytes (text input) |
| **Output Memory BW** | L × 4 bytes (INT32 token IDs) |
| **Bottleneck** | Sequential algorithm, hash table lookups |
| **Hardware Suitability** | CPU ★★★★★, GPU ☆☆☆☆☆, NPU ☆☆☆☆☆ |

---

### 1. Token Embedding

| Field | Value |
|-------|-------|
| **Layer Index** | 1 |
| **Layer Name** | Token Embedding (wte) |
| **Processing Type** | Embedding Lookup (Indexed Memory Read) |
| **Input Dimension** | [B, L] (token IDs) |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `E[x[b,l]] → h[b,l]` where `E ∈ ℝ^(50257×768)` |
| **Parameters** | 50,257 × 768 = 38,597,376 |
| **Memory (FP32)** | 154 MB |
| **Required Ops** | B × L lookups (no arithmetic ops) |
| **Input Memory BW** | B × L × 4 bytes (token IDs) |
| **Output Memory BW** | B × L × 768 × 4 = B × L × 3 KB |
| **FLOPs** | 0 (pure memory operation) |
| **Bottleneck** | Memory bandwidth (DRAM access) |
| **Cache Behavior** | Random access, poor locality |
| **Hardware Suitability** | CPU ★★☆☆☆, GPU ★★★★★, NPU ★★★★☆ |

**Performance (B=1, L=1024):**
- CPU: ~1 ms (DRAM-bound)
- GPU: ~3.3 μs (900 GB/s bandwidth)
- NPU: ~30 μs (with SRAM caching: ~10 μs)

---

### 2. Position Embedding

| Field | Value |
|-------|-------|
| **Layer Index** | 2 |
| **Layer Name** | Position Embedding (wpe) |
| **Processing Type** | Embedding Lookup (Sequential Memory Read) |
| **Input Dimension** | [B, L] (position IDs: 0, 1, ..., L-1) |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `P[l] → h_pos[b,l]` where `P ∈ ℝ^(1024×768)` |
| **Parameters** | 1,024 × 768 = 786,432 |
| **Memory (FP32)** | 3.1 MB |
| **Required Ops** | B × L sequential lookups |
| **Input Memory BW** | B × L × 4 bytes (position IDs) |
| **Output Memory BW** | B × L × 768 × 4 = B × L × 3 KB |
| **FLOPs** | 0 (pure memory operation) |
| **Bottleneck** | Memory read (but sequential, cache-friendly) |
| **Cache Behavior** | Sequential access, excellent prefetching |
| **Hardware Suitability** | CPU ★★★☆☆, GPU ★★★★★, NPU ★★★★★ |

**Performance (B=1, L=1024):**
- CPU: ~124 μs (prefetch-optimized)
- GPU: ~3.3 μs
- NPU: ~5 μs (entire matrix in SRAM)

---

### 3. Embedding Addition

| Field | Value |
|-------|-------|
| **Layer Index** | 3 |
| **Layer Name** | Embedding Addition |
| **Processing Type** | Element-wise Vector Addition |
| **Input Dimension** | 2 × [B, L, 768] |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `h₀ = E[x] + P[pos]` (element-wise) |
| **Parameters** | 0 |
| **Required Ops** | B × L × 768 additions |
| **Input Memory BW** | 2 × B × L × 768 × 4 bytes |
| **Output Memory BW** | B × L × 768 × 4 bytes |
| **FLOPs** | B × L × 768 (FP32 adds) |
| **Arithmetic Intensity** | 1/12 FLOPs/byte (memory-bound) |
| **Bottleneck** | Memory bandwidth (but L1 cache-friendly) |
| **Hardware Suitability** | CPU ★★★★☆, GPU ★★★★★, NPU ★★★★★ |

**Performance (B=1, L=1024):**
- CPU (scalar): 1.24 ms
- CPU (AVX-512): 39 μs
- GPU: 5 μs
- NPU (fused): ~0 μs (hidden in pipeline)

---

### 4-15. Transformer Block (×12 layers)

Each Transformer block contains the following sublayers:

#### 4.1 Layer Normalization 1 (Pre-Attention)

| Field | Value |
|-------|-------|
| **Layer Index** | 4 + (block_idx × 6) |
| **Layer Name** | LayerNorm 1 |
| **Processing Type** | Normalization (Mean/Variance computation) |
| **Input Dimension** | [B, L, 768] |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `LN(x) = γ ⊙ (x - μ)/σ + β` |
| **Parameters** | 2 × 768 = 1,536 (γ, β) |
| **Required Ops** | B × L × (2×768 + 768 + 768 + 768) ≈ B × L × 3,072 ops |
| **Input Memory BW** | B × L × 768 × 4 bytes |
| **Output Memory BW** | B × L × 768 × 4 bytes |
| **FLOPs** | B × L × 4 × 768 ≈ 3.1M (B=1, L=1024) |
| **Arithmetic Intensity** | ~0.5 FLOPs/byte (memory-bound) |
| **Bottleneck** | Reduction operations (mean, variance) |
| **Hardware Suitability** | CPU ★★★☆☆, GPU ★★★★☆, NPU ★★★★☆ |

**Notes**: Requires synchronization across d dimension

---

#### 4.2 Multi-Head Self-Attention (MHSA)

| Field | Value |
|-------|-------|
| **Layer Index** | 4 + (block_idx × 6) + 1 |
| **Layer Name** | Multi-Head Attention |
| **Processing Type** | Matrix Multiplication + Softmax |
| **Input Dimension** | [B, L, 768] |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `Attention(Q,K,V) = softmax(QK^T/√d_k)V` |
| **Components** | Q, K, V projections + Output projection |
| **Parameters** | 4 × (768 × 768) = 2,359,296 |
| **Memory (FP32)** | 9.4 MB per block |

**Sub-operations:**

**4.2.1 QKV Projection:**
- **Ops**: 3 × (B × L × 768 × 768) MatMuls
- **FLOPs**: 3 × B × L × 768 × 768 = 1.8B (B=1, L=1024)
- **Input Memory BW**: B × L × 768 × 4 = 3 MB
- **Weight Memory BW**: 3 × 768 × 768 × 4 = 7 MB
- **Output Memory BW**: 3 × B × L × 768 × 4 = 9 MB

**4.2.2 Attention Scores (QK^T):**
- **Dimension**: [B, H, L, L] where H=12
- **Ops**: B × H × L × L × d_k (d_k=64)
- **FLOPs**: 1 × 12 × 1024 × 1024 × 64 = 805M
- **Output Memory BW**: B × H × L × L × 4 = 48 MB (huge!)
- **Bottleneck**: Quadratic memory O(L²)

**4.2.3 Softmax:**
- **Ops**: B × H × L × L × (exp + sum + div)
- **FLOPs**: ~1 × 12 × 1024 × 1024 × 5 ≈ 63M
- **Bottleneck**: Reduction, numerically sensitive

**4.2.4 Attention × V:**
- **Dimension**: [B, H, L, L] × [B, H, L, d_k] → [B, H, L, d_k]
- **FLOPs**: B × H × L × L × d_k = 805M

**4.2.5 Output Projection:**
- **FLOPs**: B × L × 768 × 768 = 603M

**Total MHSA:**
- **Total FLOPs**: ~4.1B (B=1, L=1024)
- **Total Parameters**: 2,359,296
- **Arithmetic Intensity**: ~400 FLOPs/byte (compute-bound)
- **Bottleneck**: Attention matrix (O(L²) memory)
- **Hardware Suitability**: CPU ★★☆☆☆, GPU ★★★★★, NPU ★★★★☆

**Performance (B=1, L=1024):**
- CPU: ~100-200 ms
- GPU: ~500 μs
- NPU: ~2-5 ms (depends on SRAM size)

---

#### 4.3 Residual Connection 1

| Field | Value |
|-------|-------|
| **Layer Index** | 4 + (block_idx × 6) + 2 |
| **Layer Name** | Residual Add 1 |
| **Processing Type** | Element-wise Addition |
| **Input Dimension** | 2 × [B, L, 768] |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `h = h + Attention(h)` |
| **Parameters** | 0 |
| **FLOPs** | B × L × 768 = 786K |
| **Bottleneck** | Memory bandwidth |
| **Hardware Suitability** | CPU ★★★★☆, GPU ★★★★★, NPU ★★★★★ |

---

#### 4.4 Layer Normalization 2 (Pre-FFN)

| Field | Value |
|-------|-------|
| **Layer Index** | 4 + (block_idx × 6) + 3 |
| **Layer Name** | LayerNorm 2 |
| **Processing Type** | Normalization |
| **Input Dimension** | [B, L, 768] |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `LN(x) = γ ⊙ (x - μ)/σ + β` |
| **Parameters** | 1,536 |
| **FLOPs** | B × L × 4 × 768 ≈ 3.1M |
| **Hardware Suitability** | CPU ★★★☆☆, GPU ★★★★☆, NPU ★★★★☆ |

(Same as LayerNorm 1)

---

#### 4.5 Feed-Forward Network (MLP)

| Field | Value |
|-------|-------|
| **Layer Index** | 4 + (block_idx × 6) + 4 |
| **Layer Name** | Feed-Forward Network (MLP) |
| **Processing Type** | 2-layer MLP with GELU activation |
| **Input Dimension** | [B, L, 768] |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `FFN(x) = W₂·GELU(W₁·x + b₁) + b₂` |
| **Hidden Dimension** | 3,072 (4 × 768) |
| **Parameters** | (768×3072 + 3072) + (3072×768 + 768) = 4,722,432 |
| **Memory (FP32)** | 18.9 MB |

**Sub-operations:**

**4.5.1 Linear 1 (Expansion):**
- **Dimension**: [B, L, 768] → [B, L, 3072]
- **FLOPs**: B × L × 768 × 3,072 = 2.4B (B=1, L=1024)
- **Input Memory BW**: B × L × 768 × 4 = 3 MB
- **Weight Memory BW**: 768 × 3,072 × 4 = 9.4 MB
- **Output Memory BW**: B × L × 3,072 × 4 = 12.6 MB

**4.5.2 GELU Activation:**
- **Ops**: `GELU(x) = x · Φ(x)` (Gaussian Error Linear Unit)
- **FLOPs**: B × L × 3,072 × ~8 ops ≈ 25M
- **Bottleneck**: Transcendental functions (erf approximation)

**4.5.3 Linear 2 (Projection):**
- **Dimension**: [B, L, 3072] → [B, L, 768]
- **FLOPs**: B × L × 3,072 × 768 = 2.4B

**Total MLP:**
- **Total FLOPs**: ~4.8B (B=1, L=1024)
- **Total Parameters**: 4,722,432
- **Arithmetic Intensity**: ~500 FLOPs/byte (compute-bound)
- **Bottleneck**: GEMM operations (highly optimized on GPUs)
- **Hardware Suitability**: CPU ★★☆☆☆, GPU ★★★★★, NPU ★★★★★

**Performance (B=1, L=1024):**
- CPU: ~150-300 ms
- GPU: ~300 μs
- NPU: ~1-3 ms

---

#### 4.6 Residual Connection 2

| Field | Value |
|-------|-------|
| **Layer Index** | 4 + (block_idx × 6) + 5 |
| **Layer Name** | Residual Add 2 |
| **Processing Type** | Element-wise Addition |
| **Input Dimension** | 2 × [B, L, 768] |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `h = h + FFN(h)` |
| **Parameters** | 0 |
| **FLOPs** | B × L × 768 = 786K |
| **Hardware Suitability** | CPU ★★★★☆, GPU ★★★★★, NPU ★★★★★ |

---

### Summary: Single Transformer Block

| Metric | Value (B=1, L=1024) |
|--------|---------------------|
| **Total Parameters** | 7,087,872 |
| **Total Memory (FP32)** | 28.4 MB |
| **Total FLOPs** | ~9B (Attention: 4.1B, MLP: 4.8B) |
| **Dominant Cost** | Matrix multiplications (GEMM) |

---

### 16. Final Layer Normalization

| Field | Value |
|-------|-------|
| **Layer Index** | 76 |
| **Layer Name** | Final LayerNorm |
| **Processing Type** | Normalization |
| **Input Dimension** | [B, L, 768] |
| **Output Dimension** | [B, L, 768] |
| **Mathematical Expression** | `LN(x) = γ ⊙ (x - μ)/σ + β` |
| **Parameters** | 1,536 |
| **FLOPs** | B × L × 4 × 768 ≈ 3.1M |
| **Hardware Suitability** | CPU ★★★☆☆, GPU ★★★★☆, NPU ★★★★☆ |

---

### 17. Language Model Head (Output Projection)

| Field | Value |
|-------|-------|
| **Layer Index** | 77 |
| **Layer Name** | LM Head (lm_head) |
| **Processing Type** | Matrix Multiplication (Unembedding) |
| **Input Dimension** | [B, L, 768] |
| **Output Dimension** | [B, L, 50257] (logits over vocabulary) |
| **Mathematical Expression** | `logits = h · W^T` where `W = E^T` (weight tying) |
| **Parameters** | 0 (shares weights with token embedding) |
| **FLOPs** | B × L × 768 × 50,257 = 39.5B (B=1, L=1024) |
| **Input Memory BW** | B × L × 768 × 4 = 3 MB |
| **Weight Memory BW** | 768 × 50,257 × 4 = 154 MB |
| **Output Memory BW** | B × L × 50,257 × 4 = 206 MB |
| **Arithmetic Intensity** | ~120 FLOPs/byte (compute-bound) |
| **Bottleneck** | Large output dimension (50K vocab) |
| **Hardware Suitability** | CPU ★☆☆☆☆, GPU ★★★★★, NPU ★★★☆☆ |

**Performance (B=1, L=1024):**
- CPU: ~500 ms - 1s
- GPU: ~2-5 ms
- NPU: ~10-20 ms (limited by SRAM size)

**Note**: In practice, only compute logits for last token (generation) or specific positions (training), reducing cost significantly.

---

## 전체 모델 종합

### 파라미터 분포

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Token Embedding | 38,597,376 | 31.0% |
| Position Embedding | 786,432 | 0.6% |
| 12× Transformer Blocks | 85,054,464 | 68.3% |
| Final LayerNorm | 1,536 | 0.001% |
| **Total** | **124,439,808** | **100%** |

### 연산량 분포 (B=1, L=1024, Forward Pass)

| Component | FLOPs | Percentage |
|-----------|-------|------------|
| Input Embeddings | ~0 | 0% |
| 12× Transformer Blocks | ~108B | 73% |
| LM Head | ~39.5B | 27% |
| **Total** | **~147.5B** | **100%** |

### 메모리 사용량 (FP32)

| Component | Memory |
|-----------|--------|
| Parameters | 124M × 4 = 497 MB |
| Activations (B=1, L=1024) | ~200-300 MB |
| Attention Cache (KV cache, inference) | ~24 MB per layer × 12 = 288 MB |
| **Total (Inference)** | ~1.1 GB |

### 병목 지점 분석

| Layer Type | Bottleneck | Optimization Strategy |
|------------|------------|----------------------|
| Tokenization | Sequential algorithm | Keep on CPU, optimize with caching |
| Embeddings | Memory bandwidth | Use GPU/NPU, quantization (INT8) |
| Attention | O(L²) memory, softmax | Flash Attention, sparse attention |
| MLP | GEMM compute | Tensor cores, systolic arrays |
| LM Head | Large vocabulary | Adaptive softmax, sampling strategies |

### 하드웨어 적합성 요약

| Component | CPU | GPU | NPU | Best Choice |
|-----------|-----|-----|-----|-------------|
| Tokenization | ★★★★★ | ★☆☆☆☆ | ☆☆☆☆☆ | CPU |
| Embeddings | ★★☆☆☆ | ★★★★★ | ★★★★☆ | GPU (speed), NPU (efficiency) |
| Attention | ★☆☆☆☆ | ★★★★★ | ★★★☆☆ | GPU |
| MLP | ★☆☆☆☆ | ★★★★★ | ★★★★★ | GPU (speed), NPU (efficiency) |
| LM Head | ★☆☆☆☆ | ★★★★★ | ★★★☆☆ | GPU |

**Overall**: GPU dominates for training and high-throughput inference. NPU best for mobile/edge inference with quantization.

---

## 추가 최적화 기법

### 1. Quantization
- **FP32 → INT8**: 4× memory reduction, 2-4× speedup on NPU
- **Parameters**: 497 MB → 124 MB
- **Activations**: Can use mixed precision (INT8 GEMM, FP16 softmax)

### 2. Flash Attention
- Reduces attention memory from O(L²) to O(L)
- 2-4× faster on GPU, enables longer contexts

### 3. KV Cache (Inference)
- Cache K, V from previous tokens
- Reduces redundant computation during autoregressive generation
- Memory cost: 2 × 12 layers × L × 768 × 4 bytes

### 4. Weight Tying
- LM Head shares weights with Token Embedding
- Saves 38.6M parameters (154 MB)

### 5. Gradient Checkpointing (Training)
- Trade compute for memory (recompute activations)
- Enables larger batch sizes

---

## References
- Model: GPT-2 (Radford et al., 2019)
- Batch size (B): Typically 1-128
- Sequence length (L): Up to 1024 tokens
- All FLOPs calculated for dense operations (no sparsity)
