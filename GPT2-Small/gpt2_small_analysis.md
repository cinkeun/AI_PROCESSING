# GPT-2 Small 모델 프로세싱 분석

## 목차
1. [모델 개요](#모델-개요)
2. [전체 아키텍처 플로우](#전체-아키텍처-플로우)
3. [상세 프로세싱 단계](#상세-프로세싱-단계)
4. [각 레이어별 파라미터 수](#각-레이어별-파라미터-수)
5. [메모리 및 연산량 분석](#메모리-및-연산량-분석)

---

## 모델 개요

### GPT-2 Small 주요 스펙
- **파라미터 수**: 약 117M (117,210,624)
- **레이어 수**: 12개 Transformer 블록
- **Hidden dimension (d_model)**: 768
- **Attention heads**: 12개
- **Head dimension (d_head)**: 64 (= 768 / 12)
- **FFN inner dimension**: 3072 (= 4 × 768)
- **Vocabulary size**: 50,257
- **Max sequence length**: 1024

---

## 전체 아키텍처 플로우

```
Input Text 
    ↓
Tokenization → [token_ids]
    ↓
Token Embedding (50257 × 768) → [B × L × 768]
    ↓
Position Embedding (1024 × 768) → [B × L × 768]
    ↓
Add Embeddings → [B × L × 768]
    ↓
┌─────────────────────────────────┐
│  12× Transformer Blocks         │
│  ┌──────────────────────────┐  │
│  │ LayerNorm 1              │  │
│  │ Multi-Head Attention     │  │
│  │ Residual Connection      │  │
│  │ LayerNorm 2              │  │
│  │ Feed-Forward Network     │  │
│  │ Residual Connection      │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
    ↓
Final LayerNorm → [B × L × 768]
    ↓
Language Model Head (768 × 50257) → [B × L × 50257]
    ↓
Output Logits
```

---

## 상세 프로세싱 단계

### 1. 입력 처리 (Input Processing)

#### 1.1 Tokenization
```
Input: "Hello world"
Process: BPE (Byte Pair Encoding) 토크나이징
Output: [15496, 995] (예시)
Shape: [seq_len] = [2]
```

#### 1.2 Token Embedding
```
Input Shape: [batch_size, seq_len]
Embedding Matrix: [vocab_size, d_model] = [50257, 768]
Process: 각 token ID를 768차원 벡터로 매핑
Output Shape: [B, L, 768]

예시 (B=1, L=2):
token_ids = [15496, 995]
embedded = Embedding[token_ids, :]
output = [1, 2, 768]
```

**메모리**: 50,257 × 768 × 4 bytes = 약 154MB (float32)

#### 1.3 Position Embedding
```
Input: Position indices [0, 1, 2, ..., L-1]
Position Matrix: [max_seq_len, d_model] = [1024, 768]
Process: 각 위치를 학습된 임베딩으로 매핑
Output Shape: [B, L, 768]

예시:
positions = [0, 1]
pos_embedded = PositionEmbedding[positions, :]
output = [1, 2, 768]
```

**메모리**: 1,024 × 768 × 4 bytes = 약 3.1MB

#### 1.4 Embedding Addition
```
Input 1: Token embeddings [B, L, 768]
Input 2: Position embeddings [B, L, 768]
Process: Element-wise addition
Output: [B, L, 768]

output[b, l, d] = token_emb[b, l, d] + pos_emb[b, l, d]
```

---

### 2. Transformer Block (12번 반복)

각 Transformer 블록은 다음 구조를 가집니다:

#### 2.1 LayerNorm 1
```
Input Shape: [B, L, 768]
Process: 
  - mean = mean(x, dim=-1, keepdim=True)  # [B, L, 1]
  - var = var(x, dim=-1, keepdim=True)    # [B, L, 1]
  - x_norm = (x - mean) / sqrt(var + eps)
  - output = gamma * x_norm + beta
  
Parameters:
  - gamma (scale): [768]
  - beta (shift): [768]

Output Shape: [B, L, 768]
```

**파라미터 수**: 768 × 2 = 1,536

#### 2.2 Multi-Head Self-Attention

##### 2.2.1 QKV Projection
```
Input Shape: [B, L, 768]

Query Projection:
  W_q: [768, 768]
  Q = input @ W_q
  Q Shape: [B, L, 768]

Key Projection:
  W_k: [768, 768]
  K = input @ W_k
  K Shape: [B, L, 768]

Value Projection:
  W_v: [768, 768]
  V = input @ W_v
  V Shape: [B, L, 768]

Total QKV params: 768 × 768 × 3 = 1,769,472
```

**실제 구현**: 종종 하나의 큰 행렬로 통합
```
W_qkv: [768, 2304]  # 2304 = 768 × 3
QKV = input @ W_qkv  # [B, L, 2304]
Q, K, V = split(QKV, dim=-1)  # 각각 [B, L, 768]
```

##### 2.2.2 Multi-Head Reshape
```
Input: Q, K, V each [B, L, 768]

Process:
  Q = Q.reshape(B, L, 12, 64).transpose(1, 2)  # [B, 12, L, 64]
  K = K.reshape(B, L, 12, 64).transpose(1, 2)  # [B, 12, L, 64]
  V = V.reshape(B, L, 12, 64).transpose(1, 2)  # [B, 12, L, 64]

여기서:
  - 12 = num_heads
  - 64 = d_head (768 / 12)
```

##### 2.2.3 Scaled Dot-Product Attention (각 헤드별)
```
Q Shape: [B, 12, L, 64]
K Shape: [B, 12, L, 64]

Step 1: Attention Scores
  scores = Q @ K^T
  scores = [B, 12, L, 64] @ [B, 12, 64, L]
  scores Shape: [B, 12, L, L]

Step 2: Scaling
  scores = scores / sqrt(d_head)
  scores = scores / sqrt(64) = scores / 8
  scores Shape: [B, 12, L, L]

Step 3: Causal Masking
  mask = lower_triangular_mask(L, L)
  # mask[i, j] = 0 if i >= j else -inf
  scores = scores + mask
  scores Shape: [B, 12, L, L]
  
  예시 (L=3):
  mask = [[  0, -inf, -inf],
          [  0,    0, -inf],
          [  0,    0,    0]]

Step 4: Softmax
  attn_weights = softmax(scores, dim=-1)
  attn_weights Shape: [B, 12, L, L]

Step 5: Apply to Values
  V Shape: [B, 12, L, 64]
  output = attn_weights @ V
  output = [B, 12, L, L] @ [B, 12, L, 64]
  output Shape: [B, 12, L, 64]
```

**연산량 (Attention Score 계산)**:
- Q @ K^T: B × 12 × L × 64 × L = B × 12 × L² × 64 FLOPs
- Attention @ V: B × 12 × L × L × 64 = B × 12 × L² × 64 FLOPs
- Total per layer: B × 12 × 2L² × 64 = B × 1536 × L² FLOPs

##### 2.2.4 Concat Heads
```
Input Shape: [B, 12, L, 64]

Process:
  output = output.transpose(1, 2).reshape(B, L, 768)

Output Shape: [B, L, 768]
```

##### 2.2.5 Output Projection
```
Input Shape: [B, L, 768]
W_o: [768, 768]
Output = Input @ W_o
Output Shape: [B, L, 768]
```

**파라미터 수**: 768 × 768 = 589,824

##### 2.2.6 Residual Connection
```
Input (original): [B, L, 768]
Attention Output: [B, L, 768]

Output = Input + Attention_Output
Output Shape: [B, L, 768]
```

**총 Attention 파라미터**: 
- QKV: 768 × 768 × 3 = 1,769,472
- Output: 768 × 768 = 589,824
- **Total**: 2,359,296

#### 2.3 LayerNorm 2
```
Input Shape: [B, L, 768]
Process: Same as LayerNorm 1
Parameters: gamma [768], beta [768]
Output Shape: [B, L, 768]
```

**파라미터 수**: 1,536

#### 2.4 Feed-Forward Network (FFN)

##### 2.4.1 First Linear Layer
```
Input Shape: [B, L, 768]
W1: [768, 3072]
bias1: [3072]

Output = Input @ W1 + bias1
Output Shape: [B, L, 3072]
```

**파라미터 수**: 768 × 3,072 + 3,072 = 2,362,368

**연산량**: B × L × 768 × 3,072 × 2 FLOPs (matmul)

##### 2.4.2 GELU Activation
```
Input Shape: [B, L, 3072]

GELU(x) = x * Φ(x)
where Φ(x) is the cumulative distribution function of standard normal

또는 근사:
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

Output Shape: [B, L, 3072]
```

Element-wise 연산이므로 shape 변화 없음

##### 2.4.3 Second Linear Layer
```
Input Shape: [B, L, 3072]
W2: [3072, 768]
bias2: [768]

Output = Input @ W2 + bias2
Output Shape: [B, L, 768]
```

**파라미터 수**: 3,072 × 768 + 768 = 2,360,064

**연산량**: B × L × 3,072 × 768 × 2 FLOPs

##### 2.4.4 Residual Connection
```
Input (pre-FFN): [B, L, 768]
FFN Output: [B, L, 768]

Output = Input + FFN_Output
Output Shape: [B, L, 768]
```

**총 FFN 파라미터**: 2,362,368 + 2,360,064 = 4,722,432

#### 2.5 Single Transformer Block 요약
```
Parameters per block:
- LayerNorm 1: 1,536
- Attention: 2,359,296
- LayerNorm 2: 1,536
- FFN: 4,722,432
────────────────────
Total per block: 7,084,800

Total for 12 blocks: 7,084,800 × 12 = 85,017,600
```

---

### 3. 출력 처리 (Output Processing)

#### 3.1 Final LayerNorm
```
Input Shape: [B, L, 768]
Process: Same as previous LayerNorms
Parameters: gamma [768], beta [768]
Output Shape: [B, L, 768]
```

**파라미터 수**: 1,536

#### 3.2 Language Model Head
```
Input Shape: [B, L, 768]
W_lm: [768, 50257]

Note: W_lm은 보통 Token Embedding 행렬과 가중치를 공유 (tied weights)

Logits = Input @ W_lm
Logits Shape: [B, L, 50257]
```

**파라미터 수**: 
- Tied weights 사용시: 0 (이미 embedding에서 카운트)
- Untied weights 사용시: 768 × 50,257 = 38,597,376

GPT-2는 tied weights를 사용합니다.

#### 3.3 Next Token Prediction
```
Logits Shape: [B, L, 50257]

For generation, we typically only use the last position:
last_logits = Logits[:, -1, :]  # [B, 50257]

probabilities = softmax(last_logits / temperature)
next_token = argmax(probabilities) or sample(probabilities)
```

---

## 각 레이어별 파라미터 수

### 상세 파라미터 분해

```
1. Token Embedding
   - 50,257 × 768 = 38,597,376

2. Position Embedding
   - 1,024 × 768 = 786,432

3. Transformer Blocks (×12)
   Per block:
   ├─ LayerNorm 1: 768 × 2 = 1,536
   ├─ Attention
   │  ├─ Q projection: 768 × 768 = 589,824
   │  ├─ K projection: 768 × 768 = 589,824
   │  ├─ V projection: 768 × 768 = 589,824
   │  └─ Output projection: 768 × 768 = 589,824
   │  Subtotal: 2,359,296
   ├─ LayerNorm 2: 768 × 2 = 1,536
   └─ FFN
      ├─ W1: 768 × 3,072 = 2,359,296
      ├─ b1: 3,072
      ├─ W2: 3,072 × 768 = 2,359,296
      └─ b2: 768
      Subtotal: 4,722,432
   
   Block Total: 7,084,800
   12 Blocks: 85,017,600

4. Final LayerNorm
   - 768 × 2 = 1,536

5. LM Head (tied with embedding)
   - 0 (shared)

────────────────────────────
Total Parameters: 124,439,808 (약 124M)
```

**Note**: 실제 GPT-2 Small은 약 117M 파라미터로 알려져 있는데, 이는 bias 항을 포함하지 않거나 다른 계산 방식을 사용할 때의 값입니다.

---

## 메모리 및 연산량 분석

### 메모리 사용량 (Inference, float32)

#### 모델 파라미터
```
Parameters: 124,439,808 × 4 bytes = 497.76 MB
```

#### Activation Memory (Forward Pass)
배치 크기 B=1, 시퀀스 길이 L=1024 가정:

```
1. Embeddings
   - Token + Position: 2 × (1 × 1024 × 768) = 1,572,864 floats = 6.29 MB

2. Per Transformer Block
   - LayerNorm 1 output: 1 × 1024 × 768 = 0.78M floats
   - Q, K, V: 3 × (1 × 1024 × 768) = 2.36M floats
   - Attention scores: 1 × 12 × 1024 × 1024 = 12.58M floats
   - Attention output: 1 × 1024 × 768 = 0.78M floats
   - FFN intermediate: 1 × 1024 × 3072 = 3.15M floats
   - FFN output: 1 × 1024 × 768 = 0.78M floats
   
   Per block: ~20.43M floats = 81.7 MB
   12 blocks: ~245M floats = 980.4 MB

3. Final LayerNorm + LM Head
   - LayerNorm: 0.78M floats
   - Logits: 1 × 1024 × 50257 = 51.46M floats = 205.86 MB

Total Activation: ~1.2 GB (대략)
```

**Total Memory (Inference)**: ~1.7 GB

### 연산량 (FLOPs)

배치 크기 B=1, 시퀀스 길이 L 가정:

#### Per Transformer Block

```
1. Attention QKV Projection
   - 3 × (L × 768 × 768 × 2) = 3,538,944 × L FLOPs

2. Attention Computation
   - Q @ K^T: 12 × L × L × 64 = 768 × L² FLOPs
   - Softmax: ~12 × L × L operations (무시 가능)
   - Attn @ V: 12 × L × L × 64 = 768 × L² FLOPs
   - Total: 1,536 × L² FLOPs

3. Attention Output Projection
   - L × 768 × 768 × 2 = 1,179,648 × L FLOPs

4. FFN
   - First layer: L × 768 × 3072 × 2 = 4,718,592 × L FLOPs
   - Second layer: L × 3072 × 768 × 2 = 4,718,592 × L FLOPs
   - Total: 9,437,184 × L FLOPs

Per Block Total: 14,155,776 × L + 1,536 × L² FLOPs
```

#### Total Model (12 blocks)

```
Total FLOPs ≈ 12 × (14,155,776 × L + 1,536 × L²)
            = 169,869,312 × L + 18,432 × L²

For L = 1024:
≈ 173,982,646,272 FLOPs
≈ 174 GFLOPs (약 1740억 연산)
```

### 시퀀스 길이에 따른 복잡도

| Sequence Length | Memory (Activation) | FLOPs        | Time (A100, 예상) |
|----------------|---------------------|--------------|-------------------|
| 128            | ~200 MB             | ~22 GFLOPS   | ~1 ms             |
| 512            | ~500 MB             | ~90 GFLOPS   | ~3 ms             |
| 1024           | ~1.2 GB             | ~174 GFLOPS  | ~5 ms             |
| 2048           | ~4 GB               | ~420 GFLOPS  | ~12 ms            |

**Note**: Attention의 O(L²) 복잡도로 인해 긴 시퀀스에서 급격히 증가

---

## 주요 병목 지점

### 1. Self-Attention (O(L²) 복잡도)
- 시퀀스 길이가 길어질수록 quadratic하게 증가
- 메모리: 각 레이어당 12 × L × L attention score 저장
- 연산: L²에 비례하는 행렬 곱셈

### 2. Feed-Forward Network
- 각 레이어당 가장 많은 파라미터 (약 67%)
- 768 → 3072 → 768 변환
- 총 FLOPs의 약 66% 차지

### 3. Embedding/LM Head
- Vocabulary가 크므로 (50,257) 많은 메모리 사용
- LM Head 출력: L × 50,257 크기의 logits

### 4. KV Cache (생성 시)
- Autoregressive 생성 시 이전 토큰들의 Key, Value 저장
- 메모리: 12 layers × 2 (K, V) × L × 768 × 4 bytes
- 1024 토큰: ~72 MB

---

## 최적화 기법

### 1. Quantization
- FP32 → FP16: 메모리 50% 감소
- INT8: 메모리 75% 감소, 약간의 정확도 손실

### 2. KV Cache 관리
- 생성 시 매번 재계산하지 않고 캐시 사용
- Multi-Query Attention (MQA) 또는 Grouped-Query Attention (GQA)

### 3. Flash Attention
- Attention 연산의 메모리 효율성 개선
- Tiling 기법으로 O(L²) 메모리를 O(L)로 감소

### 4. Model Parallelism
- Tensor Parallelism: 레이어 내 병렬화
- Pipeline Parallelism: 레이어 간 병렬화

### 5. Gradient Checkpointing (학습 시)
- Activation을 저장하지 않고 필요시 재계산
- 메모리 사용량 크게 감소, 연산 시간 증가

---

## 실제 예제: "Hello world" 처리

### Input
```
Text: "Hello world"
```

### Step-by-Step Processing

```python
# Step 1: Tokenization
tokens = tokenizer.encode("Hello world")
# tokens = [15496, 995] (예시)
# Shape: [2]

# Step 2: Add batch dimension
input_ids = tokens.unsqueeze(0)  # [1, 2]

# Step 3: Token Embedding
token_emb = embedding(input_ids)  # [1, 2, 768]

# Step 4: Position Embedding
positions = torch.arange(0, 2)  # [0, 1]
pos_emb = position_embedding(positions)  # [1, 2, 768]

# Step 5: Combine Embeddings
x = token_emb + pos_emb  # [1, 2, 768]

# Step 6-17: 12 Transformer Blocks
for layer in transformer_blocks:  # 12 iterations
    # LayerNorm + Attention + Residual
    residual = x
    x = layer_norm_1(x)  # [1, 2, 768]
    x = multi_head_attention(x)  # [1, 2, 768]
    x = x + residual  # [1, 2, 768]
    
    # LayerNorm + FFN + Residual
    residual = x
    x = layer_norm_2(x)  # [1, 2, 768]
    x = ffn(x)  # [1, 2, 768]
    x = x + residual  # [1, 2, 768]

# Step 18: Final LayerNorm
x = final_layer_norm(x)  # [1, 2, 768]

# Step 19: LM Head
logits = lm_head(x)  # [1, 2, 50257]

# Step 20: Get next token prediction
next_token_logits = logits[:, -1, :]  # [1, 50257]
probs = softmax(next_token_logits)  # [1, 50257]
next_token = argmax(probs)  # scalar
```

### 각 위치에서의 예측

```
Position 0 (after "Hello"):
- Input context: [15496] ("Hello")
- Predicted next token: 995 ("world") - if trained on this pattern

Position 1 (after "Hello world"):
- Input context: [15496, 995] ("Hello world")
- Predicted next token: 0 ("!") or other continuation
```

---

## 참고사항

### Causal Masking의 중요성
```
Input: [token1, token2, token3]

Attention mask ensures:
- token1 can only attend to token1
- token2 can attend to token1, token2
- token3 can attend to token1, token2, token3

This prevents information leakage from future tokens.
```

### Weight Tying
```
Token Embedding과 LM Head가 같은 가중치 행렬 사용:
- Embedding: [vocab_size, d_model] = [50257, 768]
- LM Head: [d_model, vocab_size] = [768, 50257]

LM Head = Embedding.T (transpose)

장점:
- 파라미터 수 감소 (38M 파라미터 절약)
- Input과 output space의 일관성
```

### Residual Connections
```
매 서브레이어마다 residual connection:
output = sublayer(x) + x

효과:
- Gradient flow 개선
- 깊은 네트워크 학습 가능
- Identity mapping 학습 용이
```

---

## 요약

GPT-2 Small의 핵심 처리 과정:
1. **Tokenization**: 텍스트 → 토큰 ID
2. **Embedding**: 토큰 ID → 768차원 벡터
3. **12× Transformer Blocks**: Self-attention + FFN
4. **LM Head**: 768차원 → 50,257 vocabulary logits
5. **Prediction**: Logits → 다음 토큰

주요 특징:
- **총 파라미터**: ~117-124M
- **메모리 (inference)**: ~1.7GB (FP32)
- **복잡도**: O(L²) for attention, O(L) for others
- **최적화**: KV cache, Flash Attention, Quantization

이 문서는 GPT-2 Small의 전체 처리 과정을 dimension과 함께 상세히 설명합니다.
