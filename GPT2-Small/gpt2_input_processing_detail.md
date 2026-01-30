# GPT-2 입력 처리 상세 분석 (Tokenization → Embedding)

## 목차
1. [Tokenization (BPE)](#1-tokenization-bpe)
2. [Token Embedding](#2-token-embedding)
3. [Position Embedding](#3-position-embedding)
4. [Embedding Addition](#4-embedding-addition)

---

## 1. Tokenization (BPE)

### 1.1 개요
**BPE (Byte Pair Encoding)**는 GPT-2가 사용하는 subword 토크나이징 방식입니다. 텍스트를 고정된 크기의 vocabulary(50,257개)의 토큰으로 변환합니다.

### 1.2 BPE 알고리즘 상세

#### Step 1: 텍스트를 바이트로 변환
```python
text = "Hello world"

# UTF-8 바이트로 변환
bytes_list = list(text.encode('utf-8'))
# bytes_list = [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
#                H    e    l    l    o  space  w    o    r    l    d
```

#### Step 2: 바이트를 초기 토큰으로 변환
GPT-2는 256개의 바이트를 유니코드 문자로 매핑합니다:
```python
# 바이트-to-유니코드 매핑 (간략화)
byte_to_unicode = {
    72: 'H',   # 0x48
    101: 'e',  # 0x65
    108: 'l',  # 0x6C
    111: 'o',  # 0x6F
    32: 'Ġ',   # space는 특수문자 'Ġ'로 표현
    119: 'w',  # 0x77
    114: 'r',  # 0x72
    100: 'd',  # 0x64
}

initial_tokens = ['H', 'e', 'l', 'l', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']
```

#### Step 3: BPE Merge 규칙 적용
BPE는 학습된 merge 규칙을 반복적으로 적용합니다:

```python
# 학습된 merge 규칙 (우선순위 순서)
merge_rules = [
    ('H', 'e') → 'He',        # rank 100
    ('l', 'l') → 'll',        # rank 150
    ('He', 'l') → 'Hel',      # rank 200
    ('Hel', 'l') → 'Hell',    # rank 250
    ('Hell', 'o') → 'Hello',  # rank 300
    ('w', 'o') → 'wo',        # rank 120
    ('wo', 'r') → 'wor',      # rank 180
    ('wor', 'l') → 'worl',    # rank 240
    ('worl', 'd') → 'world',  # rank 280
    # ... 50,000개 이상의 규칙
]

# Iteration 1: rank가 가장 낮은(우선순위 높은) 규칙부터 적용
tokens = ['H', 'e', 'l', 'l', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']
# 'H' + 'e' → 'He' (rank 100이 가장 낮음)
tokens = ['He', 'l', 'l', 'o', 'Ġ', 'w', 'o', 'r', 'l', 'd']

# Iteration 2:
# 'w' + 'o' → 'wo' (rank 120)
tokens = ['He', 'l', 'l', 'o', 'Ġ', 'wo', 'r', 'l', 'd']

# Iteration 3:
# 'l' + 'l' → 'll' (rank 150)
tokens = ['He', 'll', 'o', 'Ġ', 'wo', 'r', 'l', 'd']

# Iteration 4:
# 'wo' + 'r' → 'wor' (rank 180)
tokens = ['He', 'll', 'o', 'Ġ', 'wor', 'l', 'd']

# Iteration 5:
# 'He' + 'l' → 'Hel'... (계속)

# 최종 결과:
tokens = ['Hello', 'Ġworld']
```

#### Step 4: 토큰을 ID로 변환
```python
# Vocabulary: 토큰 문자열 → ID 매핑
vocab = {
    'Hello': 15496,
    'Ġworld': 995,
    'Ġ': 220,
    'the': 262,
    # ... 50,257개 항목
}

token_ids = [vocab[token] for token in tokens]
# token_ids = [15496, 995]
```

### 1.3 실제 구현 의사코드

```python
class GPT2Tokenizer:
    def __init__(self):
        self.encoder = load_vocab()        # str → int 매핑 (50,257개)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.bpe_ranks = load_bpe_merges() # merge 규칙과 우선순위
    
    def bpe(self, token):
        """단일 토큰에 BPE 적용"""
        if token in self.cache:
            return self.cache[token]
        
        # 토큰을 문자 단위로 분할
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # 가장 우선순위 높은(rank 낮은) pair 찾기
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
            
            # bigram 병합
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        
        return ' '.join(word)
    
    def encode(self, text):
        """텍스트를 토큰 ID로 변환"""
        bpe_tokens = []
        
        # 정규식으로 단어/구두점 분리
        for token in re.findall(r"\w+|[^\w\s]", text):
            # 바이트 → 유니코드 변환
            token_bytes = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            
            # BPE 적용
            bpe_token = self.bpe(token_bytes)
            
            # 토큰 ID로 변환
            bpe_tokens.extend([self.encoder[t] for t in bpe_token.split(' ')])
        
        return bpe_tokens

# 사용 예시
tokenizer = GPT2Tokenizer()
token_ids = tokenizer.encode("Hello world")
# token_ids = [15496, 995]
```

### 1.4 수학적 표현

토크나이징은 함수 `T: Σ* → ℕ^n`으로 표현할 수 있습니다:

```
T: text → [t₁, t₂, ..., tₙ]

where:
- Σ* : 모든 가능한 문자열의 집합
- ℕ^n : n개의 자연수 시퀀스
- tᵢ ∈ {0, 1, ..., 50256} : vocabulary 인덱스
- n = sequence length (가변)
```

### 1.5 실제 예시

```python
# 예시 1: 짧은 문장
text = "Hello world"
tokens = tokenize(text)
# tokens = [15496, 995]

# 예시 2: 더 긴 문장
text = "The quick brown fox"
tokens = tokenize(text)
# tokens = [464, 2068, 7586, 21831]
#          "The" "Ġquick" "Ġbrown" "Ġfox"

# 예시 3: Subword 분리
text = "unhappiness"
tokens = tokenize(text)
# tokens = [403, 71, 381, 1272]
#          "un" "h" "app" "iness"
```

### 1.6 특징 및 장점

**특징:**
1. **Subword 단위**: 단어를 더 작은 조각으로 나눔
2. **고정 vocabulary**: 50,257개로 제한
3. **OOV 문제 해결**: 어떤 텍스트든 표현 가능
4. **공백 표현**: 'Ġ' 문자로 단어 시작 공백 표시

**장점:**
1. 희귀 단어를 subword로 표현
2. 형태소적 유사성 포착
3. 다국어 지원
4. 메모리 효율적

**단점:**
1. 토큰 길이가 불규칙
2. 언어마다 효율성 차이
3. 복잡한 전처리

---

## 2. Token Embedding

### 2.1 개요
토큰 ID를 고차원 벡터 공간(768차원)으로 매핑합니다. 이는 신경망이 처리할 수 있는 연속적인 표현입니다.

### 2.2 수학적 정의

Embedding은 lookup table로 구현됩니다:

```
E ∈ ℝ^(V × d)

where:
- V = vocabulary size = 50,257
- d = embedding dimension = 768
- E[i] = i번째 토큰의 임베딩 벡터
```

### 2.3 처리 과정

#### Step 1: Embedding Matrix 준비
```python
# Embedding matrix는 학습 가능한 파라미터
E = nn.Parameter(torch.randn(50257, 768))

# Shape: [50257, 768]
# 각 행은 하나의 토큰을 표현하는 768차원 벡터
```

#### Step 2: Token IDs → Embeddings
```python
# Input
token_ids = torch.tensor([[15496, 995]])  # shape: [1, 2]
# [1, 2] = [batch_size, sequence_length]

# Embedding lookup
token_embeddings = E[token_ids]  # shape: [1, 2, 768]

# 실제로는 nn.Embedding 레이어 사용
embedding_layer = nn.Embedding(50257, 768)
token_embeddings = embedding_layer(token_ids)
```

### 2.4 상세 연산

```python
# 입력
batch_size = 1
seq_len = 2
token_ids = [[15496, 995]]  # "Hello world"

# Embedding matrix E
E.shape = [50257, 768]

# 연산 과정 (의사 코드)
output = []
for b in range(batch_size):
    batch_output = []
    for l in range(seq_len):
        token_id = token_ids[b][l]
        embedding_vector = E[token_id, :]  # [768]
        batch_output.append(embedding_vector)
    output.append(batch_output)

# output shape: [1, 2, 768]
```

### 2.5 구체적 예시

```python
# Token ID 15496 ("Hello")의 임베딩
E[15496] = [0.234, -0.891, 0.456, ..., -0.123]  # 768 values
#           ↑
#           첫 번째 dimension 값

# Token ID 995 ("world")의 임베딩
E[995] = [-0.567, 0.123, -0.789, ..., 0.456]   # 768 values

# 최종 출력
token_embeddings = [
    [  # batch 0
        [0.234, -0.891, 0.456, ..., -0.123],  # "Hello" embedding
        [-0.567, 0.123, -0.789, ..., 0.456]   # "world" embedding
    ]
]
# shape: [1, 2, 768]
```

### 2.6 수식 표현

```
Given:
- Input: x ∈ ℕ^(B×L) where xᵢⱼ ∈ {0, 1, ..., V-1}
- Embedding matrix: E ∈ ℝ^(V×d)

Output:
- h₀ ∈ ℝ^(B×L×d)
- h₀[b, l, :] = E[x[b, l], :]

For each position (b, l):
  h₀[b, l] = E[xᵦₗ]
  
where:
- B = batch size
- L = sequence length
- V = vocabulary size = 50,257
- d = embedding dim = 768
```

### 2.7 메모리 사용량

```python
# Embedding matrix 메모리
params = 50257 × 768 = 38,597,376 parameters
memory = 38,597,376 × 4 bytes = 154,389,504 bytes ≈ 154 MB (float32)

# Forward pass activation 메모리 (batch_size=1, seq_len=1024)
activation = 1 × 1024 × 768 = 786,432 floats
activation_memory = 786,432 × 4 bytes = 3,145,728 bytes ≈ 3.1 MB
```

---

## 3. Position Embedding

### 3.1 개요
Transformer는 순서 정보가 없으므로, 각 토큰의 위치 정보를 명시적으로 제공해야 합니다. GPT-2는 **학습 가능한 position embedding**을 사용합니다.

### 3.2 수학적 정의

```
P ∈ ℝ^(T×d)

where:
- T = max sequence length = 1024
- d = embedding dimension = 768
- P[i] = 위치 i의 임베딩 벡터
```

### 3.3 처리 과정

#### Step 1: Position Embedding Matrix
```python
# Position embedding matrix는 학습 가능한 파라미터
P = nn.Parameter(torch.randn(1024, 768))

# Shape: [1024, 768]
# 각 행은 하나의 위치를 표현하는 768차원 벡터
```

#### Step 2: Position Indices 생성
```python
# Input token_ids
token_ids = torch.tensor([[15496, 995]])  # shape: [1, 2]
batch_size, seq_len = token_ids.shape  # B=1, L=2

# Position indices 생성
position_ids = torch.arange(0, seq_len)  # [0, 1]
# 또는 batch에 대해
position_ids = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1)
# shape: [1, 2] = [[0, 1]]
```

#### Step 3: Position Embedding Lookup
```python
position_embeddings = P[position_ids]
# P[0, :] for position 0
# P[1, :] for position 1

# Output shape: [1, 2, 768]
```

### 3.4 상세 연산

```python
# 입력
seq_len = 2
position_ids = [0, 1]

# Position embedding matrix P
P.shape = [1024, 768]

# 연산 과정
position_embeddings = []
for pos in position_ids:
    pos_vector = P[pos, :]  # [768]
    position_embeddings.append(pos_vector)

# output shape: [2, 768]
# batch dimension 추가 → [1, 2, 768]
```

### 3.5 구체적 예시

```python
# Position 0의 임베딩
P[0] = [0.123, 0.456, -0.789, ..., 0.234]  # 768 values

# Position 1의 임베딩
P[1] = [-0.345, 0.678, 0.901, ..., -0.567]  # 768 values

# 최종 출력
position_embeddings = [
    [  # batch 0
        [0.123, 0.456, -0.789, ..., 0.234],   # position 0
        [-0.345, 0.678, 0.901, ..., -0.567]   # position 1
    ]
]
# shape: [1, 2, 768]
```

### 3.6 수식 표현

```
Given:
- Sequence length: L
- Position embedding matrix: P ∈ ℝ^(T×d)

Position indices:
- pos = [0, 1, 2, ..., L-1]

Output:
- h_pos ∈ ℝ^(B×L×d)
- h_pos[b, l, :] = P[l, :]

For each position l:
  h_pos[b, l] = P[l]
```

### 3.7 Learned vs Sinusoidal

GPT-2는 **학습된 position embedding**을 사용하지만, 원래 Transformer는 sinusoidal을 사용했습니다:

```python
# Sinusoidal (original Transformer)
def get_sinusoidal_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

# PE(pos, 2i) = sin(pos / 10000^(2i/d))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**GPT-2가 학습 가능한 embedding을 선택한 이유:**
- 더 유연함
- 데이터로부터 최적 패턴 학습
- 성능이 약간 더 좋음

### 3.8 메모리 사용량

```python
# Position embedding matrix 메모리
params = 1024 × 768 = 786,432 parameters
memory = 786,432 × 4 bytes = 3,145,728 bytes ≈ 3.1 MB (float32)

# Forward pass activation 메모리 (batch_size=1, seq_len=1024)
activation = 1 × 1024 × 768 = 786,432 floats
activation_memory = 3.1 MB
```

---

## 4. Embedding Addition

### 4.1 개요
Token embedding과 Position embedding을 **element-wise addition**으로 결합합니다.

### 4.2 수학적 표현

```
h₀ = E[x] + P[pos]

where:
- E[x] : token embeddings ∈ ℝ^(B×L×d)
- P[pos] : position embeddings ∈ ℝ^(B×L×d)
- h₀ : combined embeddings ∈ ℝ^(B×L×d)
```

### 4.3 상세 연산

```python
# Token embeddings
token_emb = E[token_ids]  # shape: [B, L, 768]

# Position embeddings
position_emb = P[position_ids]  # shape: [B, L, 768]

# Element-wise addition
combined_emb = token_emb + position_emb  # shape: [B, L, 768]
```

### 4.4 구체적 계산 예시

```python
# Token embedding for "Hello" (token_id=15496)
token_emb[0, 0] = [0.234, -0.891, 0.456, ..., -0.123]

# Position embedding for position 0
position_emb[0, 0] = [0.123, 0.456, -0.789, ..., 0.234]

# Combined embedding
combined_emb[0, 0] = token_emb[0, 0] + position_emb[0, 0]
                   = [0.234 + 0.123, -0.891 + 0.456, 0.456 + (-0.789), ..., -0.123 + 0.234]
                   = [0.357, -0.435, -0.333, ..., 0.111]
```

### 4.5 전체 예시 (배치 포함)

```python
# Input: "Hello world"
token_ids = [[15496, 995]]  # [1, 2]

# Token embeddings
token_emb = [
    [
        [0.234, -0.891, 0.456, ..., -0.123],  # "Hello"
        [-0.567, 0.123, -0.789, ..., 0.456]   # "world"
    ]
]  # shape: [1, 2, 768]

# Position embeddings
position_emb = [
    [
        [0.123, 0.456, -0.789, ..., 0.234],   # pos 0
        [-0.345, 0.678, 0.901, ..., -0.567]   # pos 1
    ]
]  # shape: [1, 2, 768]

# Combined (element-wise addition)
combined_emb = [
    [
        [0.357, -0.435, -0.333, ..., 0.111],  # "Hello" at pos 0
        [-0.912, 0.801, 0.112, ..., -0.111]   # "world" at pos 1
    ]
]  # shape: [1, 2, 768]
```

### 4.6 수식으로 표현

```
For each element (b, l, d):
  h₀[b, l, d] = E[x[b, l], d] + P[l, d]

Vectorized:
  h₀ = E[x] ⊕ P[pos]
  
where ⊕ denotes element-wise addition
```

### 4.7 Broadcasting 상세

```python
# Token embeddings
E[x].shape = [B, L, d] = [1, 2, 768]

# Position embeddings
P[pos].shape = [B, L, d] = [1, 2, 768]  # 이미 broadcast됨

# Addition (element-wise)
h₀ = E[x] + P[pos]
h₀.shape = [1, 2, 768]

# 각 차원별로
for b in range(B):
    for l in range(L):
        for d in range(768):
            h₀[b, l, d] = E[x][b, l, d] + P[pos][b, l, d]
```

### 4.8 왜 Addition인가?

**곱셈 대신 덧셈을 사용하는 이유:**
1. **선형성 유지**: 정보 손실 최소화
2. **학습 안정성**: Gradient flow가 더 안정적
3. **해석 가능성**: 토큰 정보와 위치 정보가 독립적으로 유지
4. **실험적 검증**: 덧셈이 경험적으로 더 좋은 성능

### 4.9 연산량

```python
# Element-wise addition
operations = B × L × d additions
            = 1 × 2 × 768
            = 1,536 operations

# 매우 가벼운 연산 (negligible)
```

---

## 전체 파이프라인 종합

```python
def input_processing(text):
    """
    텍스트 입력을 Transformer가 처리 가능한 형태로 변환
    """
    # Step 1: Tokenization
    token_ids = tokenizer.encode(text)
    # "Hello world" → [15496, 995]
    # shape: [L] = [2]
    
    # Add batch dimension
    token_ids = torch.tensor([token_ids])
    # shape: [B, L] = [1, 2]
    
    # Step 2: Token Embedding
    token_embeddings = embedding_layer(token_ids)
    # shape: [B, L, d] = [1, 2, 768]
    
    # Step 3: Position Embedding
    seq_len = token_ids.size(1)
    position_ids = torch.arange(0, seq_len).unsqueeze(0)
    position_embeddings = position_embedding_layer(position_ids)
    # shape: [B, L, d] = [1, 2, 768]
    
    # Step 4: Combine Embeddings
    combined_embeddings = token_embeddings + position_embeddings
    # shape: [B, L, d] = [1, 2, 768]
    
    return combined_embeddings

# 사용
input_text = "Hello world"
h0 = input_processing(input_text)
# h0.shape = [1, 2, 768]
# 이제 Transformer blocks로 전달됨
```

---

## 핵심 요약

| 단계 | 입력 Shape | 출력 Shape | 파라미터 수 | 연산 |
|-----|-----------|-----------|-----------|------|
| Tokenization | Text | [L] | 0 | O(L) |
| Token Embedding | [B, L] | [B, L, 768] | 38.6M | O(B×L) lookup |
| Position Embedding | [L] | [B, L, 768] | 0.79M | O(B×L) lookup |
| Addition | [B, L, 768] × 2 | [B, L, 768] | 0 | O(B×L×768) |

**총 파라미터**: 39.4M (전체 모델의 약 33%)

**메모리 (inference, float32)**:
- Parameters: 157.6 MB
- Activations (B=1, L=1024): 6.3 MB
