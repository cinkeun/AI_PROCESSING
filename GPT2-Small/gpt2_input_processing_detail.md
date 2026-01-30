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

**하드웨어 연산 관점:**
```
Operation Type: Memory Read + Character Encoding
├─ String Memory Read
│  ├─ Load string from DRAM/Cache (likely L1/L2 cache)
│  ├─ Memory bandwidth: ~11 bytes for "Hello world"
│  └─ Latency: L1 cache hit ~4 cycles, L2 ~12 cycles
│
└─ UTF-8 Encoding (if needed)
   ├─ ASCII characters (0x00-0x7F): Direct copy (1 byte)
   │  └─ Operation: MOV instruction (1 cycle)
   ├─ Non-ASCII characters: Multi-byte encoding
   │  ├─ Bit manipulation operations
   │  ├─ Shift, AND, OR operations (~3-5 cycles per char)
   │  └─ ALU operations for encoding logic
   └─ Output: Write to destination buffer in memory

Total Operations:
- Memory Reads: 11 bytes (sequential access, cache-friendly)
- ALU Operations: ~11 MOV/COPY (for ASCII)
- Memory Writes: 11 bytes to output buffer
- Estimated Cycles: ~50-100 cycles (cache-dependent)
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

**하드웨어 연산 관점:**
```
Operation Type: LUT (Lookup Table) Access
├─ Byte-to-Unicode Mapping Table
│  ├─ Table Size: 256 entries × ~4 bytes = 1 KB
│  ├─ Storage: L1 Cache (매우 작아서 캐시에 상주)
│  └─ Access Pattern: Random access (per byte)
│
├─ Per-Byte Operation
│  ├─ 1. Load byte value (input): 1 memory read
│  ├─ 2. Array index calculation: ADD operation (base_addr + offset)
│  │    └─ addr = LUT_base + (byte_value × entry_size)
│  ├─ 3. LUT access: 1 memory read from L1 cache (~4 cycles)
│  ├─ 4. Store unicode char: 1 memory write
│  └─ Total per byte: ~8-10 cycles
│
└─ For "Hello world" (11 bytes)
   ├─ Total LUT lookups: 11
   ├─ Memory operations: 11 reads + 11 writes = 22 ops
   ├─ ALU operations: 11 address calculations
   └─ Estimated cycles: ~90-120 cycles (fully cached)

Optimization Notes:
- LUT은 작아서 L1 캐시에 완전히 fit됨 (캐시 미스 없음)
- Sequential processing (no SIMD, character-by-character)
- Branch-free operation (직접 인덱싱)
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

**하드웨어 연산 관점:**
```
Operation Type: Iterative Pattern Matching + Merging
├─ BPE Merge Rules Table
│  ├─ Table Size: ~50,000 rules × (2 strings + 1 rank + 1 result)
│  ├─ Storage: ~几 MB (DRAM, 일부 L3/L2 cache)
│  ├─ Data Structure: Hash map 또는 sorted array
│  └─ Access Pattern: Lookup-heavy (random access)
│
├─ Per Iteration Operations
│  ├─ 1. Get all adjacent pairs
│  │    ├─ Sequential scan of token array
│  │    ├─ Memory reads: O(n) where n = current token count
│  │    └─ ~20-50 cycles per iteration start
│  │
│  ├─ 2. Find minimum rank pair
│  │    ├─ For each pair: Hash/Binary search in merge table
│  │    │  ├─ Hash lookup: ~10-50 cycles (with collisions)
│  │    │  └─ Binary search: ~log(50000) × 10 cycles ≈ 150 cycles
│  │    ├─ Comparison operations: O(pairs) × lookup_cost
│  │    ├─ Find minimum: O(pairs) compare operations (~2 cycles each)
│  │    └─ Total: ~500-1000 cycles per iteration
│  │
│  ├─ 3. Merge the selected pair
│  │    ├─ String concatenation: 2 memory reads + 1 memory write
│  │    ├─ Array reconstruction: shift elements (O(n) memory ops)
│  │    ├─ Memory bandwidth: ~50-200 bytes per merge
│  │    └─ ~100-300 cycles
│  │
│  └─ Total per iteration: ~600-1300 cycles
│
├─ Total BPE Processing
│  ├─ Iterations needed: ~5-10 for typical words
│  ├─ "Hello world" example: ~5 iterations
│  ├─ Total cycles: 5 × 1000 = ~5,000 cycles
│  └─ Memory accesses: ~100-200 random reads (merge table)
│
└─ Performance Bottlenecks
   ├─ Merge table lookups (random memory access)
   │  └─ Cache misses possible (large table, random access)
   ├─ String operations (not vectorizable)
   ├─ Sequential algorithm (hard to parallelize per-token)
   └─ Can parallelize across batch (different sentences)

Memory Access Pattern:
- Merge Table: Random access, cache-unfriendly
- Token Array: Sequential + random (during merging)
- Cache Efficiency: Low (~30-50% L2 cache hit rate)

Estimated Total for "Hello world":
- Cycles: ~5,000-10,000 cycles (cache-dependent)
- Memory Bandwidth: ~2-4 KB
- Dominant Cost: Random memory access for merge rule lookups
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

**하드웨어 연산 관점:**
```
Operation Type: Hash Table Lookup
├─ Vocabulary Hash Map
│  ├─ Entries: 50,257 tokens
│  ├─ Size: ~50,257 × (avg 8 bytes string + 4 bytes ID) ≈ 600 KB
│  ├─ Storage: L2/L3 Cache or DRAM
│  └─ Data Structure: Hash map (Python dict)
│
├─ Per-Token Lookup Operation
│  ├─ 1. String hashing
│  │    ├─ Read token string: 1 memory read (varies by length)
│  │    ├─ Hash computation: ~10-30 ALU operations
│  │    │  └─ Loop over characters, multiply + XOR operations
│  │    └─ ~30-100 cycles (depending on string length)
│  │
│  ├─ 2. Hash table lookup
│  │    ├─ Index calculation: hash % table_size (1 DIV or AND op)
│  │    ├─ Memory access: 1-3 reads (with collision handling)
│  │    │  ├─ Best case: 1 cache hit (~4-12 cycles)
│  │    │  └─ Worst case: chain traversal (~50-200 cycles)
│  │    └─ Average: ~20-50 cycles
│  │
│  ├─ 3. String comparison (collision handling)
│  │    ├─ strcmp operation: byte-by-byte comparison
│  │    ├─ ~5-20 cycles per comparison
│  │    └─ Usually 0-1 comparisons needed
│  │
│  └─ 4. Return token ID
│       ├─ Read integer value: 1 memory read (4 bytes)
│       └─ ~4-12 cycles
│
├─ Total Per Token: ~60-200 cycles
│
└─ For 2 tokens ["Hello", "Ġworld"]
   ├─ Total lookups: 2
   ├─ Estimated cycles: ~120-400 cycles
   └─ Memory reads: ~4-8 operations

Cache Behavior:
- Vocabulary table: Partially in L2/L3 cache
- Frequently used tokens (common words): High cache hit rate
- Rare tokens: May cause cache misses
- Access pattern: Random (depends on input text)

Optimization Opportunities:
- Perfect hashing (no collisions) → reduce cycles by ~30%
- Token ID caching (for repeated lookups) → near-zero cost
- Batch processing: Minimal benefit (independent lookups)
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

### 2.8 하드웨어 연산 분석

**Operation Type: Indexed Memory Read (Embedding Lookup)**

```
Embedding Matrix Storage:
├─ Size: 50,257 × 768 × 4 bytes = 154 MB (float32)
├─ Location: DRAM (too large for cache)
│  ├─ L1 Cache: 32-64 KB (can hold ~40-80 embeddings)
│  ├─ L2 Cache: 256 KB - 1 MB (can hold ~300-1000 embeddings)
│  ├─ L3 Cache: 8-32 MB (can hold ~8,000-32,000 embeddings)
│  └─ DRAM: Full matrix resides here
└─ Layout: Row-major (contiguous 768 values per token)

Per-Token Embedding Lookup:
├─ Input: token_id (32-bit integer)
├─ Address Calculation
│  ├─ base_addr = embedding_matrix_ptr
│  ├─ offset = token_id × 768 × 4 bytes = token_id × 3,072
│  ├─ target_addr = base_addr + offset
│  └─ ALU ops: 1 multiply + 1 add (~2-3 cycles)
│
├─ Memory Read Operation
│  ├─ Read 768 floats = 3,072 bytes (3 KB per token)
│  ├─ Memory access pattern: Stride access
│  │  └─ Each token reads a different row (random access across matrix)
│  │
│  ├─ Cache Behavior
│  │  ├─ Common tokens (top 1000): High L2/L3 cache hit rate
│  │  ├─ Rare tokens: Cache miss → DRAM access required
│  │  ├─ L3 cache hit: ~40-50 cycles per 64-byte cache line
│  │  │  └─ Need 3,072/64 = 48 cache lines
│  │  │  └─ Total: ~48 × 45 = 2,160 cycles
│  │  └─ DRAM miss: ~200-300 cycles per cache line
│  │     └─ Total: ~48 × 250 = 12,000 cycles (worst case)
│  │
│  └─ Memory Bandwidth Usage
│     ├─ Per token: 3 KB
│     ├─ Batch of 1024 tokens: 3 MB
│     └─ Can saturate memory bandwidth on large batches
│
└─ Output Write
   ├─ Write 768 floats to output buffer
   ├─ Sequential write (cache-friendly)
   └─ ~200-500 cycles (write-back cache)

Realistic Performance (batch_size=1, seq_len=2):
├─ Token IDs: [15496, 995]
├─ Token 15496 ("Hello")
│  ├─ Address calc: ~3 cycles
│  ├─ Memory read: ~2,160 cycles (L3 cache hit assumed)
│  └─ Output write: ~300 cycles
│  └─ Total: ~2,463 cycles
│
├─ Token 995 ("world")
│  └─ Similar: ~2,463 cycles
│
└─ Total: ~4,926 cycles ≈ 2 μs @ 2.5 GHz CPU

Batched Performance (batch_size=1, seq_len=1024):
├─ Total memory reads: 1024 × 3 KB = 3 MB
├─ Sequential token processing: ~2,500 cycles × 1024 = 2.56M cycles
├─ Memory bandwidth limited: 3 MB @ 50 GB/s = 60 μs
├─ Actual time: max(compute, memory) ≈ 1 ms @ 2.5 GHz
└─ Bottleneck: Memory bandwidth (not compute)

Hardware Optimizations:
├─ GPU Implementation
│  ├─ Parallel lookup: All 1024 tokens simultaneously
│  ├─ High memory bandwidth: 900 GB/s (A100)
│  ├─ 3 MB @ 900 GB/s ≈ 3.3 μs (300× faster than CPU)
│  └─ Coalesced memory access pattern (contiguous reads)
│
├─ CPU Vectorization (Limited benefit)
│  ├─ Cannot vectorize across tokens (random access)
│  ├─ Can vectorize the 768-element copy (AVX-512)
│  └─ Marginal speedup (~10-20%)
│
└─ Prefetching
   ├─ Software prefetch next token's embedding
   ├─ Hide DRAM latency
   └─ ~20-30% speedup with sequential token processing

Memory Access Pattern Visualization:
Token 0: E[15496] → Read row 15496 (3 KB, random location)
Token 1: E[995]   → Read row 995   (3 KB, random location)
Token 2: E[262]   → Read row 262   (3 KB, random location)
...
→ Random strided access (poor cache locality across tokens)
→ Good cache locality within each embedding (contiguous 768 floats)
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

### 3.9 하드웨어 연산 분석

**Operation Type: Sequential Indexed Memory Read**

```
Position Embedding Matrix Storage:
├─ Size: 1,024 × 768 × 4 bytes = 3.1 MB (float32)
├─ Location: L3 Cache / DRAM
│  ├─ L3 Cache: 8-32 MB (entire matrix can fit!)
│  ├─ L2 Cache: 256 KB - 1 MB (can hold ~85-330 positions)
│  └─ L1 Cache: 32-64 KB (can hold ~10-20 positions)
└─ Layout: Row-major (768 floats per position, contiguous)

Key Difference from Token Embedding:
└─ Access Pattern: SEQUENTIAL (positions 0, 1, 2, ..., n-1)
   vs. Token Embedding: RANDOM (any token IDs)
   → Much better cache behavior!

Per-Position Embedding Lookup:
├─ Input: position_id (0, 1, 2, ..., seq_len-1)
├─ Address Calculation
│  ├─ base_addr = position_matrix_ptr
│  ├─ offset = position_id × 768 × 4 = position_id × 3,072
│  ├─ target_addr = base_addr + offset
│  └─ ALU ops: 1 multiply + 1 add (~2-3 cycles)
│
├─ Memory Read Operation
│  ├─ Read 768 floats = 3,072 bytes per position
│  ├─ Access pattern: Sequential stride (highly predictable!)
│  │
│  ├─ Hardware Prefetching
│  │  ├─ CPU detects sequential pattern
│  │  ├─ Automatically prefetches next positions
│  │  ├─ Hides DRAM latency almost completely
│  │  └─ Prefetch distance: typically 2-4 cache lines ahead
│  │
│  ├─ Cache Behavior (MUCH BETTER than token embedding)
│  │  ├─ Position 0: May be cache miss (~2,000 cycles)
│  │  ├─ Position 1: Prefetched → cache hit (~200 cycles)
│  │  ├─ Position 2: Prefetched → cache hit (~200 cycles)
│  │  └─ Sequential access → ~95% cache hit rate
│  │
│  └─ Memory Bandwidth
│     ├─ Streaming read pattern (optimal for DRAM)
│     ├─ Full bandwidth utilization
│     └─ ~3 KB per position
│
└─ Output Write
   ├─ Write 768 floats sequentially
   └─ ~200-300 cycles (write combining)

Realistic Performance (seq_len=2):
├─ Position 0:
│  ├─ Address calc: ~3 cycles
│  ├─ Memory read: ~2,000 cycles (cold start, cache miss)
│  └─ Output write: ~300 cycles
│  └─ Total: ~2,303 cycles
│
├─ Position 1:
│  ├─ Address calc: ~3 cycles
│  ├─ Memory read: ~200 cycles (prefetched, cache hit)
│  └─ Output write: ~300 cycles
│  └─ Total: ~503 cycles
│
└─ Total: ~2,806 cycles ≈ 1.1 μs @ 2.5 GHz

Batched Performance (seq_len=1024):
├─ Total memory reads: 1024 × 3 KB = 3 MB
├─ First position: ~2,000 cycles (cache miss)
├─ Remaining 1023: ~300 cycles each (prefetched)
│  └─ 1023 × 300 = 306,900 cycles
├─ Total: ~309,000 cycles ≈ 124 μs @ 2.5 GHz
│
├─ Memory bandwidth: 3 MB @ 50 GB/s = 60 μs
└─ Actual: ~124 μs (compute-bound, not memory-bound)
   → Prefetching is highly effective!

Performance Comparison:
┌─────────────────────┬──────────────────┬───────────────────┐
│                     │ Token Embedding  │ Position Embedding│
├─────────────────────┼──────────────────┼───────────────────┤
│ Matrix Size         │ 154 MB           │ 3.1 MB            │
│ Access Pattern      │ Random           │ Sequential        │
│ Cache Hit Rate      │ ~30-50%          │ ~95%              │
│ Prefetch Effective? │ No               │ Yes (very!)       │
│ Cycles (1024 tokens)│ ~2.5M cycles     │ ~309K cycles      │
│ Time @ 2.5 GHz      │ ~1 ms            │ ~124 μs           │
│ Speedup             │ 1×               │ 8× faster         │
└─────────────────────┴──────────────────┴───────────────────┘

GPU Implementation:
├─ Parallel lookup: Process all positions simultaneously
├─ Memory coalescing: Perfect (sequential access)
│  └─ Multiple threads read contiguous addresses
│  └─ Maximum memory bus utilization
├─ Bandwidth: 3 MB @ 900 GB/s ≈ 3.3 μs (A100)
└─ 40× faster than CPU (due to parallelism + bandwidth)

CPU Vectorization:
├─ Can vectorize within each position (768 elements)
├─ AVX-512: Process 16 floats per instruction
│  └─ 768 / 16 = 48 vector loads
├─ Limited benefit (already memory-bound)
└─ ~10-15% speedup

Optimization Notes:
├─ Position matrix fits in L3 cache (3.1 MB < 32 MB)
│  └─ After first access, subsequent forwards are very fast
├─ Sequential access → perfect for SIMD streaming
├─ Hardware prefetcher works optimally
└─ Much more hardware-friendly than token embedding!

Memory Access Pattern Visualization:
Position 0: P[0]   → Read row 0   (3 KB, sequential start)
Position 1: P[1]   → Read row 1   (3 KB, next row) ← Prefetched
Position 2: P[2]   → Read row 2   (3 KB, next row) ← Prefetched
...
→ Sequential strided access (excellent cache locality)
→ Predictable pattern → hardware prefetch highly effective
→ Streaming memory access (optimal DRAM efficiency)
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

### 4.10 하드웨어 연산 분석

**Operation Type: Vectorized Element-wise Addition (SIMD)**

```
Input Data Layout:
├─ Token embeddings:    [B, L, 768] = [1, 2, 768] → 6 KB
├─ Position embeddings: [B, L, 768] = [1, 2, 768] → 6 KB
└─ Output buffer:       [B, L, 768] = [1, 2, 768] → 6 KB
   Total working set: 18 KB (fits in L1 cache!)

Element-wise Addition Operation:
├─ Formula: output[i] = token_emb[i] + position_emb[i]
├─ Total elements: B × L × 768 = 1 × 2 × 768 = 1,536 floats
└─ Operation type: FADD (floating-point addition)

Scalar (Non-vectorized) Implementation:
├─ Per-element operation:
│  ├─ Load token_emb[i]:     1 memory read (4 bytes)
│  ├─ Load position_emb[i]:  1 memory read (4 bytes)
│  ├─ FADD operation:        1 ALU cycle
│  ├─ Store result:          1 memory write (4 bytes)
│  └─ Total: ~4-5 cycles per element
│
├─ Total for 1,536 elements:
│  ├─ Memory reads: 1,536 × 2 = 3,072 reads (12 KB)
│  ├─ Memory writes: 1,536 writes (6 KB)
│  ├─ FADD operations: 1,536
│  └─ Total cycles: ~6,144-7,680 cycles ≈ 2.5-3 μs @ 2.5 GHz

Vectorized (SIMD) Implementation:
├─ SSE (128-bit): 4 floats per instruction
│  ├─ Instructions needed: 1,536 / 4 = 384 vector ops
│  ├─ Per vector operation:
│  │  ├─ VMOVAPS (load):  2 vector loads (token + position)
│  │  ├─ VADDPS (add):    1 vector addition
│  │  ├─ VMOVAPS (store): 1 vector store
│  │  └─ Total: ~4 cycles per vector (1 cycle throughput)
│  └─ Total: 384 × 4 = 1,536 cycles ≈ 0.6 μs
│     Speedup: 4× vs scalar
│
├─ AVX (256-bit): 8 floats per instruction
│  ├─ Instructions needed: 1,536 / 8 = 192 vector ops
│  ├─ Throughput: 1 cycle per VADDPS (Haswell+)
│  └─ Total: 192 × 4 = 768 cycles ≈ 0.3 μs
│     Speedup: 8× vs scalar
│
└─ AVX-512 (512-bit): 16 floats per instruction
   ├─ Instructions needed: 1,536 / 16 = 96 vector ops
   ├─ Throughput: 0.5 cycles per VADDPS (2 per cycle)
   └─ Total: 96 × 2 = 192 cycles ≈ 0.08 μs
      Speedup: 16× vs scalar

Memory Access Pattern:
├─ Read pattern: Sequential (cache-friendly)
│  ├─ Token embeddings: 6 KB sequential read
│  ├─ Position embeddings: 6 KB sequential read
│  └─ Both fit in L1 cache (32-64 KB)
│
├─ Write pattern: Sequential (write-combining)
│  └─ 6 KB sequential write
│
└─ Cache behavior:
   ├─ L1 hit rate: ~100% (working set = 18 KB < L1 size)
   ├─ No cache misses expected
   └─ Memory bandwidth: negligible (all in cache)

Realistic Performance (B=1, L=2, d=768):
├─ Vectorization: AVX-512 (modern Intel/AMD)
├─ Vector operations: 96
├─ Cycles per operation: ~2 cycles (including memory)
│  └─ Load 2 vectors: parallel execution
│  └─ Add: 1 cycle throughput
│  └─ Store: write buffer (non-blocking)
├─ Total: ~192 cycles ≈ 0.08 μs @ 2.5 GHz
└─ Bottleneck: Compute (not memory) but very fast!

Large Batch Performance (B=1, L=1024, d=768):
├─ Total elements: 786,432 floats
├─ AVX-512 operations: 786,432 / 16 = 49,152
├─ Cycles: ~98,304 cycles ≈ 39 μs @ 2.5 GHz
├─ Memory bandwidth: 3 MB @ 50 GB/s = 60 μs
└─ Actual: ~60 μs (memory-bound at large scale)

GPU Implementation:
├─ Operation: Massively parallel element-wise add
├─ CUDA threads: Launch 786,432 threads (one per element)
│  └─ Organized as (batch × seq_len × d) grid
├─ Per-thread operation:
│  ├─ tid = blockIdx × blockDim + threadIdx
│  ├─ output[tid] = token_emb[tid] + pos_emb[tid]
│  └─ 1 FADD instruction
├─ Memory coalescing:
│  ├─ Threads in warp access contiguous elements
│  └─ Optimal memory transaction efficiency
├─ Execution time: ~2-5 μs (includes kernel launch)
│  └─ 10-20× faster than CPU (for large batches)
└─ Compute-bound (addition is cheap, launch overhead dominates)

Assembly Example (AVX-512):
```asm
; Loop over 768 elements (16 at a time)
mov rcx, 48              ; 768 / 16 = 48 iterations
lea rsi, [token_emb]     ; source 1
lea rdi, [pos_emb]       ; source 2
lea rdx, [output]        ; destination

.loop:
    vmovaps zmm0, [rsi]       ; Load 16 floats from token_emb
    vmovaps zmm1, [rdi]       ; Load 16 floats from pos_emb
    vaddps  zmm2, zmm0, zmm1  ; Add: zmm2 = zmm0 + zmm1
    vmovaps [rdx], zmm2       ; Store 16 floats to output

    add rsi, 64               ; Advance pointers (16 floats × 4 bytes)
    add rdi, 64
    add rdx, 64

    dec rcx
    jnz .loop                 ; Repeat 48 times
```

Performance Comparison (B=1, L=1024):
┌──────────────────┬────────────┬─────────────┬──────────────┐
│ Implementation   │ Cycles     │ Time @ 2.5GHz│ Speedup     │
├──────────────────┼────────────┼─────────────┼──────────────┤
│ Scalar           │ 3.1M       │ 1.24 ms     │ 1×           │
│ SSE (4-wide)     │ 768K       │ 307 μs      │ 4×           │
│ AVX (8-wide)     │ 384K       │ 154 μs      │ 8×           │
│ AVX-512 (16-wide)│ 98K        │ 39 μs       │ 32×          │
│ GPU (A100)       │ -          │ ~5 μs       │ 248×         │
└──────────────────┴────────────┴─────────────┴──────────────┘

Key Insights:
├─ SIMD Ideal Use Case
│  ├─ No dependencies between elements
│  ├─ Uniform operation (same op for all elements)
│  └─ Contiguous memory access
│
├─ Memory Efficiency
│  ├─ Small working set fits in L1 cache
│  ├─ Sequential access pattern
│  └─ No cache thrashing
│
├─ Compute Characteristics
│  ├─ Very light operation (1 FADD per element)
│  ├─ High arithmetic intensity for SIMD
│  └─ Negligible compared to embedding lookup cost
│
└─ Optimization Level
   ├─ Modern compilers auto-vectorize this
   ├─ PyTorch/NumPy use optimized BLAS libraries
   └─ Typically achieves near-peak throughput
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

---

## 하드웨어 성능 종합 분석

### 메모리 계층 구조와 데이터 배치

```
Memory Hierarchy (전형적인 현대 CPU):
┌─────────────────────────────────────────────────────────────┐
│ L1 Cache (32-64 KB per core)                                │
│ ├─ Latency: ~4 cycles (1.6 ns @ 2.5 GHz)                   │
│ ├─ Bandwidth: ~200 GB/s per core                           │
│ └─ Contents: Hot data, current working set                 │
│    └─ Position embeddings (recent positions): ~3-6 KB      │
│    └─ Intermediate buffers: ~18 KB                         │
├─────────────────────────────────────────────────────────────┤
│ L2 Cache (256 KB - 1 MB per core)                          │
│ ├─ Latency: ~12 cycles (4.8 ns)                            │
│ ├─ Bandwidth: ~100 GB/s per core                           │
│ └─ Contents: Frequent token embeddings, position matrix    │
│    └─ Top ~300 token embeddings: ~900 KB                   │
│    └─ Position matrix (partial): ~256-512 KB               │
├─────────────────────────────────────────────────────────────┤
│ L3 Cache (8-32 MB, shared)                                 │
│ ├─ Latency: ~40-50 cycles (16-20 ns)                       │
│ ├─ Bandwidth: ~200-400 GB/s (aggregate)                    │
│ └─ Contents: Full position matrix, frequent tokens         │
│    └─ Position embedding matrix: 3.1 MB (fits!)            │
│    └─ ~8,000 token embeddings: ~24 MB                      │
├─────────────────────────────────────────────────────────────┤
│ DRAM (16-128 GB)                                            │
│ ├─ Latency: ~200-300 cycles (80-120 ns)                    │
│ ├─ Bandwidth: ~50-100 GB/s (DDR4/DDR5)                     │
│ └─ Contents: Full embedding matrix                         │
│    └─ Token embedding matrix: 154 MB (must stay in DRAM)   │
│    └─ Model weights (rest of GPT-2): ~370 MB               │
└─────────────────────────────────────────────────────────────┘

GPU Memory Hierarchy (e.g., NVIDIA A100):
┌─────────────────────────────────────────────────────────────┐
│ Registers (per SM): 256 KB                                  │
│ ├─ Latency: 0 cycles (operand ready)                       │
│ └─ Thread-private temporary values                         │
├─────────────────────────────────────────────────────────────┤
│ Shared Memory (per SM): 164 KB                             │
│ ├─ Latency: ~30 cycles                                     │
│ └─ Tile of embeddings for thread block                     │
├─────────────────────────────────────────────────────────────┤
│ L2 Cache: 40 MB                                            │
│ ├─ Latency: ~200 cycles                                    │
│ └─ Position matrix + frequent tokens                       │
├─────────────────────────────────────────────────────────────┤
│ HBM2 (High Bandwidth Memory): 40-80 GB                     │
│ ├─ Latency: ~300-400 cycles                                │
│ ├─ Bandwidth: 900 GB/s - 2 TB/s                            │
│ └─ All embedding matrices + model weights                  │
└─────────────────────────────────────────────────────────────┘
```

### 전체 파이프라인 성능 분석

**시나리오 1: Short Input (B=1, L=2) - "Hello world"**

```
CPU Performance (Intel i7-12700K @ 2.5 GHz, AVX-512):
┌──────────────────────┬──────────────┬─────────────┬──────────────┐
│ Stage                │ Cycles       │ Time        │ Bottleneck   │
├──────────────────────┼──────────────┼─────────────┼──────────────┤
│ UTF-8 Encoding       │ ~100         │ 0.04 μs     │ Negligible   │
│ Byte-to-Unicode LUT  │ ~120         │ 0.05 μs     │ Negligible   │
│ BPE Tokenization     │ ~5,000       │ 2 μs        │ Merge lookup │
│ Vocab Lookup         │ ~300         │ 0.12 μs     │ Hash lookup  │
│ Token Embedding      │ ~4,900       │ 2 μs        │ Memory read  │
│ Position Embedding   │ ~2,800       │ 1.1 μs      │ Memory read  │
│ Embedding Addition   │ ~200         │ 0.08 μs     │ Negligible   │
├──────────────────────┼──────────────┼─────────────┼──────────────┤
│ TOTAL                │ ~13,420      │ 5.4 μs      │ Memory I/O   │
└──────────────────────┴──────────────┴─────────────┴──────────────┘

Breakdown:
├─ Tokenization (CPU-only): 38% (2 μs)
├─ Embedding lookup: 58% (3.1 μs)
└─ Addition: 4% (0.08 μs)
```

**시나리오 2: Long Input (B=1, L=1024) - Full context**

```
CPU Performance (Intel i7-12700K @ 2.5 GHz, AVX-512):
┌──────────────────────┬──────────────┬─────────────┬──────────────┐
│ Stage                │ Cycles       │ Time        │ Bottleneck   │
├──────────────────────┼──────────────┼─────────────┼──────────────┤
│ Tokenization         │ ~2.5M        │ 1 ms        │ BPE merge    │
│ Token Embedding      │ ~2.5M        │ 1 ms        │ DRAM b/w     │
│ Position Embedding   │ ~309K        │ 124 μs      │ Prefetch ok  │
│ Embedding Addition   │ ~98K         │ 39 μs       │ Compute      │
├──────────────────────┼──────────────┼─────────────┼──────────────┤
│ TOTAL                │ ~5.4M        │ 2.16 ms     │ Memory b/w   │
└──────────────────────┴──────────────┴─────────────┴──────────────┘

Memory Bandwidth Usage:
├─ Token embedding read: 3 MB
├─ Position embedding read: 3 MB
├─ Output writes: 6 MB
└─ Total: 12 MB @ 50 GB/s = 240 μs (theoretical minimum)
   Actual: ~1.16 ms (embedding reads) → ~10 GB/s effective

GPU Performance (NVIDIA A100):
┌──────────────────────┬─────────────┬──────────────┐
│ Stage                │ Time        │ Speedup vs CPU│
├──────────────────────┼─────────────┼──────────────┤
│ Tokenization (CPU)   │ 1 ms        │ 1× (on CPU)  │
│ Token Embedding      │ 3.3 μs      │ 303×         │
│ Position Embedding   │ 3.3 μs      │ 38×          │
│ Embedding Addition   │ 5 μs        │ 8×           │
├──────────────────────┼─────────────┼──────────────┤
│ TOTAL (GPU part)     │ ~12 μs      │ 97×          │
│ TOTAL (with CPU)     │ ~1.01 ms    │ 2.1×         │
└──────────────────────┴─────────────┴──────────────┘

Note: Tokenization remains on CPU, limiting total speedup
```

### 최적화 기회 및 하드웨어 선택

```
CPU Optimizations:
├─ Tokenization
│  ├─ ✓ Cache BPE merge table in L3 (if possible)
│  ├─ ✓ Use perfect hashing for vocabulary lookup
│  ├─ ✓ Batch multiple sentences (parallel tokenization)
│  └─ ✗ Limited SIMD opportunities (string operations)
│
├─ Embedding Lookup
│  ├─ ✓ Prefetch next token embeddings (software prefetch)
│  ├─ ✓ Pin frequently used embeddings in cache
│  ├─ ✓ Compress embeddings (int8) → 4× memory reduction
│  └─ ✗ Cannot vectorize (random access pattern)
│
├─ Position Embedding
│  ├─ ✓✓ Already optimal (sequential, prefetched)
│  ├─ ✓ Keep in L3 cache (easily fits)
│  └─ Could compute on-the-fly (if cache pressure)
│
└─ Addition
   ├─ ✓✓ Auto-vectorized by compiler (AVX-512)
   ├─ ✓✓ Perfect SIMD workload
   └─ Already near-optimal

GPU Advantages:
├─ Embedding Lookup
│  ├─ Massive parallelism (1024 lookups simultaneously)
│  ├─ High memory bandwidth (900 GB/s vs 50 GB/s)
│  ├─ Memory coalescing optimizations
│  └─ 100-300× faster for large batches
│
├─ Addition
│  ├─ Trivially parallel across all elements
│  ├─ Thousands of concurrent threads
│  └─ 10-20× faster
│
└─ Limitations
   ├─ Tokenization must stay on CPU (sequential algorithm)
   ├─ CPU→GPU transfer overhead for small inputs
   └─ Better for large batches (B ≥ 32)

Recommendation by Use Case:
┌─────────────────────┬──────────────┬─────────────────────┐
│ Use Case            │ Hardware     │ Reason              │
├─────────────────────┼──────────────┼─────────────────────┤
│ Interactive (B=1)   │ CPU          │ Low latency, no PCIe│
│ Small batch (B<16)  │ CPU          │ Transfer overhead   │
│ Large batch (B≥32)  │ GPU          │ Bandwidth + parallel│
│ Training            │ GPU          │ Gradient computation│
│ Edge devices        │ CPU/Mobile   │ Power + availability│
└─────────────────────┴──────────────┴─────────────────────┘
```

### 성능 프로파일링 요약

**입력 처리의 특징:**
1. **Memory-bound**: 메모리 대역폭이 주요 제약
2. **Random access**: Token embedding이 성능 병목
3. **Sequential benefit**: Position embedding은 하드웨어 친화적
4. **SIMD-friendly**: Addition은 완벽한 벡터화 대상

**전체 GPT-2 추론에서의 비중:**
- Input processing: ~5% of total inference time
- Transformer blocks: ~90%
- Output layer: ~5%

→ 입력 처리 최적화는 중요하지만, Transformer block이 주요 타겟!
