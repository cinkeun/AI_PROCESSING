# GPT-2 입력 처리 상세 분석 (Tokenization → Embedding)

## 목차
1. [Tokenization (BPE)](#1-tokenization-bpe)
   - 1.x [BPE 가속화 기법](#1x-bpe-가속화-기법)
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

GPU Implementation:
├─ Generally NOT executed on GPU
│  ├─ Tokenization preprocessing stays on CPU
│  ├─ String operations not GPU-friendly
│  └─ Data transfer overhead >> computation time
│
└─ If forced to GPU (academic exercise):
   ├─ Launch 11 threads (one per character)
   ├─ Each thread: read byte, check range, copy/encode
   ├─ Execution: ~10-20 μs (kernel launch overhead dominates)
   └─ No practical benefit (1000× slower than CPU!)

NPU Implementation:
├─ NOT suitable for NPU execution
│  ├─ NPU designed for matrix/tensor operations
│  ├─ String encoding is control-flow heavy
│  ├─ No MAC (multiply-accumulate) operations
│  └─ NPU has no advantage over CPU here
│
└─ Practical approach:
   └─ Always execute on CPU, transfer results to NPU/GPU later
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

GPU Implementation:
├─ Minimal benefit for GPU execution
│  ├─ Can launch 11 threads for 11 bytes
│  ├─ Each thread: index into LUT, retrieve unicode char
│  ├─ LUT stored in constant memory (cached, fast access)
│  └─ Execution: ~5-10 μs (kernel launch overhead)
│
├─ Performance:
│  ├─ Actual computation: <0.1 μs
│  ├─ Kernel launch: ~5 μs
│  ├─ Memory copy (CPU→GPU): ~10 μs (even for 11 bytes!)
│  └─ Total: ~15 μs vs ~0.05 μs CPU = 300× SLOWER
│
└─ Conclusion: Not worth GPU transfer overhead

NPU Implementation:
├─ Completely unsuitable for NPU
│  ├─ NPU lacks general-purpose lookup operations
│  ├─ Designed for dense matrix multiplications
│  ├─ No advantage over CPU for table lookups
│  └─ Would require CPU emulation layer
│
└─ Best practice:
   ├─ Keep on CPU (ultra-fast already)
   └─ Only transfer final token IDs to NPU
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

GPU Implementation:
├─ HIGHLY INEFFICIENT for BPE
│  ├─ BPE is inherently sequential (each merge depends on previous)
│  ├─ Cannot parallelize within a single sequence
│  ├─ String manipulation is not GPU-friendly
│  └─ GPU threads would sit idle waiting for dependencies
│
├─ Possible GPU Parallelization:
│  ├─ Batch-level: Tokenize multiple sentences in parallel
│  │  └─ Launch one thread block per sentence
│  ├─ Each block runs BPE sequentially
│  └─ Speedup: ~2-5× (for large batches, B > 100)
│
├─ Performance Reality:
│  ├─ Single sequence: GPU slower than CPU (overhead)
│  ├─ Small batch (B < 32): Still slower (transfer cost)
│  ├─ Large batch (B > 100): Marginal benefit
│  └─ Kernel launch + transfer: ~20-50 μs baseline cost
│
└─ Industry Practice:
   └─ ALWAYS perform BPE tokenization on CPU
   └─ Only transfer final token IDs to GPU

NPU Implementation:
├─ COMPLETELY UNSUITABLE
│  ├─ NPU designed for fixed computation graphs
│  ├─ BPE requires dynamic control flow (while loops, conditionals)
│  ├─ String operations not supported in NPU instruction set
│  ├─ No hash table or dynamic memory allocation
│  └─ Would require full CPU emulation
│
├─ Why NPU Cannot Handle BPE:
│  ├─ NPU architecture: Systolic arrays for matrix ops
│  │  └─ Fixed dataflow, no branching
│  ├─ BPE needs: Hash lookups, string concat, array reordering
│  │  └─ All impossible on pure NPU hardware
│  └─ NPU lacks general-purpose ALU for string operations
│
└─ Practical Reality:
   ├─ BPE must execute on CPU
   ├─ NPU only activates after embedding lookup
   └─ Token IDs transferred from CPU → NPU memory

Performance Comparison (B=1, "Hello world"):
┌────────────────┬──────────────┬─────────────────────────────┐
│ Hardware       │ Time         │ Notes                       │
├────────────────┼──────────────┼─────────────────────────────┤
│ CPU (single)   │ ~2-4 μs      │ Optimal for single sequence │
│ GPU (single)   │ ~50-100 μs   │ 20× SLOWER (overhead)       │
│ GPU (batch 100)│ ~200 μs      │ ~2× faster (batch parallel) │
│ NPU            │ Not possible │ Requires CPU preprocessing  │
└────────────────┴──────────────┴─────────────────────────────┘
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

---

**LUT인가, Hash Map인가?**

직관적으로 LUT(Lookup Table)처럼 보이지만, 실제로는 **Hash Map**입니다.

```
진짜 LUT:  array[integer_key]  → O(1), 주소 = base + key × size (직접 계산)
Hash Map:  hash(string_key) → bucket → value → O(1) 평균, but 해시 계산 필요
```

`vocab.json`의 key가 **문자열(string)**이기 때문에 정수 인덱싱이 불가능합니다:
```
"Hello"  → 정수가 아니므로 array["Hello"] 불가
hash("Hello") → 버킷 위치 계산 → 15496 반환
```

**파이프라인 전체에서 자료구조 비교:**

| 단계 | 자료구조 | 이유 |
|------|----------|------|
| 문자 → 초기 ID (byte_encoder) | **진짜 LUT** (256-entry 배열) | key가 정수(byte값 0~255) |
| BPE merge rule 조회 | **Hash Map** | key가 문자열 쌍 ("e", "r") |
| 토큰 string → ID (vocab.json) | **Hash Map** | key가 문자열 "Hello" |
| token ID → embedding vector | **진짜 LUT** | key가 정수 ID → 배열 직접 인덱싱 |

---

**최적화: "토큰이 ID를 직접 가진다" — Rust 구현**

HuggingFace Fast Tokenizer(Rust)는 BPE 자체를 **처음부터 ID 공간에서** 동작시킵니다:
```
① 각 문자에 초기 ID 할당     : 'H'→72, 'e'→68, 'l'→75, ...  (LUT, O(1))
② merge rule을 ID 쌍으로 저장 : (72, 68) → 15496             (Hash Map)
③ BPE가 ID를 직접 병합       : 문자열 중간 표현 없음
→ 별도 vocab lookup 단계 자체가 사라짐
```
이 방식이 "토큰 자체에 ID가 내포된" 구조에 가장 가깝습니다.

---

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

GPU Implementation:
├─ Limited benefit (still CPU-bound)
│  ├─ Hash table lookups can be parallelized
│  ├─ Launch one thread per token
│  └─ Each thread: hash string → lookup vocab → return ID
│
├─ GPU Challenges:
│  ├─ String hashing on GPU is slower than CPU
│  │  └─ Character-by-character processing not parallel
│  ├─ Hash table in GPU memory (global memory)
│  │  └─ Random access pattern → poor memory coalescing
│  ├─ Collision handling requires divergent branches
│  │  └─ Warp efficiency drops significantly
│  └─ Small workload doesn't saturate GPU
│
├─ Performance (2 tokens):
│  ├─ Kernel launch: ~5 μs
│  ├─ Hash + lookup: ~0.5 μs (2 threads, trivial work)
│  ├─ Memory transfer: ~10 μs (CPU↔GPU)
│  └─ Total: ~15 μs vs ~0.12 μs CPU = 125× SLOWER
│
└─ Batch Performance (1024 tokens):
   ├─ Parallel hash lookups: ~2 μs (all simultaneous)
   ├─ Transfer overhead: ~20 μs
   └─ Total: ~22 μs vs ~60 μs CPU = marginal benefit

NPU Implementation:
├─ NOT APPLICABLE
│  ├─ Hash table lookup not supported on NPU
│  ├─ String operations require general-purpose CPU
│  ├─ NPU vocabulary stored differently (embedding matrix)
│  └─ This step stays on CPU entirely
│
└─ Alternative: Embedding Lookup on NPU
   ├─ Skip vocabulary hash lookup on NPU
   ├─ CPU generates token IDs (integers)
   ├─ NPU starts from token IDs → embedding lookup
   └─ See Token Embedding section for NPU details
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

### 1.x BPE 가속화 기법

#### 왜 느린가?

```
원래 BPE 복잡도:
- Training (merge 규칙 학습): O(n × V²) — n=corpus 크기, V=vocab 크기
- Inference (규칙 적용):      O(k² × M) — k=단어 길이,  M=merge rule 수
```

BPE는 매 merge step마다 이전 결과에 의존하는 **순차적(sequential) 알고리즘**이기 때문에 GPU의 대규모 병렬 연산과 근본적으로 맞지 않습니다.

---

#### 소프트웨어 레벨 가속화

**① HuggingFace Fast Tokenizers (가장 실용적)**
- Rust로 구현된 [`tokenizers`](https://github.com/huggingface/tokenizers) 라이브러리 사용
- 순수 Python 구현 대비 **10~100x 빠름**
- GPT-2도 기본적으로 이를 사용:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)  # Rust 기반
```

**② YouTokenToMe (YTTM) — VK 팀**
- C++ 기반, Training 시 **멀티스레드 병렬화**
- BPE Training 기준 SentencePiece보다 **3~5x 빠름**

**③ SentencePiece — Google**
- C++ 구현 + Python 바인딩
- Unigram LM 방식도 지원 (BPE보다 병렬화 유리)
- BERT, T5, LLaMA 등이 사용

---

#### 알고리즘 레벨 최적화

**Priority Queue 기반 BPE**
```
순진한 구현: 매 merge마다 전체 corpus 재스캔 → O(n²)
Priority Queue: heap에 pair 빈도를 유지       → O(n log n)
```

**Trie / Aho-Corasick 기반 Inference**
```
순진한 구현: 50,000개 rule을 순서대로 시도 → O(M × k²)
Trie lookup: 현재 위치에서 바로 매칭        → O(k)
```

**Word-level Caching — GPT-2가 실제로 사용**

같은 단어는 항상 동일한 토큰으로 분리되므로, 한 번만 계산하고 결과를 캐싱합니다:
```python
# HuggingFace GPT2Tokenizer 내부
self.cache = {}  # {"Hello": [15496], "world": [995], ...}

def bpe(self, token):
    if token in self.cache:
        return self.cache[token]  # 재계산 없이 즉시 반환
    # ... BPE 계산 후 cache에 저장
    self.cache[token] = result
    return result
```

---

#### 하드웨어 레벨 가속화

| 방법 | 설명 | 한계 |
|------|------|------|
| **NVIDIA nvText (RAPIDS cuDF)** | GPU 기반 토크나이저 | BPE 순차성으로 효과 제한적 |
| **Sequence 간 병렬화** | 여러 단어/문장을 독립적으로 병렬 처리 | 단어 내부는 여전히 순차 |
| **Pre-tokenization 가속** | regex 분리 단계만 GPU regex 엔진으로 병렬화 | 전체 파이프라인의 일부만 |

**GPU 가속이 근본적으로 어려운 이유:**
```
BPE merge step N의 결과 → merge step N+1의 입력
→ 데이터 의존성이 있어 단계를 건너뛸 수 없음
→ GPU의 수천 개 코어를 활용하기 어려움
```

---

#### 대안 알고리즘 (BPE 자체를 대체)

| 알고리즘 | 사용 모델 | BPE 대비 장점 |
|----------|-----------|---------------|
| **Unigram LM** (SentencePiece) | T5, LLaMA, mBART | E-M 알고리즘 → 병렬화 유리 |
| **WordPiece** | BERT, DistilBERT | likelihood 기반 merge → inference 빠름 |
| **BPE-Dropout** | 학습 다양성 증가용 | Stochastic BPE, regularization 효과 |

---

#### GPT-2 실제 파이프라인 + 가속 포인트

```
텍스트 입력
    ↓ regex pre-tokenization (공백/구두점 분리)  ← [병렬화 가능]
    ↓ byte encoder 변환 (lookup table)           ← [O(1), 매우 빠름]
    ↓ BPE merge 적용                             ← [Word Cache hit → 즉시 반환]
    ↓ vocab.json lookup (hash map)               ← [O(1)]
토큰 ID 출력
```

**실전 비중:** GPT-2 전체 inference에서 tokenization은 **< 1%** 시간을 차지합니다.
병목은 항상 **Transformer 연산 (Attention, FFN)**입니다.

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

NPU Implementation:
├─ NPU Architecture Overview
│  ├─ Systolic array for matrix operations
│  ├─ Processing-in-Memory (PIM) capabilities
│  ├─ Optimized for INT8/INT16 quantized operations
│  └─ Much lower power consumption than GPU
│
├─ Embedding Lookup on NPU
│  ├─ Similar to GPU: Indexed memory read operation
│  ├─ NOT a matrix multiplication (no MAC operations used)
│  ├─ Utilizes DMA engines for memory transfer
│  │  └─ Token IDs → Memory addresses → Embedding vectors
│  └─ Parallel lookup using multiple DMA channels
│
├─ NPU-Specific Optimizations
│  ├─ Quantization (FP32 → INT8/INT16)
│  │  ├─ Embedding matrix: 154 MB → 38-77 MB (2-4× reduction)
│  │  ├─ Faster memory transfer (fits better in on-chip SRAM)
│  │  └─ Trade-off: Slight accuracy loss (~0.5% perplexity increase)
│  │
│  ├─ On-Chip Embedding Cache
│  │  ├─ NPUs often have large SRAM (e.g., 8-16 MB on Apple Neural Engine)
│  │  ├─ Cache top-K frequent tokens (K=2000-5000)
│  │  │  └─ Covers ~90% of actual token usage (Zipf distribution)
│  │  ├─ Cache hit: ~10-20 cycles (SRAM access)
│  │  ├─ Cache miss: ~500-1000 cycles (DRAM access)
│  │  └─ Effective speedup: ~3-5× vs full DRAM lookup
│  │
│  └─ Batch Processing
│     ├─ Process entire batch of token IDs simultaneously
│     ├─ Multiple DMA channels fetch embeddings in parallel
│     └─ Hide memory latency with computation overlap
│
├─ Performance Characteristics (Apple A17 Neural Engine example)
│  ├─ Single token lookup: ~50-100 ns (SRAM cache hit)
│  ├─ Batch 1024 tokens (uncached): ~30-50 μs
│  │  └─ Parallel memory fetch across 16 DMA channels
│  ├─ Power consumption: ~0.5-1 W (vs ~50-100 W GPU)
│  └─ Energy efficiency: 10-20× better than GPU
│
├─ Comparison with GPU (A100):
│  ┌──────────────────────┬─────────────┬──────────────┬──────────────┐
│  │ Metric               │ GPU (A100)  │ NPU (A17)    │ Advantage    │
│  ├──────────────────────┼─────────────┼──────────────┼──────────────┤
│  │ Latency (1024 tokens)│ 3.3 μs      │ 30 μs        │ GPU 9× faster│
│  │ Throughput (large B) │ 900 GB/s    │ 50-100 GB/s  │ GPU 9-18×    │
│  │ Power Consumption    │ 50-100 W    │ 0.5-1 W      │ NPU 100× less│
│  │ Energy per Inference │ ~0.17 mJ    │ ~0.03 mJ     │ NPU 6× better│
│  │ Matrix Size Support  │ Unlimited   │ Limited SRAM │ GPU better   │
│  └──────────────────────┴─────────────┴──────────────┴──────────────┘
│
└─ NPU Best Use Cases
   ├─ Mobile/Edge devices (battery-powered)
   ├─ Always-on inference (e.g., voice assistants)
   ├─ Small to medium batches (B=1-16)
   ├─ Quantized models (INT8/INT16 preferred)
   └─ When power efficiency > raw speed

Real-World NPU Examples:
├─ Apple Neural Engine (A14-A18, M1-M4)
│  ├─ 16-core design, ~35 TOPS (INT8)
│  ├─ Large on-chip SRAM for embedding cache
│  └─ Token embedding: ~20-40 μs (batch 512)
│
├─ Google Tensor TPU (Edge TPU)
│  ├─ 4-8 TOPS, optimized for MobileNet/BERT
│  ├─ INT8 quantization required
│  └─ Embedding lookup: ~50-80 μs (batch 512)
│
├─ Qualcomm Hexagon NPU (Snapdragon 8 Gen 3)
│  ├─ 12 TOPS, INT8/INT16 focus
│  ├─ Shared memory architecture
│  └─ Embedding lookup: ~30-60 μs (batch 512)
│
└─ Samsung NPU (Exynos 2400)
   ├─ 17 TOPS, INT8 optimized
   └─ Token embedding: ~25-50 μs (batch 512)

Key Insight:
├─ Embedding lookup is MEMORY-BOUND, not COMPUTE-BOUND
├─ NPU advantage comes from:
│  ├─ Lower memory latency (on-chip SRAM)
│  ├─ Better power efficiency
│  └─ Quantization support
└─ GPU wins on raw throughput, NPU wins on efficiency
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

NPU Implementation:
├─ IDEAL for NPU Execution
│  ├─ Position matrix is small (3.1 MB)
│  ├─ Sequential access pattern (perfect for DMA streaming)
│  ├─ Highly predictable memory access
│  └─ Can fit entirely in NPU on-chip SRAM
│
├─ NPU Optimizations
│  ├─ Entire Position Matrix in SRAM
│  │  ├─ Many NPUs have 8-16 MB SRAM
│  │  ├─ Position matrix: 3.1 MB (FP32) or 0.78 MB (INT8)
│  │  ├─ Load once at model initialization
│  │  └─ Zero-latency access during inference
│  │
│  ├─ DMA Streaming Read
│  │  ├─ Sequential position IDs (0, 1, 2, ..., seq_len-1)
│  │  ├─ Burst read from SRAM (optimal bandwidth)
│  │  ├─ No cache misses (all data in SRAM)
│  │  └─ Perfect prefetching from sequential pattern
│  │
│  └─ Quantization Benefits
│     ├─ FP32 → INT8: 4× smaller (3.1 MB → 0.78 MB)
│     ├─ Entire matrix easily fits in small NPU SRAM
│     ├─ 4× faster memory transfer
│     └─ Minimal accuracy loss (position info is robust)
│
├─ Performance (Apple A17 Neural Engine example)
│  ├─ Position matrix: Preloaded in SRAM (one-time cost)
│  ├─ Sequential lookup (seq_len=1024):
│  │  ├─ SRAM burst read: ~5-10 μs
│  │  ├─ No DRAM access needed
│  │  └─ Fully pipelined with token embedding
│  │
│  └─ Latency breakdown:
│     ├─ Address generation: ~100 ns (sequential, trivial)
│     ├─ SRAM read (1024 positions): ~5 μs (burst mode)
│     ├─ No stalls, no cache misses
│     └─ Total: ~5 μs (vs 124 μs CPU, 3.3 μs GPU)
│
├─ Comparison Across Hardware:
│  ┌────────────────┬──────────────┬─────────────────────────────┐
│  │ Hardware       │ Time (1024)  │ Key Advantage               │
│  ├────────────────┼──────────────┼─────────────────────────────┤
│  │ CPU            │ 124 μs       │ Prefetch helps, DRAM-bound  │
│  │ GPU (A100)     │ 3.3 μs       │ High bandwidth, parallel    │
│  │ NPU (A17)      │ ~5 μs        │ SRAM cached, power-efficient│
│  │ NPU (quantized)│ ~2 μs        │ INT8, 4× less data          │
│  └────────────────┴──────────────┴─────────────────────────────┘
│
└─ Power Efficiency:
   ├─ CPU: ~3 W for 124 μs = ~0.37 mJ
   ├─ GPU: ~50 W for 3.3 μs = ~0.17 mJ
   ├─ NPU: ~0.5 W for 5 μs = ~0.0025 mJ
   └─ NPU is 68× more energy-efficient than GPU!

NPU Best Practices for Position Embedding:
├─ Preload Strategy
│  ├─ Load position matrix to SRAM at model init
│  ├─ Keep in SRAM across all inferences
│  └─ Only reload if model changes
│
├─ Quantization Strategy
│  ├─ Use INT8 or INT16 for position embeddings
│  ├─ Minimal impact on final model accuracy
│  ├─ 4-2× memory savings
│  └─ Faster data movement
│
└─ Memory Layout
   ├─ Store positions contiguously in SRAM
   ├─ Align to DMA burst size (e.g., 64 bytes)
   └─ Enable sequential burst reads

Real-World NPU Performance:
├─ Apple Neural Engine (A17/M4)
│  ├─ Position matrix: Fully cached in 16 MB SRAM
│  ├─ Lookup 512 positions: ~2-3 μs
│  └─ Power: ~0.001 mJ
│
├─ Google Edge TPU
│  ├─ Position matrix: 8 MB SRAM (INT8 quantized)
│  ├─ Lookup 512 positions: ~4-6 μs
│  └─ Optimized for sequential DMA
│
└─ Qualcomm Hexagon NPU
   ├─ Hybrid SRAM/DRAM approach
   ├─ Frequent positions in SRAM (first 256)
   └─ Lookup 512 positions: ~6-10 μs

Key Insight:
├─ Position embedding is NPU-FRIENDLY
│  ├─ Small matrix size (fits in SRAM)
│  ├─ Sequential access (perfect for DMA)
│  └─ Predictable pattern (no cache thrashing)
├─ Much better than token embedding for NPU
└─ NPU beats CPU on power, competitive with GPU on speed
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

NPU Implementation:
├─ PERFECT for NPU Execution
│  ├─ Element-wise addition is core NPU operation
│  ├─ NPU has dedicated vector ALUs for this
│  ├─ Can fuse with other operations (activation functions)
│  └─ Zero overhead when pipelined with other ops
│
├─ NPU Architecture for Vector Addition
│  ├─ Systolic Array Bypass Mode
│  │  ├─ Matrix ops use systolic array (MAC units)
│  │  ├─ Vector ops use dedicated vector units
│  │  └─ Addition doesn't need MACs
│  │
│  ├─ Vector Processing Units (VPUs)
│  │  ├─ Separate ALUs for element-wise ops
│  │  ├─ Wide SIMD: 32-128 elements in parallel
│  │  ├─ 1 cycle throughput per vector operation
│  │  └─ Pipelined with memory operations
│  │
│  └─ Operation Fusion
│     ├─ Combine embedding lookup + addition + activation
│     ├─ No intermediate memory writes
│     ├─ Data stays in on-chip registers
│     └─ Massive latency reduction
│
├─ NPU Performance (B=1, L=1024, d=768)
│  ├─ Standalone Addition (if executed separately):
│  │  ├─ Vector width: 64 floats (typical NPU)
│  │  ├─ Total vectors: 786,432 / 64 = 12,288
│  │  ├─ Cycles: ~12,288 cycles @ 1 cycle/vector
│  │  ├─ Time: ~12 μs @ 1 GHz NPU clock
│  │  └─ Power: ~0.2 W
│  │
│  ├─ Fused Operation (typical real-world case):
│  │  ├─ Combine: Token embed lookup + Pos embed + Add
│  │  ├─ Pipeline stages:
│  │  │  └─ Stage 1: DMA fetch token embedding
│  │  │  └─ Stage 2: DMA fetch position embedding
│  │  │  └─ Stage 3: Vector add (while fetching next)
│  │  ├─ Addition fully hidden in memory latency
│  │  ├─ Effective addition time: ~0 μs (free!)
│  │  └─ Total time dominated by memory fetch
│  │
│  └─ Quantized Mode (INT8):
│     ├─ INT8 addition: Even faster than FP32
│     ├─ Wider SIMD: 128-256 INT8 elements/cycle
│     ├─ Time: ~3-5 μs (standalone)
│     └─ Fused: Still essentially free
│
├─ NPU Optimization Techniques
│  ├─ Operator Fusion
│  │  ├─ Fuse: Embed_Lookup → Add → LayerNorm
│  │  ├─ Compiler automatically fuses operations
│  │  ├─ Reduces memory traffic by 2-3×
│  │  └─ Example: CoreML, TensorFlow Lite, ONNX Runtime
│  │
│  ├─ In-Register Computation
│  │  ├─ Embeddings loaded to vector registers
│  │  ├─ Addition happens in registers (no SRAM write)
│  │  ├─ Result directly fed to next layer
│  │  └─ Zero memory bandwidth for addition
│  │
│  └─ Precision Management
│     ├─ FP32 embeddings → INT8 addition → FP32 output
│     ├─ Mixed precision for speed + accuracy
│     └─ NPU hardware supports all common formats
│
├─ Real-World NPU Performance
│  ├─ Apple Neural Engine (A17)
│  │  ├─ Vector units: 128-wide INT8, 64-wide FP16
│  │  ├─ Addition (fused): ~0 μs (hidden in pipeline)
│  │  ├─ Addition (standalone): ~3-5 μs (rarely done)
│  │  └─ Power for addition: ~0.0001 mJ (negligible)
│  │
│  ├─ Google Edge TPU
│  │  ├─ Systolic array: 64×64 for matmul
│  │  ├─ Vector ALU: Separate 128-wide unit
│  │  ├─ Addition: ~5-8 μs (standalone, INT8)
│  │  └─ Usually fused with quantization step
│  │
│  └─ Qualcomm Hexagon NPU
│     ├─ HVX vector units: 128-byte SIMD
│     ├─ Addition: ~4-6 μs (standalone)
│     └─ Fusion engine combines up to 4 ops
│
├─ Performance Comparison (B=1, L=1024):
│  ┌──────────────────┬──────────────┬─────────────┬──────────────┐
│  │ Hardware         │ Time         │ Power       │ Energy       │
│  ├──────────────────┼──────────────┼─────────────┼──────────────┤
│  │ CPU (scalar)     │ 1.24 ms      │ 3 W         │ 3.72 mJ      │
│  │ CPU (AVX-512)    │ 39 μs        │ 3 W         │ 0.12 mJ      │
│  │ GPU (A100)       │ 5 μs         │ 50 W        │ 0.25 mJ      │
│  │ NPU (standalone) │ 12 μs        │ 0.2 W       │ 0.0024 mJ    │
│  │ NPU (fused)      │ ~0 μs        │ 0 W         │ ~0 mJ        │
│  └──────────────────┴──────────────┴─────────────┴──────────────┘
│
└─ Key Takeaway:
   ├─ Addition is trivial on NPU (almost free when fused)
   ├─ NPU shines through operator fusion
   ├─ Energy efficiency: 100-1000× better than CPU/GPU
   └─ Best for mobile/edge inference

NPU Operator Fusion Example:
```
Without Fusion (3 separate operations):
1. Token_Embed: Token_IDs → DRAM read → SRAM → Registers (~30 μs)
2. Pos_Embed: Pos_IDs → SRAM read → Registers (~5 μs)
3. Add: Registers → Vector ALU → SRAM write (~12 μs)
Total: ~47 μs, multiple memory round-trips

With Fusion (single fused op):
1. Fused_Embed_Add:
   ├─ Token_IDs → DRAM read → Registers
   ├─ Pos_IDs → SRAM read → Registers
   ├─ Vector Add in registers (overlapped with fetch)
   └─ Result → Next layer (no SRAM write)
Total: ~30 μs, single memory round-trip
Speedup: 1.6×, Energy savings: 2-3×
```

Why NPU Excels at Vector Addition:
├─ Hardware Support
│  ├─ Dedicated wide SIMD vector units
│  ├─ Single-cycle throughput for common ops
│  └─ Parallel execution with memory ops
│
├─ Compiler Optimization
│  ├─ Automatic operator fusion
│  ├─ Eliminates intermediate memory traffic
│  └─ Optimizes entire computation graph
│
└─ Energy Efficiency
   ├─ Low-power vector ALUs (~0.1-0.5 W)
   ├─ On-chip SRAM (no DRAM power)
   └─ Operation fusion reduces data movement
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

---

## CPU vs GPU vs NPU 아키텍처 종합 비교

### 하드웨어 아키텍처 근본적 차이

```
CPU (Central Processing Unit):
┌─────────────────────────────────────────────────────────┐
│ Design Philosophy: General-purpose, Low-latency         │
├─────────────────────────────────────────────────────────┤
│ Core Architecture:                                      │
│ ├─ 4-32 cores (high-end: up to 128)                    │
│ ├─ Complex out-of-order execution                      │
│ ├─ Branch prediction, speculative execution            │
│ ├─ Large caches (32 MB - 256 MB total)                 │
│ └─ High clock speed (2-5 GHz)                          │
│                                                         │
│ Compute Units:                                          │
│ ├─ Scalar ALUs + Vector units (AVX-512)               │
│ ├─ ~16 FP32 ops per cycle per core (AVX-512)          │
│ ├─ Peak: ~1-3 TFLOPS (FP32)                           │
│ └─ Optimized for serial/branching code                 │
│                                                         │
│ Memory:                                                 │
│ ├─ L1: 32-64 KB/core, L2: 256KB-1MB/core              │
│ ├─ L3: 8-256 MB shared                                │
│ ├─ DRAM: 16-128 GB DDR4/DDR5                          │
│ └─ Bandwidth: 50-100 GB/s                             │
│                                                         │
│ Power: 15-280 W (desktop/server)                       │
└─────────────────────────────────────────────────────────┘

GPU (Graphics Processing Unit):
┌─────────────────────────────────────────────────────────┐
│ Design Philosophy: Massive parallelism, High throughput │
├─────────────────────────────────────────────────────────┤
│ Core Architecture:                                      │
│ ├─ 2,000-16,000+ CUDA cores (NVIDIA A100: 6,912)      │
│ ├─ Organized in SMs (Streaming Multiprocessors)       │
│ ├─ SIMT execution model (warp-based)                  │
│ ├─ Limited cache per SM                               │
│ └─ Lower clock speed (1-2 GHz)                        │
│                                                         │
│ Compute Units:                                          │
│ ├─ Thousands of simple FP32/FP16 ALUs                 │
│ ├─ Tensor Cores for matrix operations (A100: 312)     │
│ ├─ Peak: 19.5 TFLOPS (FP32), 312 TFLOPS (FP16 Tensor)│
│ └─ Optimized for data-parallel operations              │
│                                                         │
│ Memory:                                                 │
│ ├─ Registers: 256 KB/SM                               │
│ ├─ Shared Memory: 164 KB/SM (A100)                    │
│ ├─ L2 Cache: 40 MB (A100)                             │
│ ├─ HBM2/HBM2e: 40-80 GB                               │
│ └─ Bandwidth: 900 GB/s - 2 TB/s (A100)               │
│                                                         │
│ Power: 50-400 W (datacenter GPUs)                      │
└─────────────────────────────────────────────────────────┘

NPU (Neural Processing Unit / AI Accelerator):
┌─────────────────────────────────────────────────────────┐
│ Design Philosophy: Energy-efficient AI, Fixed workloads │
├─────────────────────────────────────────────────────────┤
│ Core Architecture:                                      │
│ ├─ Systolic array (8×8 to 128×128 MACs)               │
│ ├─ Fixed dataflow, no branching                       │
│ ├─ Processing-in-Memory (PIM) design                  │
│ ├─ Specialized for matrix multiply-accumulate         │
│ └─ Low clock speed (0.5-1.5 GHz)                      │
│                                                         │
│ Compute Units:                                          │
│ ├─ MAC arrays (Apple A17: ~35 TOPS INT8)              │
│ ├─ Quantization support (INT8/INT16 native)           │
│ ├─ Peak: 10-50 TOPS (INT8), 2-10 TFLOPS (FP16)       │
│ └─ Optimized ONLY for neural net operations           │
│                                                         │
│ Memory:                                                 │
│ ├─ Large on-chip SRAM: 4-16 MB                        │
│ ├─ Shared with CPU system memory (unified)            │
│ ├─ LPDDR4/LPDDR5: 4-16 GB (mobile)                   │
│ └─ Bandwidth: 50-100 GB/s (shared with CPU)          │
│                                                         │
│ Power: 0.5-5 W (mobile/edge NPUs)                     │
└─────────────────────────────────────────────────────────┘
```

### 입력 처리 단계별 하드웨어 적합성

```
┌──────────────────────┬──────────┬──────────┬──────────┬─────────────────────┐
│ Processing Stage     │ CPU      │ GPU      │ NPU      │ Reason              │
├──────────────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ UTF-8 Encoding       │ ★★★★★    │ ☆☆☆☆☆    │ ☆☆☆☆☆    │ Control flow heavy  │
│ Byte-to-Unicode LUT  │ ★★★★★    │ ★☆☆☆☆    │ ☆☆☆☆☆    │ Small lookup, fast  │
│ BPE Tokenization     │ ★★★★★    │ ★☆☆☆☆    │ ☆☆☆☆☆    │ Sequential algorithm│
│ Vocabulary Lookup    │ ★★★★★    │ ★★☆☆☆    │ ☆☆☆☆☆    │ Hash table ops      │
│ Token Embedding      │ ★★☆☆☆    │ ★★★★★    │ ★★★★☆    │ Memory bandwidth    │
│ Position Embedding   │ ★★★☆☆    │ ★★★★★    │ ★★★★★    │ Sequential access   │
│ Embedding Addition   │ ★★★★☆    │ ★★★★★    │ ★★★★★    │ Vector operation    │
└──────────────────────┴──────────┴──────────┴──────────┴─────────────────────┘

Legend: ★ = Suitability (more stars = better)
```

### 성능 및 효율성 종합 비교

**시나리오 1: 모바일/엣지 추론 (B=1, L=512)**

```
┌────────────┬──────────────┬──────────────┬─────────────┬──────────────┐
│ Hardware   │ Latency      │ Power        │ Energy      │ Notes        │
├────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ CPU        │ ~1.2 ms      │ 3-5 W        │ ~4.8 mJ     │ Always on    │
│ (Mobile)   │              │              │             │ Low latency  │
├────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ GPU        │ Not typical  │ N/A          │ N/A         │ Unavailable  │
│ (Mobile)   │ (Mali/Adreno)│              │             │ on mobile    │
├────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ NPU        │ ~800 μs      │ 0.5-1 W      │ ~0.5 mJ     │ BEST choice  │
│ (A17/M4)   │              │              │             │ 10× efficient│
└────────────┴──────────────┴──────────────┴─────────────┴──────────────┘

Winner: NPU (for battery life and sustained performance)
```

**시나리오 2: 서버 추론 (B=1, L=1024)**

```
┌────────────┬──────────────┬──────────────┬─────────────┬──────────────┐
│ Hardware   │ Latency      │ Power        │ Energy      │ Notes        │
├────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ CPU        │ ~2.2 ms      │ 15-65 W      │ ~110 mJ     │ Flexible     │
│ (Xeon)     │              │              │             │ Easy to code │
├────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ GPU        │ ~15 μs       │ 50-100 W     │ ~1.5 mJ     │ BEST latency │
│ (A100)     │ (GPU only)   │              │             │ High throughput
├────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ NPU        │ Not typical  │ N/A          │ N/A         │ Rare in      │
│ (Server)   │              │              │             │ servers      │
└────────────┴──────────────┴──────────────┴─────────────┴──────────────┘

Winner: GPU (for raw speed and throughput)
```

**시나리오 3: 대규모 배치 추론 (B=128, L=512)**

```
┌────────────┬──────────────┬──────────────┬─────────────┬──────────────┐
│ Hardware   │ Throughput   │ Latency/Item │ Efficiency  │ Notes        │
├────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ CPU        │ ~50 seq/s    │ ~20 ms       │ Low         │ Poor scaling │
│ (64 cores) │              │              │             │              │
├────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ GPU        │ ~8000 seq/s  │ ~16 μs       │ High        │ BEST for     │
│ (A100)     │              │              │             │ large batch  │
├────────────┼──────────────┼──────────────┼─────────────┼──────────────┤
│ NPU        │ ~200 seq/s   │ ~640 μs      │ Medium      │ Limited SRAM │
│ (A17)      │              │              │             │ capacity     │
└────────────┴──────────────┴──────────────┴─────────────┴──────────────┘

Winner: GPU (by far, for batch processing)
```

### 하드웨어 선택 가이드

```
Use CPU when:
├─ Prototype/development (easiest to code)
├─ Small-scale deployment (< 100 requests/day)
├─ Complex preprocessing (BPE tokenization required)
├─ Variable/unpredictable workloads
└─ No access to GPU/NPU hardware

Use GPU when:
├─ High throughput needed (> 1000 requests/sec)
├─ Large batch sizes (B ≥ 32)
├─ Training or fine-tuning
├─ Server/datacenter deployment
├─ Cost amortized over many requests
└─ Lowest latency for single requests (B=1 with GPU available)

Use NPU when:
├─ Mobile/edge devices (smartphones, tablets)
├─ Battery-powered systems
├─ Always-on inference (voice assistants, real-time translation)
├─ Small to medium batches (B=1-16)
├─ Power budget < 5 W
├─ Quantized models (INT8/INT16)
└─ Energy efficiency is critical
```

### 실전 배포 전략

```
Hybrid CPU + GPU Strategy (Common in Cloud):
├─ CPU handles:
│  ├─ Tokenization (BPE)
│  ├─ Request routing
│  ├─ Postprocessing (text generation, sampling)
│  └─ Small requests (B=1-4)
│
└─ GPU handles:
   ├─ Embedding lookup
   ├─ Transformer layers
   ├─ Large batches (B ≥ 16)
   └─ Batch accumulation from multiple requests

Hybrid CPU + NPU Strategy (Common on Mobile):
├─ CPU handles:
│  ├─ Tokenization (BPE)
│  ├─ App logic, UI updates
│  ├─ Postprocessing
│  └─ Dynamic control flow
│
└─ NPU handles:
   ├─ Embedding lookup (quantized)
   ├─ Transformer layers (quantized)
   ├─ All tensor operations
   └─ Runs in background, low power

Example: OpenAI API Backend (Speculative):
┌─────────────────────────────────────────────────────┐
│ Request arrives → CPU (tokenize)                    │
│        ↓                                            │
│ Batch accumulator (wait for B=32-128)              │
│        ↓                                            │
│ Transfer to GPU → Embedding + Transformers         │
│        ↓                                            │
│ Transfer back to CPU → Decode, sample, detokenize  │
│        ↓                                            │
│ Return response                                     │
└─────────────────────────────────────────────────────┘
Achieves: ~15-20 ms latency, 1000+ requests/sec/GPU

Example: Siri/Google Assistant (Speculative):
┌─────────────────────────────────────────────────────┐
│ Wake word detection → NPU (always-on, ultra low power)
│        ↓                                            │
│ Audio → CPU (preprocessing) → NPU (speech-to-text) │
│        ↓                                            │
│ Text → CPU (tokenize) → NPU (LLM inference)        │
│        ↓                                            │
│ Response → CPU (text-to-speech) → NPU (vocoder)    │
│        ↓                                            │
│ Audio output                                        │
└─────────────────────────────────────────────────────┘
Power: ~0.5-2 W average, ~50-100 ms latency
```

### 미래 트렌드 및 전망

```
CPU Evolution:
├─ More vector units (AVX-1024?)
├─ Specialized AI instructions (AMX, AMX-FP16)
├─ Tighter integration with accelerators
└─ Likely to remain for tokenization/control flow

GPU Evolution:
├─ Larger tensor cores (H100, B100)
├─ Higher memory bandwidth (3-5 TB/s)
├─ Better support for sparse operations
├─ FP8/FP4 precision for efficiency
└─ Dominant for training, high-throughput inference

NPU Evolution:
├─ Larger on-chip SRAM (32-64 MB)
├─ Better INT4/INT2 quantization
├─ Hybrid architectures (NPU + GPU features)
├─ Specialized units for transformers (Flash Attention)
└─ Dominant for mobile/edge, growing in servers

Emerging: Specialized LLM Accelerators
├─ Groq LPU (Language Processing Unit)
│  └─ Deterministic, ultra-low latency (~1 ms for GPT-2)
├─ Cerebras Wafer-Scale Engine
│  └─ 850,000 cores, 40 GB on-chip SRAM
├─ Google TPU v5
│  └─ Optimized for large language models
└─ Custom ASICs for specific models (e.g., AWS Inferentia)
```

### 핵심 요약

| 차원 | CPU | GPU | NPU |
|------|-----|-----|-----|
| **강점** | 범용성, 저지연, 프로그래밍 용이 | 처리량, 병렬성, 메모리 대역폭 | 전력 효율, 양자화, 온칩 메모리 |
| **약점** | 처리량 낮음, 전력 비효율 | 전력 소모 큼, 전송 오버헤드 | 유연성 부족, SRAM 제한, 처리량 낮음 |
| **Best for** | 토큰화, 전처리, 개발 | 대규모 배치, 학습, 클라우드 | 모바일, 엣지, 배터리 기기 |
| **Cost** | 낮음-중간 | 높음 (GPU 가격) | 낮음 (통합형) |
| **Maturity** | 매우 성숙 | 성숙 | 빠르게 발전 중 |

**입력 처리에 대한 최종 결론:**
- **토큰화**: CPU 필수 (BPE는 순차적)
- **임베딩 룩업**: GPU > NPU > CPU (메모리 대역폭 게임)
- **벡터 연산**: NPU ≈ GPU >> CPU (전력 효율 vs 처리량)
- **실전**: 하이브리드 접근 (CPU 토큰화 + GPU/NPU 임베딩)
