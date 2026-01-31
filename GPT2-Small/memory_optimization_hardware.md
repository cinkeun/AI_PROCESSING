# 하드웨어 메모리 시스템 최적화: GPT-2 Small 관점

## 목차
1. [이론 vs 실제 메모리 대역폭](#1-이론-vs-실제-메모리-대역폭)
2. [메모리 접근 패턴 문제](#2-메모리-접근-패턴-문제)
3. [GPU 메모리 최적화](#3-gpu-메모리-최적화)
4. [NPU 메모리 최적화](#4-npu-메모리-최적화)
5. [NoC (Network-on-Chip) 아키텍처](#5-noc-network-on-chip-아키텍처)
6. [특수 하드웨어 가속기](#6-특수-하드웨어-가속기)
7. [GPT-2 레이어별 최적화](#7-gpt-2-레이어별-최적화)

---

## 1. 이론 vs 실제 메모리 대역폭

### 1.1 이론적 대역폭의 함정

```
NVIDIA A100 HBM2e 사양:
- 이론적 대역폭: 2,039 GB/s
- 실제 달성 가능: ~1,555 GB/s (76%)
- 일반적인 코드: ~800-1,200 GB/s (40-60%)

왜 차이가 나는가?
```

### 1.2 메모리 계층구조의 현실

```
┌─────────────────────────────────────────────────────────────────┐
│ Memory Hierarchy                                                │
├─────────────────────────────────────────────────────────────────┤
│ Register (GPU)                                                  │
│ ├─ Size: 256 KB per SM                                         │
│ ├─ Latency: 0 cycles                                           │
│ └─ Bandwidth: Infinite (within SM)                             │
├─────────────────────────────────────────────────────────────────┤
│ Shared Memory (GPU) / SRAM (NPU)                               │
│ ├─ Size: 164 KB per SM (A100) / 4-16 MB (NPU)                │
│ ├─ Latency: ~20-30 cycles                                      │
│ ├─ Bandwidth: ~19 TB/s per SM (A100)                          │
│ └─ 🚨 BANK CONFLICTS can reduce bandwidth by 32×!              │
├─────────────────────────────────────────────────────────────────┤
│ L2 Cache                                                        │
│ ├─ Size: 40 MB (A100)                                          │
│ ├─ Latency: ~200 cycles                                        │
│ ├─ Bandwidth: ~7 TB/s (aggregate)                             │
│ └─ 🚨 Cache line: 128 bytes, waste if using <128 bytes        │
├─────────────────────────────────────────────────────────────────┤
│ HBM2/HBM2e (GPU) / LPDDR5 (NPU)                               │
│ ├─ Size: 40-80 GB (GPU) / 4-16 GB (NPU)                       │
│ ├─ Latency: ~300-400 cycles                                    │
│ ├─ Bandwidth: 900 GB/s - 2 TB/s (GPU) / 50-100 GB/s (NPU)    │
│ └─ 🚨 Cache line: 64-128 bytes                                 │
│    🚨 Memory coalescing required!                              │
│    🚨 Strided access can waste 90%+ bandwidth!                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Cache Line 활용률

**핵심 문제: 캐시 라인 크기와 실제 사용 데이터의 불일치**

```
GPU HBM Cache Line: 128 bytes (32 float32 values)

Example 1: Sequential Access (GOOD)
┌────────────────────────────────────────────┐
│ Cache Line: [a0, a1, a2, ..., a31]        │
│ Thread 0 reads a0  ✓ Load 128 bytes       │
│ Thread 1 reads a1  ✓ Already in cache     │
│ Thread 2 reads a2  ✓ Already in cache     │
│ ...                                        │
│ Thread 31 reads a31 ✓ Already in cache    │
└────────────────────────────────────────────┘
Utilization: 100% (all 32 values used)
Effective bandwidth: 2,039 GB/s

Example 2: Strided Access (BAD)
┌────────────────────────────────────────────┐
│ Cache Line: [a0, a1, a2, ..., a31]        │
│ Thread 0 reads a0  ✓ Load 128 bytes       │
│ Thread 1 reads a256 ✗ Different line!     │
│ Thread 2 reads a512 ✗ Different line!     │
└────────────────────────────────────────────┘
Utilization: 3.125% (1 out of 32 values used per line)
Effective bandwidth: 63.7 GB/s (32× slower!)
```

### 1.4 실제 대역폭 계산

```python
# 이론적 계산 (내가 이전에 했던 방식)
data_size = 3 MB  # Token embedding for 1024 tokens
theoretical_time = 3 MB / 2039 GB/s = 1.47 μs

# 실제 계산 (메모리 접근 패턴 고려)
cache_line_size = 128 bytes
elements_per_line = 32  # float32
stride = 768  # GPT-2 hidden size

# Random token embedding lookup
# Each token reads 768 floats = 3 KB
# But scattered across 768/32 = 24 cache lines
# Each cache line fetch wastes ~90% if tokens are random

cache_lines_per_token = 768 / 32 = 24
wasted_data_per_token = (32 - 1) * 24 * 4 = 2,976 bytes (99% waste!)
actual_bandwidth = 2039 GB/s * 0.01 = 20 GB/s

actual_time = 3 MB / 20 GB/s = 150 μs (100× slower!)
```

---

## 2. 메모리 접근 패턴 문제

### 2.1 GPT-2에서의 주요 문제 패턴

```
┌────────────────────────┬──────────────┬──────────────┬───────────┐
│ Operation              │ Access Type  │ Efficiency   │ Fix       │
├────────────────────────┼──────────────┼──────────────┼───────────┤
│ Token Embedding Lookup │ Random       │ 1-5%         │ Batching  │
│ Position Embedding     │ Sequential   │ 90-100%      │ ✓ Good    │
│ Matrix Multiply (A×B)  │ Sequential   │ 70-95%       │ Tiling    │
│ Matrix Transpose       │ Strided      │ 3-12%        │ Shared Mem│
│ Softmax (reduction)    │ Sequential   │ 60-80%       │ Warp ops  │
│ LayerNorm (reduction)  │ Sequential   │ 50-70%       │ Fusion    │
│ Sparse Attention       │ Sparse       │ 10-30%       │ CSR format│
└────────────────────────┴──────────────┴──────────────┴───────────┘
```

### 2.2 문제 1: Embedding Lookup (Random Access)

```
Token Embedding Matrix E: [50257, 768]
Token IDs: [15496, 995, 262, ...]  (random indices)

Memory Layout (row-major):
┌──────────────────────────────────────────────────┐
│ Row 0:    [e00, e01, e02, ..., e0,767]          │  3 KB
│ Row 1:    [e10, e11, e12, ..., e1,767]          │  3 KB
│ ...                                              │
│ Row 995:  [e995,0, e995,1, ..., e995,767]       │  3 KB
│ ...                                              │
│ Row 15496: [e15496,0, ..., e15496,767]          │  3 KB
│ ...                                              │
└──────────────────────────────────────────────────┘

Problem:
- Token 15496: Read from row 15496 (offset 15496 × 3 KB = 46 MB)
- Token 995:   Read from row 995    (offset 995 × 3 KB = 2.9 MB)
- Completely different memory locations!
- No cache line reuse
- Each read: 768/32 = 24 cache line fetches
- But neighboring tokens likely don't share cache lines

Effective Bandwidth: ~1-5% of peak
```

**시각화:**

```
Token IDs:     [15496,    995,      262]
                  ↓         ↓         ↓
Memory:      [  ...  ] [  ...  ] [  ...  ]  ← Scattered!
             46 MB      2.9 MB     0.8 MB

Cache Line Utilization:
┌────────────────────┐
│ Fetched: 128 bytes │
│ Used:    4 bytes   │  ← Only 1 float from this cache line
│ Wasted:  124 bytes │
└────────────────────┘
```

### 2.3 문제 2: Matrix Transpose

```
Original Matrix A: [M, N] = [1024, 768]
Transposed A^T:    [N, M] = [768, 1024]

Row-major layout in memory:
A[i, j] is at: base + (i × N + j) × sizeof(float)

Reading A^T column-by-column = Reading A row-by-row (stride N)

Example: Read column 0 of A^T (= row of A)
A[0,0], A[0,1], A[0,2], ..., A[0,767]
↓
Memory addresses: 0, 4, 8, ..., 3,068 (sequential, GOOD!)

Example: Read row 0 of A^T (= column of A)
A[0,0], A[1,0], A[2,0], ..., A[1023,0]
↓
Memory addresses: 0, 3,072, 6,144, ..., 3,141,632 (strided, BAD!)

Stride = 768 floats = 3,072 bytes
Cache line = 128 bytes = 32 floats
→ Only 1 float used per cache line!
→ 96.875% bandwidth waste
```

**Visualization:**

```
Transpose Access Pattern:

Cache Line 0:  [A[0,0], A[0,1], ..., A[0,31]]
                 ↑ Use this, waste others
Cache Line 1:  [A[1,0], A[1,1], ..., A[1,31]]
                 ↑ Use this, waste others
...

Effective bandwidth: 3.125% of peak
```

### 2.4 문제 3: Sparse Matrix

```
Sparse Attention Matrix (example):
- Size: [1024, 1024]
- Sparsity: 90% zeros
- Non-zero pattern: Band diagonal + random positions

Dense Storage:
Memory: 1024 × 1024 × 4 = 4 MB
Useful: 10% = 400 KB
Wasted: 3.6 MB (90%)

Memory Access:
- Must load entire rows/columns
- Compute on zeros (waste)
- Cannot skip zero regions easily
```

---

## 3. GPU 메모리 최적화

### 3.1 Memory Coalescing (CUDA)

**핵심 개념:** Warp 내 32개 스레드가 연속된 메모리에 접근해야 함

```
GOOD: Coalesced Access
┌──────────────────────────────────────────────────────────┐
│ Warp (32 threads)                                        │
│ Thread 0: reads address 0                                │
│ Thread 1: reads address 4                                │
│ Thread 2: reads address 8                                │
│ ...                                                       │
│ Thread 31: reads address 124                             │
│                                                           │
│ Result: Single 128-byte transaction                      │
│ Bandwidth utilization: 100%                              │
└──────────────────────────────────────────────────────────┘

BAD: Uncoalesced Access
┌──────────────────────────────────────────────────────────┐
│ Warp (32 threads)                                        │
│ Thread 0: reads address 0                                │
│ Thread 1: reads address 3072  (stride!)                  │
│ Thread 2: reads address 6144                             │
│ ...                                                       │
│ Thread 31: reads address 95,232                          │
│                                                           │
│ Result: 32 separate 128-byte transactions                │
│ Bandwidth utilization: 3.125%                            │
└──────────────────────────────────────────────────────────┘
```

**GPT-2 예시: Matrix Multiplication (QK^T)**

```cuda
// BAD: Uncoalesced (naive transpose)
__global__ void matmul_bad(float* C, float* A, float* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // A[row, k]: stride K (bad for coalescing)
        // B[k, col]: stride K (bad for coalescing)
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
// Bandwidth: ~50 GB/s (2.5% of peak)

// GOOD: Tiled with shared memory
__global__ void matmul_tiled(float* C, float* A, float* B, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Tile across K dimension
    for (int t = 0; t < K / TILE_SIZE; t++) {
        // Coalesced load into shared memory
        tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute using shared memory (no global memory access)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
// Bandwidth: ~1,500 GB/s (75% of peak)
// Speedup: 30×
```

### 3.2 Shared Memory Bank Conflicts

**구조:**
```
Shared Memory (A100): 164 KB, divided into 32 banks

Bank Structure:
┌────────┬────────┬────────┬─────┬────────┐
│ Bank 0 │ Bank 1 │ Bank 2 │ ... │ Bank 31│
│ 0      │ 4      │ 8      │     │ 124    │  ← First 128 bytes
│ 128    │ 132    │ 136    │     │ 252    │  ← Next 128 bytes
│ ...    │ ...    │ ...    │     │ ...    │
└────────┴────────┴────────┴─────┴────────┘

Consecutive 4-byte words go to different banks
```

**Bank Conflict 예시:**

```cuda
__shared__ float shared[32][32];

// BAD: Bank conflict (32-way conflict!)
float value = shared[threadIdx.x][0];
// All 32 threads access Bank 0
// Serialized: 32× slower

// GOOD: No bank conflict
float value = shared[0][threadIdx.x];
// Thread i accesses Bank i
// Parallel: 1× time
```

**GPT-2 Matrix Transpose 최적화:**

```cuda
__global__ void transpose_optimized(float* out, float* in, int M, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts!

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Coalesced read from input
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    // Transpose indices
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Coalesced write to output (transposed)
    out[y * M + x] = tile[threadIdx.x][threadIdx.y];
}

// Key optimization: tile[TILE_SIZE][TILE_SIZE + 1]
// The +1 padding ensures no bank conflicts during transpose read
```

### 3.3 Tensor Cores (NVIDIA)

**특수 하드웨어: 행렬 연산 가속**

```
Tensor Core (A100):
- Operates on small matrix tiles: 16×16 or 8×8
- Mixed precision: FP16/BF16 input, FP32 accumulate
- Throughput: 312 TFLOPS (vs 19.5 TFLOPS FP32 CUDA cores)

Example: D = A × B + C
A: [M, K] FP16
B: [K, N] FP16
C: [M, N] FP32
D: [M, N] FP32

Operation:
for each 16×16 tile:
    D_tile = A_tile (16×16 FP16) × B_tile (16×16 FP16) + C_tile (16×16 FP32)
    → Single Tensor Core instruction
    → 4,096 FMA ops in ~1 cycle!
```

**GPT-2 QKV Projection:**

```cuda
// Using CUDA Tensor Cores via CUTLASS/cuBLAS
// Q = X @ W_q where X: [1024, 768], W_q: [768, 768]

cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    768, 1024, 768,  // M, N, K
    &alpha,
    W_q, CUDA_R_16F, 768,  // FP16 weight
    X,   CUDA_R_16F, 768,  // FP16 input
    &beta,
    Q,   CUDA_R_32F, 768,  // FP32 output
    CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Use Tensor Cores!
);

// Performance:
// FP32 CUDA cores: ~500 GFLOPS → 2 ms
// Tensor Cores:    ~150 TFLOPS → 6.5 μs
// Speedup: 300×!

// Effective bandwidth from HBM:
// W_q: 768 × 768 × 2 bytes = 1.18 MB (FP16)
// X:   1024 × 768 × 2 bytes = 1.57 MB (FP16)
// Total: 2.75 MB in 6.5 μs = 423 GB/s (21% of peak)
// But compute-bound, not memory-bound!
```

---

## 4. NPU 메모리 최적화

### 4.1 Systolic Array 아키텍처

**Google TPU v1 구조:**

```
Systolic Array: 256×256 multiply-accumulate units

┌─────┬─────┬─────┬─────┬─────┐
│ PE  │ PE  │ PE  │ PE  │ ... │  ← Processing Elements (PEs)
├─────┼─────┼─────┼─────┼─────┤
│ PE  │ PE  │ PE  │ PE  │ ... │
├─────┼─────┼─────┼─────┼─────┤
│ PE  │ PE  │ PE  │ PE  │ ... │
├─────┼─────┼─────┼─────┼─────┤
│ ... │ ... │ ... │ ... │ ... │
└─────┴─────┴─────┴─────┴─────┘

Data Flow:
- Weights: Flow from left to right (stationary in some designs)
- Activations: Flow from top to bottom
- Results: Accumulate locally, drain out

Each PE:
  output += weight × activation
  pass weight to right neighbor
  pass activation to down neighbor
```

**메모리 접근 패턴:**

```
Weight Stationary (Google TPU):
┌────────────────────────────────────────────┐
│ 1. Load weights into array (once)          │
│    Memory: 256 × 256 × 2 bytes = 128 KB    │
│    Time: 128 KB / 100 GB/s = 1.28 μs       │
│                                            │
│ 2. Stream activations (continuously)       │
│    For each activation row:                │
│      Load 256 × 2 bytes = 512 bytes        │
│      Time: 512 bytes / 100 GB/s = 5 ns     │
│                                            │
│ 3. Compute 256×256 matrix multiply         │
│    Ops: 256 × 256 × 256 = 16.7M MACs       │
│    Time: 16.7M / (256×256 @ 700 MHz)       │
│        = 16.7M / 46M = 0.36 ms             │
│                                            │
│ Arithmetic Intensity:                      │
│   = OPs / Memory Traffic                   │
│   = 16.7M MACs / 128 KB                    │
│   = 130 OPs/byte (highly compute-bound!)   │
└────────────────────────────────────────────┘
```

**Benefit:** Weights reused 256× (each weight used 256 times)

### 4.2 Dataflow 최적화

**3가지 주요 Dataflow:**

```
1. Weight Stationary (WS) - Google TPU
   ┌──────────────────────────────────────┐
   │ Weights: Stored in PEs (stationary)  │
   │ Inputs:  Streamed from SRAM          │
   │ Outputs: Accumulated in PEs          │
   └──────────────────────────────────────┘
   + Minimal weight memory traffic
   - Partial sums need storage

2. Output Stationary (OS) - Eyeriss
   ┌──────────────────────────────────────┐
   │ Weights: Streamed from SRAM          │
   │ Inputs:  Streamed from SRAM          │
   │ Outputs: Accumulated in PEs (stay)   │
   └──────────────────────────────────────┘
   + Minimal output memory traffic
   + Good for large output channels
   - Higher weight/input traffic

3. Row Stationary (RS) - Eyeriss v2
   ┌──────────────────────────────────────┐
   │ Hybrid: Maximize data reuse          │
   │ Adaptively choose based on layer     │
   └──────────────────────────────────────┘
```

**GPT-2 MLP에 적용:**

```
MLP: [1024, 768] × [768, 3072] → [1024, 3072]

Weight Stationary (best for NPU):
- Load W: [768, 3072] → 9.4 MB (one time)
- Stream X: [1024, 768] per row
- Compute: 1024 iterations
- Each weight element reused 1024×

Memory Traffic:
- Weights: 9.4 MB (once)
- Inputs:  3 MB × 1024 = 3 GB (streaming)
- Outputs: 12.6 MB
- Total: 3.02 GB

Arithmetic Intensity:
= 2 × 1024 × 768 × 3072 / 3.02 GB
= 4.8 GFLOPS / 3.02 GB
= 1.6 OPs/byte (memory-bound on this layer!)

Solution: Tiling to fit weights in SRAM
```

### 4.3 On-Chip SRAM Tiling

**Apple Neural Engine 예시:**

```
SRAM: 16 MB (much larger than GPU shared memory!)
Strategy: Tile to maximize SRAM reuse

GPT-2 Attention (QK^T): [1024, 768] × [768, 1024] → [1024, 1024]

Naive:
- Q: 3 MB
- K: 3 MB
- Output: 4 MB
- Total: 10 MB → Fits in SRAM!

Tiled Execution:
┌──────────────────────────────────────────────┐
│ 1. Load full Q into SRAM: 3 MB               │
│ 2. Load full K into SRAM: 3 MB               │
│ 3. Compute QK^T entirely in SRAM             │
│    → No external memory access during compute│
│ 4. Write result: 4 MB                        │
└──────────────────────────────────────────────┘

Memory Traffic (from DRAM):
- Read: 6 MB (Q + K)
- Write: 4 MB (output)
- Total: 10 MB

vs GPU (limited shared memory, must tile):
- Read: 6 MB (Q + K) × num_tiles
- Write: 4 MB (output)
- Total: 6 MB × 16 + 4 MB = 100 MB (10× more traffic!)

NPU Advantage: Large SRAM enables aggressive tiling
```

### 4.4 Processing-in-Memory (PIM)

**Samsung HBM-PIM, UPMEM:**

```
Traditional:                PIM:
┌───────────┐              ┌───────────┐
│    CPU    │              │  Memory   │
│           │              │  + Compute│
│  Compute  │              │           │
└─────┬─────┘              │  ┌─────┐  │
      │                    │  │ PE  │  │
      │ Bus                │  ├─────┤  │
      │                    │  │ PE  │  │
┌─────▼─────┐              │  ├─────┤  │
│  Memory   │              │  │ PE  │  │
│           │              │  └─────┘  │
│  Storage  │              │  Storage  │
└───────────┘              └───────────┘

Memory    Compute       Memory Compute
Read        ↓           Read
 ↓        Process        ↓      ↓
Send     ← Result      Process in-place
 ↓          ↓             ↓
Process   Send back    Result already
                       in memory!
```

**GPT-2 Embedding Lookup on PIM:**

```
Traditional GPU:
1. Token IDs in GPU memory
2. Embedding matrix in HBM
3. For each token:
   - Send token ID to GPU
   - GPU computes address
   - GPU requests embedding from HBM
   - HBM sends 3 KB back
   - GPU stores in local memory
Total bandwidth: 1024 tokens × 3 KB = 3 MB (read from HBM)

PIM (hypothetical):
1. Token IDs sent to PIM controller
2. PIM controller in HBM:
   - Computes addresses locally
   - Reads embeddings internally (high bandwidth)
   - Sends only results to GPU
Total bandwidth: 1024 tokens × 3 KB = 3 MB (write to GPU)

Benefit:
- Eliminates one direction of traffic
- Higher effective bandwidth (internal HBM bandwidth >> external)
- But: Limited compute in PIM (mainly for simple lookups)
```

---

## 5. NoC (Network-on-Chip) 아키텍처

### 5.1 NPU 내부 interconnect

**구조:**

```
Large NPU (e.g., Cerebras Wafer-Scale Engine):

┌─────────────────────────────────────────────────────────┐
│                    NoC Router Network                    │
│                                                          │
│  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐         │
│  │ PE + │────│ PE + │────│ PE + │────│ PE + │         │
│  │ SRAM │    │ SRAM │    │ SRAM │    │ SRAM │         │
│  └──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘         │
│     │           │           │           │              │
│  ┌──▼───┐    ┌──▼───┐    ┌──▼───┐    ┌──▼───┐         │
│  │ PE + │────│ PE + │────│ PE + │────│ PE + │         │
│  │ SRAM │    │ SRAM │    │ SRAM │    │ SRAM │         │
│  └──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘         │
│     │           │           │           │              │
│     ... (thousands of PEs) ...                         │
│                                                          │
│  Each PE:                                               │
│  - 512 KB SRAM                                          │
│  - MAC unit                                             │
│  - Router (5 ports: N, S, E, W, Local)                 │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Topology 비교

```
1. Mesh (2D Grid) - Most NPUs
   Pros: Simple, scalable, short wires
   Cons: Long paths for distant PEs

   Diameter: O(√N) hops
   Bisection bandwidth: O(√N)

   ┌───┬───┬───┬───┐
   │ PE│ PE│ PE│ PE│
   ├───┼───┼───┼───┤
   │ PE│ PE│ PE│ PE│
   ├───┼───┼───┼───┤
   │ PE│ PE│ PE│ PE│
   └───┴───┴───┴───┘

2. Ring - Apple Neural Engine (speculation)
   Pros: Simple routing, balanced traffic
   Cons: High diameter

   Diameter: O(N)
   Bandwidth: O(1)

   PE ─ PE ─ PE ─ PE
    └───────────────┘

3. Crossbar - Small NPUs
   Pros: Single hop, non-blocking
   Cons: O(N²) wires, doesn't scale

   Bandwidth: O(N²)
   Area: O(N²)

      PE  PE  PE  PE
       │  │  │  │
   PE─┼──┼──┼──┼─
   PE─┼──┼──┼──┼─
   PE─┼──┼──┼──┼─

4. Tree - Reduction operations
   Pros: Efficient for broadcast/reduce
   Cons: Root bottleneck

   Diameter: O(log N)

        PE
       ╱  ╲
     PE    PE
    ╱ ╲  ╱ ╲
   PE PE PE PE
```

### 5.3 GPT-2 Attention의 NoC 부하

**Softmax Reduction:**

```
Softmax(QK^T): Requires global reduction

Without NoC optimization:
┌────────────────────────────────────────────┐
│ 1. Compute exp(x) for each element        │
│    → Local computation (parallel)          │
│                                            │
│ 2. Sum all exp(x) across row              │
│    → Needs communication!                  │
│    → 1024 values → 1 PE                    │
│    → Tree reduction: log₂(1024) = 10 hops │
│                                            │
│ 3. Broadcast sum back to all PEs          │
│    → 10 hops again                         │
│                                            │
│ 4. Divide by sum                           │
│    → Local computation                     │
└────────────────────────────────────────────┘

NoC Traffic:
- Per row: 1024 × 4 bytes × 2 (gather + broadcast) = 8 KB
- For [1024, 1024] matrix: 1024 rows × 8 KB = 8 MB
- Hops: 10 (reduction tree depth)

Latency:
- Wire delay: ~1 ns per hop
- Router delay: ~2 ns per hop
- Total: (1 + 2) × 10 = 30 ns per row
- 1024 rows × 30 ns = 30 μs (NoC latency alone!)

Optimization: Hierarchical reduction in SRAM
```

### 5.4 Bandwidth Partitioning

**Multi-tenant NoC:**

```
Total NoC Bandwidth: 10 TB/s (Cerebras example)

Partitioning for GPT-2:
┌────────────────────────────────────────────┐
│ Layer 0-3:   Use Quadrant A (2.5 TB/s)    │
│ Layer 4-7:   Use Quadrant B (2.5 TB/s)    │
│ Layer 8-11:  Use Quadrant C (2.5 TB/s)    │
│ Control:     Use Quadrant D (2.5 TB/s)    │
└────────────────────────────────────────────┘

Benefits:
- Avoid congestion (each layer has dedicated BW)
- Pipeline multiple layers simultaneously
- Predictable performance

Drawback:
- Underutilization if layers are unbalanced
```

---

## 6. 특수 하드웨어 가속기

### 6.1 NVIDIA Tensor Cores (Hopper H100)

**4th Gen Tensor Core 특징:**

```
New Features:
1. FP8 support (E4M3, E5M2 formats)
2. Sparsity acceleration (2:4 structured sparsity)
3. Thread block cluster

FP8 GEMM:
D (FP32) = A (FP8) × B (FP8) + C (FP32)

Tile size: 16×16
Throughput: 3,958 TFLOPS (FP8)

GPT-2 QKV Projection in FP8:
Q = X @ W_q
X: [1024, 768] FP8
W: [768, 768] FP8

FLOPs: 2 × 1024 × 768 × 768 = 1.2 GFLOPS
Time: 1.2 G / 3,958 T = 0.3 μs (!!)

Memory bandwidth:
X: 1024 × 768 × 1 byte = 786 KB (FP8)
W: 768 × 768 × 1 byte = 590 KB (FP8)
Total: 1.4 MB

Time for memory: 1.4 MB / 3 TB/s = 0.47 μs
→ Compute-bound (good!)
```

**2:4 Structured Sparsity:**

```
Definition: Out of every 4 values, at least 2 are zero

Example Weight Matrix:
[1.2,  0,   0.5,  0  ]
[ 0,  3.4,  0,   1.1]
[2.1,  0,   0,   0.8]

Stored in compressed format:
Values:  [1.2, 0.5, 3.4, 1.1, 2.1, 0.8]
Indices: [0, 2, 1, 3, 0, 3]

Hardware:
- Decompressor unit
- 2× effective throughput (skip zeros)
- No accuracy loss if sparsity is learned during training

GPT-2 with 50% sparsity:
- FLOPs: 1.2 G → 0.6 G
- Speedup: 2× (theoretical)
- Actual: ~1.6-1.8× (decompression overhead)
```

### 6.2 Apple M4 Matrix Engine

**Architecture (inferred):**

```
Matrix Coprocessor:
- Dedicated matrix ALU
- Integrated with Unified Memory
- 128-bit wide SIMD

Capability:
- INT8 MatMul: ~7 TOPS (M4 Max)
- FP16 MatMul: ~2 TFLOPS

Memory Bandwidth:
- Unified memory: 400 GB/s (M4 Max)
- Matrix engine has priority access
- No CPU↔GPU transfer overhead

GPT-2 Embedding Lookup:
1. CPU tokenizes: "Hello world" → [15496, 995]
2. Matrix engine loads E[15496], E[995] from unified memory
   - Memory: 2 × 3 KB = 6 KB
   - Time: 6 KB / 400 GB/s = 15 ns
3. Matrix engine returns to CPU
   - Zero copy (unified memory)

Total latency: ~50 ns (10× faster than discrete GPU!)
```

### 6.3 Intel AMX (Advanced Matrix Extensions)

**Sapphire Rapids Xeon:**

```
Tile Registers: 8 tiles, each 1 KB (16 rows × 64 bytes)

Tile Configuration:
Tile A: [16, 64] INT8
Tile B: [16, 64] INT8
Tile C: [16, 16] INT32 (accumulator)

Instruction:
TDPBSSD (Tile Dot Product Signed/Signed)
C += A × B^T (in one instruction!)

Throughput:
- 2 AMX units per core
- 16×64×16 MACs per instruction = 16,384 INT8 MACs
- Frequency: 2.0 GHz (base)
- Peak: 2 × 16,384 × 2.0 GHz = 65.5 GOPS per core

GPT-2 MLP (INT8):
X: [1024, 768] INT8
W: [768, 3072] INT8

Tiling:
- Tile size: 16×64
- Tiles in X: (1024/16) × (768/64) = 64 × 12 = 768 tiles
- Tiles in W: (768/64) × (3072/16) = 12 × 192 = 2,304 tiles

Total tile ops: 768 × 2,304 = 1.77M tile operations
Time: 1.77M / (2 × 2 GHz) = 442 μs

vs CPU VNNI (AVX-512):
Time: ~2-3 ms
Speedup: 5-7×
```

### 6.4 Google TPU v4

**Architecture:**

```
Chip:
- Systolic array: 128×128
- Clock: 1.15 GHz
- INT8 throughput: 275 TOPS
- BF16 throughput: 137 TFLOPS

Memory:
- HBM2: 32 GB per chip
- Bandwidth: 1.2 TB/s
- On-chip SRAM: 144 MB (huge!)

Interconnect:
- 3D torus topology
- ICI bandwidth: 4.8 TB/s per chip (!!!)

GPT-2 Full Model on TPU Pod:
- 12 layers distributed across chips
- Each chip handles 1 layer
- Pipeline parallelism

Layer 0 (Chip 0):
- Receive input: [1024, 768]
- Compute Attention + MLP
- Send to Chip 1: [1024, 768]
- Overlap: While processing batch N, receive batch N+1

Bandwidth:
- Inter-chip: 4.8 TB/s (ICI)
- Intra-chip: 1.2 TB/s (HBM)
- Transfer [1024, 768] FP16: 1.5 MB
- Time: 1.5 MB / 4.8 TB/s = 0.3 μs (negligible!)

Bottleneck: Compute (not communication!)
```

---

## 7. GPT-2 레이어별 최적화

### 7.1 Token Embedding: Gather Optimization

**문제: Random access, poor coalescing**

```
Input: token_ids = [15496, 995, 262, ...]
Embedding: E[50257, 768]

Naive CUDA:
__global__ void embedding_lookup(float* out, int* ids, float* emb, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int id = ids[i];
        // Each thread reads different row (uncoalesced!)
        for (int j = 0; j < 768; j++) {
            out[i * 768 + j] = emb[id * 768 + j];
        }
    }
}
Bandwidth: ~20-50 GB/s (1-2.5%)
```

**최적화 1: Batch Gather**

```cuda
__global__ void embedding_lookup_optimized(float* out, int* ids, float* emb, int n) {
    int token_idx = blockIdx.x;  // One block per token
    int dim_idx = threadIdx.x;   // Threads handle dimensions

    if (token_idx < n && dim_idx < 768) {
        int id = ids[token_idx];
        // Coalesced: All threads in block access consecutive elements
        out[token_idx * 768 + dim_idx] = emb[id * 768 + dim_idx];
    }
}
// Launch: <<<n, 768>>>
Bandwidth: ~400-600 GB/s (20-30%)
```

**최적화 2: Prefetch + Cache**

```cuda
__global__ void embedding_lookup_cached(
    float* out, int* ids, float* emb, int n,
    int* cache_ids, float* cache_emb, int cache_size
) {
    __shared__ int local_ids[BATCH_SIZE];
    __shared__ float local_emb[BATCH_SIZE][768];

    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    // Load token IDs into shared memory
    if (dim_idx < BATCH_SIZE && token_idx + dim_idx < n) {
        local_ids[dim_idx] = ids[token_idx + dim_idx];
    }
    __syncthreads();

    // Check cache (top-K frequent tokens)
    for (int i = 0; i < BATCH_SIZE; i++) {
        int id = local_ids[i];
        bool in_cache = false;

        // Binary search in cache
        // ... (omitted for brevity)

        if (in_cache) {
            // Load from cache (faster)
            if (dim_idx < 768) {
                local_emb[i][dim_idx] = cache_emb[cache_pos * 768 + dim_idx];
            }
        } else {
            // Load from global embedding
            if (dim_idx < 768) {
                local_emb[i][dim_idx] = emb[id * 768 + dim_idx];
            }
        }
    }
    __syncthreads();

    // Write output
    if (dim_idx < 768) {
        out[token_idx * 768 + dim_idx] = local_emb[0][dim_idx];
    }
}

// Cache top 1000 tokens (covers ~80% of usage)
// Hit rate: 80%
// Effective bandwidth: 600 GB/s (cached) + 50 GB/s (miss) = ~500 GB/s average
```

### 7.2 Attention (QK^T): Flash Attention

**문제: O(L²) memory for attention matrix**

```
Standard Attention:
Input: Q, K, V [1024, 768]

Steps:
1. S = Q @ K^T → [1024, 1024] (4 MB!)
2. P = softmax(S) → [1024, 1024] (4 MB)
3. O = P @ V → [1024, 768] (3 MB)

Total memory: 4 + 4 + 3 = 11 MB
Bandwidth: Read 6 MB (Q,K,V) + Write 11 MB = 17 MB
```

**Flash Attention: Tiled + Recomputation**

```
Algorithm:
1. Partition Q, K, V into blocks (tiles)
   - Q tiles: [block_size, 768]
   - K tiles: [block_size, 768]

2. For each Q tile:
   a. Load Q_tile into SRAM
   b. For each K tile:
      i.   Load K_tile into SRAM
      ii.  Compute S_tile = Q_tile @ K_tile^T (in SRAM)
      iii. Compute softmax_tile (online algorithm)
      iv.  Load V_tile
      v.   Accumulate O_tile += softmax_tile @ V_tile
      vi.  Discard S_tile, softmax_tile (recompute if needed)
   c. Write O_tile to HBM

Key Innovation:
- Never materialize full [1024, 1024] attention matrix
- Recompute instead of storing (trade compute for memory)
- Fused kernel (no intermediate writes)

Memory:
- Tiles in SRAM: block_size × 768 × 4 bytes
- Example: block_size = 64 → 192 KB per tile (fits in shared mem!)
- Total HBM traffic: Read Q,K,V (6 MB) + Write O (3 MB) = 9 MB

Speedup:
- Memory: 17 MB → 9 MB (1.9× less)
- Latency: 2-4× faster (no large intermediate write)
- Enables longer sequences (up to 8K+ tokens)
```

**Implementation:**

```cuda
__global__ void flash_attention(
    float* Q, float* K, float* V, float* O,
    int N, int d, int block_size
) {
    extern __shared__ float smem[];
    float* Q_tile = smem;
    float* K_tile = smem + block_size * d;
    float* S_tile = K_tile + block_size * d;

    int q_block = blockIdx.x;
    int q_start = q_block * block_size;

    // Load Q tile into shared memory
    for (int i = threadIdx.x; i < block_size * d; i += blockDim.x) {
        if (q_start + i / d < N) {
            Q_tile[i] = Q[(q_start + i / d) * d + (i % d)];
        }
    }
    __syncthreads();

    // Initialize output accumulator
    float O_local[d] = {0};
    float max_val = -INFINITY;
    float sum_exp = 0;

    // Iterate over K tiles
    for (int k_block = 0; k_block < (N + block_size - 1) / block_size; k_block++) {
        int k_start = k_block * block_size;

        // Load K tile
        for (int i = threadIdx.x; i < block_size * d; i += blockDim.x) {
            if (k_start + i / d < N) {
                K_tile[i] = K[(k_start + i / d) * d + (i % d)];
            }
        }
        __syncthreads();

        // Compute S_tile = Q_tile @ K_tile^T (block-wise GEMM)
        // ... (matrix multiply in shared memory)

        // Online softmax (numerically stable)
        // ... (update max_val, sum_exp, O_local)

        __syncthreads();
    }

    // Final normalization and write output
    // ...
}

// Launch with large shared memory:
// <<<num_blocks, num_threads, shared_mem_size>>>
// shared_mem_size = (2 * block_size * d + block_size^2) * sizeof(float)
```

### 7.3 MLP: Quantization + Tiling

**INT8 Quantization:**

```
Original: X [1024, 768] FP32, W [768, 3072] FP32

Quantized:
X_int8 = round(X / scale_X)
W_int8 = round(W / scale_W)

Compute:
Y_int32 = X_int8 @ W_int8  (INT8 GEMM)
Y_fp32 = Y_int32 * scale_X * scale_W

Benefits:
- Memory: 4× reduction (32-bit → 8-bit)
- Bandwidth: 4× reduction
- Compute: 4× faster (INT8 ops)

NVIDIA Tensor Cores INT8:
- Throughput: 624 TOPS (A100)
- vs FP32: 19.5 TFLOPS
- Speedup: 32×!

GPT-2 MLP INT8:
FLOPs: 2 × 1024 × 768 × 3072 = 4.8 GFLOPS
Time (FP32): 4.8 G / 19.5 T = 246 μs
Time (INT8): 4.8 G / 624 T = 7.7 μs
Speedup: 32×

Memory bandwidth:
FP32: (3 + 9.4 + 12.6) MB = 25 MB
INT8: (0.75 + 2.35 + 3.15) MB = 6.25 MB
Reduction: 4×
```

**Tiling for SRAM:**

```
Tile configuration:
- Tile size: 128×128 (fits in shared memory/SRAM)
- X tiled: (1024/128) × (768/128) = 8 × 6 tiles
- W tiled: (768/128) × (3072/128) = 6 × 24 tiles

NPU execution:
for i in range(8):      # X row tiles
    for j in range(24):  # W column tiles
        accumulator = 0
        for k in range(6):  # Inner dimension tiles
            X_tile = load_X_tile(i, k)    # 128×128 from DRAM to SRAM
            W_tile = load_W_tile(k, j)    # 128×128 from DRAM to SRAM
            accumulator += X_tile @ W_tile  # Compute in SRAM
        store_Y_tile(i, j, accumulator)  # 128×128 from SRAM to DRAM

Memory traffic:
- X reads: 8 × 6 × 128 × 128 × 1 byte = 768 KB (each tile read 24 times)
- W reads: 6 × 24 × 128 × 128 × 1 byte = 2.3 MB (each tile read 8 times)
- Y writes: 8 × 24 × 128 × 128 × 4 bytes = 12.6 MB (INT32 output)
- Total: 768 KB + 2.3 MB + 12.6 MB = 15.7 MB

vs No tiling:
- X: 1024 × 768 × 1 byte = 768 KB × 6 (read 6 times) = 4.6 MB
- W: 768 × 3072 × 1 byte = 2.3 MB × 8 (read 8 times) = 18.4 MB
- Y: 12.6 MB
- Total: 35.6 MB

Tiling saves: 35.6 - 15.7 = 19.9 MB (56% reduction!)
```

---

## 8. 실전 측정 예시

### 8.1 GPT-2 Embedding Lookup Profiling

```python
import torch
import torch.utils.benchmark as benchmark

# Setup
embedding = torch.randn(50257, 768, device='cuda')
token_ids = torch.randint(0, 50257, (1024,), device='cuda')

# Benchmark
def embedding_lookup_naive():
    return embedding[token_ids]

timer = benchmark.Timer(
    stmt='embedding_lookup_naive()',
    globals={'embedding_lookup_naive': embedding_lookup_naive}
)

result = timer.blocked_autorange(min_run_time=1.0)
print(f"Time: {result.median * 1e6:.2f} μs")
print(f"Bandwidth: {(1024 * 768 * 4 / 1e9) / result.median:.2f} GB/s")

# Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    _ = embedding[token_ids]

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Output:**
```
Time: 3.2 μs
Bandwidth: 931 GB/s (46% of A100 peak)

CUDA Kernel Analysis:
- Memory read: 1024 × 768 × 4 = 3 MB
- Cache line utilization: ~40% (random access pattern)
- L2 cache hit rate: ~15%
- DRAM transactions: 48,000 (should be 24,000 if coalesced)
```

### 8.2 Matrix Multiply Roofline Analysis

```python
# GPT-2 QKV projection
def qkv_projection(X, W_q, W_k, W_v):
    Q = X @ W_q  # [1024, 768] @ [768, 768]
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

X = torch.randn(1024, 768, device='cuda', dtype=torch.float16)
W_q = torch.randn(768, 768, device='cuda', dtype=torch.float16)
W_k = torch.randn(768, 768, device='cuda', dtype=torch.float16)
W_v = torch.randn(768, 768, device='cuda', dtype=torch.float16)

# Benchmark
import nvtx

with nvtx.annotate("QKV_Projection"):
    Q, K, V = qkv_projection(X, W_q, W_k, W_v)

# Use Nsight Compute for detailed metrics:
# $ ncu --set full python script.py

# Example output:
# Achieved Occupancy: 85.2%
# Memory Throughput: 1,450 GB/s (71% of peak)
# Compute Throughput: 145 TFLOPS (47% of tensor core peak)
# → Memory-bound (compute waiting on memory)
```

---

## 9. 요약 및 Best Practices

### 9.1 메모리 효율성 체크리스트

| Layer | Issue | Solution | Speedup |
|-------|-------|----------|---------|
| Token Embedding | Random access | Batch + cache top-K | 10-20× |
| Position Embedding | ✓ Sequential | None needed | - |
| Attention (QK^T) | O(L²) memory | Flash Attention | 2-4× |
| Softmax | Reduction overhead | Fused kernel | 2-3× |
| MLP | Large weights | Quantization + tiling | 4-8× |

### 9.2 하드웨어별 권장사항

**GPU (NVIDIA A100):**
- Use Tensor Cores (FP16/BF16/INT8)
- Tile to shared memory (164 KB)
- Fuse operations (avoid intermediate writes)
- Profile with Nsight Compute

**NPU (Apple Neural Engine):**
- Maximize SRAM usage (16 MB)
- Use INT8 quantization
- Align data to DMA burst size
- Avoid dynamic shapes

**CPU (Intel Xeon):**
- Use AMX tiles for INT8
- Vectorize with AVX-512
- Prefetch data (software prefetch)
- Use OpenMP for parallelism

### 9.3 메모리 대역폭 최적화 우선순위

1. **Avoid unnecessary memory traffic**
   - Operator fusion
   - In-place operations
   - Recomputation vs storage

2. **Improve cache utilization**
   - Tiling to fit in SRAM/L2
   - Reorder operations for locality
   - Prefetching

3. **Maximize bus utilization**
   - Memory coalescing (GPU)
   - Burst transfers (NPU)
   - Alignment (64-byte boundaries)

4. **Reduce data size**
   - Quantization (INT8/FP16)
   - Sparsity
   - Compression

### 9.4 실제 달성 가능한 대역폭

```
Hardware: NVIDIA A100 (2,039 GB/s peak)

Operation          | Achieved | % of Peak
-------------------|----------|----------
Sequential copy    | 1,800    | 88%
Strided copy (×2)  | 900      | 44%
Strided copy (×4)  | 450      | 22%
Random lookup      | 100-200  | 5-10%
Matrix multiply    | 1,400    | 69%
Fused attention    | 1,200    | 59%

Lesson: Access pattern matters more than raw bandwidth!
```

---

## References

- **CUDA Programming Guide** (NVIDIA)
- **Roofline Model** (Williams et al., 2009)
- **FlashAttention** (Dao et al., 2022)
- **In-Datacenter Performance Analysis of a Tensor Processing Unit** (TPU, Jouppi et al., 2017)
- **Eyeriss: An Energy-Efficient Reconfigurable Accelerator** (Chen et al., 2016)
- **Cerebras: A Wafer-Scale Deep Learning Accelerator** (2021)
- **Apple Neural Engine** (Reverse engineering, 2023)
