# Baseline Update: Using Best Submission (1.1194 BPB)

## What Was Done

As requested, I have:
1. ✅ Canceled the previous PR approach (removed old improvements)
2. ✅ Studied the best submission: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
3. ✅ Replaced `train_gpt.py` with the best performing submission code

## Best Submission Overview

**Score: 1.1194 BPB** (3-seed mean, std 0.0006)
**Author: abaybektursun**
**Date: 2026-03-23**

### Key Innovations

#### 1. LeakyReLU(0.5)² Activation (-0.003 BPB)
```python
# Instead of: x = torch.relu(self.fc(x)).square()
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```
- Preserves negative gradient flow through MLP
- Eliminates dead neurons while maintaining relu² inductive bias
- One-line change with substantial impact

#### 2. Legal Score-First TTT (-0.0025 BPB)
- Backward-looking test-time training on validation set
- 1,893 non-overlapping 32K-token chunks
- **SCORE first** under `torch.inference_mode()` (no gradients, no mutation)
- **TRAIN second** with SGD(lr=0.002, momentum=0.9), 3 epochs
- Hard guarantee: no weight mutation during scoring
- Takes ~410s of the 10-minute eval budget

#### 3. Parallel Muon Optimizer
- Batched Newton-Schulz orthogonalization via `torch.bmm`
- 4 contiguous 3D parameter banks replace 66 separate Linear weights
- Async reduce-scatter → local NS → async all-gather
- No DDP for banks, custom gradient handling
- **83.3ms/step** vs ~85ms baseline

### Architecture Details

| Component | Configuration |
|-----------|--------------|
| **Layers** | 11 (512d, 8 heads, 4 KV heads) |
| **MLP** | 3× expansion with LeakyReLU(0.5)² |
| **Embeddings** | BigramHash (1536 vocab, 128d projection) |
| **Attention** | XSA on last 4 layers, Flash Attention 3 |
| **Positional** | Partial RoPE (16 of 64 dims) |
| **Normalization** | LN Scale: 1/√(layer+1) |
| **Value Embed** | VE128 on layers 9-10 |
| **Weight Avg** | EMA(decay=0.997) + SWA(every 50 steps) |
| **Quantization** | GPTQ-lite int6 + lzma compression |
| **Optimizer** | Parallel Muon (matrices) + Adam (scalars) |

### Training Hyperparameters

```python
NUM_LAYERS=11
BIGRAM_VOCAB_SIZE=1536
XSA_LAST_N=4
MLP_MULT=3.0
TRAIN_SEQ_LEN=2048
TRAIN_BATCH_TOKENS=786_432
MATRIX_LR=0.025
SCALAR_LR=0.025
TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99
MUON_WD=0.04
WARMDOWN_ITERS=3500
EVAL_STRIDE=64
```

### Results

| Seed | Steps | Pre-TTT BPB | Post-TTT BPB | TTT Gain | Artifact Size |
|------|-------|-------------|--------------|----------|---------------|
| 1337 | 7,179 | 1.1217 | **1.1192** | -0.0025 | 15,977,386 B |
| 42   | 7,182 | 1.1227 | **1.1200** | -0.0027 | 15,876,510 B |
| 2025 | 7,193 | 1.1212 | **1.1189** | -0.0023 | 15,990,006 B |
| **Mean** | **7,185** | **1.1218** | **1.1194** | **-0.0025** | **~15.95 MB** |

### Timing Budget

- Training: ≤600s (10 minutes)
- Standard eval (int6 roundtrip + sliding window): ~120s
- Legal TTT (score-first + adaptation): ~410s
- **Total eval**: ~530s (under 10 minutes)

## Technical Stack

The submission builds on multiple PRs:

1. **PR #493, #518**: LeakyReLU² activation
2. **PR #399**: Parameter Banking + Parallel Muon optimizer
3. **PR #461**: Legal TTT protocol (adapted: freeze=0 vs original freeze=2)
4. **PR #414**: Base architecture (11L, BigramHash, XSA, Partial RoPE, etc.)

## File Information

- **Source**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Lines**: 1,898 (exceeds original 1,500 line guideline, but acceptable for records)
- **Size**: ~90 KB (code only, before compression)
- **Syntax**: ✅ Verified with Python 3 compilation

## Next Steps

This file now serves as the baseline for any future improvements. Some potential areas to explore:

1. **Further TTT optimization**: Different chunk sizes, learning rates, or freeze strategies
2. **Architecture refinements**: Layer count, MLP ratio, attention patterns
3. **Quantization improvements**: Different bit widths, calibration methods
4. **Optimizer tuning**: Learning rates, momentum schedules, weight decay
5. **Training efficiency**: Reduce step time to fit more iterations in 10 minutes

The strong baseline (1.1194 BPB) means any improvements need to demonstrate statistical significance (p < 0.01) with at least 0.005 nats gain per the challenge rules.

## Credits

- **Main author**: @abaybektursun
- **LeakyReLU²**: @parinzee (PR #493), @sofiabod (PR #518)
- **Parallel Muon**: @abaybektursun (PR #399)
- **TTT protocol**: @Christopher-Lee-McClendon (PR #461)
- **Base architecture**: @signalrush (PR #414)
