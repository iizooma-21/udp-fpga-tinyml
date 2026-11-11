# ResNet-8 End-to-End Evaluation Summary

## Accuracy (CIFAR-10 top-1)

| Scenario | Task | Top-1 | MLPerf (>=85%) |
|---|---|---:|---|
| FP32 baseline | Task 2 | 90.49% | PASS |
| Naive INT4 PTQ (per-tensor) | Task 3 | 49.28% | FAIL |
| UDP no post-proc (weight-only INT4) | Task 4 | 78.59% | FAIL |
| UDP with post-proc (full INT4) | Task 5 | 74.13% | FAIL |

Post-proc activation cost (Task 5 - Task 4): **-4.46%**

## Timeloop performance (3 layers)

UDP cycles = Timeloop cycles / 6 (6 int4 MACs per PE per cycle)

| Architecture | PEs | Layer | Utilization | Timeloop cycles | UDP cycles |
|---|---:|---|---:|---:|---:|
| Baseline 12x12 | 144 | stem_conv | 50% | 9984 | 1664 |
| Baseline 12x12 | 144 | s1_conv1 | 67% | 64896 | 10816 |
| Baseline 12x12 | 144 | classifier_dense | 56% | 13 | 2 |
| Opt-A 8x12 | 96 | stem_conv | 75% | 9984 | 1664 |
| Opt-A 8x12 | 96 | s1_conv1 | 100% | 64896 | 10816 |
| Opt-A 8x12 | 96 | classifier_dense | 42% | 26 | 4 |
| Opt-B 8x9 | 72 | stem_conv | 100% | 9984 | 1664 |
| Opt-B 8x9 | 72 | s1_conv1 | 67% | 129792 | 21632 |
| Opt-B 8x9 | 72 | classifier_dense | 56% | 26 | 4 |

### Total UDP cycles (3 layers)

| Architecture | Total UDP cycles | Latency @200 MHz |
|---|---:|---:|
| Baseline 12x12 | 12,482 | 0.0624 ms |
| Opt-A 8x12 | 12,484 | 0.0624 ms |
| Opt-B 8x9 | 23,300 | 0.1165 ms |

**Recommendation:** Opt-A (8x12, 96 PEs) — best utilization/area trade-off.

