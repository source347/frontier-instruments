# Frontier Instruments

**Interactive scientific instruments for exploring cryptography, AI, and computational science.**

Frontier Instruments is an open-source, browser-native laboratory by Source347. The project emphasizes inspectable models, reproducible exploration, and explicit limitations.

## Instruments

| Instrument | Status | Description |
|---|---|---|
| Post-Quantum Cryptography Lab | Available | Toy LWE encryption trace, lattice visualization, and NIST standards references |
| LLM Architecture Explorer | Available | Interactive attention heatmap, positional encoding, KV-cache estimator, and MoE routing |
| Agent Memory Simulator | Available | Interactive memory encoding, lexical retrieval, recency decay, consolidation, and context budgeting |
| WebGPU Compute Laboratory | Available | WGSL vector addition and parallel reduction, configurable dispatch geometry, CPU comparison, and result validation |

## Run locally

No build step or dependencies are required. Open `index.html` in a modern browser, or serve the directory:

```bash
python -m http.server 8000
```

Then visit http://localhost:8000.

## Platform notes

The WebGPU instrument requires a supported browser and GPU in a secure context (HTTPS or localhost). A JavaScript CPU baseline remains available when WebGPU is unavailable. Performance measurements are illustrative, include browser/runtime overhead, and are not controlled benchmarks.

## Security notice

The cryptography instrument is an educational toy model. It is not ML-KEM, not a secure cryptographic implementation, and must never be used to protect real data. The memory simulator uses deterministic illustrative heuristics, not a production agent memory system. The WebGPU lab runs synthetic workloads locally and does not transmit workload data.

## License

MIT. See [LICENSE](LICENSE).
