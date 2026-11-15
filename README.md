# Frontier Instruments

**Interactive scientific instruments for exploring cryptography, AI, and computational science.**

Frontier Instruments is an open-source, browser-native laboratory by Source347. The project emphasizes inspectable models, reproducible exploration, and explicit limitations.

## Instruments

| Instrument | Status | Description |
|---|---|---|
| Post-Quantum Cryptography Lab | Available | Toy LWE encryption trace, lattice visualization, and NIST standards references |
| LLM Architecture Explorer | Planned | Attention, positional encoding, KV caches, and mixture-of-experts |
| Agent Memory Simulator | Planned | Retrieval, consolidation, decay, and context selection |
| WebGPU Compute Laboratory | Planned | Parallel workloads and browser-based compute |

## Run locally

No build step or dependencies are required. Open `index.html` in a modern browser, or serve the directory:

```bash
python -m http.server 8000
```

Then visit http://localhost:8000.

## Security notice

The cryptography instrument is an educational toy model. It is not ML-KEM, not a secure cryptographic implementation, and must never be used to protect real data.

## License

MIT. See [LICENSE](LICENSE).
