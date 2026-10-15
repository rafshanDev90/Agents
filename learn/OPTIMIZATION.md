# Optimization Research — agent-harness-build

Date: 2026-10-14 · Machine: HP EliteBook 840 G6, i5-8365U (4C/8T, AVX2, no AVX-512),
7.1 GB RAM, no GPU · Serving: Ollama 0.35.1, qwen3.5:4b (Q4_K_M, 4.2B + 333M CLIP
projector, thinking default ON) and qwen3.5:2b (Q8_0, 1.9B + 331M CLIP, thinking ON).

All numbers below were measured on this machine (diagnose.py, lowlevel_bench.py,
timing.py traces), not copied from the internet.

---

## 1. Baseline measurements

| Scenario | Result |
|---|---|
| Model cold load (after idle >5 min) | **9–19 s** (up to 42 s during churn) |
| Prefill, small warm prompt | ~40–45 tok/s |
| Prefill, 1505-token agent prompt | **~20 tok/s → ~75–85 s per round** |
| Prefill, identical replay (cache hit) | 0.7 s (1249 tok/s) — **exact match only** |
| Decode, qwen3.5:4b | **2.7–4.6 tok/s** (contention-dependent) |
| Decode, qwen3.5:2b | **6.8 tok/s** (2.4× faster) |
| Full tool turn, fresh session | 58.1 s |
| Full tool turn, resumed 25-message session | **153.5 s** (round 1 = 116 s) |
| SQLite save | 0.008–0.021 s (negligible ✓) |
| DuckDuckGo search | ~2.8 s per call (serial) |

Turn anatomy (big history): **~50% prefill** (history re-processing every round),
**~45% decode** (hardware floor), tools/save ~0.

Root cause of the prefill share: Ollama's prompt cache is **exact-match only**
(measured: identical request 0.7 s; same prefix + new user tail 39 s = full
recompute). An append-only agent loop NEVER repeats an identical prompt → cache
never hits in real usage.

---

## 2. System-level findings (this machine)

1. **CPU contention**: VS Code (snap) GPU + renderer processes spin 30–100% CPU
   continuously; idle load average ~2.0 on 4 physical cores. Inference competes
   the whole time. Evidence: same benchmark decode varied 2.7 ↔ 4.6 tok/s.
2. **Swap pressure**: 2.8 GB of 4 GB swap used on 7 GB RAM. Best explanation for
   the pathological 73 s first prefill observed once (mmap'd model pages evicted
   → swap reads mid-prefill).
3. **Governor `powersave`, EPP `balance_power`** — both have `performance`
   available (`scaling_available_governors`: performance powersave).
4. **Thread oversubscription is catastrophic on this CPU** (see §3.1).

---

## 3. Experiments and results

### 3.1 Thread sweep — qwen3.5:4b, num_predict=48

| Config | Decode | Note |
|---|---|---|
| **default (Ollama picks ≈ physical cores)** | **2.8 tok/s** | best; prefill 44 tok/s |
| num_thread=4 | 2.7 tok/s | ≈ default; forced a model reload (14.7 s) |
| num_thread=6 | 1.3 tok/s | 2× worse |
| num_thread=8 (all HT threads) | **0.1 tok/s** | 15× worse — 381 s for 48 tokens! |

**Conclusion: never set num_thread=nproc here.** Hyperthreads + background load
destroy decode. Ollama's default is already optimal. Bonus lesson: an explicit
per-request num_thread forces a runner respawn + model reload (9–42 s).

### 3.2 Thinking tax — qwen3.5:4b, same prompt

| Mode | Decode rate |
|---|---|
| think: true (default) | 4.6 tok/s |
| think: false | 4.6 tok/s |

Rate is identical — the tax is **volume**, not speed: real agent turns spent
86–98 output tokens per round mostly on reasoning (from /timing traces). At
~0.25–0.35 s/token, `think: false` is worth roughly 20–40 s per typical turn,
quality tradeoff permitting (per-request: `extra_body={"think": False}`).

### 3.3 Model comparison

| Model | Decode | Load | Notes |
|---|---|---|---|
| qwen3.5:4b Q4_K_M (4.2B) | 2.8 tok/s | 9–19 s | current default |
| qwen3.5:2b Q8_0 (1.9B) | **6.8 tok/s** | ~9 s | 2.4× faster; Q8_0 is oversized for 1.9B — a Q4_K_M build (~1.2 GB) would load faster and likely decode slightly faster still (bandwidth-bound), at some quality cost |

---

## 4. Research findings (external, verified sources)

1. **llama.cpp server does true prefix KV reuse** — per-slot caching,
   `--cache-reuse 256`, `--cache-ram`, `--slot-save-path`. Sources explicitly
   state "for agent loops the win is significant": an append-only loop reuses
   the whole shared prefix → your measured 75–85 s prefill collapses to ~the new
   tokens only. Ollama does not offer this (exact-match only).
   Migration cost: ~1 line (OpenAI-compatible `base_url` swap in `main.py`).
2. **Generic binaries leave performance on the table** — llama.cpp built with
   `-march=native` plus newer CPU kernels (and the ik_llama.cpp fork) report
   ~25–100% prompt-processing gains on AVX2 machines.
3. **Speculative decoding** (llama.cpp `-md <draft.gguf>`, e.g. the 2b drafting
   for the 4b): plausible ~1.3–1.8× decode on CPU when draft is much smaller and
   acceptance is high. Not supported by Ollama.
4. **KV-cache quantization** (`OLLAMA_KV_CACHE_TYPE=q8_0` with flash attention):
   halves KV memory — on 7 GB RAM this buys headroom (less swap pressure) more
   than raw speed.
5. **Reality check** (CPU-only 2026 tuning guides): optimized CPU inference is
   fine for deliberate chat + agent turns; it will never feel GPU-snappy. Goal:
   minimize and *stream* the wait, don't hide it.

---

## 5. The optimization stack (priority order, expected impact)

| # | Action | Attacks | Cost | Expected |
|---|---|---|---|---|
| 1 | Close/fix spinning VS Code procs; keep RAM free of swap | contention + 73 s pathology | free | decode toward 4+ tok/s |
| 2 | `sudo cpupower frequency-set -g performance` (+ EPP performance) | sustained clocks | 1 command | +10–30% |
| 3 | `OLLAMA_KEEP_ALIVE=-1` (systemd drop-in) | 9–19 s reloads | config | cold starts gone |
| 4 | Streaming (`stream=True` + Rich Live) | perceived wait | code | first token after prefill, not after whole answer |
| 5 | Sliding window + tool-output truncation | prefill 50% | code (L6) | 1505 → ~400 tok prompts |
| 6 | Brevity in system prompt (+ `num_predict` cap) | decode 45% | 1 line | ~90 tok answers → ~30 |
| 7 | Parallel tool execution | serial 2.8 s searches | code | max() instead of sum() |
| 8 | Switch to llama-server (`-c 4096 -np 1 --cache-reuse 256 -t 4 --jinja`) | append-loop prefill | 1-line base_url + install | prefill ≈ new tokens only |
| 9 | qwen3.5:2b (or Q4 build of it) | decode floor | model choice | 2.4× decode |
| 10 | `-march=native` llama.cpp build / ik_llama.cpp | kernels | build | +25%+ prefill |
| 11 | Speculative draft (2b → 4b) on llama.cpp | decode | config | +30–80% decode |
| 12 | `OLLAMA_KV_CACHE_TYPE=q8_0` | RAM headroom | config | less swap pressure |

Realistic combined ceiling for the resumed-session turn (153.5 s today):
**~30–50 s with visible streaming from ~2–4 s** (items 1–8), or **~15–25 s**
if you also accept the 2b (item 9).

### Reproducing the measurements
- `uv run python learn/diagnose.py` — load/prefill/cache breakdown of real history
- `uv run python learn/timing.py` traces appear after every turn; `/timing`, `/tokens` in the CLI
- `/tmp/opencode/lowlevel_bench.py` — thread sweep / thinking tax / model compare
