---
summary: "Research notes: agentic video generation landscape (models, frameworks, open problems)"
read_when:
  - Evaluating video generation capabilities for OpenClaw media pipeline
  - Deciding between open-source vs proprietary video models
  - Designing multi-step agentic video workflows
title: "Agentic Video Generation Research"
updated: "2026-03-16"
---

# Agentic Video Generation: Research Notes

## Overview

The field has shifted from "text → single clip" to **multi-agent pipelines** that plan, generate, critique, and iterate — essentially an automated film production crew. Key development: GQA-style efficiency gains in language models are now mirrored in video by native audio integration and multi-shot storyboarding.

---

## Top Models (as of March 2026)

### Proprietary

| Model | Provider | Highlights | Pricing |
|---|---|---|---|
| **Veo 3.1** | Google | Native audio + lip-sync, 60s @ 1080p, best naturalistic humans | ~$0.20/sec |
| **Sora 2** | OpenAI | Best physics accuracy, complex prompt adherence, multi-subject | — |
| **Kling 3.0** | Kuaishou | Multi-shot storyboards (6 cuts), 4K/60fps, native audio | — |
| **Runway Gen-4** | Runway | Persistent characters across scenes, 3D camera control, VFX | ~$0.07/sec |

### Open-Source

| Model | Params | VRAM | Highlights |
|---|---|---|---|
| **Wan 2.2** | 14B (1.3B lite) | 8GB min | Best open model; competitive with closed systems; ~$0.05/sec hosted |
| **HunyuanVideo** | 13B+ | A100/H100 | 720p/24fps, outperformed Runway Gen-3 at launch |
| **CogVideoX-5B** | 5B | Modest | 6s clips, 720x480, good for research/fine-tuning |

---

## Agentic Frameworks

### VISTA (Google, Oct 2025) — `arXiv:2510.15831`
Multi-agent loop: decompose idea → generate → tournament selection → trio critique (visual/audio/contextual) → reasoning agent rewrites prompt → iterate.
- 60% pairwise win vs baselines; humans preferred VISTA 66.4% of the time.
- Key insight: **test-time self-improvement** via structured feedback loops.

### ViMax (HKUDS, 2025) — [github.com/HKUDS/ViMax](https://github.com/HKUDS/ViMax)
Four-agent pipeline: Script agent (RAG-powered) → Shot planner (cinematography-aware) → Asset manager (cross-shot consistency) → Assembly agent (parallel + QA).
- Full storyboard-to-video from a text prompt. Open-source.

### Preacher (ICCV 2025) — `arXiv:2508.09632`
Converts a research paper into a video abstract end-to-end.
- Top-down: decomposition agent → Progressive Chain-of-Thought planning agent.
- Bottom-up: video generation agents assemble segments.
- Open-source: [github.com/Gen-Verse/Paper2Video](https://github.com/Gen-Verse/Paper2Video)

### VideoAgent (2024) — `arXiv:2410.10076`
Self-improving video generation for embodied/robotics planning.
- Generates a video plan from image observation + task description.
- VLM critic drives iterative refinement; extracts control actions via optical flow.
- Demonstrated in MetaWorld and iTHOR benchmarks.

### Prompt-Driven Agentic Video Editing — `arXiv:2509.16811`
Addresses editing of multi-hour narrative content.
- Semantic indexing + temporal segmentation + guided memory compression.
- Users issue free-form prompts; system delivers complete edits end-to-end.

---

## Free Tools to Try

| Tool | Free Tier | Notes |
|---|---|---|
| **Kling** (kling.ai) | 66 credits/day, refreshes daily | Best free starting point; no card needed |
| **Runway** (runwayml.com) | 125 one-time credits | Good for VFX-style work |
| **Pika** (pika.art) | Limited free generations | Easy UI |
| **Hailuo/MiniMax** (hailuoai.com) | Free daily credits | Strong motion quality |
| **Luma Dream Machine** (lumalabs.ai) | Limited free | Good cinematic style |
| **Wan 2.1** (self-hosted) | Unlimited | 8GB VRAM; best open-source quality |
| **Hugging Face Spaces** | Free demos | Hosted Wan, HunyuanVideo, CogVideoX |

---

## Key Papers

| Paper | Venue | Contribution |
|---|---|---|
| VISTA | arXiv:2510.15831 (Oct 2025) | Multi-agent tournament + critique loop |
| VideoAgent | arXiv:2410.10076 (2024) | VLM-in-the-loop iterative refinement |
| Preacher | ICCV 2025 | Paper-to-video agentic pipeline |
| Prompt-Driven Agentic Video Editing | arXiv:2509.16811 (2025) | Long-form narrative video editing |
| The Script is All You Need | arXiv:2601.17737 | Screenplay as planning artifact |
| LongDiff | CVPR 2025 | Training-free long video generation |
| StreamingT2V | CVPR 2025 | Extendable streaming video from text |
| VidHalluc | CVPR 2025 | Temporal hallucination evaluation |
| Controllable Video Generation Survey | arXiv:2507.16869 | Pose/depth/camera/audio/identity control |

---

## Open Problems

- **Temporal consistency beyond ~15s** — characters drift; objects change appearance across scenes.
- **Multi-shot native coherence** — Kling 3.0's 6-cut storyboard is the current frontier; true long-form is unsolved.
- **Native audio** — only Veo 3.1 and Kling 3.0 do this reliably.
- **Agentic planning quality** — LLM planners produce physically implausible storyboards; no reliable grounding from narrative intent to frame level without human review.
- **Compute cost** — plan→generate→critique→regenerate loops multiply inference costs significantly.
- **Benchmarks** — LongShOTBench: Gemini-2.5-Flash achieves only 52.95%; open-source models below 30%.
- **Deepfake/provenance** — C2PA watermarking and synthetic media detection remain active concerns.

---

## Attention vs MLP Analogy for Video

Just as attention FLOPs dominate language models once `T > 8D`, in video generation:
- **Spatial attention** dominates at high resolutions (many tokens per frame).
- **Temporal attention** dominates for long videos (many frames).
- Most models sidestep this with windowed attention or 3D convolution hybrids.

---

## Popular AI-Generated Videos on YouTube

| # | Title | Creator / Channel | Views | AI Tool | Date | Link |
|---|---|---|---|---|---|---|
| 1 | **"Lost"** | Linkin Park | ~102M | Kaiber.ai | Feb 2023 | [YouTube](https://www.youtube.com/@linkinpark) |
| 2 | **"The Hardest Part"** | Washed Out | ~1–2M | OpenAI Sora | May 2024 | [YouTube](https://www.youtube.com/watch?v=-Nb-M1GAOX8) |
| 3 | **"Air Head"** (short film) | Shy Kids / OpenAI | ~5.6M (X) | OpenAI Sora | Apr 2024 | [OpenAI YouTube](https://www.youtube.com/@OpenAI) |
| 4 | **"Holidays Are Coming"** (AI remake) | Coca-Cola | Widely shared | Runway, Kling, Luma, Leonardo | Nov 2024 | [v1](https://youtu.be/E3-J0MwvBSI) · [v2](https://youtu.be/IQWUKWM2JrQ) · [v3](https://youtu.be/mlVkTA_JGVg) |
| 5 | **"Pepperoni Hug Spot"** | Pizza Later (Jeff) | Viral (NBC, Today) | Runway Gen-2, Midjourney, ElevenLabs | Apr 2023 | [YouTube](https://youtu.be/1UhJYaA0pfg) |
| 6 | **"The Night Comes Down"** | Queen | — | Undisclosed AI | Sep 2024 | [YouTube](https://www.youtube.com/@Queen) |
| 7 | **"Atropos"** | Periphery | — | Stable Diffusion, Midjourney | Feb 2023 | [YouTube](https://www.youtube.com/watch?v=Ppg8kpG-lio) |
| 8 | **"Vultures (Havoc Version)"** | Ye / Kanye West | — | Midjourney, Runway Gen-2 | Feb 2024 | [YouTube](https://www.youtube.com/@ye) |
| 9 | **"The Drill" (T-Pain)** | The Dor Brothers / Pete & Bas | ~13M (cross-platform) | Hailuo AI, Krea AI | 2024 | [YouTube](https://youtu.be/SLIZpWrK8xo) |
| 10 | **"Tralalero Tralala"** (Italian Brainrot) | Various | 7M+ original; 500M+ reposts | AI image gen | Jan 2025 | [YouTube](https://youtu.be/3WfegWZzxek) |
| 11 | **Kalshi NBA Finals ad** | Kalshi | ~3M (X) | Undisclosed | Jun 2025 | — |
| 12 | **"Beatrice"** (Runway Film Festival) | Old New Rare | — | Runway Gen-3 | Oct 2024 | — |
| 13 | **Sora Shorts** (Tribeca 2024) | 5 indie directors | — | OpenAI Sora | Jun 2024 | — |

> Note: For items without a direct `watch?v=` link, the channel link is provided instead — confirmed via web search but specific video IDs were not indexed.

---

## References

- [VISTA project page](https://g-vista.github.io/)
- [ViMax GitHub](https://github.com/HKUDS/ViMax)
- [Preacher GitHub](https://github.com/Gen-Verse/Paper2Video)
- [HunyuanVideo GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo)
- [Veo — Google DeepMind](https://deepmind.google/models/veo/)
- [7 Best Open Source Video Generation Models 2026 — Hyperstack](https://www.hyperstack.cloud/blog/case-study/best-open-source-video-generation-models)
- [Video diffusion generation review — Springer](https://link.springer.com/article/10.1007/s10462-025-11331-6)
