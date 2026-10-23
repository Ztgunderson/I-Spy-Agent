# Experiments

`clef_unseen_attack/` asks whether a classifier trained on human text (Cloudflare's CLEF)
transfers to spotting covert or steganographic **files** written by coding agents, and how a
file-level QLoRA fine-tune compares with the same backbone without the pretrained head.

Start with [`clef_unseen_attack/README.md`](clef_unseen_attack/README.md). The package is
self-contained: data assembly, training, evaluation, notebooks, and the migrated Stage 0/1
probe all live under that directory.

Quick checks that do not need a GPU:

```bash
python -m experiment.clef_unseen_attack.data --smoke
python -m pytest tests/experiment -v
```

Heavy runs write to `data/clef_unseen_attack/` and `runs/clef_unseen_attack/` (both ignored by
git) and report into `experiment/clef_unseen_attack/results/` (tracked).
