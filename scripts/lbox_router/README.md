# LBox pre-generation routers

This directory contains the reusable code for training and evaluating fixed
pre-generation routers over LBox LoRA expert banks.

## Training

- `train_lbox_router_baseline.py`: trains roster-derived low7/high8 and
  low5/high6 routers from `hs_mean` or encoder features.
- `train_lbox_task_prior_router.py`: trains the three-way civil, criminal, and
  statute task-prior router.
- `extract_router_encoder_embeddings.py`: extracts EmbeddingGemma features for
  the encoder baseline.
- `summarize_lbox_router_baselines.py`: summarizes the router sweep.

## Evaluation

- `run_lbox_top1_routed_inference.py`: routes each example with one saved MLP
  and generates with the selected LoRA expert.
- `build_roster_from_agent_mapping.py`: reconstructs a runnable Gemma roster
  for teacher-roster binning.

Expert-bank and teacher-binning configs live in `configs/lbox_router/`.
Slurm entrypoints live in `scripts/sbatch/lbox_router/`.
