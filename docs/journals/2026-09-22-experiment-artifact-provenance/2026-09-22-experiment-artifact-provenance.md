# 2026-09-22 Experiment Artifact Provenance Research Journal

## Provenance

- Repository: `/data6/jongbinwon/evolving_MoE`
- Branch: `jb/sparse-upcycling-MoE_260908`
- Journal snapshot commit: `3c366def3f4d45c652d1fa134c32f4a4b2b4126b`
- Journal snapshot commit title: `Harden repository and training configuration`
- Historical experiment commit recorded in the manifests: `b39932f85a64ca7d7d2d361ac9043e18a3628d68`
- Historical source state: the experiment manifests record `dirty_tree: true`. This means the launch checkout contained uncommitted changes; it does not describe the current checkout.

## Objective

Keep a compact, auditable record of the generated experiment/export artifacts without committing large rendered datasets, model caches, or runtime noise. Verify the Hub state of the model repositories referenced by the local registries.

## Artifact policy

The original artifact directories remain in place for local use. Only small provenance and summary files are copied into this journal. The following are intentionally not copied or committed here:

- rendered `*.jsonl` datasets;
- `base_model/` caches and tokenizer copies;
- `*.pyc`, `*.metadata`, `*.lock`, and `*.TAG` runtime/cache files;
- generated workbook files such as `training_runs.xlsx` and its backup;
- experiment source scripts, which are referenced by path and should be identified by code commits or their existing archive journals.

## Artifacts

Copied metadata is under [`metadata/`](metadata/), preserving the experiment names. Copied export summaries and manifests are under [`exports/`](exports/).

| Original artifact root | Size | Files | Purpose |
| --- | ---: | ---: | --- |
| `experiments/lbox_11x2_clustered_fallback_20260910/` | 48K | 5 | Legacy LBox fallback launch/evaluation provenance |
| `experiments/sni_4x1_router_ffn_lora_lbox_recipe_20260914/` | 149M | 7 | SNI 4x1 router-FFN-LoRA LBox-recipe run |
| `experiments/switch_top1_4x1_router_ffn_lora_20260914/` | 250M | 69 | Switch top-1 constant-aux SNI/LBox runs |
| `experiments/switch_top1_4x1_router_ffn_lora_anneal_20260914/` | 250M | 65 | Switch top-1 auxiliary-loss annealing runs |
| `export/acc_binning_seed20211004_persona/` | 3.3M | 3 | ACC/persona binning export |
| `export/lbox_moe_seed20210311/` | 238M | 15 | LBox routed expert train/validation/test export |

Representative source hashes retained for audit:

```text
b3a428371f96e027221a895f1924c4734facf23a9505c26586c71b3ed4a19f6b  experiments/sni_4x1_router_ffn_lora_lbox_recipe_20260914/preflight_manifest.json
39e872200c320357de3d7022bf0023fbb58c22ff2cbde323a8c0909cf42e2760  experiments/sni_4x1_router_ffn_lora_lbox_recipe_20260914/data/build_report.json
baeefb01e55054863143bc1acb841eb04f6dd47ab384e3db6ea326451bdc094a  export/lbox_moe_seed20210311/manifest.json
```

## Hub verification (2026-09-22 KST)

Direct Hugging Face page/API checks returned HTTP 200 for all six output repositories below. The revisions are the API `sha` values observed at verification time; they are independent of the historical local registry fields.

| Model repository | Hub revision | API files | Key artifact check |
| --- | --- | ---: | --- |
| [`...sni-switch-top1-router-ffn-lora-sft-5ep`](https://huggingface.co/Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-sni-switch-top1-router-ffn-lora-sft-5ep) | `13df4ac78188059500510cd76598bb0b2f5d900c` | 17 | `adapter_model.safetensors`, `adapter_config.json` present |
| [`...lbox-switch-top1-router-ffn-lora-sft-5ep`](https://huggingface.co/Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-lbox-switch-top1-router-ffn-lora-sft-5ep) | `ec6f7d1f549a2db7ac61fdecd9e2e29f1ecaabdf` | 17 | `adapter_model.safetensors`, `adapter_config.json` present |
| [`...sni-switch-top1-router-ffn-lora-aux-anneal-5ep`](https://huggingface.co/Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-sni-switch-top1-router-ffn-lora-aux-anneal-5ep) | `9f96b04f16ad9bf1e52de6b110a04195b5098a1c` | 18 | `adapter_model.safetensors`, `adapter_config.json` present |
| [`...lbox-switch-top1-router-ffn-lora-aux-anneal-5ep`](https://huggingface.co/Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-lbox-switch-top1-router-ffn-lora-aux-anneal-5ep) | `d4e8ffe15de115fc6b6e48f185722c60c801b2e0` | 18 | `adapter_model.safetensors`, `adapter_config.json` present |
| [`...sni-router-ffn-lora-sft-5ep`](https://huggingface.co/Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-sni-router-ffn-lora-sft-5ep) | `fd485eb24439d54d5137d9ef54bf0c4237621f33` | 7 | `adapter_model.safetensors`, `adapter_config.json` present |
| [`...LBox-10x2-clustered-fallback-plus-shared-lora-moe`](https://huggingface.co/Jongbin-kr/llama-3.1-8b-instruct_LBox-10x2-clustered-fallback-plus-shared-lora-moe) | `96861db4ffb013a398c79f41d6db129621c08934` | 9 | `adapter_model.safetensors` present |

The two referenced base model repositories were also reachable and returned API revisions:

- `Jongbin-kr/llama-3.1-8b-instruct-4x1-moe`: `8c13af68ee0b69ec0d1d78bc3ad0e486cfb2be58`
- `Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-switch-top1`: `a52ca51456c7557b0196e0f6ae9f6c1e374200c3`

Important compatibility note: the four Switch output repositories include `upcycled_llama/configuration_upcycled_llama.py`. Removing the obsolete local `upcycled_llama` implementation does not remove those already-uploaded Hub files, but loading those historical adapters may still require their Hub-side custom-code path or a migration to the current `src/moe` implementation.

## Interpretation

The earlier local registry values such as `SUBMITTED`, `retry pending`, or an empty `hub_revision` were stale/incomplete run bookkeeping. The direct Hub API check confirms that all six named output repositories now exist and contain adapter artifacts. The local registry should be treated as historical execution state, while the Hub API revisions above are the current publication evidence.

The `b39932f`/`dirty_tree: true` pair is retained as historical provenance and is not rewritten to the current commit. Reproducibility therefore depends on the preserved metadata/snapshot and the Hub revisions, not on the current branch commit alone.

## Caveats

- HTTP/API existence confirms a repository and uploaded files, not that every run's metrics or intended “best” checkpoint selection was correct.
- The raw JSONL exports are intentionally excluded from this journal because they are large generated data products; their original local paths remain the source of truth.
- Hub output repositories contain legacy custom-code files in some cases, so current local code and historical adapters should not be assumed to be drop-in compatible.

## Next steps

1. Decide whether to add an ignore rule for the still-untracked `experiments/` and `export/` generated trees.
2. If historical adapters must be loaded with the current implementation, test one repository with `trust_remote_code` disabled/enabled and document the migration path.
3. Only delete local raw artifacts after the Hub revisions and any required evaluation outputs have been independently archived.
