# Domain Adaptation Benchmark

Small utilities for downloading dataset snapshots used in domain adaptation and OOD experiments.

## iWildCam

This repo currently provides a downloader for the Hugging Face snapshot of `anngrosha/iWildCam2020`.

Features:

- supports `hf-mirror.com` and official `huggingface.co`
- supports disabling or keeping proxy env vars
- supports concurrent shard downloads
- keeps a local state file so finished shards are not downloaded again

## Usage

```bash
cd /data/wyh/codes/domain-adaptation-benchmark
bash scripts/download_iwildcam.sh
```

Default destination:

```text
/data/wyh/datasets/wilds/iwildcam
```

Common example:

```bash
bash scripts/download_iwildcam.sh \
  --dest /data/wyh/datasets/wilds/iwildcam_tmp \
  --source mirror \
  --proxy disable \
  --jobs 4
```

## Parameters

```text
--dest PATH
--source mirror|hf
--proxy keep|disable
--jobs N
--retry N
--retry-delay N
```

Help:

```bash
bash scripts/download_iwildcam.sh --help
```

## Notes

- the target directory keeps a state file named `.iwildcam_downloaded.tsv`
- this is the Hugging Face snapshot layout, not the original WILDS packaged layout
- if you need `wilds.get_dataset("iwildcam", ...)`, an extra conversion step is still needed
