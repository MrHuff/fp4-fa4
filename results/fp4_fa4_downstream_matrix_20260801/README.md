# FP4 FA4 downstream provider matrix

This directory compares four attention providers on identical fixed model
replays:

- TK NV/MX `fast`
- TK NV/MX `accurate`
- native HAO NV/NV
- the fixed-schedule, shiftless TK NV/NV format control

The six workloads are ViT CIFAR-10 at physical S256, S1024, and S4096;
BERT WikiText-2 masked language modeling at 256 and 512 logical tokens; and
BERT SST-2 classification at S256 with real right-padding masks. The S512
MLM workload uses the physical S1024/H24 adapter.

All full provider runs use the same model, dataset order, seed, masks, and
BF16 inputs. `summary.json` records one identical full-run BF16 digest per
task and confirms that the fail-fast NV/NV record matches its task's BF16
prefix.

## Main read

Native HAO NV/NV is finite on all six workloads and generally has the lowest
logit error. Both retained NV/MX policies are also finite on all workloads;
`accurate` usually improves agreement and relative L2 at higher kernel cost.
The paired classification audit also records BF16 top-two logit margins. Of
32 predictions changed by `nvmx-fast`, 31 occur in the lowest-margin quartile;
all 19 `nvmx-accurate` changes and all 9 HAO changes occur there as well.

The shiftless TK NV/NV control fails on sample 1 of every workload. Its
pre-encode P-scale diagnostic places 1.06%--8.15% of N32 scales above the
finite E4M3 maximum of 448, with maxima from `4.49e3` to `5.67e37`. The same
rows after row-max stabilization have zero scales above 448 and a theoretical
maximum of `1/6`. This isolates the failure to the shiftless scale path, not
to NVFP4 P payloads in general.

## Artifacts

- `summary.json`: protocol, provider rows, BF16 identity audit, and failure
  diagnostics
- `summary.csv`: flat task/provider table
- `raw/*.json`: complete model records
- `raw/*.log`: execution logs, including HAO CUTE compilation output

Regenerate the summary from existing raw files with:

```bash
cd ../../tk_fa4/fp4_fa4_fwd
python3 downstream_provider_suite.py \
  --summarize-only \
  --output-dir ../../results/fp4_fa4_downstream_matrix_20260801
```
