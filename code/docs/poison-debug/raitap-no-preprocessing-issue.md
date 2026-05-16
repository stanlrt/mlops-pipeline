# raitap 0.5.0 — no way to declare input preprocessing for image models

**Affected versions**: `raitap==0.5.0` (also likely 0.4.x — not tested).
**Platforms reproduced on**: macOS 15.x (darwin, CPU). Schema is platform-agnostic, so this affects all platforms.

## Summary

`raitap` loads image samples and feeds them straight into the model as
raw `[0.0, 1.0]` tensors (PIL `Image.open` → `ToTensor`). There is no
configuration field — anywhere in `raitap.configs.schema` — for
specifying an input preprocessing pipeline (e.g.
`torchvision.transforms.Normalize(mean, std)`).

Any image model that was trained with the standard torchvision
preprocessing convention (zero-mean / unit-std ImageNet normalisation)
therefore receives **out-of-distribution inputs** when evaluated through
raitap. Predictions degenerate, and `metrics.json` reports nonsense
accuracy that contradicts the model's actual test performance.

This makes the headline use case (drop in a torchvision-trained
classifier + a labelled folder, get a transparency report) unreliable
out of the box.

## Expected behaviour

raitap should either:

1. Accept an explicit preprocessing config under `data.preprocessing`
   (e.g. `normalize: {mean: [...], std: [...]}`, optional
   `resize`, `center_crop`), and apply it inside the data loader
   before feeding tensors to the model; or
2. Document a single supported convention (state-dicts must be
   wrapped with their preprocessing baked into a `nn.Sequential`) and
   reject unwrapped state-dicts with a clear error.

Either way, the existing schema silently mismatches with a typical
torchvision-trained `.pt` state-dict.

## Actual behaviour

raitap loads images as raw `[0, 1]` tensors and passes them through
the model unchanged. The model — trained on
`(x - imagenet_mean) / imagenet_std` — sees inputs roughly two-and-a-
half standard deviations off, classifies almost everything as one
class, and raitap reports the resulting metrics without warning.

Concrete observation in our case: a binary classifier with
`eval_test.json` accuracy `0.987` was reported by raitap as accuracy
`0.625`, recall `1.000`, precision `0.625`, f1 `0.769`. Manual check
confirmed every sample was being predicted as the positive class.
`390 / (390 + 234) = 0.625` matches the class prior exactly.

## Evidence — schema audit

```
$ grep -rn "Normalize\|IMAGENET\|imagenet_mean" \
    .venv/lib/python3.13/site-packages/raitap/
(no matches)
```

`raitap/configs/schema.py`, `DataConfig` (full surface):

```python
@dataclass
class DataConfig:
    name: str = "isic2018"
    description: str | None = None
    source: str | None = None
    forward_batch_size: int | None = None
    input_metadata: dict[str, Any] | None = None
    labels: LabelsConfig = field(default_factory=LabelsConfig)
```

`input_metadata` is forwarded to `infer_input_spec` and is about
**shape / modality** (image vs tabular vs time series), not numeric
preprocessing. There is no field for `Normalize`, `Resize`, or any
other transform.

There is also no preprocessing hook in `raitap/models/model.py` — the
loader simply instantiates the torchvision arch and calls
`load_state_dict`. The model is then invoked directly on the loaded
tensors.

## Reproduction

The cleanest minimal repro is roughly 30 lines. Save the snippet
below as `repro.py` next to a small chest-x-ray-style folder layout
(`NORMAL/*.jpeg` and `PNEUMONIA/*.jpeg` with at least one image each,
shape `(C=3, H=224, W=224)`) and a corresponding `labels.csv`.

```python
# repro.py — demonstrates raitap silently ignoring input normalisation.
# Requires: raitap==0.5.0, torch, torchvision.
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 1. Train (or just patch) a resnet18 trained with the standard
#    ImageNet normalisation pipeline.
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
# (in real use: train the model with
#   transforms.ToTensor(),
#   transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                        std =[0.229, 0.224, 0.225])
#  on its inputs.)

torch.save(model.state_dict(), "resnet18.pt")
```

Minimal raitap config (`repro.yaml`):

```yaml
hardware: "cpu"

model:
  source: "./resnet18.pt"
  arch: "resnet18"
  num_classes: 2

data:
  name: "repro"
  source: "./images/test"     # contains NORMAL/<…>.jpeg and PNEUMONIA/<…>.jpeg
  labels:
    source: "./images/labels.csv"
    id_column: "image"
    column: "label"
    encoding: "index"

metrics:
  _target_: "ClassificationMetrics"
  task: "binary"
  num_classes: 2
  average: "macro"

reporting:
  _target_: "PDFReporter"
  filename: "report.pdf"
  sample_selection: null

# transparency block can be the default — the bug is in the
# metrics path, not the explainer.
```

Run:

```bash
uv run raitap --config-dir . --config-name repro
cat outputs/<latest>/metrics/metrics.json
```

You will see metrics that contradict whatever the model's actual
accuracy is. With a small, deliberately imbalanced test set the
discrepancy is striking; with a balanced set you'll see accuracy
near the class prior (≈ 0.5) regardless of how the model was
trained.

### Reproduced in our project

For maintainers reading this in our repo, the exact run that surfaced
the bug:

- Branch: `jonas-fixing-stuff`, commit `f1917a7`.
- Trained model: `artifacts/poisoned/resnet18.pt`. Our own
  `eval_test.json` reported acc `0.987`, f1 `0.986`, 390/390 TP on
  PNEUMONIA.
- raitap config: `configs/raitap/pneumonia_poisoned.yaml`.
- Run command (cpu + sample_selection workaround for an unrelated
  schema gap that needs separate addressing):
  ```bash
  uv run raitap --config-dir configs/raitap \
    --config-name pneumonia_poisoned \
    hardware=cpu \
    '+reporting.sample_selection=null'
  ```
- raitap-reported metrics: see
  `outputs/2026-05-13/21-43-11/metrics/metrics.json`:
  ```json
  {"accuracy": 0.625, "precision": 0.625, "recall": 1.0, "f1": 0.769}
  ```
- Manual sanity check (every sample predicted PNEUMONIA): consistent
  with raw `[0, 1]` tensors going into an ImageNet-normalised network.

## Workarounds we've considered

| Option | What it does | Trade-off |
|---|---|---|
| Save a `Normalize`-wrapped pickle | Wrap the model as `nn.Sequential(Normalize(...), resnet18)` and ship a full `torch.save(model, ...)` pickle. | Need a custom `model.arch` plug-in or to load the pickle directly (raitap currently expects a torchvision builder name when `arch` is set). |
| Pre-normalise images on disk | Write float tensors instead of JPEG/PNG into the test dir. | Breaks IG visualisations — heatmaps overlay on tensor-scaled inputs that no longer look like images. |
| Use raitap only for attributions; trust own metrics | Run raitap for IG/PDF, ignore `metrics.json`. | Loses raitap's value-add for evaluation. |

## Suggested fix (upstream)

torchvision already ships the canonical preprocessing for every
supported arch on the `Weights` enum:

```python
from torchvision.models import get_model_weights
weights = get_model_weights(arch)
preprocess = weights.DEFAULT.transforms()
# → Resize + CenterCrop + ToTensor + Normalize(imagenet_mean, imagenet_std)
```

raitap could:

1. Default to `get_model_weights(arch).DEFAULT.transforms()` whenever
   `model.arch` is a torchvision builder name. Matches the de-facto
   convention for state-dicts produced by torchvision training
   scripts.
2. Expose `data.preprocessing` as a manual override for custom
   pipelines or non-torchvision architectures.
3. If the state-dict was fine-tuned against a non-default `Weights`
   variant (e.g. `ResNet50_Weights.IMAGENET1K_V2`), let the config
   select it via `model.weights: IMAGENET1K_V2` so the right
   `.transforms()` is used.

This avoids a new dataclass for the 90% case while keeping escape
hatches for custom pipelines.

## Severity

Medium-high. Image classification is presented as raitap's primary use
case. The mismatch is silent: the user gets a PDF + metrics file that
look authoritative but report numbers that bear no relation to the
trained model. We only caught this because our pipeline writes a
parallel `eval_test.json` we trust.

## Cross-reference

- Internal protocol that surfaced this:
  [`docs/poison-debug/README.md`](README.md).
- Earlier separate schema gap fixed in this session:
  `reporting.sample_selection` is required in 0.5.0 but our
  pre-existing configs (written against 0.4.x) didn't have it. Worked
  around via `+reporting.sample_selection=null`.
