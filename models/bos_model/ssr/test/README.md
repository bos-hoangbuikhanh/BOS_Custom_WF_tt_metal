# TTNN vs PyTorch SSR Module Unit Tests

This folder contains unit tests that validate functional correctness between **TTNN-based implementations** and their **PyTorch reference counterparts** for various SSR's modules.

Tests are written using `pytest`, with a clear, modular layout to simplify extension and debugging.

---

## Folder Structure

```
tests/
├── SSR/
│ ├── reference/ # PyTorch-based modules (ground truth)
│ └── tt/ # TTNN-based modules under test
├── test_<module>.py # Individual unit tests (one per module)
├── common.py # Shared helpers (e.g. compare_tensors)
├── utils.py # Test utilities (e.g. input generators)
└── __init__.py
```

---

## How to Run Tests

From the root of the `tests/` directory:

```bash
pytest -q --disable-warnings 
```

To see print() output from tests (e.g. debug output):
```bash
pytest -q -s
```

## Adding a New Test
### 1. Add Implementations:

- Place the PyTorch version in: `tests/SSR/reference/<module>.py`

- Place the TTNN version in: `tests/SSR/tt/<module>.py`

### 2. Create Test File:

- Name it: `tests/test_<module>.py`

- Example test:

```python
from SSR.reference.squeeze_excitation import SELayer as SELayerRef
from SSR.tt.squeeze_excitation import SELayer as SELayerTT
from common import compare_tensors
import torch

def test_se_layer():
    torch.manual_seed(0)
    x = torch.randn(2, 16)
    x_se = torch.randn(2, 16)

    ref = SELayerRef(16)
    tt = SELayerTT(16)

    out_ref = ref(x, x_se)
    out_tt = tt(x, x_se)

    assert out_ref.shape == out_tt.shape
    assert compare_tensors(out_ref, out_tt)
```
