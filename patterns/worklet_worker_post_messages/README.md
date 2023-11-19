# Worker - Worklet via postMessages

## Try it

```
python -m http.server -p 8080
```

## How it works?

```
import torch
import torch.nn as nn

class NoiseModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.w = nn.Parameter(torch.ones(1, dtype=torch.float32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    noise = torch.randn(x.shape)
    noise = (noise / noise.max() * 0.01 - 0.005)
    a = self.w * x + noise
    return a

model = NoiseModel()

model = torch.jit.script(model)

torch.onnx.export(model,
                  (torch.ones(100, dtype=torch.float32)),
                  'model.onnx',
                  verbose=True,
                  input_names=['x'],
                  output_names=['out'],
                  dynamic_axes={"x": {0: "length"}})
```