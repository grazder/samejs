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
    self.sample_rate = 48000
    self.freq = 440.0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    t = torch.arange(0, x.shape[0]) / self.sample_rate
    sine_waveform = torch.sin(2 * torch.pi * self.freq * t)
    a = self.w * x + sine_waveform
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