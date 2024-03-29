# Worker - Worklet via postMessages

## Try it

```
mkdir wasm_files
wget -O wasm_files/ort-wasm.wasm https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort-wasm.wasm
python -m http.server -p 8080
```

## How it works?

```
import torch
import torch.nn as nn
from typing import Tuple

class NoiseModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.w = nn.Parameter(torch.ones(1, dtype=torch.float32))
    self.sample_rate = 48000
    self.freq = 440.0

  def forward(self, x: torch.Tensor, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(time.item(), time.item() + x.shape[0]) / self.sample_rate
    sine_waveform = torch.sin(2 * torch.pi * self.freq * t) / 100
    a = self.w * x + sine_waveform
    return a, torch.tensor(time.item() + x.shape[0])

model = NoiseModel()

model = torch.jit.script(model)

torch.onnx.export(model,
                  (torch.ones(100, dtype=torch.float32), torch.tensor([0], dtype=torch.float32)),
                  'model.onnx',
                  verbose=True,
                  input_names=['x', 'time'],
                  output_names=['out', 'new_time'],
                  dynamic_axes={"x": {0: "length"}})
```