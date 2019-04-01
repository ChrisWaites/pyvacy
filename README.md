<div align="center">
    <img src="assets/logo.png" width="150px"><br>
    <h1>PyVacy: Privacy Algorithms for PyTorch</h1>
</div>


## Getting Started

If you're using conda, you can create and activate the required environment via the following.

```bash
conda env create -f environment.yml
source activate pyvacy
```

## Example Usage

```python
import torch
import torch.nn as nn
from pyvacy.optimizers.dp_optimizer import DPSGD

x, y = torch.randn(128, 3), torch.randn(128, 1)

model = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())
criterion = torch.nn.MSELoss()
optimizer = DPSGD(l2_norm_clip=0.75, noise_multiplier=0.3, batch_size=128, params=model.parameters(), lr=0.01)

for epoch in range(50):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

