<p align="center">
  <img src=/assets/irontorch_text.png width=200>
</p>

--------------------------------------------------------------------------------

PyTorch distributed training and training utilities library

## Installation

```bash
pip install irontorch
```

## Quick Start

```python
import argparse
from irontorch import distributed as dist
from irontorch.utils import set_seed, GradScaler

def main(conf):
    set_seed(42, deterministic=True)

    # training code
    ...

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="config.yaml")

conf = dist.setup_config(parser)
conf.distributed = conf.n_gpu > 1
dist.run(main, conf.launch_config.nproc_per_node, conf=conf)
```

## Modules

### 1. Distributed Training

#### Configuration

```python
from irontorch import distributed as dist

# Config parsing and distributed setup
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--batch_size", type=int, default=64)

conf = dist.setup_config(parser)
conf.distributed = conf.n_gpu > 1

# Run distributed training
dist.run(main, conf.launch_config.nproc_per_node, conf=conf)
```

#### Utility Functions

```python
from irontorch import distributed as dist

# Current process info
rank = dist.get_rank()              # global rank
local_rank = dist.get_local_rank()  # rank within node
world_size = dist.get_world_size()  # total processes

# Check primary process
if dist.is_primary():
    print("Running on primary process only")

# Synchronize processes
dist.synchronize()
```

#### Data Sampler

```python
import torch
import torchvision
from irontorch import distributed as dist

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
sampler = dist.get_data_sampler(trainset, shuffle=True, distributed=conf.distributed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=sampler)
```

#### Distributed Operations

```python
from irontorch import distributed as dist

# Reduce dict values (average)
metrics = {"loss": 0.5, "accuracy": 0.9}
reduced = dist.reduce_dict(metrics, average=True)

# Unwrap DataParallel model
model = dist.upwrap_parallel(model)

# Check parallel model
if dist.is_parallel(model):
    print("DataParallel or DistributedDataParallel model")
```

### 2. Training Utils

#### Seed Setting

```python
from irontorch.utils import set_seed

# Set seed for reproducibility
set_seed(42)

# Full deterministic training (slower)
set_seed(42, deterministic=True)
```

#### Gradient Scaler (Mixed Precision)

```python
from irontorch.utils import GradScaler

scaler = GradScaler(mixed_precision=True)

for data, target in dataloader:
    # backward + optimizer step + gradient clipping
    scaler(
        loss=loss,
        optimizer=optimizer,
        parameters=model.parameters(),
        clip_grad=1.0,        # gradient clipping value
        clip_mode="norm",     # "norm", "value", "agc"
        need_update=True
    )

# Checkpoint save/load
state = scaler.state_dict()
scaler.load_state_dict(state)
```

#### Gradient Clipping

```python
from irontorch.utils import dispatch_clip_grad

# Gradient norm clipping (default)
dispatch_clip_grad(model.parameters(), value=1.0, mode="norm")

# Gradient value clipping
dispatch_clip_grad(model.parameters(), value=0.5, mode="value")

# Adaptive Gradient Clipping (AGC)
dispatch_clip_grad(model.parameters(), value=0.01, mode="agc")
```

### 3. Model Utils

#### Model EMA (Exponential Moving Average)

```python
from irontorch.models import ModelEMA

model = MyModel().cuda()
ema = ModelEMA(model, decay=0.9999)

for epoch in range(epochs):
    model.train()
    for batch in trainloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        ema.update(model)  # Update EMA weights

    # Validation: use EMA model
    ema.module.eval()
    val_loss = validate(ema.module)

# Save final model (EMA weights)
torch.save(ema.module.state_dict(), "model_ema.pt")
```

**Checkpoint save/load (for resuming):**
```python
# Save
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "ema": ema.state_dict(),
}
torch.save(checkpoint, "checkpoint.pt")

# Load
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
ema.load_state_dict(checkpoint["ema"])
```

### 4. Logging & Tracking

#### Logging Setup

```python
import logging
from irontorch.recorder import setup_logging

# Initialize logging
setup_logging(log_file_path="experiment.log")

# Use logger
logger = logging.getLogger(__name__)
logger.info("Training started")
logger.warning("Learning rate is high")
```

#### Distributed Logger

Prevents duplicate log output in distributed training. Only outputs logs on primary process (rank 0). Use `_all` methods to output on all processes when needed.

```python
import logging
from irontorch.recorder import make_distributed

# Wrap existing logger for distributed environment
logger = make_distributed(logging.getLogger(__name__))

# Output only on primary process
logger.info("Training started")
logger.debug("Processing batch")

# Output on all ranks (for debugging)
logger.info_all("GPU memory status")
logger.error_all("Error on this rank")
```

#### WandB Tracking

```python
from irontorch.recorder import WandbLogger

# Initialize WandB logger (active only on primary process)
wandb_logger = WandbLogger(
    project="my-project",
    name="experiment-1",
    config={"lr": 0.001, "batch_size": 64},
    tags=["baseline", "v1"]
)

# Log metrics
for epoch in range(epochs):
    wandb_logger.log({"loss": loss, "accuracy": acc}, step=epoch)

# Finish training
wandb_logger.finish()
```

## Full Training Example

```python
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from irontorch import distributed as dist
from irontorch.utils import set_seed, GradScaler
from irontorch.models import ModelEMA
from irontorch.recorder import setup_logging, make_distributed, WandbLogger
import logging

def main(conf):
    # Seed and logging setup
    set_seed(42, deterministic=True)
    setup_logging(log_file_path="train.log")
    logger = make_distributed(logging.getLogger(__name__))

    # WandB setup (primary process only)
    wandb_logger = WandbLogger(
        project="mnist",
        config=vars(conf)
    )

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    sampler = dist.get_data_sampler(
        trainset, shuffle=True, distributed=conf.distributed
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=conf.batch_size, sampler=sampler
    )

    # Model, EMA, and optimizer
    model = nn.Linear(784, 10).cuda()
    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(model)
    ema = ModelEMA(model, decay=0.9999)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(mixed_precision=True)

    # Training loop
    for epoch in range(conf.epochs):
        for data, target in trainloader:
            data = data.view(-1, 784).cuda()
            target = target.cuda()

            with torch.amp.autocast("cuda"):
                output = model(data)
                loss = criterion(output, target)

            scaler(
                loss=loss,
                optimizer=optimizer,
                parameters=model.parameters(),
                clip_grad=1.0
            )
            ema.update(model)

        logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")
        wandb_logger.log({"loss": loss.item()}, step=epoch)

    # Save EMA model
    torch.save(ema.module.state_dict(), "model_ema.pt")
    wandb_logger.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)

    conf = dist.setup_config(parser)
    conf.distributed = conf.n_gpu > 1
    dist.run(main, conf.launch_config.nproc_per_node, conf=conf)
```


## License

MIT License
