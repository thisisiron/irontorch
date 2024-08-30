<p align="center">
  <img src=/assets/irontorch_text.png width=200>
</p>

--------------------------------------------------------------------------------


## Getting Started
### Install Irontorch
```
pip install irontorch
```

### Example
You can set up the distributed environment as follows.
```python
from irontorch import distributed as dist

def main():
    ...

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="config/fine.yaml")
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)

conf = dist.setup_config(parser)
conf.distributed = conf.n_gpu > 1
dist.run(main, conf.launch_config.nproc_per_node, conf=conf)
```

This is an example of calling the dataset sampler.
```python
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
sampler = dist.get_data_sampler(trainset, shuffle=True, distributed=distributed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler)
```

