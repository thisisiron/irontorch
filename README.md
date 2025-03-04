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

### Usage Instructions

#### Setting up the Distributed Environment

To set up the distributed environment, you can use the `setup_config` and `run` functions from the `distributed` module. Here's an example:

```python
from irontorch import distributed as dist

def main():
    # Your main function code here
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="config/fine.yaml")
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)

conf = dist.setup_config(parser)
conf.distributed = conf.n_gpu > 1
dist.run(main, conf.launch_config.nproc_per_node, conf=conf)
```

In this example, the `setup_config` function is used to parse the command-line arguments and load the configuration file. The `run` function is then used to launch the distributed training.

#### Using the Dataset Sampler

The `get_data_sampler` function from the `distributed` module can be used to create a data sampler for your dataset. Here's an example:

```python
import torchvision
from irontorch import distributed as dist

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
sampler = dist.get_data_sampler(trainset, shuffle=True, distributed=distributed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler)
```

In this example, the `get_data_sampler` function is used to create a data sampler for the MNIST dataset. The sampler is then passed to the `DataLoader` to create the data loader.

### Contributing

We welcome contributions to the IronTorch library! If you would like to contribute, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear and concise commit messages.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.

### Running Tests

To run the tests for the IronTorch library, follow these steps:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the tests using `pytest`:
   ```
   pytest
   ```

The tests will be executed, and you will see the test results in the terminal.
