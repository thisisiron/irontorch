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



### Using the Logging System

The improved logging system is configured via `logging_config.yaml`. You can initialize it in your main script and get logger instances:

```python
import logging
from irontorch.recorder import setup_logging

# Setup logging (ideally at the start of your application)
# You can optionally provide a path to a custom config file
# or override the log file path.
setup_logging(log_file_path="my_experiment.log")

# Get a logger instance for your module
logger = logging.getLogger("my_module")

# Log messages
logger.debug("This is a debug message (will go to file by default).")
logger.info("Starting the training process...")
logger.warning("Learning rate seems high.")

try:
    # Simulate an error
    result = 1 / 0
except ZeroDivisionError:
    logger.exception("An error occurred during calculation!") # Logs exception with traceback

logger.info("Training finished.")
```

This setup provides:
*   Console logging using `rich` for better readability (default level INFO).
*   File logging in JSON format to `irontorch.log` (or the path specified in `setup_logging`) with rotation (default level DEBUG).
*   Configuration via `logging_config.yaml` for easy customization.

#### Ensuring Reproducibility with `set_seed`

The `irontorch.utils.helper.set_seed(seed: int, deterministic: bool = False)` function is available to help ensure reproducibility in your experiments.

```python
from irontorch.utils.helper import set_seed

# Set a global seed for random, numpy, and torch
set_seed(42)

# For stricter determinism with PyTorch >= 2.0.0
set_seed(42, deterministic=True)
```

When `deterministic=True` is passed, and if you are using PyTorch version 2.0.0 or newer, `set_seed` will also configure `torch.use_deterministic_algorithms(True)`. This setting is stricter than the default and will cause PyTorch to raise an error if a non-deterministic CUDA algorithm is encountered, rather than just issuing a warning. This helps ensure that your operations are fully deterministic when required. For older PyTorch versions, or when `deterministic=False`, it still sets seeds for `random`, `numpy`, and `torch`, but PyTorch's behavior regarding deterministic algorithms will depend on its version and default settings.

#### Integrating with Weights & Biases using `WandbLogger`

IronTorch also includes `WandbLogger` for easy integration with Weights & Biases, available in `irontorch.recorder.recorder`.

```python
from irontorch.recorder import WandbLogger
# from irontorch import distributed as dist # Assuming you use irontorch.distributed

# Example initialization (typically only on the primary process)
# if dist.is_primary():
#     wandb_logger = WandbLogger(project="my_awesome_project", name="experiment_run_1", config={"learning_rate": 0.01})
#     # Log metrics during your training loop
#     # wandb_logger.log({"loss": 0.5, "accuracy": 0.9}, step=100)
#     # ...
#     # wandb_logger.finish() # finish is called automatically by __del__ or can be called explicitly
```

The `WandbLogger` facilitates the initialization of a WandB run and logging of metrics and configurations to your WandB dashboard. A key feature is its robust `finish` method: it incorporates enhanced error handling to gracefully manage and log any potential exceptions that might occur during the finalization of the WandB run. This prevents issues like unsynchronized data or abrupt crashes at the end of training due to WandB communication problems, providing clearer feedback on such occurrences.

