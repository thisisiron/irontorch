<p align="center">
  <img src=/assets/irontorch_text.png width=200>
</p>

--------------------------------------------------------------------------------

## Install Irontorch
```
pip install irontorch
```

## Example

```python
from irontorch import distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)

conf = dist.parse_and_load_config('test/config.yaml', parser)

...

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
sampler = dist.get_data_sampler(trainset, shuffle=True, distributed=distributed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler)
```


## References

[detectron2](https://github.com/facebookresearch/detectron2)
[ultralytics](https://github.com/ultralytics/ultralytics)
