import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist_torch
import torch.multiprocessing as mp
import os

import irontorch
print(irontorch.__version__)
from irontorch import distributed as dist
from irontorch.utils import GradScaler, set_seed


def cleanup():
    # 프로세스 그룹 정리
    dist_torch.destroy_process_group()


def load_data(batch_size=64, distributed=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=dist.get_data_sampler(trainset, shuffle=True, distributed=distributed))

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=dist.get_data_sampler(testset, shuffle=False, distributed=distributed))

    return trainloader, testloader


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(conf):
    set_seed()

    conf.distributed = conf.n_gpu > 1
    rank = dist.get_rank()
    print(f'Rank: {rank}')
    trainloader, testloader = load_data(distributed=conf.distributed)
    
    device = torch.device(f'cuda:{rank}')
    model = SimpleNN().to(device)
    if conf.distributed:
        print('DDP running!!!')
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9)
    grad_scaler = GradScaler(mixed_precision=True)
    
    for epoch in range(conf.epochs):
        model.train()
        running_loss = 0.0
        # trainloader.sampler.set_epoch(epoch)
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            grad_scaler(loss, optimizer)
            # loss.backward()
            # optimizer.step()
            running_loss += loss.item()
        print(f"Rank {rank}, Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

    # print(f"Rank {rank}, Model Parameters: {list(model.state_dict().items())[:2]}")
    test_model(rank, model, testloader, conf.distributed)
    # dist.synchronize()
    # if conf.distributed:
    #     cleanup()
    # exit(0)
    

@torch.no_grad()
def test_model(rank, model, testloader, distributed=False):
    device = torch.device(f'cuda:{rank}')
    model.eval()
    correct = torch.tensor(0).to(rank)
    total = torch.tensor(0).to(rank)

    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Rank {rank}, Accuracy: {100 * correct / total}%")

    if distributed:
        # Gather all the correct and total counts from all ranks
        dist_torch.all_reduce(correct, op=dist_torch.ReduceOp.SUM)
        dist_torch.all_reduce(total, op=dist_torch.ReduceOp.SUM)
        accuracy = 100 * correct.item() / total.item()
        if rank == 0:
            print(f"Accuracy: {accuracy}%")
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)

    conf = dist.parse_and_load_config('test/config.yaml', parser)

    try:
        dist.run(train_model, conf.launch_config.nproc_per_node, conf=conf)
    except Exception as e:
        print('err', e)
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
