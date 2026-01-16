<p align="center">
  <img src=/assets/irontorch_text.png width=200>
</p>

--------------------------------------------------------------------------------

PyTorch 분산 학습 및 학습 유틸리티 라이브러리

## 설치

```bash
pip install irontorch
```

## 빠른 시작

```python
import argparse
from irontorch import distributed as dist
from irontorch.utils import set_seed, GradScaler

def main(conf):
    set_seed(42, deterministic=True)

    # 학습 코드 작성
    ...

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="config.yaml")

conf = dist.setup_config(parser)
conf.distributed = conf.n_gpu > 1
dist.run(main, conf.launch_config.nproc_per_node, conf=conf)
```

## 모듈

### 1. 분산 학습 (Distributed Training)

#### 분산 환경 설정

```python
from irontorch import distributed as dist

# 설정 파싱 및 분산 환경 구성
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--batch_size", type=int, default=64)

conf = dist.setup_config(parser)
conf.distributed = conf.n_gpu > 1

# 분산 학습 실행
dist.run(main, conf.launch_config.nproc_per_node, conf=conf)
```

#### 분산 유틸리티 함수

```python
from irontorch import distributed as dist

# 현재 프로세스 정보
rank = dist.get_rank()              # 전체 rank
local_rank = dist.get_local_rank()  # 노드 내 rank
world_size = dist.get_world_size()  # 전체 프로세스 수

# 메인 프로세스 확인
if dist.is_primary():
    print("메인 프로세스에서만 실행")

# 프로세스 동기화
dist.synchronize()
```

#### 데이터 샘플러

```python
import torch
import torchvision
from irontorch import distributed as dist

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
sampler = dist.get_data_sampler(trainset, shuffle=True, distributed=conf.distributed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=sampler)
```

#### 분산 연산

```python
from irontorch import distributed as dist

# 딕셔너리 값 reduce (평균)
metrics = {"loss": 0.5, "accuracy": 0.9}
reduced = dist.reduce_dict(metrics, average=True)

# DataParallel 모델 언래핑
model = dist.upwrap_parallel(model)

# 병렬 모델 확인
if dist.is_parallel(model):
    print("DataParallel 또는 DistributedDataParallel 모델")
```

### 2. 학습 유틸리티 (Utils)

#### 시드 설정

```python
from irontorch.utils import set_seed

# 재현성을 위한 시드 설정
set_seed(42)

# 완전한 결정론적 학습 (속도 저하 있음)
set_seed(42, deterministic=True)
```

#### Gradient Scaler (Mixed Precision)

```python
from irontorch.utils import GradScaler

scaler = GradScaler(mixed_precision=True)

for data, target in dataloader:
    # backward + optimizer step + gradient clipping 통합
    scaler(
        loss=loss,
        optimizer=optimizer,
        parameters=model.parameters(),
        clip_grad=1.0,        # gradient clipping 값
        clip_mode="norm",     # "norm", "value", "agc"
        need_update=True
    )

# 체크포인트 저장/로드
state = scaler.state_dict()
scaler.load_state_dict(state)
```

#### Gradient Clipping

```python
from irontorch.utils import dispatch_clip_grad

# Gradient norm clipping (기본)
dispatch_clip_grad(model.parameters(), value=1.0, mode="norm")

# Gradient value clipping
dispatch_clip_grad(model.parameters(), value=0.5, mode="value")

# Adaptive Gradient Clipping (AGC)
dispatch_clip_grad(model.parameters(), value=0.01, mode="agc")
```

### 3. 로깅 및 실험 추적 (Logging & Tracking)

#### 로깅 설정

```python
import logging
from irontorch.recorder import setup_logging

# 로깅 초기화
setup_logging(log_file_path="experiment.log")

# 로거 사용
logger = logging.getLogger(__name__)
logger.info("학습 시작")
logger.warning("학습률이 높습니다")
```

#### WandB 실험 추적

```python
from irontorch.recorder import WandbLogger

# WandB 로거 초기화 (메인 프로세스에서만 활성화)
wandb_logger = WandbLogger(
    project="my-project",
    name="experiment-1",
    config={"lr": 0.001, "batch_size": 64},
    tags=["baseline", "v1"]
)

# 메트릭 로깅
for epoch in range(epochs):
    wandb_logger.log({"loss": loss, "accuracy": acc}, step=epoch)

# 학습 종료
wandb_logger.finish()
```

## 전체 학습 예제

```python
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from irontorch import distributed as dist
from irontorch.utils import set_seed, GradScaler
from irontorch.recorder import setup_logging, WandbLogger
import logging

def main(conf):
    # 시드 및 로깅 설정
    set_seed(42, deterministic=True)
    setup_logging(log_file_path="train.log")
    logger = logging.getLogger(__name__)

    # WandB 설정 (메인 프로세스만)
    wandb_logger = WandbLogger(
        project="mnist",
        config=vars(conf)
    )

    # 데이터 로드
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

    # 모델 및 옵티마이저
    model = nn.Linear(784, 10).cuda()
    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(mixed_precision=True)

    # 학습 루프
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

        if dist.is_primary():
            logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")
            wandb_logger.log({"loss": loss.item()}, step=epoch)

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

## 테스트 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 테스트 실행
pytest
```

## 기여하기

1. 저장소 Fork
2. 기능 브랜치 생성 (`git checkout -b feature/새기능`)
3. 변경사항 커밋 (`git commit -m 'feat: 새 기능 추가'`)
4. 브랜치에 Push (`git push origin feature/새기능`)
5. Pull Request 생성

## 라이선스

MIT License
