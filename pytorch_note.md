github: pytorch_segmentation_tool is ok, but it is a bit old. Build may encounter some error (cuda architecture is different: build.sh) (Cuda 8.0 is not feasible, Cuda 9.0 is fine)


# Synchronized-BN (https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
When uses `nn.DataParallel` to wrap the network during training, normal bn computes the statistics in each device, which is inaccruate.

To use Sync-BN, we add a data parallel replication callback.

```
from sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback

sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
```

https://github.com/CSAILVision/semantic-segmentation-pytorch (worthy trying) https://zhuanlan.zhihu.com/p/69940683
Put mode into `nn.DataParallel` interface, the real batch size for each GPU will be divided by the number of GPUs.
在forward阶段，当前GPU上的module会被复制到其他GPU上，输入数据则会被切分，分别传到不同的GPU上进行计算；在backward阶段，每个GPU上的梯度会被求和并传回当前GPU上，并更新参数。
(只有当前一个GPU在backward的时候更新gradient，其他的都是复制当前GPU的参数)
也就是复制module -> forward -> 计算loss -> backward -> 汇总gradients -> 更新参数 -> 复制module -> ...的不断重复执行

SyncBN will compute 'sum & square sum' for each device, and reduce (collect all them to compute mean & std in main device), then broadcast the statistics to the other devices.



_For offical pytorch implementation_
https://github.com/dougsouza/pytorch-sync-batchnorm-example
We can not use `nn.SyncBN` when using `nn.DataParallel`. We need to use `torch.parallel.DistributedDataParallel(...)` with multi-processing single GPU.
We need to launch a separate process for each GPU.

Basic Idea: We'll launch one process for each GPU. Our training script will be provided a `rank` argument, which is simply an integer that tells us which process is being launched. 
`rank 0` is our master. This way we can control what each process do. For example, we may want to print losses and stuff to the console only on the master process.

```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)  # local rank is set for every single GPU

# we need to init the process
torch.cuda.set_device(args.local_rank) 

world_size = args.ngpu
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=args.local_rank,
)

...
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # convert BatchNorm layer to SyncBatchNorm before wrapping Network with DDP.
device = torch.device('cuda:{}'.format(args.local_rank))
net = net.to(device)

for it, (input, target) in enumerate(self.data_loader):
    input, target = input.to(device), target.to(device)  # do the same for the inputs of the model

# wraping you model with distributeddataparallel
net = torch.nn.parallel.DistributedDataParallel(
    net,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
)


# adapting your dataloader
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=config.ngpu,
    rank=local_rank,
)
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
    sampler=sampler,
    drop_last=True,
)

# launching the process
python -m torch.distributed.launch --nproc_per_node=3 distributed_train.py \
--arg1=arg1 --arg2=arg2 --arg3=arg3 --arg4=arg4 --argn=argn

```
A test: https://github.com/AndyYuan96/pytorch-distributed-training/blob/master/multi_gpu_distributed.py

https://www.zhihu.com/question/67209417  pytorch bug (load to the model to cpu first)


# Transformer 
https://towardsdatascience.com/divide-hugging-face-transformers-training-time-by-2-or-more-21bf7129db9q-21bf7129db9e  (Uniform length batching for reduing the training time)

Training NN on a batch of sequences requires them to have the exact same length to build the batch matrix representation.   
Dynamic Padding: we limit the number of added pad tokens to reach the length of the longest sequence of each mini batch instead of a fixed value set for the whole set.    
Uniform length batching: we generate batches made of similar length sequences, so we avoid extreme cases where most sequences in the mini batch are short and we are required to add lots of pad tokens to each of them because few of them are very long.

https://www.bilibili.com/video/BV1sU4y1G7CN (general variants)   
Module-level: positional encoding, attentions (with prior), multi-head   
Pre-trained Models: encoder, decoder,   
Applications: Text, Vision, Audio, multi-modal, etc,






