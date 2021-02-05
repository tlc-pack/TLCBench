# TLCBench

Benchmark scripts for TVM

## Content
- [Requirement](#requirement)
- [Intel CPU](#intel-cpu)
- [NVIDIA GPU](#nvidia-gpu)


## Requirement
Tested with  
TVM commit id: 91e07e1f3a7 (Feb. 5, 2021)  
mxnet==1.7.0  
gluonnlp==0.10.0  

## Intel CPU

### Results on AWS c5.9xlarge (Intel Xeon Platinum 8124m @ 3.00GHz 18-core)
- AutoTVM
```bash
-------------------------------------------------------------
Network Name       Batch size   Mean Inference Time (std dev)
-------------------------------------------------------------
resnet_50          1            5.40 ms             (0.08 ms)
mobilenet_v2       1            1.33 ms             (0.05 ms)
bert               1            31.31 ms            (0.11 ms)
-------------------------------------------------------------
```

- AutoScheduler
```bash
-------------------------------------------------------------
Network Name       Batch size   Mean Inference Time (std dev)
-------------------------------------------------------------
resnet_50          1            5.30 ms             (0.05 ms)
mobilenet_v2       1            0.91 ms             (0.02 ms)
bert               1            16.52 ms            (0.16 ms)
-------------------------------------------------------------
```


### Benchmark All Networks
The following commands read pre-tuned logs from directory `saved_logs/latest` and benchmark the latency for all networks.

- Commands for AutoTVM
```bash
python3 benchmark_autotvm.py --network all --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m" --logdir saved_logs/latest
```

- Commands for AutoScheduler
```bash
python3 benchmark_autoscheduler.py --network all --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m" --logdir saved_logs/latest
```

### Benchmark One Network
The following commands read pre-tuned logs from directory `saved_logs/latest` and benchmark the latency for one network.
You can replace "resnet_50" below with "mobilenet_v2" or "bert".

- Commands for AutoTVM
```bash
python3 benchmark_autotvm.py --network resnet_50 --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m" --logdir saved_logs/latest
```

- Commands for AutoScheduler
```bash
python3 benchmark_autoscheduler.py --network resnet_50 --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"  --logdir saved_logs/latest
```

### Tuning
The following commands perform auto-tuning for one or all networks and save tuning logs to directory `tmp_logs`.
After tuning, you can use these logs to run benchmark by using benchmark commands above and replace the last argument with `--logdir tmp_logs`

- Commands for AutoTVM
```bash
# Tune one network
python3 tune_autotvm.py --network resnet_50 --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"
# Tune all networks
python3 tune_autotvm.py --network all --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"
```

- Commands for AutoScheduler
```bash
# Tune one network
python3 tune_autoscheduler.py --network resnet_50 --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"
# Tune all networks
python3 tune_autoscheduler.py --network all --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"
```

## Nvidia GPU

### Results on AWS g4dn.4xlarge (NVIDIA T4)
- AutoTVM
```bash
-------------------------------------------------------------
Network Name       Batch size   Mean Inference Time (std dev)
-------------------------------------------------------------
resnet_50          1            3.54 ms             (0.02 ms)
mobilenet_v2       1            0.74 ms             (0.00 ms)
bert               1            89.06 ms            (1.22 ms)
-------------------------------------------------------------
```

- AutoScheduler
```bash
-------------------------------------------------------------
Network Name       Batch size   Mean Inference Time (std dev)
-------------------------------------------------------------
resnet_50          1            2.90 ms             (0.01 ms)
mobilenet_v2       1            0.57 ms             (0.00 ms)
bert               1            9.95 ms             (0.01 ms)
-------------------------------------------------------------
```


### Benchmark All Networks
The following commands read pre-tuned logs from directory `saved_logs/latest` and benchmark the latency for all networks.

- Commands for AutoTVM
```bash
python3 benchmark_autotvm.py --network all --target "cuda -model=t4" --logdir saved_logs/latest
```

- Commands for AutoScheduler
```bash
python3 benchmark_autoscheduler.py --network all --target "cuda -model=t4" --logdir saved_logs/latest
```

### Benchmark One Network
The following commands read pre-tuned logs from directory `saved_logs/latest` and benchmark the latency for one network.
You can replace "resnet_50" below with "mobilenet_v2" or "bert".

- Commands for AutoTVM
```bash
python3 benchmark_autotvm.py --network resnet_50 --target "cuda -model=t4" --logdir saved_logs/latest
```

- Commands for AutoScheduler
```bash
python3 benchmark_autoscheduler.py --network resnet_50 --target "cuda -model=t4"  --logdir saved_logs/latest
```

### Tuning
The following commands perform auto-tuning for one or all networks and save tuning logs to directory `tmp_logs`.
After tuning, you can use these logs to run benchmark by using benchmark commands above and replace the last argument with `--logdir tmp_logs`

- Commands for AutoTVM
```bash
# Tune one network
python3 tune_autotvm.py --network resnet_50 --target "cuda -model=t4"
# Tune all networks
python3 tune_autotvm.py --network all --target "cuda -model=t4"
```

- Commands for AutoScheduler
```bash
# Tune one network
python3 tune_autoscheduler.py --network resnet_50 --target "cuda -model=t4"
# Tune all networks
python3 tune_autoscheduler.py --network all --target "cuda -model=t4"
```

