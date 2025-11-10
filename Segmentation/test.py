import torch
print(torch.cuda.is_available())  # should return True if GPU + CUDA ready
print(torch.cuda.get_device_name(0))  # name of first GPU
