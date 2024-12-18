import torch

print("Is CUDA available?:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current device index:", torch.cuda.current_device())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))