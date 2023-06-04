import torch

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA Available:", torch.cuda.is_available())

cuda_available = torch.cuda.is_available()
if cuda_available:
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print("Number of CUDA devices:", device_count)
    print("Current CUDA device:", current_device)
    print("Current CUDA device name:", device_name)


print("Device:", device)

# Create a tensor on the GPU
tensor = torch.tensor([1, 2, 3]).to(device)

# Perform tensor operations on the GPU
result = tensor + 2

# Move the result back to CPU if needed
result = result.to("cpu")

# Print the result
print("Result:", result)
