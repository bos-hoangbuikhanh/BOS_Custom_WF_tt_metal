import os
import torch

def load_tensors_to_dict(directory):
    """Load all .pt files in the given directory into a dictionary."""
    tensor_dict = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".pt"):
            file_path = os.path.join(directory, file_name)
            tensor_name = os.path.splitext(file_name)[0]  # Use the file name (without extension) as the key
            tensor_dict[tensor_name] = torch.load(file_path).detach().clone()
    return tensor_dict

# Specify the directory containing the .pt files
directory = os.getcwd()  # Current working directory
tensor_dict = load_tensors_to_dict(directory)

# Save the dictionary to a file
torch.save(tensor_dict, "tensor_dict.pth")
print("All tensors have been loaded into a dictionary and saved as 'tensor_dict.pth'.")
