import torch
import sys
import os


def torch_pcc(x, y):
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return [torch_pcc(x[i], y[i]) for i in range(len(x))]
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

    x_mean = x.mean()
    y_mean = y.mean()
    x_diff = x - x_mean
    y_diff = y - y_mean
    numerator = torch.sum(x_diff * y_diff)
    denominator = torch.sqrt(torch.sum(x_diff**2)) * torch.sqrt(torch.sum(y_diff**2))
    pcc = numerator / (denominator + 1e-8)
    return pcc.item()


if __name__ == "__main__":
    compare_list = {}
    if len(sys.argv) > 1:
        if ".pt" in sys.argv[1] and ".pt" in sys.argv[2]:
            a_file = sys.argv[1]
            b_file = sys.argv[2]
            compare_list["0"] = (a_file, b_file)
        else:
            a_name = sys.argv[1].strip().strip(".") + "."
            b_name = sys.argv[2].strip().strip(".") + "."
    else:
        a_name = "refer_output."
        b_name = "ultra_output."

    if not compare_list:
        all_files = os.listdir()
        for i in list(range(23)):
            a_file = a_name + str(i) + ".pt"
            b_file = b_name + str(i) + ".pt"
            if a_file in all_files and b_file in all_files:
                compare_list[f"{i}"] = (a_file, b_file)

    for name, (a_file, b_file) in compare_list.items():
        a_tensor = torch.load(a_file)
        b_tensor = torch.load(b_file)
        print(f"{name}: ", torch_pcc(a_tensor, b_tensor))

        # if name == 'output.22':
        #     print(f'{name}[0]: ', torch_pcc(a_tensor[0], b_tensor[0]))
        #     print(f'{name}[1][0]: ', torch_pcc(a_tensor[1][0], b_tensor[1][0]))
        #     print(f'{name}[1][1]: ', torch_pcc(a_tensor[1][1], b_tensor[1][1]))
        #     print(f'{name}[1][2]: ', torch_pcc(a_tensor[1][2], b_tensor[1][2]))
        # else:
        #     a_tensor = torch.load(a_file)
        #     b_tensor = torch.load(b_file)
