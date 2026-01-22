import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Compare torch and ttnn pytorch files")
    parser.add_argument("a_file", nargs="?", type=str, default="", help="First file to compare")
    parser.add_argument("b_file", nargs="?", type=str, default="", help="Second file to compare")
    parser.add_argument("--trace", type=str, nargs="?", metavar="KEYWORD", default="", help="Keyword for breakpoint")
    parser.add_argument("--save_graph", action="store_true", help="Save all comparison factors as a graph")
    parser.add_argument("--save_tensors", type=str, nargs="*", metavar="KEYWORD", default=[], help="List of keywords for tensor names to save as images. ex) 'conv' or '0'")
    parser.add_argument("--save_tensor", type=str, default="", metavar="TENSOR_NAME", help="Exact single tensor names to save as images. ex) 'conv_0' or '0'")
    return parser.parse_args()


def torch_pcc(x, y):
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return [torch_pcc(x[i], y[i]) for i in range(len(x))]
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape} in {name}"

    x_mean = x.mean()
    y_mean = y.mean()
    x_diff = x - x_mean
    y_diff = y - y_mean
    numerator = torch.sum(x_diff * y_diff)
    denominator = torch.sqrt(torch.sum(x_diff**2)) * torch.sqrt(torch.sum(y_diff**2))
    pcc = numerator / (denominator + 1e-8)
    return pcc.item()


def torch_euclidean(x, y):
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return [torch_euclidean(x[i], y[i]) for i in range(len(x))]
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    euclidean = torch.sqrt(torch.sum((x - y) ** 2))
    return euclidean.item()


def torch_cosine(x, y):
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return [torch_cosine(x[i], y[i]) for i in range(len(x))]
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    cosine = torch.sum(x * y) / (torch.norm(x) * torch.norm(y) + 1e-8)
    return cosine.item()


def torch_mahalanobis(x, y):
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return [torch_mahalanobis(x[i], y[i]) for i in range(len(x))]
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    x_flat = x.flatten()
    y_flat = y.flatten()
    x_mean = x_flat.mean()
    y_mean = y_flat.mean()
    cov = torch.cov(torch.stack([x_flat, y_flat], dim=0)) + torch.eye(2) * 1e-8
    try:
        cov_inv = torch.inverse(cov)
    except:
        cov_inv = torch.pinverse(cov)
    diff = torch.tensor([x_mean - y_mean, x_mean - y_mean])
    mahalanobis = torch.sqrt(torch.matmul(torch.matmul(diff, cov_inv), diff))
    return mahalanobis.item()


def save_tensor_image(tensor: torch.Tensor, file_name: str, pixel_size: int = 4, max_batch: int = 1):
    """
    Visualize a 4D tensor and save as image.

    - Positive values → blue (dark for large values, light/white for near-zero)
    - Negative values → red (same mapping)
    - Only first `max_channels` channels per batch are shown
    - Saves to `file_name`

    Args:
        tensor (torch.Tensor): 4D or less tensor [N, C, H, W]
        file_name (str): Output image path
        pixel_size (int): Scaling factor per value
        max_channels (int): Max number of channels to show
    """
    assert tensor.ndim <= 4, "Tensor must be 4D or less"
    # if tensor.ndim == 3 and tensor.shape[0] == 1:
    #     C, H, W = tensor.shape
    #     mod_size = 20
    #     if mod_size and tensor.shape[-1] > mod_size:
    #         tensor = tensor.reshape(C, H, -1, mod_size)
    #         print(f"Tensor is too wide -> {tensor.shape}")
    while tensor.ndim < 4:
        tensor = tensor.unsqueeze(0)
    N, C, H, W = tensor.shape
    N = min(N, max_batch)
    tensor = tensor[:N, :]
    norm_tensor = tensor / tensor.abs().max()

    batch_border_thickness = 1 if N > 1 else 0
    channel_border_thickness = 1 if C > 1 else 0
    channel_grid_ncol = int(np.ceil(np.sqrt(C)))
    channel_grid_nrow = int(np.ceil(C / channel_grid_ncol))

    channel_grid_width = W * pixel_size
    channel_grid_height = H * pixel_size
    batch_grid_width = channel_grid_ncol * (channel_grid_width + channel_border_thickness) + channel_border_thickness
    batch_grid_height = channel_grid_nrow * (channel_grid_height + channel_border_thickness) + channel_border_thickness
    # total canvas size (stack batch grid in horizontal)
    canvas_width = N * (batch_grid_width + batch_border_thickness) + batch_border_thickness
    canvas_height = batch_grid_height + 2 * batch_border_thickness

    full_img = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))  # White background

    for n in range(N):
        bx_offsets = batch_border_thickness + n * (batch_grid_width + batch_border_thickness)
        by_offsets = batch_border_thickness
        for c in range(C):
            data = norm_tensor[n, c].numpy()
            abs_val = np.abs(data)
            alpha = (abs_val * 255).astype(np.uint8)

            R = np.where(data < 0, 255, 255 - alpha)  # negative is red
            G = 255 - alpha
            B = np.where(data > 0, 255, 255 - alpha)  # positive is blue
            img_array = np.stack([R, G, B], axis=-1).astype(np.uint8)

            img = Image.fromarray(img_array, mode="RGB")
            img = img.resize((channel_grid_width, channel_grid_height), resample=Image.NEAREST)

            # 채널 2D 그리드 위치
            c_row = c // channel_grid_ncol
            c_col = c % channel_grid_ncol
            ix = bx_offsets + c_col * (channel_grid_width + channel_border_thickness) + channel_border_thickness
            iy = by_offsets + c_row * (channel_grid_height + channel_border_thickness) + channel_border_thickness
            full_img.paste(img, (ix, iy))

            draw = ImageDraw.Draw(full_img)
            if channel_border_thickness > 0:
                draw.rectangle(
                    [
                        ix - channel_border_thickness,
                        iy - channel_border_thickness,
                        ix + channel_grid_width,
                        iy + channel_grid_height,
                    ],
                    outline="black",
                    width=channel_border_thickness,
                )

        if batch_border_thickness > 0:
            # 배치별 전체 그리드 테두리
            draw = ImageDraw.Draw(full_img)
            bx = bx_offsets - batch_border_thickness
            by = by_offsets - batch_border_thickness
            draw.rectangle(
                [
                    bx,
                    by,
                    bx + batch_grid_width + batch_border_thickness,
                    by + batch_grid_height + batch_border_thickness,
                ],
                outline="red",
                width=batch_border_thickness,
            )

    full_img.save(file_name)


def save_pcc_image(x_mat, y_mat, name):
    save_file_name = f"pcc_{name}.png"
    pcc = torch_pcc(x_mat, y_mat)
    x_mat = x_mat.cpu().numpy().reshape(-1)
    y_mat = y_mat.cpu().numpy().reshape(-1)
    plt.figure(figsize=(6, 6))
    plt.xlabel("torch")
    plt.ylabel("ttnn")
    plt.title(f"{name}\n PCC: {pcc:.4f}")
    plt.grid(True)
    plt.scatter(x_mat, y_mat)
    plt.savefig(save_file_name)
    plt.close()


if __name__ == "__main__":
    compare_list = {}
    args = parse_args()
    if args.a_file and args.b_file:
        if ".pt" in args.a_file and ".pt" in args.b_file:
            a_file = args.a_file
            b_file = args.b_file
            compare_list["0"] = (a_file, b_file)
        else:
            a_name = args.a_file.strip().strip(".") + "."
            b_name = args.b_file.strip().strip(".") + "."

    # if not compare_list:
    #     a_name = "refer_output."
    #     b_name = "ultra_output."
    #     all_files = os.listdir()
    #     for i in list(range(23)):
    #         a_file = a_name + str(i) + ".pt"
    #         b_file = b_name + str(i) + ".pt"
    #         if a_file in all_files and b_file in all_files:
    #             compare_list[f"{i}"] = (a_file, b_file)
    if not compare_list:
        layers = [
            "im",
            "conv_0",
            "conv_1",
            "c2f_2",
            "c2f_2.cv1_a",
            "c2f_2.cv1_b",
            "c2f_2.m0.cv1",
            "c2f_2.m0.cv2",
            "c2f_2.m0.add",
            "c2f_2.concat",
            "c2f_2.cv2",
            "conv_3",
            "c2f_4",
            "c2f_4.cv1_a",
            "c2f_4.cv1_b",
            "c2f_4.m0.cv1",
            "c2f_4.m0.cv2",
            "c2f_4.m0.add",
            "c2f_4.m1.cv1",
            "c2f_4.m1.cv2",
            "c2f_4.m1.add",
            "c2f_4.concat",
            "c2f_4.cv2",
            "conv_5",
            "c2f_6",
            "c2f_6.cv1_a",
            "c2f_6.cv1_b",
            "c2f_6.m0.cv1",
            "c2f_6.m0.cv2",
            "c2f_6.m0.add",
            "c2f_6.m1.cv1",
            "c2f_6.m1.cv2",
            "c2f_6.m1.add",
            "c2f_6.concat",
            "c2f_6.cv2",
            "conv_7",
            "c2f_8",
            "c2f_8.cv1_a",
            "c2f_8.cv1_b",
            "c2f_8.m0.cv1",
            "c2f_8.m0.cv2",
            "c2f_8.m0.add",
            "c2f_8.concat",
            "c2f_8.cv2",
            "sppf_9",
            "sppf_9.cv1",
            "sppf_9.maxpool0",
            "sppf_9.maxpool1",
            "sppf_9.maxpool2",
            "sppf_9.cv2",
            "upsample_10",
            "concat_11",
            "c2f_12",
            "c2f_12.cv1_a",
            "c2f_12.cv1_b",
            "c2f_12.m0.cv1",
            "c2f_12.m0.cv2",
            "c2f_12.m0.add",
            "c2f_12.concat",
            "c2f_12.cv2",
            "upsample_13",
            "concat_14",
            "c2f_15",
            "c2f_15.cv1_a",
            "c2f_15.cv1_b",
            "c2f_15.m0.cv1",
            "c2f_15.m0.cv2",
            "c2f_15.m0.add",
            "c2f_15.concat",
            "c2f_15.cv2",
            "conv_16",
            "concat_17",
            "c2f_18",
            "c2f_18.cv1_a",
            "c2f_18.cv1_b",
            "c2f_18.m0.cv1",
            "c2f_18.m0.cv2",
            "c2f_18.m0.add",
            "c2f_18.concat",
            "c2f_18.cv2",
            "conv_19",
            "concat_20",
            "c2f_21",
            "c2f_21.cv1_a",
            "c2f_21.cv1_b",
            "c2f_21.m0.cv1",
            "c2f_21.m0.cv2",
            "c2f_21.m0.add",
            "c2f_21.concat",
            "c2f_21.cv2",
            # "detect_22.in", # 1.0 when regen_input = True
            "detect_22.0.cv2",
            "detect_22.0.cv2.0",
            "detect_22.0.cv2.1",
            "detect_22.0.cv2.2",
            "detect_22.0.cv3",
            "detect_22.0.cv3.0",
            "detect_22.0.cv3.1",
            "detect_22.0.cv3.2",
            "detect_22.0.concat",
            "detect_22.1.cv2",
            "detect_22.1.cv2.0",
            "detect_22.1.cv2.1",
            "detect_22.1.cv2.2",
            "detect_22.1.cv3",
            "detect_22.1.cv3.0",
            "detect_22.1.cv3.1",
            "detect_22.1.cv3.2",
            "detect_22.1.concat",
            "detect_22.2.cv2",
            "detect_22.2.cv2.0",
            "detect_22.2.cv2.1",
            "detect_22.2.cv2.2",
            "detect_22.2.cv3",
            "detect_22.2.cv3.0",
            "detect_22.2.cv3.1",
            "detect_22.2.cv3.2",
            "detect_22.2.concat",
            "detect_22._inference.concat",
            "detect_22._inference.box",
            "detect_22._inference.cls",
            "detect_22._inference.dfl",
            "detect_22._inference.dfl.reshape",
            "detect_22._inference.dfl.softmax",
            "detect_22._inference.dfl.conv",
            "detect_22._inference.dfl.out",
            "detect_22._inference.dbox",
            "detect_22._inference.dbox.lt",
            "detect_22._inference.dbox.rb",
            "detect_22._inference.dbox.x1y1",
            "detect_22._inference.dbox.x2y2",
            "detect_22._inference.dbox.c_xy",
            "detect_22._inference.dbox.wh",
            "detect_22._inference.dbox_strides",
            "detect_22._inference.cls_sigmoid",
            "detect_22.y",
            "detect_22.y.0",
            "detect_22.y.1",
        ]
        for i, layer in enumerate(layers):
            a_file = "torch_" + str(layer) + ".pt"
            b_file = "ttnn_" + str(layer) + ".pt"
            compare_list[f"{layer}"] = (a_file, b_file)

    pcc_all = []
    for name, (a_file, b_file) in compare_list.items():
        try:
            a_tensor = torch.load(a_file)
            b_tensor = torch.load(b_file)
        except Exception as e:
            print(f"{name}: {e} -> skip")
            continue
        # assert a_tensor.shape == b_tensor.shape, breakpoint()
        if abs(a_tensor.dim() - b_tensor.dim()) == 1:
            if a_tensor.dim() > b_tensor.dim():
                print(f"** Warning ** Dimension mismatch for {name}: {a_tensor.shape} vs {b_tensor.shape}")
                a_tensor = a_tensor.squeeze(0)
            else:
                b_tensor = b_tensor.squeeze(0)
                print(f"** Warning ** Dimension mismatch for {name}: {a_tensor.shape} vs {b_tensor.shape}")

        print(f"{name}", torch_pcc(a_tensor, b_tensor))

        # print(
        #     f"{name}",
        #     torch_pcc(a_tensor, b_tensor),
        #     torch_euclidean(a_tensor, b_tensor),
        #     torch_cosine(a_tensor, b_tensor),
        #     torch_mahalanobis(a_tensor, b_tensor),
        # )

        pcc_all.append(
            [
                name,
                torch_pcc(a_tensor, b_tensor),
                torch_euclidean(a_tensor, b_tensor) / 1000,
                torch_cosine(a_tensor, b_tensor),
                torch_mahalanobis(a_tensor, b_tensor),
            ]
        )

        if args.save_tensors and any(keyword in name for keyword in args.save_tensors):
            print(f"Saving tensor images for {name} with shape {a_tensor.shape} and {b_tensor.shape}")
            save_tensor_image(a_tensor, f"{a_file}.png")
            save_tensor_image(b_tensor, f"{b_file}.png")
            save_pcc_image(a_tensor, b_tensor, name)
        if args.save_tensor == name:
            print(f"Saving tensor images for {name} with shape {a_tensor.shape} and {b_tensor.shape}")
            save_tensor_image(a_tensor, f"{a_file}.png")
            save_tensor_image(b_tensor, f"{b_file}.png")
            save_pcc_image(a_tensor, b_tensor, name)

        if args.trace and args.trace in name:
            breakpoint()

    if args.save_graph:
        df = pd.DataFrame(pcc_all, columns=["name", "pcc", "euclidean/1000", "cosine", "mahalanobis"])
        df.set_index("name", inplace=True)

        plt.figure(figsize=(24, 6))
        plt.plot(df.index, df["pcc"], marker="o", label="pcc")
        plt.plot(df.index, df["euclidean/1000"], marker="o", label="euclidean/1000")
        plt.plot(df.index, df["cosine"], marker="x", label="cosine")
        plt.plot(df.index, df["mahalanobis"], marker="+", label="mahalanobis")

        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("pcc_graph.png")
        plt.close()
