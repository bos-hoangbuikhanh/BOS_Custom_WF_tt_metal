from typing import Union, Optional, List
import warnings
import ttnn
import torch
import copy
from bos_metal import helpers, device_box
import numpy as np
import pandas as pd
import time
# import ace_tools as tools


def to_l1(input):
    """convert input to l1 memory config."""
    if isinstance(input, list):
        return [to_l1(x) for x in input]
    elif isinstance(input, dict):
        return {k: to_l1(v) for k, v in input.items()}
    elif isinstance(input, ttnn.Tensor):
        return ttnn.to_memory_config(input, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        warnings.warn(f"unsupported type: {type(input)}. return the input as is.")
        return input
    
def to_dram(input):
    """convert input to DRAM memory config."""
    if isinstance(input, list):
        return [to_dram(x) for x in input]
    elif isinstance(input, dict):
        return {k: to_dram(v) for k, v in input.items()}
    elif isinstance(input, ttnn.Tensor):
        return ttnn.to_memory_config(input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        warnings.warn(f"unsupported type: {type(input)}. return the input as is.")
        return input
    

def tt2pt(input, dtype=None):
    """Convert ttnn tensor(s) to pytorch tensor(s)."""
    if isinstance(input, list):
        return [tt2pt(x, dtype=dtype) for x in input]
    elif isinstance(input, dict):
        return {k: tt2pt(v, dtype=dtype) for k, v in input.items()}
    elif isinstance(input, ttnn.Tensor):
        if ttnn.is_sharded(input):
            input = ttnn.sharded_to_interleaved(input)
        return ttnn.to_torch(input, dtype=dtype)
    else:
        warnings.warn(f"Unsupported type: {type(input)}. Return the input as is.")
        return input
    
    
def pt2tt(input, dtype=ttnn.bfloat16, tile=None, pad_value=None, layout=ttnn.TILE_LAYOUT, device=None, memory_config=None, mesh_mapper=None):
    """Convert pytorch tensor(s) to ttnn tensor(s)."""
    assert tile is None, "tile is not supported yet."
    assert pad_value is None, "pad_value is not supported yet."
    assert mesh_mapper is None, "mesh_mapper is not supported yet." 
    
    # Convert positional arguments to kwargs
    kwargs = {
        'dtype': dtype,
        'tile': tile,  # tile is not supported yet, but keep it for API compatibility
        'pad_value': pad_value,  # pad_value is not supported yet, but keep it for API compatibility
        'layout': layout,
        'device': device,
        'memory_config': memory_config,
        'mesh_mapper': mesh_mapper,  # mesh_mapper is not supported yet, but keep it for API compatibility
    } 
     
    if isinstance(input, list):
        return [pt2tt(x, **kwargs) for x in input]
    elif isinstance(input, dict):
        return {k: pt2tt(v, **kwargs) for k, v in input.items()}
    elif isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        return ttnn.from_torch(input, **kwargs)
    else:
        warnings.warn(f"Unsupported type: {type(input)}. Return the input as is.")
        return input


def deepcopy_with_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone().detach()  # Use clone for tensors
    elif isinstance(obj, list):
        return [deepcopy_with_tensors(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: deepcopy_with_tensors(value) for key, value in obj.items()}
    else:
        return copy.deepcopy(obj)


def setup_host_input(torch_input, img_config, cq_id=0):
    """
    Split the original tensor into a list of 6 sub-tensors corresponding to the images.
        - Each image is in host and is reshaped to (1, 1, H*W, C).
        - The image channel is padded, required for Persistent L1-Sharding).
    """

    # Process only 1 batch, to channel last
    torch_input = torch_input[0].permute(0, 2, 3, 1)

    # Reshape and padding
    input_list = []
    for i in range(torch_input.shape[0]):
        # Single batch dimension
        inp = torch_input[i : i + 1]
        inp = ttnn.from_torch(
            inp,
            dtype=img_config["dtype"],
            layout=img_config["layout"],
            cq_id=cq_id,
        )
        inp = ttnn.reshape(inp, (1, 1, inp.shape[0] * inp.shape[1] * inp.shape[2], inp.shape[3]))
        inp = ttnn.pad(inp, [inp.shape[0], inp.shape[1], inp.shape[2], img_config["padding"]], [0, 0, 0, 0], 0)
        input_list.append(inp)

    return input_list


def setup_dram_sharded_config(ttnn_host_input, device):
    def divup(a, b):
        return (a + b - 1) // b

    # Memory config
    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(ttnn_host_input.volume() // ttnn_host_input.padded_shape[-1], (dram_grid_size.x * dram_grid_size.y)),
            ttnn_host_input.padded_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)

    return mem_config


def setup_l1_sharded_config(ttnn_host_input, device):
    def divup(a, b):
        return (a + b - 1) // b

    # Memory config
    l1_grid_size = device.compute_with_storage_grid_size()
    mem_config = ttnn.create_sharded_memory_config(
        shape=(
            divup(ttnn_host_input.volume() // ttnn_host_input.padded_shape[-1], l1_grid_size.x * l1_grid_size.y),
            ttnn_host_input.padded_shape[-1],
        ),
        core_grid=ttnn.CoreGrid(x=l1_grid_size.x, y=l1_grid_size.y),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    return mem_config


def setup_dram_input(ttnn_tensors, img_config, device, cq_id=0):
    # Create DRAM resident for input tensors
    num_cams = img_config["num_cams"]

    # Setup tensor config
    mem_config = setup_dram_sharded_config(ttnn_tensors[0], device)

    # Allocate persistent input on device
    input_device = [ttnn_tensors[i].to(device, mem_config, cq_id=cq_id) for i in range(num_cams)]

    return input_device


def extract_data_from_container(data, tensor="tt", output_storage=None, device=None, input_config=None, cq_id=0, **kwargs):
    if tensor == "tt":
        assert device is not None, "Device must be provided to extract data for TT Model"
        assert input_config is not None, "Input configuration must be provided to extract data for TT Model"

        data["img"] = data["img"][0].data
        data["img_metas"] = data["img_metas"][0].data
        data["ego_fut_cmd"] = data["ego_fut_cmd"][0].data

        img_config = input_config["img_config"]
        img_metas_config = input_config["img_metas_config"]

        # Convert images to ttnn tensor
        img_host = setup_host_input(data["img"][0], img_config, cq_id=cq_id)

        # Convert Image metas to ttnn tensor
        can_bus_config = img_metas_config["can_bus"]
        lidar2img_config = img_metas_config["lidar2img"]
        img_metas_data = data["img_metas"][0][0]

        ## Convert lidar2img to ttnn tensor list
        lidar2img_tt = []
        lidar2img_dram_storage = []
        for lidar2img in img_metas_data["lidar2img"]:
            lidar2img_tt.append(
                ttnn.from_torch(
                    tensor=torch.Tensor(lidar2img),
                    dtype=lidar2img_config["dtype"],
                    layout=lidar2img_config["layout"],
                )
            )

        ## Convert can_bus to ttnn tensor
        can_bus_tt = ttnn.from_torch(
            tensor=img_metas_data["can_bus"],
            dtype=can_bus_config["dtype"],
            layout=can_bus_config["layout"],
        )
        
        # Copy host to device
        # Images 
        ## Initialize DRAM input storage for images if not provided
        img_dram = setup_dram_input(img_host, img_config, device, cq_id=cq_id) if output_storage is None else output_storage["img"][0]
        ## Copy to device
        for i in range(img_config['num_cams']):
            ttnn.copy_host_to_device_tensor(
                img_host[i],
                img_dram[i],
                cq_id=cq_id
            )
        data["img"][0] = img_dram

        # lidar2img
        ## Initialize DRAM input storage for lidar2img if not provided
        if output_storage is None:
            for l2i in lidar2img_tt:
                lidar2img_dram_storage.append(
                    ttnn.to_device(
                        tensor=l2i,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        queue_id=cq_id
                    )
                )
        else:
            lidar2img_dram_storage = output_storage["img_metas"][0][0]["lidar2img"]
        ## Copy to device
        for i, l2i in enumerate(lidar2img_tt):
            ttnn.copy_host_to_device_tensor(
                l2i,
                lidar2img_dram_storage[i],
                cq_id=cq_id
            )

        # can_bus
        ## Initialize DRAM input storage for can_bus if not provided
        if output_storage is None:
            can_bus_dram = ttnn.to_device(
                tensor=can_bus_tt,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                queue_id=cq_id
            )
        else:
            can_bus_dram = output_storage["img_metas"][0][0]["can_bus"]
        ## Copy to device
        ttnn.copy_host_to_device_tensor(
            can_bus_tt,
            can_bus_dram,
            cq_id=cq_id
        )

        # Update data dictionary
        data["img_metas"][0][0]["can_bus"] = can_bus_dram
        data["img_metas"][0][0]["lidar2img"] = lidar2img_dram_storage

        return data
    elif tensor == "pt":
        """Extract data from DataContainer."""
        data["img_metas"] = data["img_metas"][0].data if "img_metas" in data else None
        data["gt_bboxes_3d"] = data["gt_bboxes_3d"][0].data if "gt_bboxes_3d" in data else None
        data["gt_labels_3d"] = data["gt_labels_3d"][0].data if "gt_labels_3d" in data else None
        data["img"] = data["img"][0].data if "img" in data else None
        # data["points"] = data["points"][0].data
        # data["fut_valid_flag"] = data["fut_valid_flag"][0].data
        data["ego_his_trajs"] = data["ego_his_trajs"][0].data if "ego_his_trajs" in data else None
        data["ego_fut_trajs"] = data["ego_fut_trajs"][0].data if "ego_fut_trajs" in data else None
        # data["ego_fut_masks"] = data["ego_fut_masks"][0].data
        data["ego_fut_cmd"] = data["ego_fut_cmd"][0].data if "ego_fut_cmd" in data else None
        data["ego_lcf_feat"] = data["ego_lcf_feat"][0].data if "ego_lcf_feat" in data else None
        data["gt_attr_labels"] = data["gt_attr_labels"][0].data if "gt_attr_labels" in data else None
        # data["gt_attr_labels"] = data["gt_attr_labels"][0]
        data["map_gt_labels_3d"] = data["map_gt_labels_3d"].data[0] if "map_gt_labels_3d" in data else None
        data["map_gt_bboxes_3d"] = data["map_gt_bboxes_3d"].data[0] if "map_gt_bboxes_3d" in data else None

        return data
    else:
        raise RuntimeError(f"Undefined tensor type for dataloader extraction: {tensor}")


def preprocess_dataloader(dataloader):
    """Preprocess all samples in the dataloader."""
    preprocessed_data = []
    for data in dataloader:
        preprocessed_data.append(extract_data_from_container(data))
    return preprocessed_data



def compare_tensors(x: Union[str, torch.Tensor, ttnn.Tensor], # type: ignore
                    y: Union[str, torch.Tensor, ttnn.Tensor],   # type: ignore
                    message: Optional[str] = None,
                    permute: Optional[List[int]] = None,
                    debug: Optional[bool] = True) -> bool: # type: ignore
    if not debug: return True
    if message is not None:
        print(message, end=' ')
    
    def load_tensor(obj):
        if isinstance(obj, str):
            if obj.endswith(".pt"):
                return torch.load(obj)
            elif obj.endswith(".tt"):
                return ttnn.load_tensor(obj)
            else:
                raise ValueError(f"Unknown file type for: {obj}")
        return obj

    def to_torch_tensor(obj):
        if isinstance(obj, ttnn.Tensor):
            return tt2pt(obj)
        return obj

    try:
        x = load_tensor(x)
        y = load_tensor(y)
    except:
        print("Loading tensors error, skip comparison.")
        return False

    x = to_torch_tensor(x)
    y = to_torch_tensor(y)

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Inputs must be or convert to torch.Tensor")

    if permute:
        y = y.permute(*permute)
        
    if x.numel() != y.numel():
        print(f"Numel mismatch: {x.numel()} vs {y.numel()}")
        return False 
     
    return helpers.compare_tensors(x.to(dtype=y.dtype).view(y.shape), y)


def pad_to_multiple(tensor: ttnn.Tensor,    # type: ignore
                    multiple: int,
                    axis: int,
                    pad_value=0):
    length = tensor.shape[axis]
    pad = (multiple - length % multiple) % multiple
    if pad == 0:
        return tensor, 0
    pad_shape = list(tensor.shape)
    pad_shape[axis] = pad
    extra = ttnn.full(tuple(pad_shape), pad_value,
                      dtype=tensor.dtype,
                      layout=tensor.layout,
                      device=tensor.device(),
                      memory_config=tensor.memory_config())
    tensor = ttnn.concat([tensor, extra], dim=axis,
                         memory_config=tensor.memory_config())
    return tensor, pad


def masked_fill(tensor, mask, value):
    """Fills elements of the input tensor with value where mask is True."""
    # mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
    if isinstance(mask, torch.Tensor):
        mask = pt2tt(mask, device=tensor.device())
    # tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    mask_value = mask * value
    not_mask = ttnn.logical_not(mask)
    mask_tensor = ttnn.add(
        ttnn.multiply(tensor, not_mask),
        mask_value,
        # memory_config=None
    )
    mask_tensor = tensor * not_mask + mask_value
    
    return mask_tensor

