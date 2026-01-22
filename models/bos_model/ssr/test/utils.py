from typing import Union, Optional, List
import warnings
import ttnn
import torch
import copy
from bos_metal import helpers
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
    
    
def find_error(bbox_result):
    keys = []
    for k, v in bbox_result.items():
        try:
            copy.deepcopy(v)
        except Exception as e:
            keys.append(k)
            print(f"Error copying {k}: {e}")
            print(f"Type of {k}: {type(v)}")
            print(f"Value of {k}: {v}")
            print(f"Length of {k}: {len(v)}")
            if isinstance(v, torch.Tensor):
                print(f"Shape of {k}: {v.shape}")
            print()
            
    return keys


def deepcopy_with_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone().detach()  # Use clone for tensors
    elif isinstance(obj, list):
        return [deepcopy_with_tensors(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: deepcopy_with_tensors(value) for key, value in obj.items()}
    else:
        return copy.deepcopy(obj)
    
    
def extract_data_from_container(data):
    """Extract data from DataContainer."""
    data["img_metas"] = data["img_metas"][0].data
    # data["points"] = data["points"][0].data
    data["gt_bboxes_3d"] = data["gt_bboxes_3d"][0].data
    data["gt_labels_3d"] = data["gt_labels_3d"][0].data
    data["img"] = data["img"][0].data
    # data["fut_valid_flag"] = data["fut_valid_flag"][0].data
    data["ego_his_trajs"] = data["ego_his_trajs"][0].data
    data["ego_fut_trajs"] = data["ego_fut_trajs"][0].data
    # data["ego_fut_masks"] = data["ego_fut_masks"][0].data
    data["ego_fut_cmd"] = data["ego_fut_cmd"][0].data
    data["ego_lcf_feat"] = data["ego_lcf_feat"][0].data
    data["gt_attr_labels"] = data["gt_attr_labels"][0].data
    # data["gt_attr_labels"] = data["gt_attr_labels"][0]
    data["map_gt_labels_3d"] = data["map_gt_labels_3d"].data[0]
    data["map_gt_bboxes_3d"] = data["map_gt_bboxes_3d"].data[0]
    return data


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
                    debug: Optional[bool] = True,
                    pcc_thresh=0.99,
                    ) -> bool: # type: ignore
    
    if not debug: return True, None
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
        msg = "Loading tensors error, skip comparison."
        print(msg)
        return False, msg

    x = to_torch_tensor(x)
    y = to_torch_tensor(y)

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Inputs must be or convert to torch.Tensor")

    if permute:
        y = y.permute(*permute)
        
    if x.numel() != y.numel():
        msg = f"Numel mismatch: {x.numel()} vs {y.numel()}"
        print(msg)
        return False, msg

    return helpers.compare_tensors(x.to(dtype=y.dtype).view(y.shape), y, pcc=pcc_thresh)


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


def create_table(tensor, save_path="trajectory_comparison.csv"):
    # tools.display_dataframe_to_user(name="Trajectory Modes DataFrame", dataframe=df)
    # Prepare data for the comparison DataFrame
    comparison_rows = []

    for mode_id in range(6):
        for time_step in range(3):
            ours_x, ours_y = tensor[0, time_step, mode_id]
            ref_x, ref_y = tensor[1, time_step, mode_id]
            comparison_rows.append({
                "Mode": mode_id,
                "Time-step": time_step + 1,
                "Ours (x)": ours_x.item(),
                "Reference (x)": ref_x.item(),
                "Ours (y)": ours_y.item(),
                "Reference (y)": ref_y.item()
            })

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_rows)

    # Optional: sort by Mode then Time-step for clarity
    comparison_df.sort_values(by=["Mode", "Time-step"], inplace=True)
    
    # Insert a blank row between each mode for spacing
    spaced_rows = []
    previous_mode = -1
    for index, row in comparison_df.iterrows():
        if row["Mode"] != previous_mode and previous_mode != -1:
            spaced_rows.append({col: None for col in comparison_df.columns})  # blank row
        spaced_rows.append(row.to_dict())
        previous_mode = row["Mode"]

    # Create new DataFrame with spacing
    comparison_df_spaced = pd.DataFrame(spaced_rows)
    
    # tools.display_dataframe_to_user(name="Trajectory Comparison DataFrame", dataframe=comparison_df)

    print(f"Saved trajectory comparison to {save_path}")
    print(comparison_df)
    print('\n'*2)
    time.sleep(5)

import ttnn
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS

from bos_metal import op

__all__ = ["ReLU", "Sigmoid", "GELU"]


@ACTIVATION_LAYERS.register_module(name="ReLU_tt", force=True)
class ReLU(op.BaseModule):
    def __init__(self, inplace=False):
        super().__init__(requires_shape=False)

    def forward(self, inputs, memory_config=None):
        return ttnn.relu(inputs, memory_config=memory_config)


@ACTIVATION_LAYERS.register_module(name="Sigmoid_tt", force=True)
class Sigmoid(op.BaseModule):
    def __init__(self, inplace=False):
        super().__init__(requires_shape=False)

    def forward(self, inputs, memory_config=None):
        return ttnn.sigmoid(inputs, memory_config=memory_config)


@ACTIVATION_LAYERS.register_module(name="GELU_tt", force=True)
class GELU(op.BaseModule):
    def __init__(self, inplace=False):
        super().__init__(requires_shape=False)

    def forward(self, inputs, memory_config=None):
        return ttnn.gelu(inputs, memory_config=memory_config)