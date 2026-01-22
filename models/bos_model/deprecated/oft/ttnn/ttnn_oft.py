import torch.nn.functional as F

import ttnn

try:
    from tracy import signpost

    use_signpost = True

except ModuleNotFoundError:
    use_signpost = False

EPSILON = 1e-6


class TtOft:
    def __init__(self, device, parameters, cell_size, grid_height, scale=1):
        self.linear_weight = parameters.conv3d.weight
        self.linear_bias = parameters.conv3d.bias
        self.inv_scale = int(1 / scale)

        self.device = device

    def __call__(self, features, pre_config=None):
        if use_signpost:
            signpost(header=f"Oft{self.inv_scale}")

        if pre_config is None:
            raise ValueError("Preprocessed config must be passed")

        bbox_corners = pre_config["bbox_corners"]  # In torch - used in gridsample
        area = pre_config["area"]
        visible = pre_config["visible"]
        batch, _, depth, width, _ = pre_config["bbox_corners_shape"]

        # Preprocess above till here...
        integral_img = integral_image(features)
        integral_img = ttnn.to_torch(integral_img)

        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]]).reshape([batch, -1, depth * width])
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]]).reshape([batch, -1, depth * width])
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]]).reshape([batch, -1, depth * width])
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]]).reshape([batch, -1, depth * width])

        top_left = ttnn.from_torch(top_left, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        btm_right = ttnn.from_torch(btm_right, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        top_right = ttnn.from_torch(top_right, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        btm_left = ttnn.from_torch(btm_left, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)

        vox_feats = ttnn.add(top_left, btm_right)
        ttnn.deallocate(top_left)
        ttnn.deallocate(btm_right)
        vox_feats = ttnn.subtract(vox_feats, top_right)
        ttnn.deallocate(top_right)
        vox_feats = ttnn.subtract(vox_feats, btm_left)
        ttnn.deallocate(btm_left)
        vox_feats = ttnn.reshape(vox_feats, (batch, -1, 7, depth * width))
        vox_feats = ttnn.div(vox_feats, area, dtype=ttnn.float32)
        ttnn.deallocate(area)

        vox_feats = ttnn.multiply(vox_feats, visible, dtype=ttnn.float32)
        ttnn.deallocate(visible)

        vox_feats = ttnn.permute(vox_feats, (0, 3, 1, 2))
        vox_feats = ttnn.reshape(
            vox_feats, (-1, vox_feats.shape[2] * vox_feats.shape[3]), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ortho_feats = ttnn.linear(vox_feats, self.linear_weight, bias=self.linear_bias)
        ttnn.deallocate(vox_feats)

        ortho_feats = ttnn.reshape(ortho_feats, (batch, depth, width, -1))
        ortho_feats = ttnn.permute(ortho_feats, (0, 3, 1, 2))
        ortho_feats = ttnn.relu(ortho_feats)

        return ortho_feats


def integral_image(features):
    return ttnn.cumsum(ttnn.cumsum(features, dim=-1), dim=-2)
