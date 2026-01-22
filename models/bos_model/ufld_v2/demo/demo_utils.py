# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import ttnn


def loader_func(path):
    return Image.open(path)


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1.0, 0.0)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception("Format of lanes error.")
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0.0, 0.0, 1.0
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0.0, 0.0
        matched = 0.0
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.0
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return (
            s / max(min(4.0, len(gt)), 1.0),
            fp / len(pred) if len(pred) > 0 else 0.0,
            fn / max(min(len(gt), 4.0), 1.0),
        )

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        json_pred = [json.loads(line) for line in open(pred_file, "r").readlines()]
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception("We do not get the predictions of all the test tasks")
        gts = {l["raw_file"]: l for l in json_gt}
        accuracy, fp, fn = 0.0, 0.0, 0.0
        for pred in json_pred:
            if "raw_file" not in pred or "lanes" not in pred or "run_time" not in pred:
                raise Exception("raw_file or lanes or run_time not in some predictions.")
            raw_file = pred["raw_file"]
            pred_lanes = pred["lanes"]
            run_time = pred["run_time"]
            if raw_file not in gts:
                raise Exception("Some raw_file from your predictions do not exist in the test tasks.")
            gt = gts[raw_file]
            gt_lanes = gt["lanes"]
            y_samples = gt["h_samples"]
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception("Format of lanes error.")
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        pr = 1 - fp / num
        re = 1 - fn / num
        if (pr + re) == 0:
            f1 = 0
        else:
            f1 = 2 * pr * re / (pr + re)
        return json.dumps(
            [
                {"name": "Accuracy", "value": accuracy / num, "order": "desc"},
                {"name": "FP", "value": fp / num, "order": "asc"},
                {"name": "FN", "value": fn / num, "order": "asc"},
                {"name": "F1", "value": f1, "order": "asc"},
            ]
        )


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_transform=None, crop_size=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.crop_size = crop_size
        self.image_files = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if os.path.isfile(os.path.join(self.path, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __getitem__(self, index):
        img_path = self.image_files[index]
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)
        img = img[:, -self.crop_size :, :]

        return img, os.path.basename(img_path)

    def __len__(self):
        return len(self.image_files)


def get_test_loader(batch_size, data_root, dataset, distributed, crop_ratio, train_width, train_height):
    if dataset == "Tusimple":
        img_transforms = transforms.Compose(
            [
                transforms.Resize((int(train_height / crop_ratio), train_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_dataset = LaneTestDataset(
            os.path.join(data_root, "images"),
            img_transform=img_transforms,
            crop_size=train_height,
        )
    else:
        raise NotImplementedError

    sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return loader


def generate_tusimple_lines(row_out, row_ext, col_out, col_ext, row_anchor=None, col_anchor=None, mode="2row2col"):
    tusimple_h_sample = np.linspace(160, 710, 56)
    row_num_grid, row_num_cls, row_num_lane = row_out.shape
    row_max_indices = row_out.argmax(0).cpu()
    row_valid = row_ext.argmax(0).cpu()
    row_out = row_out.cpu()

    col_num_grid, col_num_cls, col_num_lane = col_out.shape
    col_max_indices = col_out.argmax(0).cpu()
    col_valid = col_ext.argmax(0).cpu()
    col_out = col_out.cpu()

    if mode == "normal" or mode == "2row2col":
        row_lane_list = [1, 2]
        col_lane_list = [0, 3]
    elif mode == "4row":
        row_lane_list = range(row_num_lane)
        col_lane_list = []
    elif mode == "4col":
        row_lane_list = []
        col_lane_list = range(col_num_lane)
    else:
        raise NotImplementedError

    local_width_row = 14
    local_width_col = 14
    min_lanepts_row = 3
    min_lanepts_col = 3

    all_lanes = []

    for row_lane_idx in row_lane_list:
        if row_valid[:, row_lane_idx].sum() > min_lanepts_row:
            cur_lane = []
            for row_cls_idx in range(row_num_cls):
                if row_valid[row_cls_idx, row_lane_idx]:
                    ## === confidence filtering : START ===#
                    prob = row_out[:, row_cls_idx, row_lane_idx].softmax(0)
                    best_p = prob.max().item()
                    if best_p < 0.25:  # grid class confidence threshold
                        cur_lane.append(-2)
                        continue
                    # === confidence filtering : END ===#

                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, row_max_indices[row_cls_idx, row_lane_idx] - local_width_row),
                                min(row_num_grid - 1, row_max_indices[row_cls_idx, row_lane_idx] + local_width_row) + 1,
                            )
                        )
                    )
                    coord = (row_out[all_ind, row_cls_idx, row_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    coord_x = coord / (row_num_grid - 1) * 1280
                    coord_y = row_anchor[row_cls_idx] * 720
                    cur_lane.append(int(coord_x))
                else:
                    cur_lane.append(-2)
            all_lanes.append(cur_lane)
        else:
            pass

    for col_lane_idx in col_lane_list:
        if col_valid[:, col_lane_idx].sum() > min_lanepts_col:
            cur_lane = []
            for col_cls_idx in range(col_num_cls):
                if col_valid[col_cls_idx, col_lane_idx]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, col_max_indices[col_cls_idx, col_lane_idx] - local_width_col),
                                min(col_num_grid - 1, col_max_indices[col_cls_idx, col_lane_idx] + local_width_col) + 1,
                            )
                        )
                    )
                    coord = (col_out[all_ind, col_cls_idx, col_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    coord_y = coord / (col_num_grid - 1) * 720
                    coord_x = col_anchor[col_cls_idx] * 1280
                    cur_lane.append((coord_x, coord_y))
            cur_lane = np.array(cur_lane)
            top_lim = min(cur_lane[:, 1])
            bot_lim = max(cur_lane[:, 1])

            p = np.polyfit(cur_lane[:, 1], cur_lane[:, 0], deg=2)
            lanes_on_tusimple = np.polyval(p, tusimple_h_sample)

            lanes_on_tusimple = np.round(lanes_on_tusimple)
            lanes_on_tusimple = lanes_on_tusimple.astype(int)
            lanes_on_tusimple[lanes_on_tusimple < 0] = -2
            lanes_on_tusimple[lanes_on_tusimple > 1280] = -2
            lanes_on_tusimple[tusimple_h_sample < top_lim] = -2
            lanes_on_tusimple[tusimple_h_sample > bot_lim] = -2
            all_lanes.append(lanes_on_tusimple.tolist())
        else:
            pass
    return all_lanes


def run_test_tusimple(
    net,
    data_root,
    work_dir,
    exp_name,
    distributed,
    crop_ratio,
    train_width,
    train_height,
    batch_size=1,
    row_anchor=None,
    col_anchor=None,
    device=None,
    num_cls_row=56,
    num_grid_col=100,
    num_cls_col=41,
    num_lane_on_row=4,
    num_lane_on_col=4,
    num_grid_row=100,
    n_images=100,
    is_overlay=False,
    json_path="models/bos_model/ufld_v2/demo/image_data/tusimple/test_label.json",
    is_eval=False,
    model_location_generator=None,
    input_mode="dram_sharded",
):
    output_path = os.path.join(work_dir, exp_name + ".txt")
    fp = open(output_path, "w")
    if is_overlay:
        os.makedirs(os.path.join(data_root, f"{exp_name}"), exist_ok=True)

    if is_eval:
        loader = get_test_loader_from_json(
            batch_size, data_root, json_path, n_images, train_width, train_height, crop_ratio
        )
    else:
        loader = get_test_loader(batch_size, data_root, "Tusimple", distributed, crop_ratio, train_width, train_height)

    dim1 = num_grid_row * num_cls_row * num_lane_on_row
    dim2 = num_grid_col * num_cls_col * num_lane_on_col
    dim3 = 2 * num_cls_row * num_lane_on_row
    dim4 = 2 * num_cls_col * num_lane_on_col
    performant_runner = None
    for data in tqdm(loader, desc="Processing images", ncols=n_images):
        imgs, names = data
        if exp_name.startswith("reference_model"):
            with torch.no_grad():
                out, pred = net(imgs)
        else:
            if performant_runner is None:
                performant_runner = net(
                    device=device,
                    model_location_generator=model_location_generator,
                    torch_input_tensor=imgs,
                    device_batch_size=(batch_size // device.get_num_devices()),
                    input_mode=input_mode,
                )
                if input_mode == "dram_interleaved":
                    performant_runner._capture_ufldv2_trace_2cqs_interleaved()
                elif input_mode == "dram_interleaved_double_buffer":
                    performant_runner._capture_ufldv2_trace_2cqs_double_buffer()
                else:
                    performant_runner._capture_ufldv2_trace_2cqs()
            out = performant_runner.run(imgs)
            out = (
                ttnn.to_torch(out, mesh_composer=performant_runner.runner_infra.output_mesh_composer)
                .squeeze(dim=1)
                .squeeze(dim=1)
            )
            pred = {
                "loc_row": out[:, :dim1].view(-1, num_grid_row, num_cls_row, num_lane_on_row),
                "loc_col": out[:, dim1 : dim1 + dim2].view(-1, num_grid_col, num_cls_col, num_lane_on_col),
                "exist_row": out[:, dim1 + dim2 : dim1 + dim2 + dim3].view(-1, 2, num_cls_row, num_lane_on_row),
                "exist_col": out[:, -dim4:].view(-1, 2, num_cls_col, num_lane_on_col),
            }

        for b_idx, name in enumerate(names):
            tmp_dict = {}
            tmp_dict["lanes"] = generate_tusimple_lines(
                pred["loc_row"][b_idx],
                pred["exist_row"][b_idx],
                pred["loc_col"][b_idx],
                pred["exist_col"][b_idx],
                row_anchor=row_anchor,
                col_anchor=col_anchor,
                mode="4row",
            )
            tmp_dict["h_samples"] = [
                160,
                170,
                180,
                190,
                200,
                210,
                220,
                230,
                240,
                250,
                260,
                270,
                280,
                290,
                300,
                310,
                320,
                330,
                340,
                350,
                360,
                370,
                380,
                390,
                400,
                410,
                420,
                430,
                440,
                450,
                460,
                470,
                480,
                490,
                500,
                510,
                520,
                530,
                540,
                550,
                560,
                570,
                580,
                590,
                600,
                610,
                620,
                630,
                640,
                650,
                660,
                670,
                680,
                690,
                700,
                710,
            ]
            tmp_dict["raw_file"] = name
            tmp_dict["run_time"] = 10
            json_str = json.dumps(tmp_dict)
            fp.write(json_str + "\n")
            if is_overlay:
                if (
                    exp_name == "reference_model_results"
                    or exp_name == "ttnn_model_results"
                    or exp_name == "reference_model_results_dp"
                    or exp_name == "ttnn_model_results_dp"
                ):
                    folder = "images"
                    full_path = os.path.join(data_root, f"{exp_name}", name)
                else:
                    folder = os.path.join("image_data", "tusimple")
                    parts = name.split(os.sep)
                    partial_dir = os.path.join(data_root, exp_name, parts[0], parts[1], parts[2])
                    os.makedirs(partial_dir, exist_ok=True)
                    full_path = os.path.join(data_root, f"{exp_name}", parts[0], parts[1], parts[2], parts[3])

                overlay_lanes(
                    os.path.join(data_root, folder, name),
                    tmp_dict,
                    full_path,
                )
    fp.close()


def overlay_lanes(image_path, res, output_path, radius=5, point_length=2, mask_constant=-2):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    lanes = res["lanes"]
    h_samples = res["h_samples"]

    h_samples = np.array(h_samples)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # green, red, blue

    thickness = 3

    for lane_idx, lane_xs in enumerate(lanes):
        lane_xs = np.array(lane_xs)

        valid_mask = lane_xs != mask_constant
        valid_xs = lane_xs[valid_mask]
        valid_ys = h_samples[valid_mask]

        points = np.array([valid_xs, valid_ys]).T.astype(np.int32)

        if len(points) >= point_length:
            cv2.polylines(img, [points], isClosed=False, color=colors[lane_idx % len(colors)], thickness=thickness)

        for x, y in points:
            cv2.circle(img, (x, y), radius=radius, color=colors[lane_idx % len(colors)], thickness=-1)
    cv2.imwrite(output_path, img)
    logger.info(f"Saved overlay image to {output_path}")


class LaneTestDatasetFromJSON(torch.utils.data.Dataset):
    def __init__(self, data_root, json_path, n_images=100, img_transform=None, crop_size=None):
        super(LaneTestDatasetFromJSON, self).__init__()
        self.data_root = data_root
        self.img_transform = img_transform
        self.crop_size = crop_size
        self.annotations = []
        with open(json_path, "r") as f:
            for line in f:
                if len(self.annotations) >= n_images:
                    break
                line = line.strip()
                if line:
                    try:
                        self.annotations.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.info(f"Skipping invalid JSON line: {e}")

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_rel_path = ann["raw_file"].replace("\\", "/")
        img_path = os.path.join(self.data_root, "image_data", "tusimple", img_rel_path)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.crop_size is not None:
            img = img[:, -self.crop_size :, :]

        return img, img_rel_path

    def __len__(self):
        return len(self.annotations)


def get_test_loader_from_json(batch_size, data_root, json_path, n_images, train_width, train_height, crop_ratio=1.0):
    img_transforms = transforms.Compose(
        [
            transforms.Resize((int(train_height / crop_ratio), train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_dataset = LaneTestDatasetFromJSON(
        data_root=data_root,
        json_path=json_path,
        n_images=n_images,
        img_transform=img_transforms,
        crop_size=train_height,
    )

    sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return loader


def preprocess_frame(frame_bgr, train_width, train_height, crop_ratio=1.0):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    img_transforms = transforms.Compose(
        [
            transforms.Resize((int(train_height / crop_ratio), train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = img_transforms(img_pil)  # (3, H', W')
    img = img[:, -train_height:, :]  # bottom crop
    img = img.unsqueeze(0)  # (1, 3, H, W)
    return img


def overlay_lanes_on_frame(
    frame_bgr,
    res,
    radius=4,
    point_length=1,
    mask_constant=-2,
    y_min_ratio=0.3,
    y_max_ratio=1.05,
    dataset="culane",
    draw_mode="both",  # "dot", "line", "both"
):
    img = frame_bgr.copy()
    H, W = img.shape[:2]

    lanes = res["lanes"]
    h_samples = np.array(res["h_samples"], dtype=np.float32)

    if dataset == "culane":
        base_w, base_h = 1640.0, 590.0
    elif dataset == "tusimple":
        base_w, base_h = 1280.0, 720.0
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    colors = [
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        (0, 255, 0),
        # (0, 0, 255),
        # (0, 255, 0),
        # (255, 0, 0),
        # (255, 255, 0),
    ]

    for lane_idx, lane_xs in enumerate(lanes):
        lane_xs = np.array(lane_xs, dtype=np.float32)
        valid_mask = lane_xs != mask_constant
        if not np.any(valid_mask):
            continue

        xs_valid = lane_xs[valid_mask]
        ys_valid = h_samples[valid_mask]

        # Coordinate scaling (dataset → frame)
        xs_frame = xs_valid / base_w * W
        ys_frame = ys_valid / base_h * H

        # ROI filtering
        y_norm = ys_frame / float(H)
        roi_mask = (y_norm >= y_min_ratio) & (y_norm <= y_max_ratio)

        xs_frame = xs_frame[roi_mask]
        ys_frame = ys_frame[roi_mask]

        if len(xs_frame) < point_length:
            continue

        # ---------- draw Line ----------
        if draw_mode in ("line", "both"):
            points = np.array([(int(round(x)), int(round(y))) for x, y in zip(xs_frame, ys_frame)])
            if len(points) > 1:
                cv2.polylines(
                    img,
                    [points],
                    isClosed=False,
                    color=colors[lane_idx % len(colors)],
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )

        # ---------- draw Dot ----------
        if draw_mode in ("dot", "both"):
            for x, y in zip(xs_frame, ys_frame):
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H:
                    cv2.circle(
                        img,
                        (xi, yi),
                        radius=radius,
                        color=colors[lane_idx % len(colors)],
                        thickness=-1,
                    )

    return img


def generate_culane_lines(
    row_out,
    row_ext,
    col_out,
    col_ext,
    row_anchor,
    col_anchor,
    h_samples,
    img_w=1640,
    img_h=590,
    min_pts=3,
    local_width=14,
    mode="4col",
):
    """
    CULane line decode (row + col hybrid)
    Final output: all lanes are a list of length len(h_samples) (CULane format)
    """

    row_num_grid, row_num_cls, row_num_lane = row_out.shape
    col_num_grid, col_num_cls, col_num_lane = col_out.shape

    row_out = row_out.cpu()
    row_ext = row_ext.cpu()
    col_out = col_out.cpu()
    col_ext = col_ext.cpu()

    # grid argmax
    row_max = row_out.argmax(0)  # (row_num_cls, row_num_lane)
    col_max = col_out.argmax(0)  # (col_num_cls, col_num_lane)

    # exist argmax -> 0/1 map
    # exist shape: (2, num_cls, num_lane)
    row_exist_map = row_ext.argmax(0)  # (row_num_cls, row_num_lane)
    col_exist_map = col_ext.argmax(0)  # (col_num_cls, col_num_lane)

    if mode in ("normal", "2row2col"):
        row_lane_list = [1, 2]
        col_lane_list = [0, 3]
    elif mode == "4row":
        row_lane_list = range(row_num_lane)
        col_lane_list = []
    elif mode == "4col":
        row_lane_list = []
        col_lane_list = range(col_num_lane)
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")

    h_samples = np.array(h_samples, dtype=np.float32)
    lanes = []

    # ---------------------------------------------------------
    # 1) Convert row-based lanes to (len(h_samples),) format
    # ---------------------------------------------------------
    for lane_idx in row_lane_list:
        valid_mask = row_exist_map[:, lane_idx] == 1  # (row_num_cls,)
        if valid_mask.sum().item() < min_pts:
            continue

        pts = []
        for cls_idx in range(row_num_cls):
            if not valid_mask[cls_idx].item():
                continue

            center = row_max[cls_idx, lane_idx].item()

            inds = torch.arange(max(0, center - local_width), min(row_num_grid - 1, center + local_width) + 1)

            prob = row_out[inds, cls_idx, lane_idx].softmax(0)
            coord = (prob * inds.float()).sum().item() + 0.5

            # grid -> pixel
            px = coord / (row_num_grid - 1) * img_w
            py = float(row_anchor[cls_idx]) * img_h

            pts.append((px, py))

        if len(pts) < min_pts:
            continue

        pts = np.array(pts, dtype=np.float32)
        ys = pts[:, 1]
        xs = pts[:, 0]

        # 2nd degree polynomial fitting
        p = np.polyfit(ys, xs, deg=2)

        # Evaluate x values from h_samples
        xs_eval = np.polyval(p, h_samples)
        xs_eval = np.round(xs_eval).astype(int)

        # Set to -2 for out of range / invalid range
        xs_eval[h_samples < ys.min()] = -2
        xs_eval[h_samples > ys.max()] = -2
        xs_eval[xs_eval < 0] = -2
        xs_eval[xs_eval > img_w] = -2

        lanes.append(xs_eval.tolist())

    # ---------------------------------------------------------
    # 2) Convert col-based lanes to (len(h_samples),) format
    # ---------------------------------------------------------
    for lane_idx in col_lane_list:
        valid_mask = col_exist_map[:, lane_idx] == 1  # (col_num_cls,)
        if valid_mask.sum().item() < min_pts:
            continue

        pts = []
        for cls_idx in range(col_num_cls):
            if not valid_mask[cls_idx].item():
                continue

            center = col_max[cls_idx, lane_idx].item()

            inds = torch.arange(max(0, center - local_width), min(col_num_grid - 1, center + local_width) + 1)

            prob = col_out[inds, cls_idx, lane_idx].softmax(0)
            coord = (prob * inds.float()).sum().item() + 0.5

            py = coord / (col_num_grid - 1) * img_h
            px = float(col_anchor[cls_idx]) * img_w

            pts.append((px, py))

        if len(pts) < min_pts:
            continue

        pts = np.array(pts, dtype=np.float32)
        ys = pts[:, 1]
        xs = pts[:, 0]

        p = np.polyfit(ys, xs, deg=2)
        xs_eval = np.polyval(p, h_samples)
        xs_eval = np.round(xs_eval).astype(int)

        xs_eval[h_samples < ys.min()] = -2
        xs_eval[h_samples > ys.max()] = -2
        xs_eval[xs_eval < 0] = -2
        xs_eval[xs_eval > img_w] = -2

        lanes.append(xs_eval.tolist())

    return lanes
