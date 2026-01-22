import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion import arcline_path_utils
from tqdm import tqdm
import shapely
from shapely.strtree import STRtree 
from shapely.geometry import box

 # --- Style dictionaries (shared) ---
LINE_STYLE = {
    "lane":           dict(color="#6C757D", linewidths=0.8, linestyles=(0, (4, 4)), alpha=0.60, zorder=2),
    "lane_connector": dict(color="#8E6BBE", linewidths=0.9, linestyles=(0, (4, 3)), alpha=0.45, zorder=1),
    "lane_divider":   dict(color="#1F77B4", linewidths=1.1, linestyles="solid", alpha=0.95, zorder=4),
    "road_divider":   dict(color="#1F77B4", linewidths=1.6, linestyles="solid", alpha=1.00, zorder=5),
    "stop_line":      dict(color="#111111", linewidths=0.5, linestyles="solid", alpha=0.90, zorder=6),
}

FILL_STYLE = {
    "drivable_area": dict(face="#EDF5FF", alpha=0.18, zorder=0),
    "road_segment":  dict(face="#F3F4FA", alpha=0.10, zorder=0),
    "road_block":    dict(face="#FFF0F0", alpha=0.10, zorder=0),
    "walkway":       dict(face="#F6F7FB", alpha=0.25, zorder=0),
    "carpark_area":  dict(face="#FFF7EA", alpha=0.18, zorder=0),
    "ped_crossing":  dict(face="#FFF3C4", alpha=0.55, zorder=1),
}

FILL_KEYS = sorted(FILL_STYLE.keys(), key=lambda k: FILL_STYLE[k].get("zorder", 0))
LINE_KEYS = sorted(LINE_STYLE.keys(), key=lambda k: LINE_STYLE[k].get("zorder", 0))

LANE_FILL_COLOR = "#d9ead3"
LANE_HALF_WIDTH_M = 1.8
RESOLUTION_M = 0.5
DENSIFY_STEP_M = 0.25

# =====================  COORDINATE MAPPING  =====================
XMIN, XMAX = -30.0, 30.0
YMIN, YMAX = -30.0, 30.0
LANE_RES_M = 1.5  # coarser sampling -> faster

class BEVMapLoader:
    def __init__(self, dataset, bev_size=800):
        self.dataset = dataset
        self.data_version = dataset.version
        self.data_root = dataset.data_root
        self.len_data = len(dataset.data_infos)
        self.nusc = NuScenes(version=str(self.data_version), dataroot=str(self.data_root), verbose=True)
        self.nusc_map = NuScenesMap
        self.bev_size = bev_size
        
        # BEV
        self.get_bev_map_info()
        
        # cache
        self._map_cache = {}
        self._bev_cache = {}

        self._strtree_cache   = {}  # map_id -> {key: STRtree}
        self._polys_cache     = {}  # map_id -> {key: [polys]}
        self._tokens_cache    = {}  # map_id -> {key: [tokens]}
        self._polyid2tok_cache= {}  # map_id -> {key: {id(poly): tok}}


    # =====================  COLOR / ALPHA HELPERS  =====================
    def _hex_to_bgr(self, hexstr: str):
        s = hexstr.lstrip('#')
        return (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))  # B, G, R

    def _blend_fill_poly(self, img, cnt, face_hex, alpha):
        color = self._hex_to_bgr(face_hex)
        overlay = img.copy()
        cv2.fillPoly(overlay, [cnt], color)
        cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0.0, img)

    def _blend_polyline(self, img, cnt, line_hex, alpha, thickness=1):
        color = self._hex_to_bgr(line_hex)
        overlay = img.copy()
        cv2.polylines(overlay, [cnt], False, color, int(max(1, round(thickness))), cv2.LINE_AA)
        cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0.0, img)

    def _blend_dashed(self, img, cnt_points, line_hex, alpha, thickness=1, on_px=8, off_px=8):
        color = self._hex_to_bgr(line_hex)
        pts = cnt_points.reshape(-1, 2).astype(np.int32)
        overlay = img.copy()

        def _seglen(a, b): return float(np.linalg.norm(b - a))
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i+1]
            L = _seglen(a, b)
            if L < 1: 
                continue
            dirv = (b - a) / L
            t = 0.0
            draw = True
            while t < L:
                run = min(on_px if draw else off_px, L - t)
                if draw:
                    p0 = (a + dirv * t).astype(int)
                    p1 = (a + dirv * (t + run)).astype(int)
                    cv2.line(overlay, tuple(p0), tuple(p1), color,
                            int(max(1, round(thickness))), cv2.LINE_AA)
                t += run
                draw = not draw
        cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0.0, img)

    # =====================  COORDINATE MAPPING  =====================
    def _bev_setup(self, pose_record, size=800):
        t_xy = np.array(pose_record["translation"][:2], dtype=np.float32)
        R2 = Quaternion(pose_record["rotation"]).inverse.rotation_matrix[:2, :2].astype(np.float32)
        sx = size / (XMAX - XMIN)
        sy = size / (YMAX - YMIN)
        return t_xy, R2, sx, sy, size

    # GLOBAL(world) -> BEV pixels (used for map layers)
    def _world_to_cv_px(self, coords_xy, t_xy, R2, sx, sy, size):
        p = (coords_xy.astype(np.float32) - t_xy) @ R2.T
        x_plot = -p[:, 1]
        y_plot =  p[:, 0]
        u = (x_plot - XMIN) * sx
        v = (YMAX - y_plot) * sy
        return np.round(np.stack([u, v], axis=1)).astype(np.int32)

    # EGO(frame) -> BEV pixels (use THIS for ego car & planned traj)
    def _ego_to_cv_px(self, coords_ego, sx, sy, size):
        # In your BEV: x_plot = -y_ego ; y_plot = +x_ego
        x_plot = -coords_ego[:, 1].astype(np.float32)
        y_plot =  coords_ego[:, 0].astype(np.float32)
        u = (x_plot - XMIN) * sx
        v = (YMAX - y_plot) * sy
        return np.round(np.stack([u, v], axis=1)).astype(np.int32)

    def _line_cnt_from_world(self, xy_world, t_xy, R2, sx, sy, size):
        if xy_world.shape[0] < 2: return None
        uv = self._world_to_cv_px(xy_world, t_xy, R2, sx, sy, size)
        return uv.reshape(-1, 1, 2)

    def _line_cnt_from_ego(self, xy_ego, sx, sy, size):
        if xy_ego.shape[0] < 2: return None
        uv = self._ego_to_cv_px(xy_ego, sx, sy, size)
        return uv.reshape(-1, 1, 2)

    def _poly_to_cv2(self, pol, t_xy, R2, sx, sy, size):
        cc = np.asarray(pol.exterior.coords, dtype=np.float32)
        uv = self._world_to_cv_px(cc, t_xy, R2, sx, sy, size)
        return uv.reshape(-1, 1, 2)

    def _line_cnt_from_xy(self, xy, t_xy, R2, sx, sy, size):
        if xy.shape[0] < 2: return None
        uv = self._world_to_cv_px(xy, t_xy, R2, sx, sy, size)
        return uv.reshape(-1, 1, 2)

    # =====================  BEV MAP DRAWING  =====================
    def get_batch_bev_map_info(self, index):
        info = self.dataset.data_infos[index]
        token = info['token']
        scene_token = info['scene_token']

        log = self.nusc.get("log", self.nusc.get("scene", scene_token)["log_token"])
        map_id = log['location']

        sample = self.nusc.get("sample", token)
        sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        
        return {
            "token": token,
            "scene_token": scene_token,
            "map_id": map_id,
            "lidar_sd_token": sd["token"],
            "pose": pose,
        }
    
    def get_bev_map_info(self):
        self.map_samples = []

        for i in range(self.len_data):
            self.map_samples.append(self.get_batch_bev_map_info(i))
            
    def draw_batch_bev_map(self, sample):
        token  = sample["token"]
        map_id = sample["map_id"]
        pose   = sample["pose"]

        # Get (or create) the NuScenesMap for this location once
        nusc_map = self._map_cache.get(map_id)
        if nusc_map is None:
            nusc_map = NuScenesMap(dataroot=self.nusc.dataroot, map_name=map_id)
            self._map_cache[map_id] = nusc_map
        # Setup transforms & canvas
        t_xy, R2, sx, sy, size = self._bev_setup(pose, size=self.bev_size)

        # World patch covering the viewport (slightly padded)
        patch_box = (
            pose["translation"][0] + XMIN - 3,
            pose["translation"][1] + YMIN - 3,
            pose["translation"][0] + XMAX + 3,
            pose["translation"][1] + YMAX + 3,
        )
        patch_geom = box(*patch_box)

        # White background
        img = np.full((size, size, 3), 255, dtype=np.uint8)

        # ---- Polygon fills (background) ----
        for key in FILL_KEYS:
            fs = FILL_STYLE[key]

            if key == "road_block":
                # per-map caches (created once per map_id)
                trees_for_map   = self._strtree_cache.setdefault(map_id, {})
                polys_for_map   = self._polys_cache.setdefault(map_id, {})
                tokens_for_map  = self._tokens_cache.setdefault(map_id, {})

                # build STRtree once per (map_id, key)
                if key not in trees_for_map:
                    # guard if layer missing in this location
                    if not hasattr(nusc_map, key):
                        continue

                    toks, polys = [], []
                    # NuScenesMap tables are list-like of dicts (each has 'token', may have 'polygon_token')
                    for ent in getattr(nusc_map, key):
                        if "polygon_token" not in ent:
                            continue
                        poly = nusc_map.extract_polygon(ent["polygon_token"])
                        # optional simplification for speed (tune tolerance)
                        # poly = poly.simplify(0.2, preserve_topology=True)
                        toks.append(ent["token"])
                        polys.append(poly)

                    trees_for_map[key]  = STRtree(polys)
                    polys_for_map[key]  = polys
                    tokens_for_map[key] = toks

                # query the spatial index for this patch
                tree  = trees_for_map[key]
                polys = polys_for_map[key]

                # shapely 2.x: returns ndarray[int] of indices; fast predicate filter
                idxs = tree.query(patch_geom, predicate="intersects")
                if idxs is None:
                    idxs = []
                for i in (idxs.tolist() if hasattr(idxs, "tolist") else idxs):
                    poly = polys[i]
                    cnt = self._poly_to_cv2(poly, t_xy, R2, sx, sy, size)
                    self._blend_fill_poly(img, cnt, fs["face"], fs["alpha"])

            else:
                # original path for non-heavy layers
                try:
                    recs = nusc_map.get_records_in_patch(patch_box, [key], mode="intersect")[key]
                except Exception:
                    recs = []
                for tok in recs:
                    ent = getattr(nusc_map, key)[nusc_map.getind(key, tok)]
                    if "polygon_token" not in ent:
                        continue
                    poly = nusc_map.extract_polygon(ent["polygon_token"])
                    cnt = self._poly_to_cv2(poly, t_xy, R2, sx, sy, size)
                    self._blend_fill_poly(img, cnt, fs["face"], fs["alpha"])


        # ---- Optional lane underlay band (wide stroke under centerlines) ----
        lane_fill_col = self._hex_to_bgr(LANE_FILL_COLOR)
        thickness_px  = int(round((LANE_HALF_WIDTH_M * 2) * sx))
        overlay = img.copy()

        # ---- Line layers (lanes, dividers, stop lines) ----
        for key in LINE_KEYS:
            ls = LINE_STYLE[key]
            thickness = max(1, int(round(ls["linewidths"])))
            dashed = isinstance(ls["linestyles"], tuple)

            try:
                recs = nusc_map.get_records_in_patch(patch_box, [key], mode="intersect")[key]
            except Exception:
                recs = []

            for tok in recs:
                if key in ("lane", "lane_connector"):
                    al = nusc_map.get_arcline_path(tok)
                    poses = arcline_path_utils.discretize_lane(al, resolution_meters=LANE_RES_M)
                    if poses is None or len(poses) < 2:
                        continue
                    xy = np.asarray(poses, dtype=np.float32)[:, :2]
                    cnt = self._line_cnt_from_xy(xy, t_xy, R2, sx, sy, size)
                else:
                    ent = getattr(nusc_map, key)[nusc_map.getind(key, tok)]
                    geom = None
                    if "line_token" in ent:
                        geom = nusc_map.extract_line(ent["line_token"])
                    elif "polygon_token" in ent:
                        poly = nusc_map.extract_polygon(ent["polygon_token"])
                        geom = poly.boundary if hasattr(poly, "boundary") else None
                    if geom is None or not hasattr(geom, "coords"):
                        continue
                    xy = np.asarray(geom.coords, dtype=np.float32)
                    cnt = self._line_cnt_from_xy(xy, t_xy, R2, sx, sy, size)

                if cnt is None:
                    continue

                # Underlay band for lanes/connectors
                if key in ("lane", "lane_connector"):
                    cv2.polylines(overlay, [cnt], False, lane_fill_col, thickness_px, cv2.LINE_AA)

                # Final stroke
                if dashed:
                    _, (on, off) = ls["linestyles"]
                    scale = max(1, thickness)
                    self._blend_dashed(
                        img, cnt, ls["color"], ls["alpha"],
                        thickness=thickness, on_px=int(on*scale), off_px=int(off*scale)
                    )
                else:
                    self._blend_polyline(img, cnt, ls["color"], ls["alpha"], thickness=thickness)

        # Commit the lane underlay after all lines
        cv2.addWeighted(overlay, 0.30, img, 0.70, 0.0, img)

        # ---- Precompute ego rectangle + front-center anchor ----
        ego_m  = np.array([[-2.0, -0.9], [-2.0, 0.9], [2.0, 0.9], [2.0, -0.9]], np.float32)
        ego_cnt = self._line_cnt_from_ego(ego_m, sx, sy, size)

        p_fc_ego = np.array([[2.0, 0.0]], dtype=np.float32)   # front-center (L/2, 0) in ego meters
        p_fc_px  = self._ego_to_cv_px(p_fc_ego, sx, sy, size).reshape(-1).astype(np.float32)

        # ---- Store cache entry ----
        self._bev_cache[token] = dict(
            bev_bg=img,
            t_xy=t_xy, R2=R2, sx=sx, sy=sy, size=size,
            ego_rect_cnt=ego_cnt,
            p_frontcenter_px=p_fc_px,
        )

    def draw_bev_map(self):
        assert hasattr(self, "map_samples") and len(self.map_samples) > 0, \
            "Call get_bev_map_info() first."

        for sample in tqdm(self.map_samples):
            self.draw_batch_bev_map(sample)


    # =====================  TRAJECTORY DRAWING (DURING INFERENCE TIME)  =====================
    # Get token_id and trajectory
    def get_token_id(self, visualize_result):
        self.token_id = list(visualize_result['plan_results_ttnn'].keys())[0]
        logits = visualize_result["plan_results_ttnn"][self.token_id][1][0,0,0].cpu().float().numpy()
        self.traj = visualize_result["plan_results_ttnn"][self.token_id][0][int(np.argmax(logits))].cpu().float().numpy().astype(np.float32)

    # Convert trajectory to bev view
    def make_bev_trajectory(self, traj, n_segments=6, samples_per_seg=51, front_center_anchor=True):
        curr_bev = self._bev_cache[self.token_id]
        sx, sy, size = curr_bev['sx'], curr_bev['sy'], curr_bev['size']
        traj = np.asarray(traj, dtype=np.float32)

        # remove tiny jitter
        traj[np.abs(traj) < 0.01] = 0.0
        traj = traj[:, [1, 0]]
        traj[:, 1] *= -1

        # cumulative ego points (meters), include origin
        pts_ego = np.vstack([np.zeros((1, 2), np.float32), np.cumsum(traj, axis=0)])  # (M,2)

        # optional front-center anchor (shift by vehicle half-length along +x_ego)
        if front_center_anchor:
            pts_ego = pts_ego + np.array([2.0, 0.0], dtype=np.float32)

        # clamp number of segments and bail if empty
        nsegs = max(0, min(n_segments, len(pts_ego) - 1))
        if nsegs == 0:
            return None

        # densify linearly in ego meters
        starts = pts_ego[:nsegs]            # (nsegs,2)
        ends   = pts_ego[1:nsegs+1]         # (nsegs,2)
        t = np.linspace(0.0, 1.0, samples_per_seg, dtype=np.float32)[:, None]  # (S,1)
        dense_ego = starts[:, None, :] * (1 - t) + ends[:, None, :] * t        # (nsegs,S,2)
        dense_ego = dense_ego.reshape(-1, 2)                                   # (nsegs*S,2)

        # ---- ego meters -> BEV pixels (CONSISTENT with _ego_to_cv_px) ----
        # x_plot = -y_ego ; y_plot = +x_ego
        x_plot = -dense_ego[:, 1]
        y_plot =  dense_ego[:, 0]

        # pixel mapping (image y down)
        XMIN_local, XMAX_local = -30.0, 30.0
        YMIN_local, YMAX_local = -30.0, 30.0
        u = (x_plot - XMIN_local) * sx
        v = (YMAX_local - y_plot) * sy

        uv = np.round(np.stack([u, v], axis=1)).astype(np.int32).reshape(-1, 1, 2)
        return uv
    
    def get_current_bev_map(self):
        # for debug only
        return self._bev_cache[self.token_id]['bev_bg'].copy()

    def create_output_bev_map(self, draw_ego=True, color=(0, 255, 0),
                            canvas=None, paste_xy=(20, 20), bev_out_size=400):
        cache = self._bev_cache[self.token_id]
        bev = cache['bev_bg'].copy()

        if draw_ego and cache.get('ego_rect_cnt') is not None:
            cv2.polylines(bev, [cache['ego_rect_cnt']], True, (60, 180, 120), 2, cv2.LINE_AA)

        if self.traj is not None:
            uv = self.make_bev_trajectory(self.traj)
            if uv is not None:
                cv2.polylines(bev, [uv], False, color, 1, cv2.LINE_AA)

        if canvas is None:
            return cv2.resize(bev, (int(bev_out_size), int(bev_out_size)), interpolation=cv2.INTER_AREA)

        H, W = canvas.shape[:2]
        sx, size = 80, 500
        sy = (H - size) // 2

        bev_resized = cv2.resize(bev, (size, size), interpolation=cv2.INTER_AREA)

        x, y = sx, sy
        h, w = bev_resized.shape[:2]

        if x >= W or y >= H or x + w <= 0 or y + h <= 0:
            return canvas

        x0d, y0d = max(0, x), max(0, y)
        x1d, y1d = min(W, x + w), min(H, y + h)
        x0s, y0s = x0d - x, y0d - y

        canvas[y0d:y1d, x0d:x1d] = bev_resized[y0s:y0s + (y1d - y0d), x0s:x0s + (x1d - x0d)]
        return canvas
