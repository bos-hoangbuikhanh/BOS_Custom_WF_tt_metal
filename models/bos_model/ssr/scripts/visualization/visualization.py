import cv2
import numpy as np

class Visualizer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_version = dataset.version
        self.data_root = dataset.data_root
        self.cam_names = list(dataset.data_infos[0]['cams'].keys())
        self.cam_imgs = {}
        # ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.direction_cmd = ['Turn Right', 'Turn Left', 'Go Straight']
        self.window_name = "TTNN Visualizer"
        self._win_open = False

        self.canvas_shape = (1080, 1920) # fhd size
        self.visualize_img = None
        self.video = None

    def add_cam_label(self, cam_img, cam_name):
        """Add camera name label on top-left of each camera view image."""
        lw, tf = 6, 3
        h = cv2.getTextSize(cam_name, 0, fontScale=lw / 6, thickness=tf)[0][1]
        cv2.putText(cam_img, cam_name, (10, h + 10), 0, lw / 6, (255, 255, 255), tf, cv2.LINE_AA)
        return cam_img
    
    def get_plan_vecs(self, plan_traj, lidar2img): 
        """Using lidar2img to project 3d trajectory to 2d"""
        plan_traj[abs(plan_traj) < 0.01] = 0.0
        plan_traj = np.cumsum(plan_traj, axis=0)

        plan_traj = np.concatenate((
            plan_traj[:, [0]],
            plan_traj[:, [1]],
            -1.0 * np.ones((6, 1)),
            np.ones((6, 1)),
        ), axis=1)

        plan_traj = np.concatenate((np.zeros((1, 4)), plan_traj), axis=0)
        plan_traj[0, 0] = 0.3
        plan_traj[0, 2] = -1.0
        plan_traj[0, 3] = 1.0

        plan_traj = lidar2img @ plan_traj.T
        plan_traj = plan_traj[0:2, ...] / np.maximum(
            plan_traj[2:3, ...], np.ones_like(plan_traj[2:3, ...]) * 1e-5
        )
        plan_traj = plan_traj.T
        plan_traj = np.stack((plan_traj[:-1], plan_traj[1:]), axis=1)

        plan_vecs = None
        for i in range(6):
            plan_vec_i = plan_traj[i]
            x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
            y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
            xy = np.stack((x_linspace, y_linspace), axis=1)
            xy = np.stack((xy[:-1], xy[1:]), axis=1)
            if plan_vecs is None:
                plan_vecs = xy
            else:
                plan_vecs = np.concatenate((plan_vecs, xy), axis=0)

        return plan_vecs

    def valid_points(self, pts, h, w):
        """Check if points is in image range, if not, pass"""
        x = pts[..., 0]
        y = pts[..., 1]
        finite = np.isfinite(x) & np.isfinite(y)
        return finite & (x > 0) & (x <= w) & (y > 0) & (y <= h)

    def _unit(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _draw_arrowhead(self, img, center, direction, size, color):
        d = self._unit(direction).astype(np.float32)
        n = np.array([-d[1], d[0]], dtype=np.float32)
        tip  = (center + d * size).astype(np.int32)
        base = center - d * (0.3 * size)
        w = 0.6 * size
        p1 = (base + w * n).astype(np.int32)
        p2 = (base - w * n).astype(np.int32)
        thickness = max(1, int(0.3 * size))
        cv2.line(img, tuple(tip), tuple(p1), color, thickness, cv2.LINE_AA)
        cv2.line(img, tuple(tip), tuple(p2), color, thickness, cv2.LINE_AA)

    def _draw_ribbon(self, img, pts, color=(0, 165, 255), thickness=70, alpha=0.35):
        if len(pts) < 2:
            return
        overlay = img.copy()
        cv2.polylines(overlay, [pts.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    def draw_arrows(self, img, plan_vecs, max_arrows):
        """Draw arrowhead to planned trajectory"""
        h, w, _ = img.shape

        v0 = self.valid_points(plan_vecs[:, 0], h, w)
        v1 = self.valid_points(plan_vecs[:, 1], h, w)
        keep = v0 & v1
        kept = plan_vecs[keep]
        if kept.shape[0] == 0:
            return img
        
        # Find lowest points and bring it to bottom
        flat_idx = np.argmax(kept[:, :, 1])
        seg_idx, end_idx = divmod(flat_idx, 2)
        kept[seg_idx, end_idx, 1] = h

        ptsf = np.vstack((kept[0, 0], kept[:, 1])).astype(np.float32)
        if ptsf.shape[0] < 2:
            return img

        pts = np.rint(ptsf).astype(np.int32)
        self._draw_ribbon(img, pts, thickness=100)
        cv2.polylines(img, [pts], isClosed=False, color=(255, 0, 0), thickness=5)

        seg_vecs = ptsf[1:] - ptsf[:-1]
        seg_len = np.linalg.norm(seg_vecs, axis=1).astype(np.float32)
        total = float(seg_len.sum())
        if total == 0 or max_arrows <= 0:
            return img

        if max_arrows == 1:
            targets = np.array([0.0], dtype=np.float32)
        else:
            step = total / float(max_arrows - 1)
            targets = np.linspace(0.0, total, max_arrows, dtype=np.float32)

        cum = np.concatenate(([0.0], np.cumsum(seg_len, dtype=np.float32)))

        idx = np.searchsorted(cum, targets, side='right') - 1
        idx = np.clip(idx, 0, len(seg_len) - 1)
        good = seg_len[idx] > 0

        if not np.any(good):
            return img

        idx = idx[good]
        tg = targets[good]

        u = (tg - cum[idx]) / seg_len[idx]
        centers = ptsf[idx] + seg_vecs[idx] * u[:, None]

        left_i = np.clip(idx - 1, 0, len(ptsf) - 1)
        right_i = np.clip(idx + 1, 0, len(ptsf) - 1)
        dir_vec = ptsf[right_i] - ptsf[left_i]

        zero = np.linalg.norm(dir_vec, axis=1) == 0
        if np.any(zero):
            dir_vec[zero] = seg_vecs[idx[zero]]

        color = (255, 0, 0)
        for c, d in zip(centers.astype(np.float32), dir_vec.astype(np.float32)):
            self._draw_arrowhead(img, c, d, size=25, color=color)

    def draw_ttnn_trajectory(self, cam_img, token_id, visualize_result, lidar2img):
        plan_cmd = np.argmax(visualize_result['plan_results_ttnn'][token_id][1][0, 0, 0].cpu().float().numpy())
        plan_traj = visualize_result['plan_results_ttnn'][token_id][0][plan_cmd].cpu().float().numpy()
        plan_vecs = self.get_plan_vecs(plan_traj, lidar2img)
        self.draw_arrows(cam_img, plan_vecs, max_arrows=5)
        return cam_img
    
    def resize_hw(self, img, scale=1, interp=cv2.INTER_AREA):
        h, w = img.shape[:2]
        nh = max(1, int(h / float(scale)))
        nw = max(1, int(w / float(scale)))

        return cv2.resize(img, (nw, nh), interpolation=interp)

    def paste_rgba(self, dst: np.ndarray, src: np.ndarray, x: int, y: int, *, border: bool = False,
               thickness: int = 2, color=(255, 255, 255)) -> None:
        H, W = dst.shape[:2]
        h, w = src.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return
        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        patch = src[sy1:sy2, sx1:sx2]
        if patch.shape[2] == 4:
            a = (patch[:, :, 3:4].astype(np.float32) / 255.0)
            roi = dst[y1:y2, x1:x2].astype(np.float32)
            dst[y1:y2, x1:x2] = (roi * (1 - a) + patch[:, :, :3].astype(np.float32) * a).astype(np.uint8)
        else:
            dst[y1:y2, x1:x2] = patch[:, :, :3]
        if border:
            cv2.rectangle(dst, (x1, y1), (x2 - 1, y2 - 1), color, thickness, lineType=cv2.LINE_AA)

    # --- positioning in gray area ---
    def place_bottom_left_in_rect(self, rx, ry, rw, rh, w, h, left_gap, bottom_gap):
        return int(rx + left_gap), int(ry + rh - bottom_gap - h)

    def place_bottom_right_in_rect(self, rx, ry, rw, rh, w, h, right_gap, bottom_gap):
        return int(rx + rw - right_gap - w), int(ry + rh - bottom_gap - h)

    def place_top_left_in_rect(self, rx, ry, rw, rh, w, h, left_gap, top_gap):
        return int(rx + left_gap), int(ry + top_gap)

    def place_top_right_in_rect(self, rx, ry, rw, rh, w, h, right_gap, top_gap):
        return int(rx + rw - right_gap - w), int(ry + top_gap)

    def center_by_anchor_in_rect(self, rx, ry, rw, rh, coord, w, h, dx=0, dy=0):
        ax, ay = coord
        cx, cy = rx + rw // 2, ry + rh // 2
        return int(cx - ax + dx), int(cy - ay + dy)
    
    # --- draw surrounding image line ---
    def draw_connectors(self, img, x_src, y_src, uv, x_dst, y_dst, w_dst, h_dst=None,
                        color=(255, 255, 255), thickness=2, use_bottom=False):
        p0 = (int(x_src + uv[0]), int(y_src + uv[1]))
        y_corner = int(y_dst if not use_bottom else y_dst + (h_dst - 1 if h_dst is not None else 0))
        p1 = (int(x_dst), y_corner)
        p2 = (int(x_dst + w_dst - 1), y_corner)
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.line(img, p0, p2, color, thickness, cv2.LINE_AA)

    # --- layout ---
    def draw_canvas(self, canvas, car_img, logo_img, is_ver1):
        H, W = canvas.shape[:2]
        # bev map placement coordinates
        sx, size = 80, 500
        sy = (H - size) // 2
        # right gray panel + soft wedge blend
        rh, rw = 750, 1120
        gap_r = 80
        rx = W - rw - gap_r
        ry = (H - rh) // 2
        cv2.rectangle(canvas, (rx, ry), (W - gap_r - 1, ry + rh - 1), (128, 128, 128), -1)
        pts = np.array([[rx, ry], [rx, ry + rh - 1], [rx - 100, ry + rh // 2]], np.int32)
        s = 4
        mask = np.zeros((H * s, W * s), np.uint8)
        cv2.fillPoly(mask, [(pts * s).astype(np.int32)], 255)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        canvas[:] = (canvas.astype(np.float32) * (1 - alpha) + np.array([128, 128, 128], np.float32) * alpha).astype(np.uint8)

        # draw bos logo
        h0, w0 = logo_img.shape[:2]
        gap_logo = 20
        nh = min(540, max(1, H - (sy + size + gap_logo)))
        nw = int(w0 * (nh / float(h0)))
        logo = cv2.resize(logo_img, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas[sy + size + gap_logo:sy + size + gap_logo + nh, sx:sx + nw] = logo[:nh, :nw]

        # titles
        title1, title2 = "Eagle-N A0", "SSR Live Demo"
        font = cv2.FONT_ITALIC
        s1, t1 = 2.5, 7
        s2, t2 = 1.5, 6
        (w2, h2), _ = cv2.getTextSize(title2, font, s2, t2)
        (w1, h1), _ = cv2.getTextSize(title1, font, s1, t1)
        x2, y2 = sx + (size - w2) // 2, sy - 50
        x1, y1 = sx + (size - w1) // 2, y2 - h2 - 30
        cv2.putText(canvas, title1, (x1, y1), font, s1, (255, 255, 255), t1, cv2.LINE_AA)
        cv2.putText(canvas, title2, (x2, y2), font, s2, (0, 192, 255), t2, cv2.LINE_AA)

        # car with anchor points (expects RGBA; keep given alpha)
        car = car_img.copy()
        if is_ver1:
            coord = {
                'front': (214, 250),
                'front_left': (455, 245),
                'front_right': (231, 196),
                'back': (517, 161),
                'back_left': (559, 231),
                'back_right': (330, 150),
            }
            cv2.circle(car, coord['front'], 5, (50, 113, 233, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(car, coord['back_left'], 5, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(car, coord['front_left'], 5, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)

            x_car, y_car = self.center_by_anchor_in_rect(rx, ry, rw, rh, coord['front'], car.shape[1], car.shape[0], dx=-150, dy=-30)
        elif not is_ver1:
            car = self.resize_hw(car, scale=0.6)
            coord = {
                'front': (87, 266),
                'front_left': (304, 249),
                'front_right': (114, 194),
                'back': (405, 167),
                'back_left': (379, 224),
                'back_right': (220, 141),
            }
            cv2.circle(car, coord['front'], 5, (50, 113, 233, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(car, coord['back_left'], 5, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(car, coord['front_left'], 5, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)

            x_car, y_car = self.center_by_anchor_in_rect(rx, ry, rw, rh, coord['front'], car.shape[1], car.shape[0], dx=-50, dy=0)

        # cam_imgs is a dictionary with keys = cam_name
        cam_front        = self.cam_imgs['CAM_FRONT']
        cam_front_right  = self.cam_imgs['CAM_FRONT_RIGHT']
        cam_front_left   = self.cam_imgs['CAM_FRONT_LEFT']
        cam_back         = self.cam_imgs['CAM_BACK']
        cam_back_left    = self.cam_imgs['CAM_BACK_LEFT']
        cam_back_right   = self.cam_imgs['CAM_BACK_RIGHT']

        # camera tiles (all derived from cam_img scale factors)
        cam_front       = self.resize_hw(cam_front, scale=3.7)
        cam_front_left  = self.resize_hw(cam_front_left, scale=5.5)
        cam_front_right = self.resize_hw(cam_front_right, scale=5.5)
        cam_back        = self.resize_hw(cam_back, scale=5.5)
        cam_back_left   = self.resize_hw(cam_back_left, scale=5.5)
        cam_back_right  = self.resize_hw(cam_back_right, scale=5.5)

        # positions (fix width/height variables per tile)
        f_h,  f_w  = cam_front.shape[:2]
        fl_h, fl_w = cam_front_left.shape[:2]
        fr_h, fr_w = cam_front_right.shape[:2]
        b_h,  b_w  = cam_back.shape[:2]
        bl_h, bl_w = cam_back_left.shape[:2]
        br_h, br_w = cam_back_right.shape[:2]

        # identify coordinates of all 6 cam images
        x_front, y_front = self.place_bottom_left_in_rect(rx, ry, rw, rh, f_w, f_h, 10, 10)
        x_back_left, y_back_left = self.place_bottom_right_in_rect(rx, ry, rw, rh, bl_w, bl_h, 10, 200)
        x_front_left, y_front_left = self.place_bottom_right_in_rect(rx, ry, rw, rh, fl_w, fl_h, 350, 140)
        x_front_right,y_front_right = self.place_top_left_in_rect(rx, ry, rw, rh, fr_w, fr_h, 50, 100)
        x_back_right, y_back_right = self.place_top_right_in_rect(rx, ry, rw, rh, br_w, br_h, 420, 30)
        x_back, y_back = self.place_top_right_in_rect(rx, ry, rw, rh, b_w, b_h, 50, 60)

        # add cam images to canvas
        self.paste_rgba(canvas, cam_front, x_front, y_front, border=True, thickness=4, color=(0, 192, 255))
        self.paste_rgba(canvas, cam_back_left, x_back_left, y_back_left, border=True)
        self.paste_rgba(canvas, cam_front_left, x_front_left, y_front_left, border=True)
        self.paste_rgba(canvas, cam_front_right, x_front_right, y_front_right, border=True)
        self.paste_rgba(canvas, cam_back_right, x_back_right, y_back_right, border=True)
        self.paste_rgba(canvas, cam_back, x_back, y_back, border=True)

        # connectors (surrounding lines of cam images)
        self.draw_connectors(canvas, x_car, y_car, coord['front_right'], x_front_right, y_front_right, fr_w, h_dst=fr_h, use_bottom=True)
        self.draw_connectors(canvas, x_car, y_car, coord['back_right'], x_back_right, y_back_right, br_w, h_dst=br_h, use_bottom=True)
        self.draw_connectors(canvas, x_car, y_car, coord['back'], x_back, y_back, b_w, h_dst=b_h, use_bottom=True)
        self.paste_rgba(canvas, car, x_car, y_car, border=False)
        self.draw_connectors(canvas, x_car, y_car, coord['front'], x_front, y_front, f_w, color=(0, 192, 255))
        self.draw_connectors(canvas, x_car, y_car, coord['front_left'], x_front_left, y_front_left, fl_w)
        self.draw_connectors(canvas, x_car, y_car, coord['back_left'], x_back_left, y_back_left, bl_w, use_bottom=True)
        return canvas

    def create_canvas(self, canvas_shape, car_img, logo_img, is_ver1):
        canvas_h, canvas_w = canvas_shape
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        return self.draw_canvas(canvas, car_img, logo_img, is_ver1)

    def create_visual(self, index, visualize_result, car_path, logo_path, fps):
        car_img = cv2.imread(car_path, cv2.IMREAD_UNCHANGED)
        logo_img = cv2.imread(logo_path)
        token_id = list(visualize_result['plan_results_ttnn'].keys())[0]

        plan_cmd = np.argmax(visualize_result['plan_results_ttnn'][token_id][1][0, 0, 0].cpu().numpy())
        direction = self.direction_cmd[plan_cmd]
        for cam_idx, cam_name in enumerate(self.cam_names):
            cam_img = cv2.imread(self.dataset.get_data_info(index)['img_filename'][cam_idx])
            if cam_name == 'CAM_FRONT':
                lidar2img = self.dataset.get_data_info(index)['lidar2img'][cam_idx]
                cam_img = self.draw_ttnn_trajectory(cam_img, token_id, visualize_result, lidar2img)
            cam_img = self.add_cam_label(cam_img, cam_name)
            self.cam_imgs[cam_name] = cam_img

        # add visualization version checking
        is_ver1 = None
        if "1" in car_path:
            is_ver1 = True
        elif "2" in car_path:
            is_ver1 = False
        self.visualize_img = self.create_canvas(canvas_shape=self.canvas_shape, car_img=car_img, logo_img=logo_img, is_ver1=is_ver1)
        cv2.putText(self.visualize_img, f'Direction: {direction}', (750, self.canvas_shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        if fps != 0:
            cv2.putText(self.visualize_img, f'FPS: {fps}', (750, self.canvas_shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    def get_visual(self):
        return self.visualize_img
    
    def set_visual(self, new_visual):
        self.visualize_img = new_visual

    def show_realtime(self, resize_to=(1280, 720)):
        """
        Show current visualize_img once (non-blocking).
        Returns:
            True  -> keep running
            False -> user requested to quit (Esc or 'q')
        """
        if self.visualize_img is None:
            return True  

        if not self._win_open:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            if resize_to is not None:
                cv2.resizeWindow(self.window_name, *resize_to)
            self._win_open = True

        cv2.imshow(self.window_name, self.visualize_img)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # Esc or 'q'
            self.close_realtime()
            return False
        return True

    def close_realtime(self):
        if self._win_open:
            cv2.destroyWindow(self.window_name)
            self._win_open = False
    
    def set_output(self, video_path):
        self.video_path = video_path

    def show_video(self):
        if self.video is None:  
            h, w = self.visualize_img.shape[:2]
            fourccs = [cv2.VideoWriter_fourcc(*"mp4v"), cv2.VideoWriter_fourcc(*"avc1")]
            opened = False
            for f4 in fourccs:
                v = cv2.VideoWriter(str(self.video_path), f4, float(5), (w, h), True)
                if v.isOpened():
                    self.video = v
                    opened = True
                    break
            if not opened:  
                print("[ERROR] Cannot open VideoWriter for", self.video_path)

        # Write frame to video
        if self.video is not None:
            self.video.write(self.visualize_img)
            
    def close_video(self):
        self.video.release()