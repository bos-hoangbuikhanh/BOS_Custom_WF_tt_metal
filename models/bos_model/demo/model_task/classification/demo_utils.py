import math
import queue

import cv2
import numpy as np

GRAY_BOX_CACHE = {}  # (w, h, (b,g,r)) -> np.ndarray
SIDEBAR_BASE_CACHE = {}  # (width, height, batch_size) -> np.ndarray (except FPS)
LOGO_CACHE = None  # logo image cache

GRAY = (128, 128, 128)  # cv2 COLOR (B, G, R)
WHITE = (255, 255, 255)  # cv2 COLOR (B, G, R)
GREEN = (0, 255, 0)  # cv2 COLOR (B, G, R)
RED = (0, 0, 255)  # cv2 COLOR (B, G, R)


def get_gray_box(w, h):
    key = (w, h, GRAY)
    if key not in GRAY_BOX_CACHE:
        GRAY_BOX_CACHE[key] = np.full((h, w, 3), GRAY, dtype=np.uint8)
    return GRAY_BOX_CACHE[key]


def load_RGBA_logo(path, target_w):
    global LOGO_CACHE
    if LOGO_CACHE is not None:
        return LOGO_CACHE
    else:
        logo = np.repeat(np.expand_dims(cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, 3], axis=2), 3, axis=2)
        logo_h, logo_w = logo.shape[:2]
        scale = min(1.0, float(target_w) / max(1, logo_w))
        logo_h_rs, logo_w_rs = max(1, int(logo_h * scale)) - 20, max(1, int(logo_w * scale)) - 20
        LOGO_CACHE = cv2.resize(logo, (logo_w_rs, logo_h_rs), interpolation=cv2.INTER_AREA)
        return LOGO_CACHE


def draw_sidebar(model_name, batch_size, fps, width=480, height=1080):
    key = (width, height, batch_size)

    if key not in SIDEBAR_BASE_CACHE:
        base = np.zeros((height, width, 3), dtype=np.uint8)

        logo = load_RGBA_logo("models/bos_model/demo/resources/BOS_logo.png", target_w=width)
        lh, lw = logo.shape[:2]
        base[10 : 10 + lh, 10 : 10 + lw] = logo

        x_pad = 20
        y_cursor = 10 + lh + 40

        cv2.putText(base, "Eagle-N based", (x_pad, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 1.8, WHITE, 4, cv2.LINE_AA)
        y_cursor += 60
        cv2.putText(
            base, "Object Classification", (x_pad, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 1.35, WHITE, 3, cv2.LINE_AA
        )
        y_cursor += 60
        cv2.line(base, (x_pad, y_cursor), (width - 20, y_cursor), WHITE, 2, cv2.LINE_AA)  # split line
        y_cursor += 110
        fps_origin = (x_pad, y_cursor)  # FPS space
        y_cursor += 60
        cv2.line(base, (x_pad, y_cursor), (width - 20, y_cursor), WHITE, 2, cv2.LINE_AA)  # split line
        y_cursor += 70
        cv2.putText(
            base, f"Model: {model_name}", (x_pad, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA
        )
        y_cursor += 100
        cv2.putText(
            base, "Dataset: ImageNet-1K", (x_pad, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA
        )
        y_cursor += 100
        cv2.putText(
            base, f"Batch Size: {batch_size}", (x_pad, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA
        )

        y_cursor += 80
        cv2.putText(
            base, "Space: Pause/Resume", (x_pad, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.95, WHITE, 2, cv2.LINE_AA
        )
        y_cursor += 40
        cv2.putText(base, "Q or ESC: Quit", (x_pad, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.95, WHITE, 2, cv2.LINE_AA)

        SIDEBAR_BASE_CACHE[key] = (base, fps_origin)

    base, fps_origin = SIDEBAR_BASE_CACHE[key]
    sidebar = base.copy()
    cv2.putText(
        sidebar, f"FPS: {fps:.2f}", fps_origin, cv2.FONT_HERSHEY_SIMPLEX, 2.2, WHITE, 6, cv2.LINE_AA
    )  # FPS overwrite

    return sidebar


def draw_result_window(batch, batch_images, batch_predictions, label_dict, width=1440, height=1080):
    tile_w, tile_h = width // int(math.ceil(math.sqrt(batch))), height // int(math.ceil(math.sqrt(batch)))
    grid = np.zeros((height, width, 3), dtype=np.uint8)

    result_box_h, result_box_w = 120, 660
    result_box_alpha = 0.7
    box_x1, box_y1 = 30, tile_h - 185
    gray_box = get_gray_box(result_box_w, result_box_h)

    batch_predictions = np.array([label_dict[batch_predictions[i]] for i in range(batch)])
    batch_expected = np.array([label_dict[batch_images["label"][i]] for i in range(batch)])
    batch_is_correct = batch_predictions == batch_expected

    for i in range(batch):
        img = cv2.imread(batch_images["image"][i], cv2.IMREAD_COLOR)
        tile = cv2.resize(
            img,
            (tile_w, tile_h),
            interpolation=cv2.INTER_AREA if (img.shape[1] > tile_w or img.shape[0] > tile_h) else cv2.INTER_LINEAR,
        )  # resize
        roi = tile[box_y1 : box_y1 + result_box_h, box_x1 : box_x1 + result_box_w]

        correctness_color = GREEN if batch_is_correct[i] else RED
        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), correctness_color, 18, lineType=cv2.LINE_8)
        cv2.addWeighted(gray_box, result_box_alpha, roi, 1 - result_box_alpha, 0, roi)
        cv2.putText(
            tile,
            f"EXPT: {batch_expected[i]}",
            (50, box_y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            correctness_color,
            2,
            cv2.LINE_8,
        )
        cv2.putText(
            tile,
            f"PRED: {batch_predictions[i]}",
            (50, box_y1 + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            correctness_color,
            2,
            cv2.LINE_8,
        )

        r, c = divmod(i, 2)
        y0, x0 = r * tile_h, c * tile_w
        grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    return grid


def compose_sidebar_result(sidebar_img, grid_img):
    frame = np.empty((grid_img.shape[0], sidebar_img.shape[1] + grid_img.shape[1], 3), dtype=np.uint8)
    sw = sidebar_img.shape[1]
    frame[:, :sw] = sidebar_img
    frame[:, sw:] = grid_img
    return frame


def draw_paused_messsage(frame):
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    overlay_box = overlay.copy()
    cv2.rectangle(overlay_box, (0, 0), (w, 80), GRAY, -1)
    overlay = cv2.addWeighted(overlay_box, 0.6, overlay, 0.4, 0)
    cv2.putText(
        overlay,
        "PAUSED - Press SPACE to resume, Q/ESC to quit",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    return overlay


def ui_thread(frame_queue, stop_flag, paused_flag, demo_window_name, demo_fullscreen=False):
    cv2.namedWindow(demo_window_name, cv2.WINDOW_NORMAL)
    if demo_fullscreen:
        cv2.setWindowProperty(demo_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    last_frame = None
    while not stop_flag.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
            if frame is None:
                break
            last_frame = frame
        except queue.Empty:
            pass

        if paused_flag.is_set() and last_frame is not None:
            display_frame = draw_paused_messsage(last_frame)
        else:
            display_frame = last_frame

        if display_frame is not None:
            cv2.imshow(demo_window_name, display_frame)

        k = cv2.waitKey(50) & 0xFF
        if k == ord(" ") or k == 32:
            if paused_flag.is_set():
                paused_flag.clear()
            else:
                paused_flag.set()
        elif k in (ord("q"), 27):
            stop_flag.set()
            break
    cv2.destroyAllWindows()
