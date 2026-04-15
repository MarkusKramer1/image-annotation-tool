"""Page 1 — Data Extraction: extract frames from a video file or ROS bag."""

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.common import DATA_DIR, TEST_VIDEO_PATH, TEST_DATASET_NAME, IMAGE_ROOT

# ─── Optional ROS bag dependencies ────────────────────────────────────────────
try:
    import imagehash
    import yaml
    from PIL import Image as PILImage
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag2 import Reader as Ros2Reader

    ROSBAGS_AVAILABLE = True
    _ROSBAGS_IMPORT_ERROR: str | None = None
except Exception as _exc:
    ROSBAGS_AVAILABLE = False
    _ROSBAGS_IMPORT_ERROR = str(_exc)


# ==============================================================================
# ROS BAG UTILITY FUNCTIONS
# ==============================================================================

def msg_to_bgr(msg) -> np.ndarray | None:
    """Convert a deserialized ROS Image or CompressedImage message to BGR numpy array."""
    # CompressedImage: decode directly with OpenCV
    if hasattr(msg, "format"):
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Raw Image: interpret based on encoding
    enc = msg.encoding.lower()
    raw = bytes(msg.data)
    h, w = int(msg.height), int(msg.width)

    if enc in ("rgb8", "rgb"):
        return cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if enc in ("bgr8", "bgr"):
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(h, w), cv2.COLOR_GRAY2BGR)
    if enc in ("mono16", "16uc1"):
        img16 = np.frombuffer(raw, np.uint16).reshape(h, w)
        return cv2.cvtColor((img16 >> 8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if enc.startswith("bayer_"):
        _BAYER = {
            "bayer_rggb8": cv2.COLOR_BayerBG2BGR,
            "bayer_bggr8": cv2.COLOR_BayerRG2BGR,
            "bayer_gbrg8": cv2.COLOR_BayerGR2BGR,
            "bayer_grbg8": cv2.COLOR_BayerGB2BGR,
        }
        code = _BAYER.get(enc, cv2.COLOR_BayerBG2BGR)
        return cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(h, w), code)
    return None


def _compute_timestamp_offset(timestamp_ns: int, bag_start_ns: int) -> float:
    return float(timestamp_ns - bag_start_ns) / 1e9


def variance_of_laplacian(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def contrast_score(img_gray: np.ndarray) -> float:
    return float(np.std(img_gray)) / 255.0


def brightness_mean(img_gray: np.ndarray) -> float:
    return float(img_gray.mean())


def calculate_motion_score(frame1_gray, frame2_gray) -> float:
    if frame1_gray is None or frame2_gray is None:
        return 0.0
    return float(np.mean(cv2.absdiff(frame1_gray, frame2_gray)))


def hamming(a, b) -> int:
    return int(a - b)


def color_bar(colors_and_stops: list, legend_html: str | None = None) -> None:
    gradient_css = ", ".join(f"{c} {s}%" for c, s in colors_and_stops)
    bar = (
        f'<div style="height:10px; border-radius:5px; '
        f'background:linear-gradient(90deg, {gradient_css}); '
        f'margin-top:5px; margin-bottom:10px;"></div>'
    )
    if legend_html:
        bar += (
            f'<div style="font-size:0.75rem; color:#666; text-align:center; '
            f'margin-top:4px; margin-bottom:15px;">{legend_html}</div>'
        )
    st.markdown(bar, unsafe_allow_html=True)


# ─── 360° Camera undistortion ─────────────────────────────────────────────────

_map_cache: dict = {}


def generate_rectify_map(img_shape, K_dist, D_dist, xi, K_rect):
    h, w = img_shape
    key = (h, w, tuple(K_dist.flatten()), tuple(D_dist.flatten()), xi, tuple(K_rect.flatten()))
    if key in _map_cache:
        return _map_cache[key]
    u_rect, v_rect = np.meshgrid(np.arange(w), np.arange(h))
    pixels_rect = np.stack([u_rect, v_rect, np.ones_like(u_rect)], axis=-1).reshape(-1, 3)
    rays_rect = (np.linalg.inv(K_rect) @ pixels_rect.T).T
    X, Y, Z = rays_rect[:, 0], rays_rect[:, 1], rays_rect[:, 2]
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    denom = Z + xi * norm
    x_p = X / denom
    y_p = Y / denom
    r2 = x_p**2 + y_p**2
    r4 = r2 * r2
    k1, k2, p1, p2 = D_dist[0], D_dist[1], D_dist[2], D_dist[3]
    x_d = x_p * (1 + k1 * r2 + k2 * r4) + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p**2)
    y_d = y_p * (1 + k1 * r2 + k2 * r4) + p1 * (r2 + 2 * y_p**2) + 2 * p2 * x_p * y_p
    map1 = (K_dist[0, 0] * x_d + K_dist[0, 2]).reshape(h, w).astype(np.float32)
    map2 = (K_dist[1, 1] * y_d + K_dist[1, 2]).reshape(h, w).astype(np.float32)
    _map_cache[key] = (map1, map2)
    return map1, map2


def undistort_kalibr_image(image_bgr, K_dist, D_dist, xi, zoom: float = 1.0):
    h, w = image_bgr.shape[:2]
    K_rect = K_dist.copy().astype(float)
    K_rect[0, 0] *= zoom
    K_rect[1, 1] *= zoom
    map1, map2 = generate_rectify_map((h, w), K_dist, D_dist, xi, K_rect)
    return cv2.remap(image_bgr, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _validate_camera_matrix(arr, name: str) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64)
    if arr.size != 9:
        raise ValueError(f"{name}: expected 9 elements, got {arr.size}")
    return arr.reshape(3, 3)


def _validate_distortion_coeffs(arr, name: str) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64).flatten()
    if arr.size < 4:
        raise ValueError(f"{name}: expected ≥4 coefficients, got {arr.size}")
    return arr


# ─── ROS bag I/O helpers ──────────────────────────────────────────────────────

def open_reader(bag_path: Path):
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")
    try:
        return AnyReader([bag_path])
    except Exception:
        try:
            return Ros2Reader(str(bag_path))
        except Exception as exc:
            raise RuntimeError(f"Cannot open {bag_path.name}: {exc}") from exc


@st.cache_data(show_spinner=False, ttl=300)
def find_bagfiles_cached(parent_dir: str) -> list:
    bagfiles = []
    parent = Path(parent_dir)
    if parent.is_dir() and (parent / "metadata.yaml").exists():
        if any(f.suffix == ".mcap" for f in parent.iterdir()):
            bagfiles.append(parent)
    for p in parent.iterdir():
        if p.is_file() and p.suffix == ".bag":
            bagfiles.append(p)
    for p in parent.rglob("*"):
        if p.is_dir() and (p / "metadata.yaml").exists():
            if any(f.suffix == ".mcap" for f in p.iterdir()):
                bagfiles.append(p)
        elif p.is_file() and p.suffix == ".bag":
            bagfiles.append(p)
    return list(dict.fromkeys(bagfiles))


@st.cache_data(show_spinner=False, ttl=300)
def get_bag_fps_cached(bag_path_str: str, topic: str) -> float:
    """Estimate the recording frame rate of *topic* in the bag (frames / second)."""
    try:
        with open_reader(Path(bag_path_str)) as reader:
            t_start = getattr(reader, "start_time", None)
            t_end = getattr(reader, "end_time", None)
            conns = [c for c in reader.connections if c.topic == topic]
            total_msgs = sum(c.msgcount for c in conns)
            if t_start and t_end and total_msgs > 1:
                duration_s = (t_end - t_start) / 1e9
                if duration_s > 0:
                    return total_msgs / duration_s
    except Exception:
        pass
    return 0.0


@st.cache_data(show_spinner=False, ttl=300)
def get_image_topics_cached(bag_path_str: str) -> list:
    topics: set[str] = set()
    try:
        with open_reader(Path(bag_path_str)) as reader:
            for conn in reader.connections:
                if "Image" in conn.msgtype or "CompressedImage" in conn.msgtype:
                    topics.add(conn.topic)
    except Exception as exc:
        st.warning(f"Error reading topics from {Path(bag_path_str).name}: {exc}")
    return sorted(topics)


@st.cache_data(show_spinner="Building frame index…", ttl=600)
def build_frame_index(bag_path_str: str, topic: str, max_frames: int, width: int) -> list:
    frames = []
    bag_path = Path(bag_path_str)
    try:
        with open_reader(bag_path) as reader:
            bag_start = getattr(reader, "start_time", None)
            conns = [c for c in reader.connections if c.topic == topic]
            if not conns:
                return []
            processed = 0
            for conn, ts, raw in reader.messages(connections=conns):
                if processed >= max_frames:
                    break
                try:
                    msg = reader.deserialize(raw, conn.msgtype)
                    frame_bgr = msg_to_bgr(msg)
                    if frame_bgr is None:
                        continue
                    h, w_img = frame_bgr.shape[:2]
                    if w_img > width:
                        frame_bgr = cv2.resize(
                            frame_bgr, (width, int(h * width / w_img)), interpolation=cv2.INTER_AREA
                        )
                    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ok:
                        continue
                    ts_s = (
                        _compute_timestamp_offset(int(ts), int(bag_start))
                        if bag_start is not None
                        else float(ts) / 1e9
                    )
                    frames.append((buf.tobytes(), ts_s))
                    processed += 1
                except Exception:
                    continue
    except Exception as exc:
        st.error(f"Failed to index frames: {exc}")
    return frames


# ─── Core bag processing ──────────────────────────────────────────────────────

def process_bag(
    bag_path: Path,
    image_topic: str,
    out_dir: Path,
    discard_dir: Path,
    blur_gate: float,
    hash_thresh: int,
    motion_thresh: float,
    min_contrast: float,
    min_bright: float,
    max_bright: float,
    max_frames: int | None,
    start_time: int | None,
    end_time: int | None,
    progress_bar,
    status_text,
    save_discarded: bool = False,
    fisheye: bool = False,
    calib: dict | None = None,
    zoom: float = 1.0,
    sample_fps: float = 0.0,
) -> dict:
    seen_hashes: list = []
    total_images = 0
    exported_images = 0
    prev_frame_gray = None
    resolution: tuple | None = None
    _sample_interval_ns: int | None = int(1e9 / sample_fps) if sample_fps > 0 else None
    _last_kept_ts_ns: int | None = None
    discard_stats = {
        "discarded_motion": 0,
        "discarded_blur": 0,
        "discarded_duplicate": 0,
        "discarded_quality": 0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    if save_discarded:
        discard_dir.mkdir(parents=True, exist_ok=True)

    with open_reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == image_topic]
        if not connections:
            st.error(f"Topic '{image_topic}' not found in '{bag_path.name}'.")
            return {"total_images": 0, "exported_images": 0, "resolution": None, **discard_stats}

        bag_start = getattr(reader, "start_time", None)
        total_messages = sum(c.msgcount for c in connections)
        if total_messages == 0:
            st.warning(f"No messages on '{image_topic}' in {bag_path.name}.")
            return {"total_images": 0, "exported_images": 0, "resolution": None, **discard_stats}

        frame_count = 0
        processed_messages = 0

        for conn, ts, rawdata in reader.messages(connections=connections):
            processed_messages += 1
            if (start_time is not None and ts < start_time) or (end_time is not None and ts > end_time):
                continue
            if max_frames is not None and frame_count >= max_frames:
                break

            if _sample_interval_ns is not None:
                if _last_kept_ts_ns is not None and (ts - _last_kept_ts_ns) < _sample_interval_ns:
                    progress_bar.progress(min(processed_messages / total_messages, 1.0))
                    continue
                _last_kept_ts_ns = ts

            total_images += 1
            frame_count += 1
            progress_bar.progress(min(processed_messages / total_messages, 1.0))

            msg = reader.deserialize(rawdata, conn.msgtype)
            frame_bgr = msg_to_bgr(msg)
            if frame_bgr is None:
                continue

            if resolution is None:
                h_f, w_f = frame_bgr.shape[:2]
                resolution = (w_f, h_f)

            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            try:
                ts_s = (
                    _compute_timestamp_offset(int(ts), int(bag_start))
                    if bag_start is not None
                    else float(ts) / 1e9
                )
            except Exception:
                ts_s = float(frame_count)

            stem = bag_path.stem

            # ── Fisheye branch ────────────────────────────────────────────────
            if fisheye and calib is not None:
                try:
                    h_f, w_f = frame_bgr.shape[:2]
                    half = w_f // 2
                    left_dist = frame_bgr[:, :half]
                    right_dist = frame_bgr[:, half:]

                    K_l = _validate_camera_matrix(calib["left_camera"]["camera_matrix_K"], "left K")
                    D_l = _validate_distortion_coeffs(calib["left_camera"]["distortion_coeffs_D"], "left D")
                    xi_l = float(calib["left_camera"]["xi"])
                    K_r = _validate_camera_matrix(calib["right_camera"]["camera_matrix_K"], "right K")
                    D_r = _validate_distortion_coeffs(calib["right_camera"]["distortion_coeffs_D"], "right D")
                    xi_r = float(calib["right_camera"]["xi"])

                    status_text.text(f"Undistorting frame {frame_count}…")
                    und_left = cv2.rotate(
                        undistort_kalibr_image(left_dist, K_l, D_l, xi_l, zoom), cv2.ROTATE_90_CLOCKWISE
                    )
                    und_right = cv2.rotate(
                        undistort_kalibr_image(right_dist, K_r, D_r, xi_r, zoom), cv2.ROTATE_90_COUNTERCLOCKWISE
                    )
                    gray_l = cv2.cvtColor(und_left, cv2.COLOR_BGR2GRAY)
                    gray_r = cv2.cvtColor(und_right, cv2.COLOR_BGR2GRAY)

                    if motion_thresh > 0:
                        if prev_frame_gray is not None and calculate_motion_score(prev_frame_gray, gray_l) < motion_thresh:
                            discard_stats["discarded_motion"] += 1
                            if save_discarded:
                                cv2.imwrite(str(discard_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_motion.png"), frame_bgr)
                            prev_frame_gray = gray_l
                            continue
                    prev_frame_gray = gray_l

                    if blur_gate > 0:
                        if variance_of_laplacian(gray_l) < blur_gate or variance_of_laplacian(gray_r) < blur_gate:
                            discard_stats["discarded_blur"] += 1
                            if save_discarded:
                                cv2.imwrite(str(discard_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_blur.png"), frame_bgr)
                            continue

                    if hash_thresh > 0:
                        try:
                            ph = imagehash.dhash(PILImage.fromarray(gray_l))
                            if any(hamming(ph, old) <= hash_thresh for old in seen_hashes):
                                discard_stats["discarded_duplicate"] += 1
                                if save_discarded:
                                    cv2.imwrite(str(discard_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_duplicate.png"), frame_bgr)
                                continue
                            seen_hashes.append(ph)
                        except Exception:
                            pass

                    if min_contrast > 0.0 or min_bright > 0 or max_bright < 255:
                        cl = contrast_score(gray_l)
                        cr = contrast_score(gray_r)
                        bl = brightness_mean(gray_l)
                        br = brightness_mean(gray_r)
                        if not (cl >= min_contrast and min_bright <= bl <= max_bright
                                and cr >= min_contrast and min_bright <= br <= max_bright):
                            discard_stats["discarded_quality"] += 1
                            if save_discarded:
                                reason = (
                                    "lowcontrast" if cl < min_contrast or cr < min_contrast
                                    else "dark" if bl < min_bright or br < min_bright
                                    else "bright"
                                )
                                cv2.imwrite(str(discard_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_{reason}.png"), frame_bgr)
                            continue

                    cv2.imwrite(str(out_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_left.png"), und_left)
                    cv2.imwrite(str(out_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_right.png"), und_right)
                    exported_images += 2

                except Exception as exc:
                    status_text.text(f"⚠️ Undistortion error: {exc}")
                    cv2.imwrite(str(out_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}.png"), frame_bgr)
                    exported_images += 1

            # ── Standard branch ───────────────────────────────────────────────
            else:
                if motion_thresh > 0:
                    if prev_frame_gray is not None and calculate_motion_score(prev_frame_gray, frame_gray) < motion_thresh:
                        discard_stats["discarded_motion"] += 1
                        if save_discarded:
                            cv2.imwrite(str(discard_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_motion.png"), frame_bgr)
                        prev_frame_gray = frame_gray
                        continue
                prev_frame_gray = frame_gray

                if blur_gate > 0:
                    if variance_of_laplacian(frame_gray) < blur_gate:
                        discard_stats["discarded_blur"] += 1
                        if save_discarded:
                            cv2.imwrite(str(discard_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_blur.png"), frame_bgr)
                        continue

                if hash_thresh > 0:
                    try:
                        ph = imagehash.dhash(PILImage.fromarray(frame_gray))
                        if any(hamming(ph, old) <= hash_thresh for old in seen_hashes):
                            discard_stats["discarded_duplicate"] += 1
                            if save_discarded:
                                cv2.imwrite(str(discard_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_duplicate.png"), frame_bgr)
                            continue
                        seen_hashes.append(ph)
                    except Exception:
                        pass

                if min_contrast > 0.0 or min_bright > 0 or max_bright < 255:
                    contr = contrast_score(frame_gray)
                    bri = brightness_mean(frame_gray)
                    if not (contr >= min_contrast and min_bright <= bri <= max_bright):
                        discard_stats["discarded_quality"] += 1
                        if save_discarded:
                            reason = (
                                "lowcontrast" if contr < min_contrast
                                else "dark" if bri < min_bright
                                else "bright"
                            )
                            cv2.imwrite(str(discard_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}_{reason}.png"), frame_bgr)
                        continue

                cv2.imwrite(str(out_dir / f"{stem}_{ts_s:.3f}_{frame_count:04d}.png"), frame_bgr)
                exported_images += 1

    return {
        "total_images": total_images,
        "exported_images": exported_images,
        "resolution": resolution,
        **discard_stats,
    }


# ==============================================================================
# PAGE
# ==============================================================================

st.set_page_config(page_title="Data Extraction", page_icon="🎞️", layout="wide")

st.title("Data Extraction")
st.caption(
    "Extract frames from a video file or a ROS bag (ROS1 `.bag` / ROS2 `.mcap`). "
    "The extracted images become the shared image source for all subsequent pipeline stages."
)
st.divider()

# ─── Source type selector ──────────────────────────────────────────────────────

source_type = st.radio(
    "Source type",
    ["Video file", "ROS bag"],
    index=1,
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO FILE  (original functionality, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

if source_type == "Video file":
    st.subheader("Configuration")

    frame_step = st.number_input(
        "Extract every N-th frame",
        min_value=1,
        max_value=200,
        value=5,
        help="1 = every frame. For a 30 fps video, 5 gives ~6 fps.",
    )

    st.divider()
    st.subheader("Video Source")

    col_upload, col_test = st.columns([3, 1])

    with col_upload:
        video_file = st.file_uploader("Video file (.mp4, .mov)", type=["mp4", "mov"])

    with col_test:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        test_available = TEST_VIDEO_PATH.exists()
        test_clicked = st.button(
            "Test Extraction",
            disabled=not test_available,
            help=(
                f"Extract frames from `{TEST_VIDEO_PATH.name}` into `data/{TEST_DATASET_NAME}/`."
                if test_available
                else "Test video not found."
            ),
            use_container_width=True,
        )

    if "extraction_test_mode" not in st.session_state:
        st.session_state.extraction_test_mode = False
    if test_clicked:
        st.session_state.extraction_test_mode = True

    is_test_mode = st.session_state.extraction_test_mode

    _default_name = ""
    if is_test_mode:
        _default_name = TEST_DATASET_NAME
    elif video_file is not None:
        _default_name = Path(video_file.name).stem

    dataset_name = st.text_input(
        "Dataset name",
        value=_default_name,
        placeholder="my_video",
        help=f"Frames will be saved to `data/<name>/{IMAGE_ROOT}/`.",
    )

    if dataset_name.strip():
        st.caption(f"Output directory: `{DATA_DIR / dataset_name.strip() / IMAGE_ROOT}`")

    st.divider()

    has_video = is_test_mode or video_file is not None
    can_run = has_video and bool(dataset_name.strip())

    run_clicked = st.button("Extract Frames", type="primary", disabled=not can_run)

    if not can_run and not is_test_mode:
        missing = []
        if not has_video:
            missing.append("a video file (or click Test Extraction)")
        if not dataset_name.strip():
            missing.append("a dataset name")
        if missing:
            st.info(f"Please provide {', '.join(missing)} to continue.")

    should_run = run_clicked or (is_test_mode and test_clicked)

    if should_run and can_run:
        out_dir = (DATA_DIR / dataset_name.strip()).resolve()
        frames_dir = out_dir / IMAGE_ROOT
        ann_dir = out_dir / "annotations"

        frames_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        status_text = st.empty()
        cleanup_video = False

        if is_test_mode:
            status_text.info(f"Using test video: `{TEST_VIDEO_PATH.name}`")
            tmp_video_path = str(TEST_VIDEO_PATH)
        else:
            status_text.info("Saving uploaded video to disk…")
            upload_suffix = Path(video_file.name).suffix.lower()
            if upload_suffix not in {".mp4", ".mov"}:
                st.error(f"Unsupported file type `{upload_suffix}`.")
                st.stop()
            with tempfile.NamedTemporaryFile(suffix=upload_suffix, delete=False) as tmp:
                tmp.write(video_file.read())
                tmp_video_path = tmp.name
            cleanup_video = True

        status_text.info("Extracting frames…")
        cap = cv2.VideoCapture(tmp_video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_progress = st.progress(0.0, text="Extracting frames…")
        extract_count = 0
        video_frame_idx = 0
        resolution: tuple[int, int] | None = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if video_frame_idx % int(frame_step) == 0:
                cv2.imwrite(str(frames_dir / f"frame_{extract_count:06d}.jpg"), frame)
                if resolution is None:
                    h, w = frame.shape[:2]
                    resolution = (w, h)
                extract_count += 1
            video_frame_idx += 1
            if total_video_frames > 0:
                frame_progress.progress(
                    video_frame_idx / total_video_frames,
                    text=f"Extracting frames ({video_frame_idx}/{total_video_frames})…",
                )

        cap.release()
        if cleanup_video:
            os.unlink(tmp_video_path)
        frame_progress.empty()

        if extract_count == 0:
            st.error("No frames could be extracted from the video.")
            st.stop()

        source_name = TEST_VIDEO_PATH.name if is_test_mode else video_file.name
        meta = {
            "source_video": source_name,
            "frame_step": int(frame_step),
            "num_frames": extract_count,
            "resolution": list(resolution) if resolution else None,
            "fps": round(fps, 2) if fps else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(ann_dir / "extraction.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        st.session_state.extraction_test_mode = False
        status_text.empty()

        st.success(
            f"Extracted **{extract_count}** frames to "
            f"`{frames_dir.relative_to(out_dir.parent.parent)}`"
        )

        sample_frames = sorted(frames_dir.iterdir())[:20]
        if sample_frames:
            cols = st.columns(5)
            for i, fp in enumerate(sample_frames):
                cols[i % 5].image(str(fp), caption=fp.name, use_container_width=True)
            if extract_count > 20:
                st.caption(f"Showing first 20 of {extract_count} frames.")


# ══════════════════════════════════════════════════════════════════════════════
# ROS BAG
# ══════════════════════════════════════════════════════════════════════════════

else:
    if not ROSBAGS_AVAILABLE:
        st.error(
            f"Required packages failed to import: `{_ROSBAGS_IMPORT_ERROR}`. "
            "Install `rosbags`, `ImageHash`, and `pyyaml` into the environment and restart the app."
        )
        st.stop()

    # ── Bag source ─────────────────────────────────────────────────────────────

    bag_parent_input = st.text_input(
        "Parent directory containing ROS bags",
        value=st.session_state.get("_rb_bag_parent", ""),
        placeholder="/data/bags",
        help=(
            "Path to a folder containing `.bag` files (ROS1) or `.mcap` folders (ROS2). "
            "Subdirectories are searched recursively."
        ),
    )
    st.session_state["_rb_bag_parent"] = bag_parent_input

    uploaded_bag_files = st.file_uploader(
        "Or upload bag file(s) directly",
        type=["bag", "mcap", "db3"],
        accept_multiple_files=True,
        help="Upload ROS1 `.bag`, ROS2 `.mcap`, or `.db3` files. These are added to (or used instead of) the directory scan above.",
        key="_rb_bag_uploader",
    )

    # Persist uploaded bags to temp files across reruns
    if uploaded_bag_files:
        saved_uploads: dict[str, str] = st.session_state.get("_rb_uploaded_bag_paths", {})
        for uf in uploaded_bag_files:
            if uf.name not in saved_uploads:
                suffix = Path(uf.name).suffix or ".bag"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uf.read())
                    saved_uploads[uf.name] = tmp.name
        st.session_state["_rb_uploaded_bag_paths"] = saved_uploads
    else:
        # Clear saved uploads when uploader is cleared
        for p in st.session_state.get("_rb_uploaded_bag_paths", {}).values():
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        st.session_state["_rb_uploaded_bag_paths"] = {}

    bagfile_paths: list = []

    if bag_parent_input:
        try:
            bagfile_paths = find_bagfiles_cached(bag_parent_input)
        except Exception as exc:
            st.error(f"Error scanning directory: {exc}")

    uploaded_paths = [
        Path(p) for p in st.session_state.get("_rb_uploaded_bag_paths", {}).values()
    ]
    bagfile_paths = list(dict.fromkeys(bagfile_paths + uploaded_paths))

    if bag_parent_input or uploaded_paths:
        if not bagfile_paths:
            st.warning("No ROS bag files found.")
        else:
            dir_count = len([p for p in bagfile_paths if p not in uploaded_paths])
            up_count = len(uploaded_paths)
            parts = []
            if dir_count:
                parts.append(f"**{dir_count}** from directory")
            if up_count:
                parts.append(f"**{up_count}** uploaded")
            st.success(f"Found {' + '.join(parts)} bag file(s).")

    # ── Topic discovery ────────────────────────────────────────────────────────

    image_topic: str = ""
    if bagfile_paths:
        all_topics: set[str] = set()
        for bp in bagfile_paths:
            all_topics.update(get_image_topics_cached(str(bp)))
        all_topics_sorted = sorted(all_topics)

        st.subheader("Image Topic")
        if all_topics_sorted:
            image_topic = st.selectbox(
                "Select image topic",
                all_topics_sorted,
                help="Topic used for preview and as default extraction topic.",
            )
        else:
            st.warning("No image topics detected automatically.")
            image_topic = st.text_input(
                "Image topic name",
                placeholder="/camera/image_raw",
            )

    # ── Bag analysis (optional) ────────────────────────────────────────────────

    if bagfile_paths:
        if st.button("Analyze bag files (optional)", help="Count frames and durations for all bags."):
            bag_stats_list = []
            with st.spinner("Analyzing…"):
                for bp in bagfile_paths:
                    try:
                        with open_reader(bp) as reader:
                            t_start = getattr(reader, "start_time", None)
                            t_end = getattr(reader, "end_time", None)
                            duration = (t_end - t_start) / 1e9 if t_start and t_end else None
                            conns = [c for c in reader.connections if c.topic == image_topic]
                            total_msgs = sum(c.msgcount for c in conns)
                    except Exception:
                        total_msgs = 0
                        duration = None
                    bag_stats_list.append({
                        "Path": str(bp),
                        "Topic frames": total_msgs,
                        "Duration (s)": round(duration, 1) if duration else "—",
                    })
            st.session_state["_rb_bag_stats"] = bag_stats_list

        if "_rb_bag_stats" in st.session_state:
            st.subheader(f"Bag analysis ({len(bagfile_paths)} file(s))")
            st.dataframe(pd.DataFrame(st.session_state["_rb_bag_stats"]), use_container_width=True)

    # ── Frame preview ──────────────────────────────────────────────────────────

    if bagfile_paths:
        with st.expander("Frame preview (interactive slider)", expanded=False):
            preview_bag = st.selectbox(
                "Bag to preview",
                bagfile_paths,
                format_func=lambda p: Path(p).name,
                key="_rb_preview_bag",
            )
            topics_for_preview = get_image_topics_cached(str(preview_bag))
            if topics_for_preview:
                preview_topic = st.selectbox("Preview topic", topics_for_preview, key="_rb_preview_topic")
            else:
                preview_topic = st.text_input("Preview topic", value=image_topic, key="_rb_preview_topic_manual")

            col_pv1, col_pv2 = st.columns(2)
            with col_pv1:
                preview_max = st.number_input(
                    "Max preview frames", min_value=10, max_value=2000, value=300, step=50,
                )
            with col_pv2:
                preview_width = st.number_input(
                    "Preview width (px)", min_value=320, max_value=1920, value=640, step=160,
                )

            if st.button("Build frame index"):
                frames_idx = build_frame_index(
                    str(preview_bag), preview_topic, int(preview_max), int(preview_width)
                )
                st.session_state["_rb_preview_frames"] = frames_idx
                if frames_idx:
                    st.success(f"Indexed {len(frames_idx)} frames.")
                else:
                    st.warning(f"No frames found for topic '{preview_topic}'.")

            frames_idx = st.session_state.get("_rb_preview_frames", [])
            if frames_idx:
                sel = st.slider("Frame", 0, len(frames_idx) - 1, 0)
                jpg_bytes, ts_val = frames_idx[sel]
                st.image(jpg_bytes, use_container_width=True)
                st.markdown(f"**Timestamp:** {ts_val:.6f} s")
            else:
                st.info("Click 'Build frame index' to start.")

    # ── Extraction configuration ───────────────────────────────────────────────

    if bagfile_paths:
        st.divider()
        st.subheader("Extraction Configuration")

        _rb_selected = st.session_state.get("_rb_selected_bags", None)
        if _rb_selected is None:
            _rb_selected = bagfile_paths
        else:
            _rb_selected = [b for b in _rb_selected if b in bagfile_paths] or bagfile_paths

        selected_extract_bags = st.multiselect(
            "Bags to extract",
            bagfile_paths,
            default=_rb_selected,
            format_func=lambda p: Path(p).name,
        )
        if selected_extract_bags != st.session_state.get("_rb_selected_bags"):
            st.session_state.pop("_rb_dataset_name", None)
        st.session_state["_rb_selected_bags"] = selected_extract_bags

        extraction_topics: set[str] = set()
        for b in selected_extract_bags:
            extraction_topics.update(get_image_topics_cached(str(b)))
        extraction_topics_sorted = sorted(extraction_topics)

        if extraction_topics_sorted:
            extraction_topic = st.selectbox(
                "Extraction topic",
                extraction_topics_sorted,
                help="Topic used during extraction (may differ from preview topic).",
            )
        else:
            extraction_topic = st.text_input("Extraction topic", placeholder="/camera/image_raw")

        # Dataset name — default to the first selected bag's stem
        _uploaded_orig_names = list(st.session_state.get("_rb_uploaded_bag_paths", {}).keys())
        _uploaded_temp_set = {str(p) for p in uploaded_paths}
        if selected_extract_bags:
            _first = selected_extract_bags[0]
            if str(_first) in _uploaded_temp_set and _uploaded_orig_names:
                _rb_default_name = Path(_uploaded_orig_names[0]).stem.replace(" ", "_")
            else:
                _rb_default_name = Path(_first).stem.replace(" ", "_")
        elif bag_parent_input:
            _rb_default_name = Path(bag_parent_input).name.replace(" ", "_")
        elif _uploaded_orig_names:
            _rb_default_name = Path(_uploaded_orig_names[0]).stem.replace(" ", "_")
        else:
            _rb_default_name = ""
        rb_dataset_name = st.text_input(
            "Dataset name",
            value=st.session_state.get("_rb_dataset_name", _rb_default_name),
            placeholder="my_rosbag_dataset",
            help=f"Frames will be saved to `data/<name>/{IMAGE_ROOT}/`.",
        )
        st.session_state["_rb_dataset_name"] = rb_dataset_name
        if rb_dataset_name.strip():
            st.caption(f"Output directory: `{DATA_DIR / rb_dataset_name.strip() / IMAGE_ROOT}`")

        # Estimate max FPS across selected bags for the slider ceiling
        _fps_estimates = [
            get_bag_fps_cached(str(b), extraction_topic)
            for b in selected_extract_bags
            if extraction_topic
        ]
        _max_bag_fps = int(max(_fps_estimates)) if _fps_estimates and max(_fps_estimates) > 0 else 60

        # Quick presets
        st.subheader("Quick presets")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button("Outdoor scenes"):
                st.session_state.update({
                    "_rb_motion_thresh": 8, "_rb_blur_gate": 140, "_rb_hash_thresh": 8,
                    "_rb_min_bright": 30, "_rb_max_bright": 245, "_rb_min_contrast": 0.12,
                    "_rb_enable_blur": True, "_rb_enable_dedup": True,
                    "_rb_enable_brightness": True, "_rb_enable_contrast": True,
                })
                st.rerun()
        with col_p2:
            if st.button("Indoor scenes"):
                st.session_state.update({
                    "_rb_motion_thresh": 4, "_rb_blur_gate": 110, "_rb_hash_thresh": 6,
                    "_rb_min_bright": 40, "_rb_max_bright": 235, "_rb_min_contrast": 0.10,
                    "_rb_enable_blur": True, "_rb_enable_dedup": True,
                    "_rb_enable_brightness": True, "_rb_enable_contrast": True,
                })
                st.rerun()

        # Filter form
        with st.form("_rb_filters_form"):
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                st.markdown("**Temporal bounds**")
                start_time_s = st.number_input(
                    "Start time (s)", min_value=0,
                    value=st.session_state.get("_rb_start_time_s", 0), step=1,
                    help="Skip frames before this offset from bag start.",
                )
                end_time_s = st.number_input(
                    "End time (s)", min_value=0,
                    value=st.session_state.get("_rb_end_time_s", 0), step=1,
                    help="Stop after this offset. 0 = process entire bag.",
                )
                st.markdown("**Sampling rate**")
                sample_fps_val = st.slider(
                    "Frames per second (0 = all frames)",
                    min_value=0,
                    max_value=_max_bag_fps,
                    value=st.session_state.get("_rb_sample_fps", 0),
                    help=(
                        f"Limit extraction to at most this many frames per second. "
                        f"0 = keep every frame. Max detected bag rate: {_max_bag_fps} fps."
                    ),
                )
                st.markdown("**Motion filtering**")
                motion_thresh = st.slider(
                    "Motion threshold", 0, 20,
                    st.session_state.get("_rb_motion_thresh", 6),
                    help="Minimum mean pixel difference between frames. 0 = disabled.",
                )
                st.markdown("**Frame limit**")
                max_frames_val = st.number_input(
                    "Max frames per bag", min_value=0,
                    value=st.session_state.get("_rb_max_frames", 0),
                    help="0 = unlimited.",
                )

            with col_f2:
                st.markdown("**Quality filters**")
                enable_blur = st.checkbox("Enable blur filter",
                    value=st.session_state.get("_rb_enable_blur", False))
                enable_dedup = st.checkbox("Enable deduplication",
                    value=st.session_state.get("_rb_enable_dedup", False))
                enable_brightness = st.checkbox("Enable brightness filter",
                    value=st.session_state.get("_rb_enable_brightness", False))
                enable_contrast = st.checkbox("Enable contrast filter",
                    value=st.session_state.get("_rb_enable_contrast", False))

                st.markdown("---")

                blur_gate_val = st.slider(
                    "Blur gate (Laplacian variance)", 0, 500,
                    st.session_state.get("_rb_blur_gate", 160),
                    help="Minimum sharpness score. Higher = stricter.",
                )
                color_bar(
                    [("red", 0), ("red", 12), ("yellow", 24), ("green", 36), ("orange", 56), ("red", 100)],
                    legend_html="Very blurry &larr; &nbsp; <strong>Optimal (120–240)</strong> &nbsp; &rarr; May discard good frames",
                )
                hash_thresh_val = st.slider(
                    "Deduplication threshold (Hamming)", 0, 16,
                    st.session_state.get("_rb_hash_thresh", 6),
                    help="Max perceptual-hash distance. Lower = stricter.",
                )
                color_bar(
                    [("orange", 0), ("green", 6), ("green", 50), ("yellow", 62), ("red", 88)],
                    legend_html="Stricter &larr; &nbsp; <strong>Optimal (2–8)</strong> &nbsp; &rarr; More permissive",
                )
                min_bright_val = st.slider("Min brightness", 0, 255,
                    st.session_state.get("_rb_min_bright", 30))
                max_bright_val = st.slider("Max brightness", 0, 255,
                    st.session_state.get("_rb_max_bright", 235))
                color_bar(
                    [("black", 0), ("#333", 12), ("green", 50), ("#cc0", 88), ("white", 100)],
                    legend_html="Too dark &larr; &nbsp; <strong>Optimal (30–235)</strong> &nbsp; &rarr; Too bright",
                )
                min_contrast_val = st.slider("Min contrast", 0.0, 0.5,
                    st.session_state.get("_rb_min_contrast", 0.12), step=0.01)
                color_bar(
                    [("red", 0), ("yellow", 15), ("green", 30), ("orange", 50), ("red", 80)],
                    legend_html="Low contrast &larr; &nbsp; <strong>Optimal (0.10–0.20)</strong> &nbsp; &rarr; High contrast",
                )

            submitted = st.form_submit_button("Apply filter settings", type="primary", use_container_width=True)
            if submitted:
                st.session_state.update({
                    "_rb_start_time_s": start_time_s, "_rb_end_time_s": end_time_s,
                    "_rb_sample_fps": sample_fps_val,
                    "_rb_motion_thresh": motion_thresh,
                    "_rb_enable_blur": enable_blur, "_rb_blur_gate": blur_gate_val,
                    "_rb_enable_dedup": enable_dedup, "_rb_hash_thresh": hash_thresh_val,
                    "_rb_enable_brightness": enable_brightness,
                    "_rb_min_bright": min_bright_val, "_rb_max_bright": max_bright_val,
                    "_rb_enable_contrast": enable_contrast, "_rb_min_contrast": min_contrast_val,
                    "_rb_max_frames": max_frames_val,
                })
                st.success("Filter settings applied.")

        # Additional options
        with st.expander("Additional options"):
            save_discarded = st.checkbox(
                "Save discarded frames for evaluation",
                value=st.session_state.get("_rb_save_discarded", False),
                help=(
                    "Save rejected frames to `data/<name>/discarded/` with a suffix "
                    "indicating the reason (_motion, _blur, _duplicate, _lowcontrast, _dark, _bright)."
                ),
            )
            st.session_state["_rb_save_discarded"] = save_discarded

            enable_fisheye = st.checkbox(
                "Enable 360° camera undistortion (Kalibr omni-radtan)",
                value=st.session_state.get("_rb_enable_fisheye", False),
                help="Requires a Kalibr calibration YAML with left_camera / right_camera sections.",
            )
            st.session_state["_rb_enable_fisheye"] = enable_fisheye

            if enable_fisheye:
                calib_file = st.file_uploader(
                    "Calibration YAML file", type=["yaml", "yml"], key="_rb_calib_uploader",
                )
                if calib_file:
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".yaml", delete=False) as tmp:
                        tmp.write(calib_file.getbuffer())
                        st.session_state["_rb_calib_path"] = tmp.name
                    st.success(f"Loaded calibration: {calib_file.name}")
                zoom = st.slider(
                    "Rectification zoom", 0.2, 2.0,
                    value=st.session_state.get("_rb_zoom", 0.4), step=0.1,
                    help="< 1 = wider FOV, > 1 = zoom in.",
                )
                st.session_state["_rb_zoom"] = zoom

        # ── Extraction button ──────────────────────────────────────────────────

        st.divider()
        can_extract_rb = bool(selected_extract_bags) and bool(extraction_topic) and bool(rb_dataset_name.strip())

        if st.button("Extract Frames", type="primary", disabled=not can_extract_rb, key="_rb_extract_btn"):
            _start_s = st.session_state.get("_rb_start_time_s", 0)
            _end_s = st.session_state.get("_rb_end_time_s", 0)
            _sample_fps = float(st.session_state.get("_rb_sample_fps", 0))
            _motion = st.session_state.get("_rb_motion_thresh", 0)
            _blur = st.session_state.get("_rb_blur_gate", 0) if st.session_state.get("_rb_enable_blur") else 0
            _hash = st.session_state.get("_rb_hash_thresh", 0) if st.session_state.get("_rb_enable_dedup") else 0
            _min_b = st.session_state.get("_rb_min_bright", 0) if st.session_state.get("_rb_enable_brightness") else 0
            _max_b = st.session_state.get("_rb_max_bright", 255) if st.session_state.get("_rb_enable_brightness") else 255
            _min_c = st.session_state.get("_rb_min_contrast", 0.0) if st.session_state.get("_rb_enable_contrast") else 0.0
            _max_f = st.session_state.get("_rb_max_frames", 0) or None
            _fisheye = st.session_state.get("_rb_enable_fisheye", False)
            _zoom = st.session_state.get("_rb_zoom", 0.4)
            _save_disc = st.session_state.get("_rb_save_discarded", False)

            calib = None
            if _fisheye:
                calib_path = Path(st.session_state.get("_rb_calib_path", ""))
                if not calib_path.exists():
                    st.error("Please upload a calibration YAML file before extracting with fisheye undistortion.")
                    st.stop()
                try:
                    with open(calib_path) as fh:
                        calib = yaml.safe_load(fh)
                    for key in ("left_camera", "right_camera"):
                        if key not in calib:
                            st.error(f"Calibration file missing '{key}' section.")
                            st.stop()
                        for sub in ("camera_matrix_K", "distortion_coeffs_D", "xi"):
                            if sub not in calib[key]:
                                st.error(f"Calibration missing '{sub}' in {key}.")
                                st.stop()
                except Exception as exc:
                    st.error(f"Failed to load calibration: {exc}")
                    st.stop()

            out_root = (DATA_DIR / rb_dataset_name.strip()).resolve()
            frames_dir = out_root / IMAGE_ROOT
            discard_dir = out_root / "discarded"
            ann_dir = out_root / "annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)

            all_stats = []
            total = {k: 0 for k in (
                "total_images", "exported_images",
                "discarded_motion", "discarded_blur", "discarded_duplicate", "discarded_quality",
            )}
            resolution_out: tuple | None = None
            n_bags = len(selected_extract_bags)

            for i, bag_path in enumerate(selected_extract_bags):
                bag_path = Path(bag_path)
                st.info(f"Processing bag {i + 1}/{n_bags}: `{bag_path.name}`")
                pbar = st.progress(0)
                stext = st.empty()

                try:
                    start_ts = end_ts = None
                    with open_reader(bag_path) as reader:
                        b_start = getattr(reader, "start_time", None)
                        if _start_s > 0 and b_start is not None:
                            start_ts = b_start + int(_start_s * 1e9)
                        if _end_s > 0 and b_start is not None:
                            end_ts = b_start + int(_end_s * 1e9)

                    bag_result = process_bag(
                        bag_path, extraction_topic, frames_dir, discard_dir,
                        _blur, _hash, _motion, _min_c, _min_b, _max_b,
                        _max_f, start_ts, end_ts, pbar, stext,
                        save_discarded=_save_disc, fisheye=_fisheye, calib=calib, zoom=_zoom,
                        sample_fps=_sample_fps,
                    )

                    if resolution_out is None and bag_result["resolution"] is not None:
                        resolution_out = bag_result["resolution"]

                    for key in total:
                        total[key] += bag_result[key]

                    all_stats.append({
                        "Bag": bag_path.name,
                        "Total frames": bag_result["total_images"],
                        "Exported": bag_result["exported_images"],
                        "Discarded (motion)": bag_result["discarded_motion"],
                        "Discarded (blur)": bag_result["discarded_blur"],
                        "Discarded (duplicate)": bag_result["discarded_duplicate"],
                        "Discarded (quality)": bag_result["discarded_quality"],
                        "Status": "✅ Success",
                    })
                except Exception as exc:
                    st.error(f"Failed: `{bag_path.name}`: {exc}")
                    all_stats.append({
                        "Bag": bag_path.name, "Total frames": 0, "Exported": 0,
                        "Discarded (motion)": 0, "Discarded (blur)": 0,
                        "Discarded (duplicate)": 0, "Discarded (quality)": 0,
                        "Status": "❌ Failed",
                    })

                pbar.progress(1.0)
                stext.empty()

            meta = {
                "source_bags": [Path(b).name for b in selected_extract_bags],
                "extraction_topic": extraction_topic,
                "num_frames": total["exported_images"],
                "resolution": list(resolution_out) if resolution_out else None,
                "filters": {
                    "sample_fps": _sample_fps if _sample_fps > 0 else "all",
                    "motion_threshold": _motion, "blur_gate": _blur,
                    "hash_threshold": _hash, "min_brightness": _min_b,
                    "max_brightness": _max_b, "min_contrast": _min_c,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(ann_dir / "extraction.json", "w") as fh:
                json.dump(meta, fh, indent=2)

            st.success("Extraction complete!")
            st.info(f"Output directory: `{frames_dir}`")
            if _save_disc:
                st.info(f"Discarded frames: `{discard_dir}`")

            st.subheader("Extraction summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total frames processed", f"{total['total_images']:,}")
            acc = total["exported_images"] / total["total_images"] * 100 if total["total_images"] else 0
            c2.metric("Frames exported", f"{total['exported_images']:,}", f"{acc:.1f}%")
            total_disc = sum(v for k, v in total.items() if k.startswith("discarded_"))
            c3.metric("Total discarded", f"{total_disc:,}")

            if _save_disc and total_disc > 0:
                st.subheader("Discard breakdown")
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Motion", f"{total['discarded_motion']:,}")
                dc2.metric("Blur", f"{total['discarded_blur']:,}")
                dc3.metric("Duplicates", f"{total['discarded_duplicate']:,}")
                dc4.metric("Quality", f"{total['discarded_quality']:,}")

            st.subheader("Per-bag statistics")
            st.dataframe(pd.DataFrame(all_stats), use_container_width=True)

            sample = sorted(frames_dir.iterdir())[:20] if frames_dir.exists() else []
            if sample:
                cols_th = st.columns(5)
                for i, fp in enumerate(sample):
                    cols_th[i % 5].image(str(fp), caption=fp.name, use_container_width=True)
                if total["exported_images"] > 20:
                    st.caption(f"Showing first 20 of {total['exported_images']} frames.")

        if not can_extract_rb:
            missing_rb = []
            if not selected_extract_bags:
                missing_rb.append("select at least one bag")
            if not extraction_topic:
                missing_rb.append("specify an image topic")
            if not rb_dataset_name.strip():
                missing_rb.append("provide a dataset name")
            if missing_rb:
                st.info(f"Please {', '.join(missing_rb)} to continue.")
