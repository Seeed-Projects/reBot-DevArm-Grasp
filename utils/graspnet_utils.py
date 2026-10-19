"""GraspNet helpers that sit above camera frames and YOLO detections."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import open3d as o3d
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
DEFAULT_NUM_VIEW = 300
DEFAULT_VOXEL_SIZE = 0.01
DEFAULT_WARMUP_FRAMES = 20
DISPLAY_FLIP_X = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def prepare_graspnet_imports(graspnet_root: Path = GRASPNET_ROOT) -> None:
    for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"):
        path = str(graspnet_root / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)
    root = str(graspnet_root)
    if root not in sys.path:
        sys.path.insert(0, root)


prepare_graspnet_imports()

try:
    from .yolo_utils import YoloDetection
except ImportError:  
    from yolo_utils import YoloDetection

from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from data_utils import CameraInfo, create_point_cloud_from_depth_image  # noqa: E402
from graspnet import GraspNet, pred_decode  # noqa: E402
from graspnetAPI import Grasp, GraspGroup  # noqa: E402

try:
    from .transforms import graspnet_rotation_to_rebot_tcp_rotation, transform_grasp_pose_to_base_with_retreat
except ImportError:
    from transforms import graspnet_rotation_to_rebot_tcp_rotation, transform_grasp_pose_to_base_with_retreat


@dataclass
class GraspNetFrameResult:
    grasps: GraspGroup
    pre_bbox_grasps: GraspGroup
    bbox_grasps: GraspGroup
    best: Optional[Grasp]
    status: str
    target_status: str
    detections: list[YoloDetection]
    selected_target: Optional[YoloDetection]
    o3d_cloud: o3d.geometry.PointCloud
    raw_cloud: np.ndarray


class Open3DGraspWindow:
    def __init__(self, title: str, top_k: int) -> None:
        self._top_k = top_k
        self._vis = o3d.visualization.Visualizer()
        if not self._vis.create_window(title, width=1280, height=720):
            self._vis.destroy_window()
            raise RuntimeError("Open3D visualizer window could not be created")
        self._geometries = []
        self._initialized = False

    def update(self, cloud: o3d.geometry.PointCloud, grasps: GraspGroup) -> None:
        for geom in self._geometries:
            self._vis.remove_geometry(geom, reset_bounding_box=False)
        self._geometries = []

        cloud_vis = o3d.geometry.PointCloud(cloud)
        cloud_vis.transform(DISPLAY_FLIP_X)
        geometries = [cloud_vis]

        if len(grasps) > 0:
            grasps_vis = GraspGroup(grasps.grasp_group_array.copy())
            try:
                grasps_vis = grasps_vis.nms()
            except Exception as exc:
                print(f"Grasp NMS skipped: {exc}")
            grasps_vis.sort_by_score()
            grasps_vis = grasps_vis[: self._top_k]
            grasps_vis.transform(DISPLAY_FLIP_X)
            geometries.extend(grasps_vis.to_open3d_geometry_list())

        for geom in geometries:
            self._vis.add_geometry(geom, reset_bounding_box=not self._initialized)
        self._geometries = geometries
        self._initialized = True
        self.poll()

    def poll(self) -> bool:
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self) -> None:
        self._vis.destroy_window()


def copy_grasp_group(grasps: GraspGroup) -> GraspGroup:
    return GraspGroup(grasps.grasp_group_array.copy())


def visualization_grasps(result: GraspNetFrameResult, mode: str) -> GraspGroup:
    """Return the grasp set used for Open3D diagnostics."""
    if mode == "pre-bbox":
        return result.pre_bbox_grasps
    if mode == "bbox":
        return result.bbox_grasps
    return result.grasps


def resolve_checkpoint_path(
    checkpoint: str,
    *,
    project_root: Path = PROJECT_ROOT,
    graspnet_root: Path = GRASPNET_ROOT,
) -> Path:
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path
    if len(checkpoint_path.parts) > 1:
        return project_root / checkpoint_path
    return graspnet_root / "checkpoints" / checkpoint_path


def build_net(checkpoint_path: str | Path, num_view: int = DEFAULT_NUM_VIEW) -> GraspNet:
    checkpoint_path = resolve_checkpoint_path(str(checkpoint_path))
    if not torch.cuda.is_available():
        raise RuntimeError("GraspNet pointnet2 operators require CUDA, but torch.cuda is unavailable.")

    net = GraspNet(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0")
    net.to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    print(f"Loaded checkpoint {checkpoint_path} (epoch: {checkpoint['epoch']})")
    return net


def _expanded_bbox_pixel_bounds(
    bbox_xyxy: tuple[int, int, int, int],
    expand_ratio: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Pixel bounds of a bbox expanded about its center, clipped to the image."""
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    expand_ratio = max(1.0, float(expand_ratio))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
    half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
    x1 = int(np.clip(int(round(cx - half_w)), 0, max(0, width - 1)))
    y1 = int(np.clip(int(round(cy - half_h)), 0, max(0, height - 1)))
    x2 = int(np.clip(int(round(cx + half_w)), 0, max(0, width - 1)))
    y2 = int(np.clip(int(round(cy + half_h)), 0, max(0, height - 1)))
    return x1, y1, x2, y2


def build_end_points(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    num_point: int,
    min_depth_m: float,
    max_depth_m: float,
    bbox_xyxy: Optional[tuple[int, int, int, int]] = None,
    expand_ratio: float = 1.0,
) -> tuple[dict, o3d.geometry.PointCloud, np.ndarray]:
    if color_bgr.shape[:2] != depth_mm.shape[:2]:
        depth_mm = cv2.resize(depth_mm, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    min_mm = int(max(0.0, min_depth_m) * 1000.0)
    max_mm = int(max_depth_m * 1000.0)
    depth = depth_mm.astype(np.uint16, copy=False)
    mask = (depth > min_mm) & (depth < max_mm)
    if int(mask.sum()) == 0:
        raise RuntimeError("No valid depth pixels in the configured depth range.")

    # Concentrate network input points on the target bbox when available;
    # collision detection / visualization still use the full masked scene cloud.
    sample_mask = mask
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = _expanded_bbox_pixel_bounds(bbox_xyxy, expand_ratio, depth.shape[1], depth.shape[0])
        bbox_mask = np.zeros_like(mask)
        bbox_mask[y1 : y2 + 1, x1 : x2 + 1] = True
        cropped = mask & bbox_mask
        if int(cropped.sum()) >= 16:
            sample_mask = cropped

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = depth.shape
    camera = CameraInfo(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), 1000.0)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    cloud_masked = cloud[sample_mask]
    color_masked = color_rgb[sample_mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs = np.concatenate(
            [
                np.arange(len(cloud_masked)),
                np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True),
            ],
            axis=0,
        )

    end_points = {
        "point_clouds": torch.from_numpy(cloud_masked[idxs].astype(np.float32)[np.newaxis]).cuda(non_blocking=True),
        "cloud_colors": color_masked[idxs],
    }

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud[mask].astype(np.float32))
    o3d_cloud.colors = o3d.utility.Vector3dVector(color_rgb[mask].astype(np.float32))
    return end_points, o3d_cloud, cloud[mask]


def infer_grasps(
    net: GraspNet,
    end_points: dict,
    raw_cloud: np.ndarray,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> tuple[GraspGroup, dict[str, int]]:
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    decoded_count = len(gg)
    collision_removed = 0
    if len(gg) > 0 and collision_thresh > 0:
        detector = ModelFreeCollisionDetector(raw_cloud, voxel_size=voxel_size)
        collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        collision_removed = int(np.count_nonzero(collision_mask))
        gg = gg[~collision_mask]

    return gg, {
        "decoded": decoded_count,
        "pre_collision": decoded_count,
        "collision_removed": collision_removed,
        "final": len(gg),
    }


def filter_grasps_by_bbox(
    grasps: GraspGroup,
    bbox_xyxy: tuple[int, int, int, int],
    K: np.ndarray,
    *,
    margin_px: int = 0,
    expand_ratio: float = 1.0,
    image_shape: Optional[tuple[int, int]] = None,
) -> GraspGroup:
    if len(grasps) == 0:
        return grasps

    translations = np.asarray(grasps.translations, dtype=np.float64)
    z = translations[:, 2]
    valid_z = z > 1e-6
    u = np.full(len(grasps), np.nan, dtype=np.float64)
    v = np.full(len(grasps), np.nan, dtype=np.float64)
    u[valid_z] = float(K[0, 0]) * translations[valid_z, 0] / z[valid_z] + float(K[0, 2])
    v[valid_z] = float(K[1, 1]) * translations[valid_z, 1] / z[valid_z] + float(K[1, 2])

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    expand_ratio = max(1.0, float(expand_ratio))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
    half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
    x1 = int(round(cx - half_w)) - int(margin_px)
    y1 = int(round(cy - half_h)) - int(margin_px)
    x2 = int(round(cx + half_w)) + int(margin_px)
    y2 = int(round(cy + half_h)) + int(margin_px)
    if image_shape is not None:
        h, w = image_shape
        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        x2 = int(np.clip(x2, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        y2 = int(np.clip(y2, 0, max(0, h - 1)))

    keep = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return grasps[keep]


def filter_grasps_by_width(grasps: GraspGroup, max_width_m: Optional[float]) -> GraspGroup:
    if max_width_m is None or len(grasps) == 0:
        return grasps
    return grasps[np.asarray(grasps.widths, dtype=np.float64) <= float(max_width_m)]


def select_best_grasp(grasps: GraspGroup) -> Optional[Grasp]:
    if len(grasps) == 0:
        return None
    ranked = GraspGroup(grasps.grasp_group_array.copy())
    try:
        ranked = ranked.nms()
    except Exception as exc:
        print(f"[WARN] GraspNet NMS skipped: {exc}")
    ranked.sort_by_score()
    return ranked[0] if len(ranked) > 0 else None


def select_target(detections: list[YoloDetection], target_class: Optional[str]) -> Optional[YoloDetection]:
    if not detections:
        return None
    candidates = detections
    if target_class:
        target_norm = target_class.casefold()
        exact = [target for target in detections if target.class_name.casefold() == target_norm]
        contains = [target for target in detections if target_norm in target.class_name.casefold()]
        candidates = exact or contains
    if not candidates:
        return None
    return max(candidates, key=lambda target: target.conf)


def selected_target_text(selected: Optional[YoloDetection], target_class: Optional[str]) -> str:
    if selected is None:
        return f"target={target_class or 'best'} not found"
    return f"target={selected.class_name} {selected.conf:.2f}"


def target_status_text(selected: Optional[YoloDetection], detections: list[YoloDetection], target_class: Optional[str]) -> str:
    if selected is not None:
        return f"target={selected.class_name} {selected.conf:.2f} detections={len(detections)}"
    if target_class:
        return f"target={target_class} not found detections={len(detections)}"
    return f"target not found detections={len(detections)}"


def draw_detections_overlay(
    frame: np.ndarray,
    detections: list[YoloDetection],
    selected: Optional[YoloDetection],
    target_class: Optional[str],
) -> np.ndarray:
    display = frame.copy()
    selected_key = None
    if selected is not None:
        selected_key = (selected.result_index, selected.detection_index)
    for target in detections:
        is_selected = selected_key == (target.result_index, target.detection_index)
        color = (0, 255, 80) if is_selected else (0, 185, 255)
        thickness = 3 if is_selected else 2
        x1, y1, x2, y2 = target.bbox_xyxy
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        label = f"{target.class_name} {target.conf:.2f}"
        if target_class and is_selected:
            label = f"TARGET {label}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(display, (x1, bg_y1), (x1 + label_size[0] + 8, y1), (0, 0, 0), -1)
        cv2.putText(display, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return display


def draw_best_grasp_projection(display: np.ndarray, grasp: Optional[Grasp], K: np.ndarray) -> None:
    if grasp is None:
        return
    x, y, z = [float(v) for v in grasp.translation]
    if z <= 1e-6:
        return
    u = int(round(float(K[0, 0]) * x / z + float(K[0, 2])))
    v = int(round(float(K[1, 1]) * y / z + float(K[1, 2])))
    if 0 <= u < display.shape[1] and 0 <= v < display.shape[0]:
        cv2.drawMarker(display, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        label = f"best score={grasp.score:.2f} width={grasp.width * 100:.1f}cm"
        cv2.putText(display, label, (u + 10, max(24, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def grasp_to_base_poses(
    grasp: Grasp,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    return transform_grasp_pose_to_base_with_retreat(
        np.asarray(grasp.translation, dtype=np.float64),
        graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix),
        T_cam2base,
        pregrasp_offset_m,
        retreat_offset_m,
        insertion_depth_m,
    )


def draw_status(
    frame: np.ndarray,
    status: str,
    target_status: str = "",
    frozen: bool = False,
    title: str = "GraspNet Full-Scene Demo",
) -> np.ndarray:
    display = frame.copy()
    lines = [
        title,
        "G/SPACE: infer   R: resume   Q/ESC: quit",
    ]
    if target_status:
        lines.append(target_status)
    lines.append(status)
    y = 28
    for line in lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    if frozen:
        cv2.putText(display, "[FROZEN]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    return display


def infer_frame(
    net: GraspNet,
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    num_point: int,
    min_depth: float,
    max_depth: float,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    yolo_model: Optional[Any] = None,
    yolo_opts: Optional[dict[str, Any]] = None,
    target_class: Optional[str] = None,
    target_margin_px: int = 0,
    target_expand_ratio: float = 1.0,
    max_grasp_width_m: Optional[float] = None,
) -> GraspNetFrameResult:
    try:
        from .yolo_utils import detect_objects
    except ImportError:
        from yolo_utils import detect_objects

    detections: list[YoloDetection] = []
    selected_target: Optional[YoloDetection] = None
    target_label = "full scene"

    if yolo_model is not None:
        _, detections = detect_objects(yolo_model, color_bgr, yolo_opts or {})
        selected_target = select_target(detections, target_class)
        if selected_target is None:
            target_status = target_status_text(selected_target, detections, target_class)
            empty = GraspGroup()
            empty_cloud = o3d.geometry.PointCloud()
            return GraspNetFrameResult(
                grasps=empty,
                pre_bbox_grasps=empty,
                bbox_grasps=empty,
                best=None,
                status=f"inference skipped: {target_status}",
                target_status=target_status,
                detections=detections,
                selected_target=None,
                o3d_cloud=empty_cloud,
                raw_cloud=np.empty((0, 3), dtype=np.float32),
            )
        target_label = f"{selected_target.class_name} {selected_target.conf:.2f}"

    tic = time.time()
    crop_bbox = selected_target.bbox_xyxy if selected_target is not None else None
    end_points, o3d_cloud, raw_cloud = build_end_points(
        color_bgr,
        depth_mm,
        K,
        num_point,
        min_depth,
        max_depth,
        bbox_xyxy=crop_bbox,
        expand_ratio=target_expand_ratio,
    )
    grasps, counts = infer_grasps(net, end_points, raw_cloud, collision_thresh, voxel_size)
    pre_bbox_grasps = copy_grasp_group(grasps)
    if selected_target is not None:
        grasps = filter_grasps_by_bbox(
            grasps,
            selected_target.bbox_xyxy,
            K,
            margin_px=target_margin_px,
            expand_ratio=target_expand_ratio,
            image_shape=color_bgr.shape[:2],
        )
    bbox_grasps = copy_grasp_group(grasps)
    grasps = filter_grasps_by_width(grasps, max_grasp_width_m)
    best = select_best_grasp(grasps)
    elapsed = time.time() - tic

    if yolo_model is None:
        status = f"grasps={len(grasps)} decoded={counts['decoded']} inference={elapsed:.2f}s"
        target_status = "YOLO disabled: full-scene GraspNet"
    else:
        status = (
            f"{target_label} grasps={len(grasps)}/{len(bbox_grasps)}/{len(pre_bbox_grasps)} decoded={counts['decoded']} "
            f"collide={counts['collision_removed']}/{counts['pre_collision']} inference={elapsed:.2f}s"
        )
        target_status = target_status_text(selected_target, detections, target_class)

    return GraspNetFrameResult(
        grasps=grasps,
        pre_bbox_grasps=pre_bbox_grasps,
        bbox_grasps=bbox_grasps,
        best=best,
        status=status,
        target_status=target_status,
        detections=detections,
        selected_target=selected_target,
        o3d_cloud=o3d_cloud,
        raw_cloud=raw_cloud,
    )
