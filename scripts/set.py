"""
Place banana into box demo based on YOLO.

Workflow:
  1. Detect banana and box
  2. Go to grasp banana (don't open gripper before placing)
  3. Grasp banana and lift to ready position
  4. Place into box (release gripper)
  5. Return to ready position

Keys:
  G: capture and execute place
  R: resume live preview
  Q/Esc: release gripper, home, and exit

Usage:
    python scripts/set.py
    python scripts/set.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.camera import make_camera
from drivers.robot.grasp_driver import GraspDriver, selected_arm_config
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from utils.camera_utils import compose_cam_to_base_transform, load_config, load_hand_eye
from utils.ordinary_grasp import GraspPose, estimate_grasps
from utils.transforms import transform_grasp_pose_to_base
from utils.yolo_utils import load_yolo


def _wait_motion(controller: RebotArmEndPose, duration: float, extra: float = 0.6) -> None:
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)


def _move_ready(controller: RebotArmEndPose, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    _wait_motion(controller, duration)


def _cam_to_base(T_hand_eye: np.ndarray, grasp_driver: GraspDriver, cfg: dict[str, Any]) -> np.ndarray:
    return compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)


def _execute_place_sequence(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    box_center: np.ndarray,
    ready_cfg: dict[str, Any],
    dry_run: bool,
) -> bool:
    """Execute the complete place sequence: grasp -> lift -> place -> return."""
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[Step2] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Step2] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    if dry_run:
        print("[Step2] dry run; skip motion")
        return True

    print("[Step2] Open gripper")
    grasp_driver.open_gripper()

    print("[Step2] Move to pregrasp")
    if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Step2] Pregrasp IK failed")
        return False
    _wait_motion(controller, 2.0)

    print("[Step2] Move to grasp")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Step2] Grasp IK failed")
        return False
    _wait_motion(controller, 1.5)

    print("[Step2] Closing gripper")
    ok = grasp_driver.grasp()
    print("[Step2] Holding object" if ok else "[Step2] Empty grasp")
    if not ok:
        return False

    print("[Step3] Lift to ready position with banana")
    _move_ready(controller, ready_cfg)

    print(f"[Step4] Move to box center: xyz=({box_center[0]:+.3f},{box_center[1]:+.3f},{box_center[2]:+.3f})")
    roll, pitch, yaw = 0.0, 1.2, 0.0
    if not controller.move_to_traj(box_center[0], box_center[1], box_center[2], roll, pitch, yaw, duration=2.0):
        print("[Step4] Place position IK failed")
        return False
    _wait_motion(controller, 2.0)

    print("[Step4] Release gripper - banana falls into box")
    grasp_driver.open_gripper(timeout=0.5)

    def _close_gripper():
        time.sleep(0.8)
        d = 0.0
        raw_target = (d / grasp_driver.MAX_DISTANCE_M) * grasp_driver._angle_open
        target = float(np.clip(raw_target, grasp_driver._open_lo, grasp_driver._open_hi))
        with grasp_driver._state_lock:
            grasp_driver._target_pos = target
            grasp_driver._state = grasp_driver._STATE_POSITION
            grasp_driver._position_reached = False

    close_thread = threading.Thread(target=_close_gripper)
    close_thread.start()

    print("[Step5] Return to ready position")
    _move_ready(controller, ready_cfg)
    close_thread.join()

    return True


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best_banana: Optional[GraspPose],
    best_box: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        color = (0, 255, 0) if grasp.is_valid else (0, 165, 255)
        cv2.rectangle(display, 
                      (int(grasp.bbox_xyxy[0]), int(grasp.bbox_xyxy[1])),
                      (int(grasp.bbox_xyxy[2]), int(grasp.bbox_xyxy[3])), 
                      color, 2)
        cv2.putText(display, f"{grasp.class_name} {grasp.conf:.2f}", 
                    (int(grasp.bbox_xyxy[0]), int(grasp.bbox_xyxy[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    
    if best_banana is not None:
        x_m, y_m, z_m = best_banana.position.tolist()
        cv2.putText(display, f"banana: xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})",
                    (10, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 140), 2)
    
    if best_box is not None:
        x_m, y_m, z_m = best_box.position.tolist()
        cv2.putText(display, f"box: xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 120), 2)
    
    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place banana into box demo")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; do not move the arm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    cam_cfg = cfg.get("camera", {})
    print(f"=== Camera: {cam_cfg.get('type')} ===")
    cam = make_camera(cfg)

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    window_name = "Set - Place Banana into Box"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=place  R=resume  Q/ESC=quit\n")

    controller: Optional[RebotArmEndPose] = None
    rebotarm: Optional[RebotArm] = None
    grasp_driver: Optional[GraspDriver] = None
    T_hand_eye: Optional[np.ndarray] = None
    yolo_opts: dict[str, Any] = {}
    robot_ready = False

    try:
        cam.open()
        cam.warm_up(15)
        K = cam.K.astype(np.float32)

        cam_type = str(cam_cfg.get("type", "")).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None

        yolo_cfg = cfg.get("yolo", {})
        gp_cfg = cfg.get("grasp_pipeline", {})
        grasp_cfg = gp_cfg.get("grasp", {})

        model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
        pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
        depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
        infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

        print(f"=== Load YOLO: {model_name} ===")
        model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

        print("=== Init robot ===")
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        robot_ready = True
        print(f"[Robot] mode: {selected.controller_mode}")

        print("[Robot] Move ready")
        _move_ready(controller, ready_cfg)

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if not frozen and (frame_index % infer_every == 0 or not last_results):
                last_results = model.predict(
                    color_bgr,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                last_grasps = estimate_grasps(last_results, depth_mm, K, depth_quantile=depth_quantile)

            best_banana = None
            best_box = None
            for grasp in last_grasps:
                if not grasp.is_valid:
                    continue
                if "banana" in grasp.class_name.lower():
                    if best_banana is None or grasp.conf > best_banana.conf:
                        best_banana = grasp
                elif "box" in grasp.class_name.lower():
                    if best_box is None or grasp.conf > best_box.conf:
                        best_box = grasp

            status = f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps | G=place R=resume Q=quit"
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display = _render_display(color_bgr, last_grasps, best_banana, best_box, status)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                continue

            if key in (ord("g"), ord("G")):
                print("\n[G] Capture and execute place")
                print("[Step1] Detect banana and box")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue

                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                snap_grasps = estimate_grasps(snap_results, snap_depth, K, depth_quantile=depth_quantile)

                snap_banana = None
                snap_box = None
                for grasp in snap_grasps:
                    if not grasp.is_valid:
                        continue
                    if "banana" in grasp.class_name.lower():
                        if snap_banana is None or grasp.conf > snap_banana.conf:
                            snap_banana = grasp
                    elif "box" in grasp.class_name.lower():
                        if snap_box is None or grasp.conf > snap_box.conf:
                            snap_box = grasp

                if snap_banana is None:
                    print("[G] No valid banana detected")
                    continue
                if snap_box is None:
                    print("[G] No valid box detected")
                    continue

                print(f"\n[G] Banana: class={snap_banana.class_name} conf={snap_banana.conf:.3f}")
                print(f"  position_xyz={snap_banana.position.tolist()}")
                print(f"\n[G] Box: class={snap_box.class_name} conf={snap_box.conf:.3f}")
                print(f"  position_xyz={snap_box.position.tolist()}")

                snap_display = _render_display(snap_color, snap_grasps, snap_banana, snap_box, "SNAPSHOT")
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_hand_eye is None:
                    print("[G] Hand-eye calibration unavailable")
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)

                grasp6d, pre6d = transform_grasp_pose_to_base(
                    snap_banana.position,
                    snap_banana.tcp_rotation,
                    T_cam2base,
                    pregrasp_offset_m,
                )

                pos = snap_box.position
                box_homogeneous = np.array([float(pos[0]), float(pos[1]), float(pos[2]), 1.0])
                box_base = T_cam2base @ box_homogeneous
                box_center_base = np.array([float(box_base[0]), float(box_base[1]), 0.08])

                ok = _execute_place_sequence(
                    controller, 
                    grasp_driver, 
                    grasp6d, 
                    pre6d,
                    box_center_base,
                    ready_cfg,
                    dry_run=args.dry_run,
                )
                
                if ok:
                    print("[G] Place completed successfully!")
                else:
                    print("[G] Place failed")
                
                _move_ready(controller, ready_cfg)

    finally:
        print("\n[Exit] Release gripper and home")
        try:
            if robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            cam.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)