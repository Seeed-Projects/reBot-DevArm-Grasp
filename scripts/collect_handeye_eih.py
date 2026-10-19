"""
Eye-in-hand calibration data collection and solving (Gemini2 + reBotArm).

Modes:
  Auto mode (default): the arm traverses 50 preset poses, captures samples
                       when ArUco is detected, and skips timed-out poses.
  Manual mode (--manual): gravity compensation lets the user move the arm by
                          hand. Release the arm to lock it, then press Enter.

Setup:
  The camera is mounted on the end effector.
  The ArUco marker is fixed on the work surface.

Usage:
    python scripts/collect_handeye_eih.py           # auto mode
    python scripts/collect_handeye_eih.py --manual  # manual gravity mode
"""

import os
import sys
import threading
import argparse
import queue
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

from drivers.camera import make_camera
from drivers.robot.grasp_driver import GraspDriver, selected_arm_config
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from calibration.hand_eye import CalibMode, HandEyeCalibrator
from utils.camera_utils import load_config
from utils.transforms import rotation_matrix_to_euler_zyx


# ==========================================
# Preset calibration poses in Cartesian space, using meters and radians.
# (x, y, z, roll, pitch, yaw)
# pitch > 0 points the end effector downward toward the ArUco marker.
#
# NOTE: pose set imported from wrc_demo's calib_wrist.py (25 poses tuned for
# the reBot DevArm wrist link: x bounded 0.20-0.40 m, gentle roll/yaw) to
# avoid exceeding joint travel limits on this arm.
# ==========================================
CALIB_POSES_XYZ = [
    (0.28, 0.00, 0.30, 0.00, 0.50, 0.00),
    (0.28, 0.00, 0.30, 0.00, 0.70, 0.00),
    (0.28, 0.00, 0.30, 0.00, 0.90, 0.00),
    (0.28, -0.15, 0.30, 0.00, 0.60, 0.00),
    (0.28, 0.15, 0.30, 0.00, 0.60, 0.00),
    (0.28, -0.20, 0.32, 0.00, 0.50, 0.00),
    (0.28, 0.20, 0.32, 0.00, 0.50, 0.00),
    (0.35, 0.00, 0.28, 0.00, 0.60, 0.00),
    (0.22, 0.00, 0.35, 0.00, 0.70, 0.00),
    (0.38, 0.00, 0.25, 0.00, 0.80, 0.00),
    (0.28, 0.00, 0.30, 0.00, 0.60, -0.30),
    (0.28, 0.00, 0.30, 0.00, 0.60, 0.30),
    (0.28, 0.00, 0.30, 0.00, 0.60, -0.60),
    (0.28, 0.00, 0.30, 0.00, 0.60, 0.60),
    (0.28, 0.00, 0.30, -0.40, 0.60, 0.00),
    (0.28, 0.00, 0.30, 0.40, 0.60, 0.00),
    (0.28, 0.00, 0.40, 0.00, 0.50, 0.00),
    (0.28, 0.10, 0.40, 0.00, 0.50, 0.30),
    (0.28, -0.10, 0.40, 0.00, 0.50, -0.30),
    (0.28, 0.00, 0.22, 0.00, 0.80, 0.00),
    (0.28, 0.08, 0.22, 0.00, 0.80, 0.20),
    (0.28, -0.08, 0.22, 0.00, 0.80, -0.20),
    (0.32, 0.15, 0.26, 0.30, 0.70, 0.40),
    (0.32, -0.15, 0.26, -0.30, 0.70, -0.40),
    (0.24, 0.12, 0.35, 0.20, 0.55, 0.25),
]

AUTO_MOVE_DURATION_S = 3.0
AUTO_SETTLE_EXTRA_S = 0.6
AUTO_MARKER_TIMEOUT_S = 2.5
AUTO_MARKER_STABLE_FRAMES = 4
MIN_CALIB_SAMPLES = 5


def make_input_thread(line_queue: queue.Queue) -> threading.Thread:
    def _loop():
        while True:
            try:
                line_queue.put(input())
            except EOFError:
                line_queue.put(None)
                break
            except KeyboardInterrupt:
                line_queue.put(None)
                break
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ==========================================
# Gravity compensation controller for manual mode.
# ==========================================
class GravityCompController:
    """MIT mode with end-effector velocity locking for hand-guided poses.

    Reference: reBotArm_control_py/example/10_gravity_compensation_lock.py
    """
    KP = 8.0
    KD = 1.5
    V_THRESH  = 0.04   # End-effector linear velocity threshold (m/s)
    W_THRESH  = 0.08   # End-effector angular velocity threshold (rad/s)

    def __init__(self, arm: RebotArm) -> None:
        from reBotArm_control_py.dynamics import compute_generalized_gravity
        from reBotArm_control_py.kinematics import (
            load_robot_model, get_end_effector_frame_id,
        )
        from reBotArm_control_py.kinematics.robot_model import pad_q_for_model
        import pinocchio as pin

        self._compute_gravity = compute_generalized_gravity
        self._pad_q_for_model = pad_q_for_model
        self._pin = pin

        self._arm = arm
        if not self._arm.has_gripper:
            raise ValueError(
                "Hardware config is missing groups.gripper. "
                "Enable the gripper group in the selected hardware YAML under "
                "reBotArm_control_py/config."
            )
        self._model = load_robot_model()
        self._data  = self._model.createData()
        self._ee_id = get_end_effector_frame_id(self._model)

        self._n = None          # Joint count, set after connect.
        self._q_target = None
        self._integral = None
        self._gc_running = threading.Event()
        self._io_lock = threading.RLock()

    def start(self) -> None:
        """Connect, enable, set MIT mode, and start gravity compensation."""
        self._arm.connect()
        print("[GravityComp] Connected")

        self._arm.arm.mode_mit(
            kp=np.full(self._arm.arm.num_joints, self.KP),
            kd=np.full(self._arm.arm.num_joints, self.KD),
        )
        self._arm.gripper.mode_mit()
        self._arm.enable_all()
        print("[GravityComp] Enabled")

        n = self._arm.arm.num_joints
        self._n = n
        self._wait_state_valid()
        with self._io_lock:
            q0 = self._arm.get_state()[0][:n]
        self._q_target = q0.copy()
        self._integral = np.zeros(n)

        print(f"[GravityComp] MIT mode, kp={self.KP} kd={self.KD}. Move the arm by hand.")

        self._gc_running.set()
        self._arm.start_control_loop(self._worker, rate=self._arm.rate)

    def _wait_state_valid(self, timeout: float = 2.0) -> None:
        """Wait until all motor feedback is available."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._io_lock:
                self._arm.get_state()
                ok = all(
                    m.get_state() is not None
                    for m in self._arm._motor_map.values()
                )
            if ok:
                return
            time.sleep(0.02)
        raise RuntimeError("[GravityComp] Arm feedback is not ready")

    def safe_home(self) -> None:
        """Stop gravity compensation, home with an SDK controller, then disconnect."""
        self._gc_running.clear()
        try:
            print("[GravityComp] Homing...")
            self._arm.stop_control_loop()
            with self._io_lock:
                q_now = self._arm.get_state()[0][: self._n]
                g_now = self._arm.gripper.get_positions()

            ctrl = RebotArmEndPose(
                self._arm,
                arm_control_mode="mit",
                use_gravity_ff=True,
            )
            ctrl.set_gripper_target(float(g_now[0]) if g_now.size else 0.0)
            ctrl._q_target[:] = q_now
            ctrl.start()
            ctrl.safe_home()
            self._arm.stop_control_loop()
        except Exception as e:
            print(f"[GravityComp] Homing failed: {e}")
        try:
            self._arm.disconnect()
        except Exception:
            pass
        print("[GravityComp] Disconnected")

    def _worker(self, r, dt: float) -> None:
        if not self._gc_running.is_set():
            return

        pin = self._pin
        model, data, ee_id = self._model, self._data, self._ee_id
        n = self._n
        KP, KD = self.KP, self.KD

        try:
            with self._io_lock:
                q_all, qd_all, _ = r.get_state()
            q = q_all[:n]
            qd = qd_all[:n]
            q_model = self._pad_q_for_model(model, q, n)
            tau_g = self._compute_gravity(model=model, q=q_model)[:n]

            q_err = self._q_target - q
            self._integral += q_err
            np.clip(self._integral, -0.5, 0.5, out=self._integral)

            qd_model = np.zeros(model.nv)
            qd_model[: min(model.nv, n)] = qd[: min(model.nv, n)]
            pin.computeJointJacobians(model, data, q_model)
            pin.updateFramePlacements(model, data)
            J = pin.getFrameJacobian(model, data, ee_id, pin.ReferenceFrame.WORLD)
            v = J @ qd_model

            if (np.linalg.norm(v[:3]) > self.V_THRESH or
                    np.linalg.norm(v[3:]) > self.W_THRESH):
                self._q_target = q.copy()
                self._integral *= 0.9

            with self._io_lock:
                r.arm.send_mit(
                    pos=self._q_target,
                    vel=np.zeros(n),
                    kp=np.full(n, KP),
                    kd=np.full(n, KD),
                    tau=tau_g + self._integral,
                )
                r.gripper.send_mit(r.gripper.get_positions())
        except Exception:
            pass


# ==========================================
# Main flow.
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Eye-in-hand calibration data collection")
    parser.add_argument("--manual", action="store_true",
                        help="manual mode: gravity compensation; move the arm by hand and press Enter to capture")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg  = load_config(root / "config" / "default.yaml")

    cam_type   = cfg["camera"]["type"]
    calib_dir  = root / "config" / "calibration" / cam_type
    aruco_cfg  = cfg["calibration"]["aruco"]
    he_method  = cfg["calibration"].get("hand_eye_method", "TSAI")
    save_path  = calib_dir / "hand_eye.npz"

    # Camera.
    cam = make_camera(cfg)
    cam.setup_aruco(
        marker_length_m=aruco_cfg["marker_length_m"],
        dict_id=aruco_cfg.get("dict_id", 0),
        target_marker_id=aruco_cfg.get("target_marker_id"),
    )

    # Calibrator.
    calibrator = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method=he_method)

    # Robot.
    mode_str = "manual (gravity compensation)" if args.manual else f"auto ({len(CALIB_POSES_XYZ)} preset poses)"
    gc_ctrl: GravityCompController | None = None
    rebotarm: RebotArm | None = None
    controller: RebotArmEndPose | None = None
    grasp_driver: GraspDriver | None = None
    auto_controller_mode: str | None = None
    auto_use_gravity_ff = False
    auto = {
        "enabled": not args.manual,
        "idx": 0,
        "pose_idx": None,
        "phase": "idle",
        "settle_until": 0.0,
        "timeout_at": 0.0,
        "stable_frames": 0,
        "status": "waiting to start",
        "finished": False,
    }
    result_saved = False

    print(f"\n=== Eye-in-Hand Calibration ===")
    print(f"Camera: {cam_type}  |  Mode: {mode_str}  |  Solver: {he_method}")
    print(f"ArUco size: {aruco_cfg['marker_length_m']*100:.0f}cm  |  Output: {save_path}")
    print()

    # Open the camera first so the arm is not enabled if camera setup fails.
    try:
        cam.open()
        print("Warming up camera...", end="", flush=True)
        cam.warm_up(20)
        print(" ready\n")
    except Exception as e:
        try:
            cam.close()
        except Exception:
            pass
        print(f"[Camera] Initialization failed: {e}")
        sys.exit(1)

    robot_cfg = cfg.get("robot", {})
    try:
        rebotarm = RebotArm()
        if args.manual:
            controller = RebotArmEndPose(rebotarm, arm_control_mode="mit", use_gravity_ff=True)
            grasp_driver = GraspDriver(
                rebotarm,
                controller,
                gripper_config=robot_cfg.get("gripper"),
                repo_root=robot_cfg.get("repo_root"),
            )
            gc_ctrl = GravityCompController(rebotarm)
            gc_ctrl.start()
            print("[Robot] Manual mode ready. Move the arm by hand, then press Enter to capture.")
        else:
            selected = selected_arm_config(robot_cfg.get("repo_root"))
            auto_controller_mode = selected.controller_mode
            auto_use_gravity_ff = auto_controller_mode == "mit"
            controller = RebotArmEndPose(rebotarm, arm_control_mode=auto_controller_mode)
            grasp_driver = GraspDriver(
                rebotarm,
                controller,
                gripper_config=robot_cfg.get("gripper"),
                repo_root=robot_cfg.get("repo_root"),
            )
            grasp_driver.start()
            print(
                f"[Robot] Auto mode ready, control mode: {selected.controller_mode}. "
                f"{len(CALIB_POSES_XYZ)} preset poses will be traversed."
            )
    except Exception as e:
        try:
            cam.close()
        except Exception:
            pass
        print(f"[Robot] Connection failed: {e}")
        sys.exit(1)

    if args.manual:
        print("[Controls] Enter=capture  c/q=finish and solve  pos=print current TCP pose")
    else:
        print("[Controls] Auto traversal and capture  c/q=stop and solve  pos=print current TCP pose")
    print()

    latest_pose = None
    line_queue: queue.Queue | None = None
    if sys.stdin.isatty():
        line_queue = queue.Queue()
        make_input_thread(line_queue)
    else:
        print("[Hint] Non-interactive terminal detected; terminal commands are disabled")

    def _print_fk() -> None:
        try:
            T = grasp_driver.get_tcp_pose()
            t = T[:3, 3]
            R = T[:3, :3]
            _r, _p, _y = rotation_matrix_to_euler_zyx(R)
            print(f"  FK: x={t[0]:+.3f} y={t[1]:+.3f} z={t[2]:+.3f} m"
                  f"  rpy=[{_r:+.2f} {_p:+.2f} {_y:+.2f}] rad")
        except Exception as e:
            print(f"  [Error] {e}")

    def capture_sample(cur, source: str) -> bool:
        if cur is None:
            print("  [Skip] Marker is not visible; adjust the pose and try again")
            return False

        print(f"\n[Sample {calibrator.n_samples + 1}] {source}")
        print(f"  ArUco: x={cur.T_marker2cam[0,3]:.3f} "
              f"y={cur.T_marker2cam[1,3]:.3f} "
              f"z={cur.T_marker2cam[2,3]:.3f} m")
        try:
            T_g2b = grasp_driver.get_tcp_pose()
            t = T_g2b[:3, 3]
            print(f"  End effector (FK): x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            calibrator.add_sample(T_g2b, cur.T_marker2cam)
            print(f"  [OK] Recorded, total samples: {calibrator.n_samples}"
                  + ("  will solve automatically on finish" if calibrator.n_samples >= 15 else ""))
            return True
        except Exception as e:
            print(f"  [Error] Failed to read TCP pose: {e}")
            return False

    def compute_and_save(reason: str) -> bool:
        nonlocal result_saved
        print(f"\n[Finish] {reason}")
        if calibrator.n_samples < MIN_CALIB_SAMPLES:
            print(f"[Result] Not enough samples ({calibrator.n_samples} < {MIN_CALIB_SAMPLES}); calibration was not solved")
            if save_path.exists():
                print("[Result] Existing hand_eye.npz was not updated")
            return False

        print(f"[Result] Solving with {calibrator.n_samples} samples...")
        try:
            result = calibrator.calibrate(min_samples=MIN_CALIB_SAMPLES)
            HandEyeCalibrator.save(result, save_path)
            t = result.T_result[:3, 3]
            R = result.T_result[:3, :3]
            print(f"[Result] T_cam2gripper translation: x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            print(f"[Result] Rotation matrix:\n{R}")
            print(f"[Result] [OK] Saved to {save_path}")
            if calibrator.n_samples < 15:
                print("[Result] Tip: fewer than 15 samples; collect more samples for better accuracy")
            result_saved = True
            return True
        except Exception as e:
            print(f"[Result] [Error] Solve failed: {e}")
            return False

    def start_next_auto_pose() -> bool:
        if not auto["enabled"] or controller is None:
            return False

        total = len(CALIB_POSES_XYZ)
        while auto["idx"] < total:
            idx = auto["idx"]
            x, y, z, roll, pitch, yaw = CALIB_POSES_XYZ[idx]
            print(f"\n[Auto] Pose {idx+1}/{total}: "
                  f"pos=({x:.2f},{y:.2f},{z:.2f}) rpy=({roll:.2f},{pitch:.2f},{yaw:.2f})")
            ok = controller.move_to_traj(x, y, z, roll=roll, pitch=pitch, yaw=yaw, duration=AUTO_MOVE_DURATION_S)
            if ok:
                now = time.monotonic()
                auto["pose_idx"] = idx
                auto["phase"] = "settling"
                auto["settle_until"] = now + AUTO_MOVE_DURATION_S + AUTO_SETTLE_EXTRA_S
                auto["timeout_at"] = auto["settle_until"] + AUTO_MARKER_TIMEOUT_S
                auto["stable_frames"] = 0
                auto["status"] = f"pose {idx+1}/{total} moving"
                return False

            print(f"[Auto] Pose {idx+1}/{total} has no IK solution, skipping")
            auto["idx"] += 1

        auto["phase"] = "done"
        auto["finished"] = True
        auto["status"] = "all poses completed"
        print("\n[Auto] All preset poses completed")
        return True

    def tick_auto(cur) -> bool:
        if not auto["enabled"] or auto["finished"]:
            return auto["finished"]

        if auto["phase"] == "idle":
            return start_next_auto_pose()

        pose_idx = auto["pose_idx"]
        total = len(CALIB_POSES_XYZ)
        now = time.monotonic()

        if auto["phase"] == "settling":
            remain = auto["settle_until"] - now
            if remain > 0.0:
                auto["stable_frames"] = 0
                auto["status"] = f"pose {pose_idx+1}/{total} moving/settling {remain:.1f}s"
                return False
            auto["phase"] = "searching"

        if cur is not None:
            auto["stable_frames"] += 1
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = (
                f"pose {pose_idx+1}/{total} marker stable "
                f"{auto['stable_frames']}/{AUTO_MARKER_STABLE_FRAMES}  remaining {remain:.1f}s"
            )
            if auto["stable_frames"] >= AUTO_MARKER_STABLE_FRAMES:
                capture_sample(cur, f"auto pose {pose_idx+1}/{total}")
                auto["idx"] += 1
                auto["phase"] = "idle"
                auto["stable_frames"] = 0
                return start_next_auto_pose()
        else:
            auto["stable_frames"] = 0
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = f"pose {pose_idx+1}/{total} waiting for ArUco {remain:.1f}s"

        if now >= auto["timeout_at"]:
            print(f"[Auto] Pose {pose_idx+1}/{total} timed out without ArUco, skipping")
            auto["idx"] += 1
            auto["phase"] = "idle"
            auto["stable_frames"] = 0
            return start_next_auto_pose()

        return False

    def safe_home_and_disconnect() -> None:
        """Stop active control, return home with a fresh SDK controller, then disconnect."""
        if rebotarm is None or auto_controller_mode is None:
            return
        try:
            print("[Robot] Homing and disconnecting...")
            rebotarm.stop_control_loop()
            q_now = rebotarm.get_state()[0][: rebotarm.arm.num_joints]
            g_now = rebotarm.gripper.get_positions() if rebotarm.has_gripper else np.array([])

            ctrl = RebotArmEndPose(
                rebotarm,
                arm_control_mode=auto_controller_mode,
                use_gravity_ff=auto_use_gravity_ff,
            )
            ctrl.set_gripper_target(float(g_now[0]) if g_now.size else 0.0)
            ctrl._q_target[:] = q_now
            ctrl.start()
            ctrl.safe_home()
            rebotarm.stop_control_loop()
        except Exception as e:
            print(f"[Robot] Homing failed: {e}")
        try:
            rebotarm.disconnect()
        except Exception:
            pass

    def handle_line(raw: str) -> bool:
        if raw is None:
            print("\n[Interrupt] Terminal input closed; stopping and trying to solve")
            return True

        line = raw.strip().lower()

        if line in {"q", "c"}:
            return True

        if line == "pos":
            _print_fk()
            return False

        if args.manual and line == "":
            capture_sample(latest_pose, "manual capture")
            return False

        if line:
            if args.manual:
                print("  Manual commands: Enter=capture  c/q=finish and solve  pos=print current TCP pose")
            else:
                print("  Auto commands: c/q=finish and solve  pos=print current TCP pose")
        return False

    # Main loop.
    WIN = "Eye-in-Hand Calibration  (operate in terminal)"
    finish_reason = "normal finish"
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

        while True:
            bgr, _ = cam.get_frame()
            if bgr is not None:
                pose = cam.detect_aruco(bgr)
                latest_pose = pose
                if tick_auto(pose):
                    finish_reason = "auto traversal completed"
                vis  = cam.draw_aruco(bgr)
                n    = calibrator.n_samples

                def osd(text, y, color=(220, 220, 220)):
                    cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 1, cv2.LINE_AA)

                if pose:
                    if args.manual:
                        osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  "
                            f"samples:{n}  Enter=capture  c/q=finish",
                            28, (80, 220, 80))
                    else:
                        osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  samples:{n}",
                            28, (80, 220, 80))
                else:
                    if args.manual:
                        osd(f"No marker  samples:{n}  move arm to see marker",
                            28, (80, 80, 220))
                    else:
                        osd(f"No marker  samples:{n}",
                            28, (80, 80, 220))

                if args.manual:
                    osd("MANUAL: Enter=capture  c/q=finish  pos=print fk", 50, (180, 180, 60))
                else:
                    osd(f"AUTO: {auto['status']}", 50, (180, 180, 60))

                filled = min(n, 15) * (400 // 15)
                cv2.rectangle(vis, (10, 70), (10 + filled, 82), (0, 200, 100), -1)
                cv2.rectangle(vis, (10, 70), (410, 82), (160, 160, 160), 1)
                cv2.putText(vis, f"{n}/15", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                mode_label = "MANUAL(GravComp)" if args.manual else "AUTO"
                cv2.putText(vis, mode_label, (vis.shape[1] - 200, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1)

                cv2.imshow(WIN, vis)

            if cv2.waitKey(30) & 0xFF in [ord('q'), ord('Q'), 27]:
                finish_reason = "window exit"
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                finish_reason = "window closed"
                break

            try:
                if line_queue is not None and handle_line(line_queue.get_nowait()):
                    finish_reason = "user interrupted"
                    break
            except queue.Empty:
                pass

            if auto["finished"]:
                break

    except KeyboardInterrupt:
        finish_reason = "Ctrl+C interrupt"
        print("\n[Ctrl+C] Stopping and trying to solve")

    finally:
        cv2.destroyAllWindows()
        cam.close()
        if gc_ctrl is not None:
            gc_ctrl.safe_home()
        elif controller is not None:
            safe_home_and_disconnect()
        compute_and_save(finish_reason)

    print(f"\nDone, total samples: {calibrator.n_samples}.")
    if calibrator.n_samples > 0 and not result_saved:
        print("Tip: hand_eye.npz was not generated; collect more samples and try again.")


if __name__ == "__main__":
    main()
    os._exit(0)
