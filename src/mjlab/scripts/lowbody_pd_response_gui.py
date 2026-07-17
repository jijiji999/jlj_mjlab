"""Tk GUI for JLJLowBody PD response tests in mjlab/MuJoCo."""

from __future__ import annotations

import csv
import json
import math
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import mujoco

from mjlab.asset_zoo.robots.jljbot.jljlowbody_constants import get_jljlowbody_robot_cfg
from mjlab.entity import Entity
from mjlab.scripts.lowbody_pd_response_test import (
  FIELDNAMES,
  JointMotorParams,
  LowbodyPdResponseTestCfg,
  _actuator_by_joint,
  _build_joint_motor_params,
  _clip_dc_motor_torque,
  _compute_summary,
  _initial_qpos,
  _joint_names,
  _joint_position,
  _joint_velocity,
  _target,
)


class PlotCanvas(tk.Canvas):
  def __init__(self, master):
    super().__init__(master, background="#111827", highlightthickness=0, height=420)
    self.samples: list[dict[str, float]] = []
    self.reference_samples: list[dict[str, float]] = []
    self.reference_time_offset = 0.0
    self.sim_zero_position = math.nan
    self.reference_zero_position = math.nan
    self.window_sec = 10.0
    self.bind("<Configure>", lambda _event: self.redraw())

  def clear(self):
    self.samples.clear()
    self.sim_zero_position = math.nan
    self.redraw()

  def set_reference(self, samples: list[dict[str, float]]):
    self.reference_samples = samples
    self.reference_zero_position = self._zero_position(samples)
    self.redraw()

  def set_reference_time_offset(self, offset: float):
    self.reference_time_offset = offset
    self.redraw()

  def add_sample(self, sample: dict[str, float]):
    if not math.isfinite(self.sim_zero_position) and math.isfinite(
      sample.get("position", math.nan)
    ):
      self.sim_zero_position = sample["position"]
    self.samples.append(sample)
    cutoff = sample["time"] - self.window_sec
    self.samples = [item for item in self.samples if item["time"] >= cutoff]
    self.redraw()

  def redraw(self):
    self.delete("all")
    width = max(1, self.winfo_width())
    height = max(1, self.winfo_height())
    left, right, top, bottom = 58, 18, 18, 34
    plot_w = max(10, width - left - right)
    plot_h = max(10, height - top - bottom)
    self.create_rectangle(left, top, left + plot_w, top + plot_h, outline="#475569")
    for i in range(1, 4):
      x = left + plot_w * i / 4
      y = top + plot_h * i / 4
      self.create_line(x, top, x, top + plot_h, fill="#334155")
      self.create_line(left, y, left + plot_w, y, fill="#334155")
    if not self.samples and not self.reference_samples:
      self.create_text(
        width / 2, height / 2, text="等待仿真数据", fill="#cbd5e1", font=("Sans", 14)
      )
      return

    sim_zero = (
      self.sim_zero_position
      if math.isfinite(self.sim_zero_position)
      else self._zero_position(self.samples)
    )
    reference_zero = (
      self.reference_zero_position
      if math.isfinite(self.reference_zero_position)
      else self._zero_position(self.reference_samples)
    )

    values = []
    for samples, zero in (
      (self.samples, sim_zero),
      (self.reference_samples, reference_zero),
    ):
      for sample in samples:
        for key in ("target", "position"):
          value = sample[key] - zero
          if math.isfinite(value):
            values.append(value)
    if not values:
      values = [0.0]
    v_min, v_max = min(values), max(values)
    pad = max(0.02, 0.18 * (v_max - v_min))
    v_min -= pad
    v_max += pad
    if abs(v_max - v_min) < 1e-9:
      v_min -= 0.05
      v_max += 0.05
    if self.samples:
      newest = self.samples[-1]["time"]
    else:
      newest = max(
        sample["time"] + self.reference_time_offset for sample in self.reference_samples
      )
    oldest = newest - self.window_sec

    for i in range(5):
      fraction = i / 4
      value = v_max - fraction * (v_max - v_min)
      y = top + fraction * plot_h
      self.create_text(
        left - 8,
        y,
        anchor="e",
        text=f"{value:.3f}",
        fill="#cbd5e1",
        font=("Sans", 9),
      )

    def xy(zero, time_offset):
      def mapper(sample, key):
        value = sample[key] - zero
        if not math.isfinite(value):
          return None
        x = left + (sample["time"] + time_offset - oldest) / self.window_sec * plot_w
        y = top + (v_max - value) / (v_max - v_min) * plot_h
        return x, y

      return mapper

    self._series(
      self.reference_samples,
      "target",
      "#94a3b8",
      xy(reference_zero, self.reference_time_offset),
      dash=(4, 4),
    )
    self._series(
      self.reference_samples,
      "position",
      "#facc15",
      xy(reference_zero, self.reference_time_offset),
      dash=(4, 4),
    )
    self._series(self.samples, "target", "#38bdf8", xy(sim_zero, 0.0))
    self._series(self.samples, "position", "#f97316", xy(sim_zero, 0.0))
    zero_y = top + (v_max - 0.0) / (v_max - v_min) * plot_h
    if top <= zero_y <= top + plot_h:
      self.create_line(left, zero_y, left + plot_w, zero_y, fill="#64748b", dash=(3, 4))
    if self.samples:
      latest = self.samples[-1]
      sim_target = latest["target"] - sim_zero
      sim_position = latest["position"] - sim_zero
      sim_error = sim_target - sim_position
      self.create_text(
        left + 6,
        top + 8,
        anchor="nw",
        text=f"sim target {sim_target:.4f} rad",
        fill="#38bdf8",
        font=("Sans", 10),
      )
      self.create_text(
        left + 6,
        top + 28,
        anchor="nw",
        text=f"sim position {sim_position:.4f} rad",
        fill="#f97316",
        font=("Sans", 10),
      )
      self.create_text(
        left + 6,
        top + 48,
        anchor="nw",
        text=f"sim error {sim_error:.4f} rad",
        fill="#f43f5e",
        font=("Sans", 10),
      )
    if self.reference_samples:
      self.create_text(
        left + 6,
        top + 70,
        anchor="nw",
        text=f"hardware dashed, x offset {self.reference_time_offset:+.3f}s",
        fill="#facc15",
        font=("Sans", 10),
      )

  def _zero_position(self, samples: list[dict[str, float]]) -> float:
    for sample in samples:
      value = sample.get("position", math.nan)
      if not math.isfinite(value):
        continue
      return value
    return 0.0

  def _draw_line(
    self,
    points: list[float],
    color: str,
    dash: str | int | list[int] | tuple[int, ...] | None,
  ) -> None:
    if dash is None:
      self.create_line(points, fill=color, width=2, smooth=True)
    else:
      self.create_line(points, fill=color, width=2, smooth=True, dash=dash)

  def _series(
    self,
    samples: list[dict[str, float]],
    key: str,
    color: str,
    mapper: Callable[[dict[str, float], str], tuple[float, float] | None],
    dash: str | int | list[int] | tuple[int, ...] | None = None,
  ) -> None:
    points: list[float] = []
    for sample in samples:
      point = mapper(sample, key)
      if point is None:
        if len(points) >= 4:
          self._draw_line(points, color, dash)
        points = []
      else:
        points.extend(point)
    if len(points) >= 4:
      self._draw_line(points, color, dash)


class LowbodyPdResponseGui:
  def __init__(self, root: tk.Tk):
    self.root = root
    self.root.title("JLJLowBody MuJoCo PD Response")
    self.root.geometry("1180x720")
    self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
    self.stop_event = threading.Event()
    self.thread: threading.Thread | None = None
    self.joints = self._load_joint_names()

    self.joint_var = tk.StringVar(value=self.joints[0] if self.joints else "")
    self.mode_var = tk.StringVar(value="step")
    self.kp_var = tk.StringVar(value="")
    self.kd_var = tk.StringVar(value="")
    self.amplitude_var = tk.StringVar(value="0.05")
    self.frequency_var = tk.StringVar(value="0.5")
    self.duration_var = tk.StringVar(value="6.0")
    self.hold_kp_scale_var = tk.StringVar(value="2.0")
    self.hold_kd_scale_var = tk.StringVar(value="2.0")
    self.reference_time_offset_var = tk.StringVar(value="0.0")
    self.viewer_var = tk.BooleanVar(value=True)
    self.status = tk.StringVar(value="ready")

    self._build_ui()
    self._load_default_gains()
    self._pump_queue()

  def _load_joint_names(self) -> list[str]:
    robot = Entity(get_jljlowbody_robot_cfg())
    model = robot.compile()
    return _joint_names(model)

  def _build_ui(self):
    main = ttk.Frame(self.root, padding=14)
    main.pack(fill="both", expand=True)
    controls = ttk.Frame(main, width=320)
    controls.pack(side="left", fill="y", padx=(0, 14))
    controls.pack_propagate(False)
    ttk.Label(controls, text="mjlab 悬空关节响应测试", font=("Sans", 15, "bold")).pack(
      anchor="w", pady=(0, 12)
    )
    self.joint_combo = ttk.Combobox(
      controls, textvariable=self.joint_var, values=self.joints, state="readonly"
    )
    self._labeled(controls, "测试关节", self.joint_combo)
    self.joint_combo.bind(
      "<<ComboboxSelected>>", lambda _event: self._load_default_gains()
    )

    mode_row = ttk.Frame(controls)
    ttk.Radiobutton(mode_row, text="阶跃", value="step", variable=self.mode_var).pack(
      side="left"
    )
    ttk.Radiobutton(mode_row, text="正弦", value="sine", variable=self.mode_var).pack(
      side="left", padx=(16, 0)
    )
    mode_row.pack(fill="x", pady=(0, 8))

    self._entry(controls, "Kp", self.kp_var)
    self._entry(controls, "Kd", self.kd_var)
    self._entry(controls, "幅度 rad", self.amplitude_var)
    self._entry(controls, "频率 Hz", self.frequency_var)
    self._entry(controls, "时长 s", self.duration_var)
    self._entry(controls, "其他关节 Kp 倍率", self.hold_kp_scale_var)
    self._entry(controls, "其他关节 Kd 倍率", self.hold_kd_scale_var)
    ttk.Checkbutton(
      controls, text="打开 MuJoCo 机器人窗口", variable=self.viewer_var
    ).pack(anchor="w", pady=(6, 12))

    row = ttk.Frame(controls)
    row.pack(fill="x")
    ttk.Button(row, text="开始", command=self._start).pack(
      side="left", expand=True, fill="x", padx=(0, 6)
    )
    ttk.Button(row, text="停止", command=self._stop).pack(
      side="left", expand=True, fill="x"
    )
    ttk.Button(controls, text="导入实机曲线", command=self._import_hardware_curve).pack(
      fill="x", pady=(10, 0)
    )
    self._entry(
      controls,
      "实机曲线时间偏移 s",
      self.reference_time_offset_var,
      callback=self._on_reference_time_offset_changed,
    )
    ttk.Label(controls, textvariable=self.status, wraplength=300).pack(
      anchor="w", fill="x", pady=(18, 0)
    )

    plot_area = ttk.Frame(main)
    plot_area.pack(side="left", fill="both", expand=True)
    self.plot = PlotCanvas(plot_area)
    self.plot.pack(fill="both", expand=True)

  def _labeled(self, parent, label, widget):
    ttk.Label(parent, text=label).pack(anchor="w")
    widget.pack(fill="x", pady=(2, 8))

  def _entry(self, parent, label, var, callback=None):
    entry = ttk.Entry(parent, textvariable=var)
    self._labeled(parent, label, entry)
    if callback is not None:
      entry.bind("<Return>", callback)
      entry.bind("<FocusOut>", callback)

  def _load_default_gains(self):
    params = _build_joint_motor_params(LowbodyPdResponseTestCfg())
    joint = self.joint_var.get()
    if joint in params:
      self.kp_var.set(str(params[joint].kp))
      self.kd_var.set(str(params[joint].kd))

  def _read_float(self, var: tk.StringVar, label: str) -> float:
    try:
      value = float(var.get())
    except ValueError as exc:
      raise ValueError(f"{label} 不是有效数字") from exc
    if not math.isfinite(value):
      raise ValueError(f"{label} 不是有限数值")
    return value

  def _start(self):
    if self.thread and self.thread.is_alive():
      return
    try:
      cfg = LowbodyPdResponseTestCfg(
        joint_regex=self.joint_var.get(),
        tests=self.mode_var.get(),
        kp=f"{self.joint_var.get()}={self._read_float(self.kp_var, 'Kp')}",
        kd=f"{self.joint_var.get()}={self._read_float(self.kd_var, 'Kd')}",
        step_amplitude=self._read_float(self.amplitude_var, "幅度"),
        sine_amplitude=self._read_float(self.amplitude_var, "幅度"),
        sine_frequency_hz=self._read_float(self.frequency_var, "频率"),
        duration_s=self._read_float(self.duration_var, "时长"),
        disable_gravity=False,
        fixed_base=True,
        hold_other_joints=True,
        hold_kp_scale=self._read_float(self.hold_kp_scale_var, "其他关节 Kp 倍率"),
        hold_kd_scale=self._read_float(self.hold_kd_scale_var, "其他关节 Kd 倍率"),
      )
    except ValueError as exc:
      messagebox.showerror("参数错误", str(exc))
      return
    self.plot.clear()
    self.stop_event.clear()
    self.thread = threading.Thread(
      target=self._run_sim, args=(cfg, self.viewer_var.get()), daemon=True
    )
    self.thread.start()

  def _stop(self):
    self.stop_event.set()

  def _import_hardware_curve(self):
    path = filedialog.askopenfilename(
      title="选择实机响应曲线 CSV",
      filetypes=(("CSV files", "*.csv"), ("All files", "*")),
    )
    if not path:
      return
    try:
      samples = []
      joint_hint = ""
      with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
          time_s = float(row.get("time_s", "nan"))
          target = float(row.get("target", "nan"))
          position = float(row.get("position", "nan"))
          error = float(row.get("error", "nan"))
          if not math.isfinite(time_s):
            continue
          if not joint_hint and row.get("joint"):
            joint_hint = str(row.get("joint"))
          samples.append(
            {
              "time": time_s,
              "target": target,
              "position": position,
              "error": error,
            }
          )
    except (OSError, ValueError) as exc:
      messagebox.showerror("导入失败", str(exc))
      return
    if not samples:
      messagebox.showwarning("导入失败", "CSV 中没有有效曲线数据")
      return
    self.plot.set_reference(samples)
    self._on_reference_time_offset_changed()
    self._load_hardware_summary(Path(path), joint_hint=joint_hint)
    self.status.set(f"已导入实机曲线: {path}")

  def _on_reference_time_offset_changed(self, _event=None):
    try:
      offset = self._read_float(self.reference_time_offset_var, "实机曲线时间偏移")
    except ValueError:
      return
    self.plot.set_reference_time_offset(offset)

  def _set_joint_from_hardware_name(self, name: str):
    if not name:
      return
    candidates = [name, f"{name}_joint"]
    if name.endswith("_joint"):
      candidates.append(name.removesuffix("_joint"))
    for joint in candidates:
      if joint in self.joints:
        self.joint_var.set(joint)
        return

  def _load_hardware_summary(self, csv_path: Path, joint_hint: str = ""):
    summary_path = csv_path.with_name("summary.json")
    self._set_joint_from_hardware_name(joint_hint)
    if not summary_path.exists():
      return
    try:
      summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      return

    motor = str(summary.get("joint", "") or summary.get("motor", ""))
    self._set_joint_from_hardware_name(motor)
    if summary.get("mode") in {"step", "sine"}:
      self.mode_var.set(str(summary["mode"]))
    for key, var in [
      ("kp", self.kp_var),
      ("kd", self.kd_var),
      ("amplitude", self.amplitude_var),
      ("frequency", self.frequency_var),
      ("duration", self.duration_var),
    ]:
      value = summary.get(key)
      if value not in (None, ""):
        var.set(str(value))

  def _pump_queue(self):
    try:
      while True:
        kind, payload = self.queue.get_nowait()
        if kind == "sample":
          self.plot.add_sample(payload)  # type: ignore[arg-type]
        elif kind == "status":
          self.status.set(str(payload))
    except queue.Empty:
      pass
    self.root.after(30, self._pump_queue)

  def _run_sim(self, cfg: LowbodyPdResponseTestCfg, show_viewer: bool):
    viewer = None
    try:
      robot = Entity(get_jljlowbody_robot_cfg())
      model = robot.compile()
      model.opt.timestep = cfg.physics_dt
      data = mujoco.MjData(model)
      data.qpos[:] = _initial_qpos(model)
      data.qvel[:] = 0.0
      mujoco.mj_forward(model, data)
      if show_viewer:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 120

      joint = cfg.joint_regex
      test_name = cfg.tests
      params = _build_joint_motor_params(cfg)
      params[joint] = JointMotorParams(
        kp=float(self.kp_var.get()),
        kd=float(self.kd_var.get()),
        effort_limit=params[joint].effort_limit,
        saturation_effort=params[joint].saturation_effort,
        velocity_limit=params[joint].velocity_limit,
      )
      all_joints = _joint_names(model)
      joint_id_by_name = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in all_joints
      }
      actuator_ids = _actuator_by_joint(model)
      center = {
        name: _joint_position(model, data, joint_id_by_name[name])
        for name in all_joints
      }
      root_qpos = data.qpos[:7].copy()
      root_qvel = data.qvel[:6].copy()
      rows = []
      sample_period = 1.0 / cfg.sample_hz
      next_sample = 0.0
      wall_start = time.monotonic()
      total_s = cfg.pre_signal_s + cfg.duration_s
      self.queue.put(("status", "running: base_link fixed in air, gravity on"))

      while not self.stop_event.is_set() and data.time <= total_s:
        data.ctrl[:] = 0.0
        target_for_test = center[joint]
        for name in all_joints:
          target = (
            _target(test_name, center[joint], data.time, cfg)
            if name == joint
            else center[name]
          )
          if name == joint:
            target_for_test = target
          joint_id = joint_id_by_name[name]
          motor_params = params[name]
          kp = motor_params.kp if name == joint else motor_params.kp * cfg.hold_kp_scale
          kd = motor_params.kd if name == joint else motor_params.kd * cfg.hold_kd_scale
          pos = _joint_position(model, data, joint_id)
          vel = _joint_velocity(model, data, joint_id)
          torque = _clip_dc_motor_torque(
            motor_params, vel, kp * (target - pos) - kd * vel
          )
          data.ctrl[actuator_ids[name]] = torque

        mujoco.mj_step(model, data)
        data.qpos[:7] = root_qpos
        data.qvel[:6] = root_qvel
        mujoco.mj_forward(model, data)
        if viewer is not None:
          viewer.sync()

        if data.time >= next_sample:
          joint_id = joint_id_by_name[joint]
          pos = _joint_position(model, data, joint_id)
          vel = _joint_velocity(model, data, joint_id)
          sample = {
            "time": float(data.time),
            "target": float(target_for_test),
            "position": float(pos),
            "velocity": float(vel),
            "error": float(target_for_test - pos),
          }
          self.queue.put(("sample", sample))
          rows.append(
            {
              "time_s": sample["time"],
              "source": "mjlab_mujoco_gui",
              "test": test_name,
              "joint": joint,
              "target": sample["target"],
              "position": sample["position"],
              "velocity": sample["velocity"],
              "error": sample["error"],
              "torque_command": float(data.ctrl[actuator_ids[joint]]),
              "actuator_force": float(data.qfrc_actuator[model.jnt_dofadr[joint_id]]),
              "kp": params[joint].kp,
              "kd": params[joint].kd,
            }
          )
          next_sample += sample_period

        sleep_s = wall_start + data.time - time.monotonic()
        if sleep_s > 0:
          time.sleep(min(sleep_s, 0.01))

      out_dir = Path("logs/lowbody_pd_response_gui") / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
      )
      out_dir.mkdir(parents=True, exist_ok=True)
      with (out_dir / "joint_response.csv").open(
        "w", newline="", encoding="utf-8"
      ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
      summary = _compute_summary(rows, test_name)
      summary["joint"] = joint
      (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
      )
      self.queue.put(("status", f"done: {out_dir}"))
    except Exception as exc:
      self.queue.put(("status", f"error: {exc}"))
    finally:
      if viewer is not None:
        viewer.close()


def main():
  root = tk.Tk()
  LowbodyPdResponseGui(root)
  root.mainloop()


if __name__ == "__main__":
  main()
