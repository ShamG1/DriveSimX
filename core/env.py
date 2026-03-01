"""基于 C++ 后端的高性能场景环境（提供 Python 友好 API）。

渲染由 C++ OpenGL/GLFW 渲染器负责。

说明：
- 多智能体模式：返回 obs (N,145)、rewards (N,)、terminated、truncated、info。
- traffic_flow 模式（单智能体）：返回 obs (145,)、reward 标量、terminated、truncated、info。
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Union
from collections import deque

import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from . import cpp_backend
except ImportError:
    import cpp_backend  # type: ignore

from .utils import build_lane_layout, ROUTE_MAP_BY_SCENARIO, DEFAULT_REWARD_CONFIG


def _apply_reward_config(env: Any, reward_cfg: Dict[str, Any]) -> None:
    if not hasattr(env, "reward_config"):
        return
    rc = env.reward_config

    if "progress_scale" in reward_cfg:
        rc.k_prog = float(reward_cfg["progress_scale"])
    if "stuck_speed_threshold" in reward_cfg:
        rc.v_min_ms = float(reward_cfg["stuck_speed_threshold"])
    if "stuck_penalty" in reward_cfg:
        rc.k_stuck = float(reward_cfg["stuck_penalty"])
    if "crash_vehicle_penalty" in reward_cfg:
        rc.k_cv = float(reward_cfg["crash_vehicle_penalty"])
    if "crash_wall_penalty" in reward_cfg:
        rc.k_cw = float(reward_cfg["crash_wall_penalty"])
    if "crash_line_penalty" in reward_cfg:
        rc.k_cl = float(reward_cfg["crash_line_penalty"])
    if "success_reward" in reward_cfg:
        rc.k_succ = float(reward_cfg["success_reward"])
    if "action_smoothness_scale" in reward_cfg:
        rc.k_sm = float(reward_cfg["action_smoothness_scale"])
    if "team_alpha" in reward_cfg:
        rc.alpha = float(reward_cfg["team_alpha"])


def _parse_num_lanes_from_scenario_name(name: str) -> int:
    # Special case for merge scenario: highway_merge_3lane uses 3 lanes
    if "highway_merge_3lane" == name:
        return 3
    # Accept patterns like: cross_2lane, t_junction_3lane_v2, roundabout_4lane
    m = re.findall(r"(?:^|_)(\d+)lane(?:$|_)", str(name))
    if m:
        return int(m[-1])

    # Allow scenarios without lane suffix by explicit mapping.
    # Keep this minimal: only handle known special cases here.
    if str(name) == "bottleneck":
        return 3

    raise ValueError(
        f"Cannot parse num_lanes from scenario_name={name!r}. Expected to contain like '_2lane', or be a known special-case scenario (e.g. 'bottleneck')."
    )


class MetricsTracker:
    """Episode + scenario-level metrics.

    Metrics are computed using *agent_id* (not slot index) so respawn-enabled
    training remains comparable to respawn-disabled evaluation.
    """

    def __init__(self):
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.last_episode_metrics: Dict[str, Any] | None = None
        self._active = False

    def reset_episode(self, scenario_name: str):
        self._active = True
        self.current_scenario = str(scenario_name)
        self.current_time = 0.0

        # agent_id -> flags + first success time
        self._agents: Dict[int, Dict[str, Any]] = {}

    def update(self, dt: float, info: Dict[str, Any]):
        if not self._active:
            return

        self.current_time += float(dt)

        status_list = list(info.get("status", []))
        agent_ids = list(info.get("agent_ids", []))

        for i, status in enumerate(status_list):
            aid = int(agent_ids[i]) if i < len(agent_ids) else int(i)

            rec = self._agents.get(aid)
            if rec is None:
                rec = {"participated": True, "arrived": False, "collided": False, "success_time": None}
                self._agents[aid] = rec

            if status == "SUCCESS" and not rec["arrived"]:
                rec["arrived"] = True
                rec["success_time"] = self.current_time
            elif status in ("CRASH_CAR", "CRASH_WALL") and not rec["collided"]:
                rec["collided"] = True

    def end_episode(self, *, terminated: bool, truncated: bool, end_reason: str):
        if not self._active:
            return

        participated = list(self._agents.keys())
        total_participated = len(participated)

        arrived_ids = [aid for aid, r in self._agents.items() if r.get("arrived")]
        collided_ids = [aid for aid, r in self._agents.items() if r.get("collided")]
        success_times = [r["success_time"] for r in self._agents.values() if r.get("arrived") and r.get("success_time") is not None]

        metrics: Dict[str, Any] = {
            "scenario": self.current_scenario,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "end_reason": str(end_reason),
            "episode_time": float(self.current_time),
            "agents_participated": total_participated,
            "arrived_count": len(arrived_ids),
            "collided_count": len(collided_ids),
            "success_rate": (len(arrived_ids) / total_participated) if total_participated > 0 else 0.0,
            "collision_rate": (len(collided_ids) / total_participated) if total_participated > 0 else 0.0,
            "avg_time_to_success": float(np.mean(success_times)) if success_times else None,
        }

        self.last_episode_metrics = metrics
        self.history.setdefault(self.current_scenario, []).append(metrics)

        self._active = False

    def get_summary(self, scenario_name: str | None = None) -> Dict[str, Dict[str, Any]]:
        scenarios = [scenario_name] if scenario_name else list(self.history.keys())
        out: Dict[str, Dict[str, Any]] = {}

        for sn in scenarios:
            eps = self.history.get(sn, [])
            if not eps:
                continue

            total_agents = sum(int(e.get("agents_participated", 0)) for e in eps)
            total_arrived = sum(int(e.get("arrived_count", 0)) for e in eps)
            total_collided = sum(int(e.get("collided_count", 0)) for e in eps)

            # avg_time_to_success: average across episodes where it's defined
            times = [e.get("avg_time_to_success") for e in eps if e.get("avg_time_to_success") is not None]

            out[sn] = {
                "episodes": len(eps),
                "total_agents_participated": total_agents,
                "success_rate": (total_arrived / total_agents) if total_agents > 0 else 0.0,
                "collision_rate": (total_collided / total_agents) if total_agents > 0 else 0.0,
                "avg_time_to_success": float(np.mean(times)) if times else None,
            }

        return out

class ScenarioEnv:
    def __init__(self, config: Dict[str, Any] | None = None):
        if config is None:
            config = {}

        self.metrics_tracker = MetricsTracker()

        self.scenario_name = config.get("scenario_name", None)
        if not self.scenario_name:
            raise ValueError("config['scenario_name'] is required")

        self.num_lanes = _parse_num_lanes_from_scenario_name(str(self.scenario_name))

        self.traffic_flow = bool(config.get("traffic_flow", False))

        if self.traffic_flow:
            self.num_agents = 1
        else:
            self.num_agents = int(config.get("num_agents", 1))

        self.render_mode = config.get("render_mode", None)
        self.show_lane_ids = bool(config.get("show_lane_ids", False))
        self.show_lidar = bool(config.get("show_lidar", False))

        # 是否启用团队奖励混合：默认关闭，可通过配置显式开启。
        # 注意：traffic_flow（单智能体）模式下会强制关闭团队奖励。
        use_team = bool(config.get("use_team_reward", False))
        if self.traffic_flow:
            use_team = False

        self.respawn_enabled = bool(config.get("respawn_enabled", True))
        respawn = self.respawn_enabled
        max_steps = int(config.get("max_steps_per_episode", config.get("max_steps", 2000)))
        self.max_steps = max_steps
        # Penalty applied only when respawn is disabled and episode is truncated by max steps.
        self.max_steps_penalty_no_respawn = float(
            config.get("max_steps_penalty_no_respawn", DEFAULT_REWARD_CONFIG.get("max_steps_penalty_no_respawn", -5.0))
        )
        self.respawn_penalty = float(config.get("respawn_penalty", DEFAULT_REWARD_CONFIG.get("respawn_penalty", -0.5)))
        self.no_progress_penalty = float(config.get("no_progress_penalty", DEFAULT_REWARD_CONFIG.get("no_progress_penalty", -0.2)))
        self.no_progress_window_steps = max(1, int(config.get("no_progress_window_steps", DEFAULT_REWARD_CONFIG.get("no_progress_window_steps", 30))))
        self.no_progress_threshold = float(config.get("no_progress_threshold", DEFAULT_REWARD_CONFIG.get("no_progress_threshold", 0.01)))

        # Environment-level cooperative shaping / lightweight interaction modeling.
        self.cooperative_mode = bool(config.get("cooperative_mode", False))
        self.cooperative_alpha = float(config.get("cooperative_alpha", 0.3))
        self.cooperative_credit_coef = float(config.get("cooperative_credit_coef", 0.0))
        self.pairwise_coordination_enabled = bool(config.get("pairwise_coordination_enabled", False))
        self.pairwise_distance_threshold = float(config.get("pairwise_distance_threshold", 80.0))
        self.pairwise_brake_scale = float(config.get("pairwise_brake_scale", 0.35))
        self.pairwise_cooldown_steps = max(1, int(config.get("pairwise_cooldown_steps", 6)))
        self._pairwise_cooldown: Dict[tuple[int, int], int] = {}

        # Metrics behavior (enabled by default)
        # Keep the public config surface minimal: a single on/off switch.
        self.metrics_enabled = bool(config.get("metrics_enabled", False))
        # Internal policy: when respawn is enabled, also finalize metrics if all agents are dead.
        # (Some training loops treat this as episode end.)
        self._metrics_end_on_all_dead = True

        self.ego_routes = config.get("ego_routes", None)
        if self.ego_routes is None:
            mapping = ROUTE_MAP_BY_SCENARIO.get(str(self.scenario_name), None)
            if mapping is None:
                raise RuntimeError(f"No route mapping defined for scenario_name={self.scenario_name!r}")

            routes = []
            for mp in mapping.values():
                for in_id, out_id in mp.items():
                    # Support both numeric indices and full string IDs
                    final_in = in_id if isinstance(in_id, str) else f"IN_{in_id}"
                    final_out = out_id if isinstance(out_id, str) else f"OUT_{out_id}"
                    routes.append((final_in, final_out))

            if not routes:
                raise RuntimeError(f"No valid routes generated for scenario_name={self.scenario_name!r}")

            self.ego_routes = [routes[i % len(routes)] for i in range(self.num_agents)]

        self.lane_layout = build_lane_layout(self.num_lanes)
        self.points = self.lane_layout["points"]

        if not cpp_backend.has_cpp_backend():
            raise RuntimeError(
                "SIM_MARL_ENV backend not available. Build it first, or run with the pure-Python environment. "
                "(cpp/build or cpp/build/Release must contain the SIM_MARL_ENV extension)"
            )

        self.env = cpp_backend.ScenarioEnv(self.num_lanes)
        self.env.set_scenario_name(str(self.scenario_name))

        # Inject explicit route intents so C++ does not infer intent from lane ids.
        mapping = ROUTE_MAP_BY_SCENARIO.get(str(self.scenario_name), None)
        if mapping is None:
            raise RuntimeError(f"No route mapping defined for scenario_name={self.scenario_name!r}")

        intent_ids = {"straight": 0, "left": 1, "right": 2}
        intent_items = []
        for turn_type, mp in mapping.items():
            intent_id = intent_ids.get(turn_type, 0)
            for in_idx, out_idx in mp.items():
                intent_items.append(((f"IN_{in_idx}", f"OUT_{out_idx}"), int(intent_id)))

        self.env.set_route_intents(intent_items)
        self.env.configure(use_team, respawn, max_steps)

        # Scenario bitmaps (required)
        scenarios_root = config.get(
            "scenarios_root",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenarios"),
        )
        scenario_dir = os.path.join(scenarios_root, str(self.scenario_name))
        drivable_png = os.path.join(scenario_dir, "drivable.png")
        yellowline_png = os.path.join(scenario_dir, "yellowline.png")
        lane_dashes_png = os.path.join(scenario_dir, "lane_dashes.png")
        lane_id_png = os.path.join(scenario_dir, "lane_id.png")
        ok = self.env.load_scenario_bitmaps(drivable_png, yellowline_png, lane_dashes_png, lane_id_png)
        if not ok:
            raise RuntimeError(
                f"Failed to load scenario bitmaps from {scenario_dir}. "
                "Expected drivable.png, yellowline.png, lane_id.png with resolution matching constants.h"
            )

        # traffic flow (single ego + NPC)
        self.traffic_density = float(config.get("traffic_density", 0.5))
        self.traffic_mode = config.get("traffic_mode", "stochastic")
        self.traffic_kmax = int(config.get("traffic_kmax", 20))

        self.env.configure_traffic(self.traffic_flow, self.traffic_density)
        self.env.set_traffic_mode(self.traffic_mode, self.traffic_kmax)

        traffic_routes = []
        for mp in mapping.values():
            for in_idx, out_idx in mp.items():
                final_in = in_idx if isinstance(in_idx, str) else f"IN_{in_idx}"
                final_out = out_idx if isinstance(out_idx, str) else f"OUT_{out_idx}"
                traffic_routes.append((final_in, final_out))
        self.env.configure_routes(traffic_routes)

        reward_cfg = config.get("reward_config", None)
        if reward_cfg is None:
            reward_cfg = DEFAULT_REWARD_CONFIG.get("reward_config", {})
        if isinstance(reward_cfg, dict):
            _apply_reward_config(self.env, reward_cfg)

        self.cars: List[cpp_backend.Car] = []
        self.traffic_cars: List[cpp_backend.Car] = []

        self.reset()


    def reset(self):
        # If the caller resets mid-episode, finalize the previous episode's metrics.
        if self.metrics_enabled and getattr(self.metrics_tracker, "_active", False):
            self.metrics_tracker.end_episode(terminated=False, truncated=False, end_reason="reset")

        self.env.reset()

        # Ensure ego_routes length matches num_agents
        if not self.ego_routes:
            raise RuntimeError("ego_routes is empty; cannot reset environment")
        if len(self.ego_routes) < self.num_agents:
            # Cycle routes if fewer routes than agents
            self.ego_routes = [self.ego_routes[i % len(self.ego_routes)] for i in range(self.num_agents)]

        for i in range(self.num_agents):
            start_id, end_id = self.ego_routes[i]
            self.env.add_car_with_route(start_id, end_id)
        self.cars = self.env.cars
        if self.traffic_flow:
            try:
                self.traffic_cars = list(self.env.traffic_cars)
            except Exception:
                self.traffic_cars = []
        obs = self._collect_obs()

        if self.metrics_enabled:
            # Only start a new episode if we aren't mid-episode (resetting after end_episode was already called)
            # This avoids the first reset in __init__ starting a tracked episode that gets counted twice.
            self.metrics_tracker.reset_episode(self.scenario_name)

        # Per-agent shaping trackers (environment-side, independent from trainer)
        self._last_status_by_id: Dict[int, str] = {}
        self._progress_hist_by_id: Dict[int, deque] = {}
        self._last_progress_by_id: Dict[int, float] = {}

        if self.traffic_flow:
            return obs[0], {}
        return obs, {}

    def _collect_obs(self) -> np.ndarray:
        # Zero-copy fast path (C++ returns a NumPy view)
        if hasattr(self.env, "get_observations_numpy"):
            try:
                obs = self.env.get_observations_numpy()
                return np.asarray(obs, dtype=np.float32)
            except Exception:
                pass

        # Fallback: flat buffer
        if hasattr(self.env, "get_observations_flat"):
            obs_flat = self.env.get_observations_flat()
            return np.asarray(obs_flat, dtype=np.float32).reshape(-1, 145)

        # Legacy fallback
        obs = self.env.get_observations()
        return np.asarray(obs, dtype=np.float32)

    def _agent_progress_proxy(self, agent_slot: int, agent_id: int, obs: np.ndarray) -> float:
        """Estimate per-agent progress for no-progress penalty.

        Priority:
        1) If C++ car object exposes route progress-like scalar, use it.
        2) Fallback to observation first feature (commonly progress-like in this env family).
        """
        # 1) Try C++ car fields
        try:
            if 0 <= int(agent_slot) < len(self.cars):
                car = self.cars[int(agent_slot)]
                for attr in ("route_progress", "progress", "s", "distance_along_route"):
                    if hasattr(car, attr):
                        v = float(getattr(car, attr))
                        if np.isfinite(v):
                            return v
        except Exception:
            pass

        # 2) Obs fallback
        try:
            row = np.asarray(obs[int(agent_slot)], dtype=np.float32)
            if row.size > 0 and np.isfinite(float(row[0])):
                return float(row[0])
        except Exception:
            pass

        return float(self._last_progress_by_id.get(int(agent_id), 0.0))

    def _apply_pairwise_coordination(self, actions: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Environment-side lightweight pairwise coordination for close agents.

        Returns:
            adjusted_actions, applied_flag_per_agent(0/1)
        """
        out = np.asarray(actions, dtype=np.float32).copy()
        applied = np.zeros((out.shape[0],), dtype=np.float32) if out.ndim == 2 else np.zeros((0,), dtype=np.float32)
        if (not self.pairwise_coordination_enabled) or out.ndim != 2 or out.shape[0] <= 1:
            return out, applied

        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim != 2 or obs_arr.shape[0] != out.shape[0] or obs_arr.shape[1] < 2:
            return out, applied

        # cooldown decay
        expired = []
        for k, v in self._pairwise_cooldown.items():
            nv = int(v) - 1
            if nv <= 0:
                expired.append(k)
            else:
                self._pairwise_cooldown[k] = nv
        for k in expired:
            self._pairwise_cooldown.pop(k, None)

        thr2 = float(self.pairwise_distance_threshold) ** 2
        n = int(out.shape[0])
        for i in range(n):
            xi, yi = float(obs_arr[i, 0]), float(obs_arr[i, 1])
            for j in range(i + 1, n):
                xj, yj = float(obs_arr[j, 0]), float(obs_arr[j, 1])
                dx, dy = xi - xj, yi - yj
                if (dx * dx + dy * dy) > thr2:
                    continue

                key = (i, j)
                if self._pairwise_cooldown.get(key, 0) > 0:
                    continue

                # simple yielding heuristic based on longitudinal projection in world axes
                if abs(dx) >= abs(dy):
                    yield_idx = i if xi < xj else j
                else:
                    yield_idx = i if yi < yj else j

                out[yield_idx, 0] = float(np.clip(out[yield_idx, 0] - self.pairwise_brake_scale, -1.0, 1.0))
                applied[yield_idx] = 1.0
                self._pairwise_cooldown[key] = int(self.pairwise_cooldown_steps)

        return out, applied

    def _apply_cooperative_reward_mixing(self, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Environment-side mixed cooperative reward + lightweight credit shaping.

        Returns:
            mixed_rewards, cooperative_mix_delta, cooperative_credit_delta
        """
        r = np.asarray(rewards, dtype=np.float32)
        mix_delta = np.zeros_like(r, dtype=np.float32)
        credit_delta = np.zeros_like(r, dtype=np.float32)
        if (not self.cooperative_mode) or r.size == 0:
            return r, mix_delta, credit_delta

        alpha = float(np.clip(self.cooperative_alpha, 0.0, 1.0))
        team_mean = float(np.mean(r))
        mixed = (1.0 - alpha) * r + alpha * team_mean
        mix_delta = (mixed - r).astype(np.float32, copy=False)

        beta = float(np.clip(self.cooperative_credit_coef, 0.0, 1.0))
        if beta > 0.0 and r.size > 1:
            others_mean = (float(np.sum(r)) - r) / float(r.size - 1)
            credit_term = team_mean - others_mean
            credit_delta = (beta * credit_term).astype(np.float32, copy=False)
            mixed = mixed + credit_delta

        return mixed.astype(np.float32, copy=False), mix_delta, credit_delta

    def step(self, actions: Union[np.ndarray, List[List[float]], List[float]], dt: float = 1.0 / 60.0):
        actions = np.asarray(actions, dtype=np.float32)

        # Accept both (2,) and (1,2) for single-agent manual control
        if self.traffic_flow:
            actions = actions.reshape(1, 2)
        else:
            if actions.ndim == 1:
                if actions.size == 2 and self.num_agents == 1:
                    actions = actions.reshape(1, 2)
                else:
                    raise ValueError(f"Expected actions shape (N,2) for multi-agent, got {actions.shape}")

        # Optional environment-side pairwise coordination before stepping physics.
        pairwise_applied = np.zeros((actions.shape[0],), dtype=np.float32) if actions.ndim == 2 else np.zeros((0,), dtype=np.float32)
        if (not self.traffic_flow) and actions.ndim == 2 and actions.shape[0] > 1:
            obs_now = self._collect_obs()
            actions, pairwise_applied = self._apply_pairwise_coordination(actions, obs_now)

        # Use optimized numpy step if available
        if hasattr(self.env, "step_numpy"):
            res = self.env.step_numpy(actions, float(dt))
        else:
            res = self.env.step(actions[:, 0].tolist(), actions[:, 1].tolist(), float(dt))

        if self.traffic_flow:
            try:
                self.traffic_cars = list(self.env.traffic_cars)
            except Exception:
                self.traffic_cars = []

        # Optimization: res.obs from C++ is often a copy. 
        # We prefer using the zero-copy observations buffer which is already updated in step().
        obs = self._collect_obs()

        rewards = np.asarray(res.rewards, dtype=np.float32)
        terminated = bool(res.terminated)
        truncated = bool(res.truncated)

        py_respawn = np.zeros_like(rewards, dtype=np.float32)
        py_no_progress = np.zeros_like(rewards, dtype=np.float32)
        py_max_steps = np.zeros_like(rewards, dtype=np.float32)

        # If respawn is disabled, penalize episodes that end by max-step truncation.
        if truncated and (not self.respawn_enabled) and rewards.size > 0:
            rewards = rewards + self.max_steps_penalty_no_respawn
            py_max_steps += np.float32(self.max_steps_penalty_no_respawn)

        collisions = {int(res.agent_ids[i]): str(res.status[i]) for i in range(len(res.status))}

        statuses = list(res.status)
        agent_ids = list(map(int, getattr(res, "agent_ids", [])))

        # Environment-side shaping: respawn penalty + no-progress window penalty.
        if rewards.size > 0:
            # Respawn penalty: apply when previous status was crash and current status is alive/success.
            # This captures "crash -> respawn" transitions in respawn-enabled mode.
            if self.respawn_enabled and self.respawn_penalty != 0.0:
                for i, status in enumerate(statuses):
                    aid = agent_ids[i] if i < len(agent_ids) else i
                    prev_status = self._last_status_by_id.get(int(aid), None)
                    if prev_status in ("CRASH_CAR", "CRASH_WALL") and status not in ("CRASH_CAR", "CRASH_WALL"):
                        rewards[i] += np.float32(self.respawn_penalty)
                        py_respawn[i] += np.float32(self.respawn_penalty)

            # No-progress penalty: if progress gain in window is too small, penalize.
            if self.no_progress_penalty != 0.0 and self.no_progress_window_steps > 0:
                for i, status in enumerate(statuses):
                    aid = agent_ids[i] if i < len(agent_ids) else i
                    if status in ("CRASH_CAR", "CRASH_WALL", "SUCCESS"):
                        continue

                    cur_prog = self._agent_progress_proxy(i, int(aid), obs)
                    hist = self._progress_hist_by_id.setdefault(int(aid), deque(maxlen=self.no_progress_window_steps))
                    if len(hist) == hist.maxlen:
                        gain = float(cur_prog - hist[0])
                        if gain < self.no_progress_threshold:
                            rewards[i] += np.float32(self.no_progress_penalty)
                            py_no_progress[i] += np.float32(self.no_progress_penalty)
                    hist.append(float(cur_prog))
                    self._last_progress_by_id[int(aid)] = float(cur_prog)

            # Update per-agent last status for next step transition checks.
            for i, status in enumerate(statuses):
                aid = agent_ids[i] if i < len(agent_ids) else i
                self._last_status_by_id[int(aid)] = str(status)

        # Cooperative mixing is applied in environment layer.
        rewards, py_coop_mix, py_coop_credit = self._apply_cooperative_reward_mixing(rewards)

        reward_components_cpp = {
            "progress": list(map(float, getattr(res, "r_progress", []))),
            "stuck": list(map(float, getattr(res, "r_stuck", []))),
            "smooth": list(map(float, getattr(res, "r_smooth", []))),
            "line": list(map(float, getattr(res, "r_line", []))),
            "crash_vehicle": list(map(float, getattr(res, "r_crash_vehicle", []))),
            "crash_wall": list(map(float, getattr(res, "r_crash_wall", []))),
            "success": list(map(float, getattr(res, "r_success", []))),
            "team_mix_cpp": list(map(float, getattr(res, "r_team_mix", []))),
        }

        reward_components_py = {
            "respawn_penalty": list(map(float, py_respawn.tolist())),
            "no_progress_penalty": list(map(float, py_no_progress.tolist())),
            "max_steps_penalty_no_respawn": list(map(float, py_max_steps.tolist())),
            "cooperative_mix_py": list(map(float, py_coop_mix.tolist())),
            "cooperative_credit_py": list(map(float, py_coop_credit.tolist())),
            "pairwise_action_adjust_applied": list(map(float, pairwise_applied.tolist())),
        }

        info = {
            "step": int(res.step),
            "rewards": rewards.tolist() if not self.traffic_flow else float(rewards[0]) if len(rewards) else 0.0,
            "collisions": collisions,
            "agents_alive": int(getattr(res, "agents_alive", 0)),
            "terminated": terminated,
            "truncated": truncated,
            "done": list(res.done),
            "status": statuses,
            "agent_ids": agent_ids,
            "reward_components_cpp": reward_components_cpp,
            "reward_components_py": reward_components_py,
        }

        # Update metrics every step (enabled by default)
        if self.metrics_enabled:
            self.metrics_tracker.update(float(dt), info)

            end_reason = None
            if truncated:
                end_reason = "truncated"
            elif terminated:
                end_reason = "terminated"
            elif self.respawn_enabled and self._metrics_end_on_all_dead and info.get("agents_alive", 1) == 0:
                end_reason = "all_dead"

            if end_reason is not None:
                self.metrics_tracker.end_episode(terminated=terminated, truncated=truncated, end_reason=end_reason)

        if self.traffic_flow:
            return obs[0], float(rewards[0]) if len(rewards) else 0.0, terminated, truncated, info
        return obs, rewards, terminated, truncated, info

    def freeze_traffic(self, freeze: bool):
        """Freeze/unfreeze NPC refills (useful for MCTS rollouts)."""
        self.env.freeze_traffic(freeze)

    def render(
        self,
        show_lane_ids: bool | None = None,
        show_lidar: bool | None = None,
        show_connections: bool | None = None,
    ):
        if self.render_mode != "human":
            return
        if show_lane_ids is None:
            show_lane_ids = self.show_lane_ids
        if show_lidar is None:
            show_lidar = self.show_lidar
        if show_connections is None:
            show_connections = bool(getattr(self, "show_connections", False))
        self.env.render(bool(show_lane_ids), bool(show_lidar), bool(show_connections))

    def metrics_summary(self, scenario_name: str | None = None) -> Dict[str, Dict[str, Any]]:
        """Convenience wrapper around MetricsTracker.get_summary()."""
        return self.metrics_tracker.get_summary(scenario_name)

    def last_metrics(self) -> Dict[str, Any] | None:
        """Return metrics for the most recently finalized episode (or None)."""
        return self.metrics_tracker.last_episode_metrics

    def reset_metrics_history(self):
        """Clear all accumulated metrics history.

        Note: ScenarioEnv.__init__ calls reset() once. If metrics are enabled, that can start
        an in-progress episode. This method clears both the stored history and any in-progress
        episode state, so subsequent benchmarking starts from a clean slate.
        """
        self.metrics_tracker.history.clear()
        self.metrics_tracker.last_episode_metrics = None

        # Also drop any in-progress episode to avoid an extra "reset"-ended episode being counted.
        if getattr(self.metrics_tracker, "_active", False):
            self.metrics_tracker._active = False
        if hasattr(self.metrics_tracker, "_agents"):
            try:
                self.metrics_tracker._agents.clear()
            except Exception:
                pass

    def close(self):
        # C++ side owns the GLFW window; nothing to close here.
        pass


if __name__ == "__main__":
    env = ScenarioEnv({"num_agents": 6, "num_lanes": 3, "render_mode": "human", "respawn_enabled": True})
    env.reset()
    for _ in range(200):
        act = np.zeros((env.num_agents, 2), dtype=np.float32)
        env.step(act)
        env.render()
