# benchmark_metrics_demo.py
import numpy as np
from core.env import ScenarioEnv

def run_benchmark(
    scenario_name: str = "roundabout_3lane",
    num_agents: int = 6,
    episodes: int = 20,
    max_steps: int = 2000,
    render: bool = False,
):
    env = ScenarioEnv({
        "scenario_name": scenario_name,
        "traffic_flow": False,
        "num_agents": num_agents,
        "render_mode": "human" if render else None,
        "show_lidar": False,
        "show_lane_ids": False,

        # 评测建议关闭重生，否则 episode 可能主要靠 truncated 结束
        "respawn_enabled": False,
        "max_steps": max_steps,

        # 显式开启 metrics（默认是 False）
        "metrics_enabled": True,
    })

    # 确保从干净的统计开始：清空历史后直接进入 episode 循环。
    # 注意：ScenarioEnv.__init__ 内部会调用一次 reset()；这里不再额外 reset，避免引入多余 episode 统计。
    env.reset_metrics_history()

    for ep in range(1, episodes + 1):
        obs, _info = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Route-following baseline policy
            heading_err = obs[:, 5].astype(np.float32)
            signed_cte = obs[:, 10].astype(np.float32)

            base_throttle = 0.35
            throttle = np.clip(base_throttle - 0.15 * np.abs(heading_err), 0.10, 0.45)

            k_heading = 1.4
            k_cte = 0.6
            steer = np.clip(k_heading * heading_err + k_cte * signed_cte, -1.0, 1.0)

            actions = np.zeros((env.num_agents, 2), dtype=np.float32)
            actions[:, 0] = throttle
            actions[:, 1] = steer

            obs, _rewards, terminated, truncated, _info = env.step(actions)
            steps += 1
            done = terminated or truncated

            if render:
                env.render()

        # 精简输出
        m = env.last_metrics() or {}
        reason = m.get('end_reason', 'unknown')
        sr = m.get('success_rate', 0.0)
        cr = m.get('collision_rate', 0.0)
        t = m.get('avg_time_to_success')
        t_str = f"{t:.2f}s" if t is not None else "N/A"
        
        print(f"[EP {ep:03d}] steps={steps:4d} | reason={reason:10s} | success={sr:.2f} | collision={cr:.2f} | avg_time={t_str}")

    print("\n" + "="*30 + " SUMMARY " + "="*30)
    summary = env.metrics_summary(scenario_name).get(scenario_name, {})
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.4f}")
        else:
            print(f"{k:25s}: {v}")
    print("="*69)

    env.close()


if __name__ == "__main__":
    run_benchmark(
        scenario_name="roundabout_3lane",
        num_agents=3,
        episodes=5,
        max_steps=2000,
        render=True,  # 想看画面就改 True
    )