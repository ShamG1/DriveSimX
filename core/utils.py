# This file is a local copy of necessary components from the 'Scenario' package
# to make C_MCTS a self-contained module.
# === From Scenario/config.py ===
WIDTH, HEIGHT = 1000, 1000
SCALE = 12
LANE_WIDTH_M = 3.5
LANE_WIDTH_PX = int(LANE_WIDTH_M * SCALE)



OBS_DIM = 145

# 默认奖励配置（可通过 config 字典覆盖）
DEFAULT_REWARD_CONFIG = {
    'use_team_reward': False,  # 是否启用团队奖励混合（多智能体可选）
    'traffic_flow': False,      # 若为 True，则为交通流单智能体模式，使用个体奖励
    # 仅在 respawn_enabled=False 且因 max_steps_per_episode 截断时生效。
    'max_steps_penalty_no_respawn': -5.0,
    # 在 respawn_enabled=True 时，车辆碰撞后重生施加的惩罚。
    'respawn_penalty': -0.5,
    # 在窗口内进度增量低于阈值时施加的惩罚。
    'no_progress_penalty': -0.2,
    'reward_config': {
        'progress_scale': 60.0,          # 前进进度奖励缩放系数
        'stuck_speed_threshold': 1.0,    # 卡住判定速度阈值（m/s）
        'stuck_penalty': -0.08,          # 卡住惩罚
        'crash_vehicle_penalty': -30,    # 车辆碰撞惩罚
        'crash_wall_penalty': -8.0,      # 撞墙/离开道路惩罚
        'crash_line_penalty': -0.5,      # 压线惩罚（通常轻于撞墙）
        'success_reward': 100.0,         # 到达目标奖励
        'action_smoothness_scale': -0.005,  # 动作平滑项系数
        'team_alpha': 0.2,               # 团队奖励混合系数
    }
}
# === From Scenario/env.py ===
ROUTE_MAP_BY_SCENARIO = {
    "cross_2lane": {
        "straight": {2: 6, 4: 8, 6: 2, 8: 4},
        "left": {1: 3, 3: 5, 5: 7, 7: 1},
    },
    "cross_3lane": {
        "straight": {2: 8, 5: 11, 8: 2, 11: 5},
        "right": {3: 12, 6: 3, 9: 6, 12: 9},
        "left": {1: 4, 4: 7, 7: 10, 10: 1},
    },
    "T_2lane": {
        "straight": {2: 6, 5: 1},
        "right": {4: 2, 6: 4},
        "left": {1: 3, 3: 5},
    },
    "T_3lane": {
        "straight": {2: 8, 3: 9, 7: 1, 8: 2},
        "left": {1: 4, 4: 7, 5: 8},
        "right": {6: 3, 9: 6, 5: 2},
    },
    "highway_2lane": {
        "straight": {1: 1, 2: 2},
    },
    "highway_4lane": {
        "straight": {1: 1, 2: 2, 3: 3, 4: 4},
    },
    "roundabout_2lane": {
        "straight": {2: 6, 4: 8, 6: 2, 8: 4},
        "left": {1: 3, 3: 5, 5: 7, 7: 1},
    },
    "roundabout_3lane": {
        "straight": {2: 8, 5: 11, 8: 2, 11: 5},
        "right": {3: 12, 6: 3, 9: 6, 12: 9},
        "left": {1: 4, 4: 7, 7: 10, 10: 1},
    },
    "onrampmerge_3lane": {
        "straight": {1: 1, 2: 2},
        "ramp": {"IN_RAMP_1": "OUT_2"},
    },
    "bottleneck": {
        "straight": {1: 1, 2: 2, 3: 3},
    },
}



# === From Scenario/agent.py ===
def build_lane_layout(num_lanes: int):
    dir_order = ['N', 'E', 'S', 'W']
    points = {}
    in_by_dir = {d: [] for d in dir_order}
    out_by_dir = {d: [] for d in dir_order}
    dir_of = {}
    idx_of = {}
    MARGIN = 30
    CX, CY = WIDTH // 2, HEIGHT // 2

    for d_idx, d in enumerate(dir_order):
        for j in range(num_lanes):
            offset = LANE_WIDTH_PX * (0.5 + j)
            in_name = f"IN_{d_idx * num_lanes + j + 1}"
            out_name = f"OUT_{d_idx * num_lanes + j + 1}"

            if d == 'N':
                points[in_name] = (CX - offset, MARGIN)
                points[out_name] = (CX + offset, MARGIN)
            elif d == 'S':
                points[in_name] = (CX + offset, HEIGHT - MARGIN)
                points[out_name] = (CX - offset, HEIGHT - MARGIN)
            elif d == 'E':
                points[in_name] = (WIDTH - MARGIN, CY - offset)
                points[out_name] = (WIDTH - MARGIN, CY + offset)
            else:  # 'W'
                points[in_name] = (MARGIN, CY + offset)
                points[out_name] = (MARGIN, CY - offset)

            in_by_dir[d].append(in_name)
            out_by_dir[d].append(out_name)
            dir_of[in_name] = d
            dir_of[out_name] = d
            idx_of[in_name] = j
            idx_of[out_name] = j

    return {
        'points': points,
        'in_by_dir': in_by_dir,
        'out_by_dir': out_by_dir,
        'dir_of': dir_of,
        'idx_of': idx_of,
        'dir_order': dir_order,
    }
