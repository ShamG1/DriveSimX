# MARL 无信号路口环境 🚦

这是一个基于 **C++ (pybind11) + OpenGL/GLFW** 的轻量级多智能体强化学习 (MARL) 无信号路口仿真环境。

项目实现了基于 **运动学自行车模型** 的车辆控制、**贝塞尔曲线** 导航、**线束激光雷达** 感知以及符合工业标准的 **语义化 RL 观测空间**。

---

## 📊 观测空间 (Observation Space)

环境提供 **145 维** 的连续向量作为观测输入：

- **0-3: Ego 状态**：`[x, y, v, heading]` (归一化)
- **4-5: 导航目标**：`[距离目标点, 航向偏差]`
- **6-13: 路面与拓扑特征 (重点)**：
    - `road_edge_dist_L/R`: 左右路边缘距离采样
    - `off_road_flag`: 离路标志
    - `on_line_flag`: 压黄线标志
    - **`signed_cte`**: 相对路径中心线的带符号横向偏差 (左正右负)
    - **`path_heading_err`**: 相对路径切线的局部航向偏差
    - **`in_lane` & `lane_id`**: 车道存在性及 ID 索引
- **14-48: 邻居车辆 (最近 5 个)**：每个邻居 7 维：
    - `[dx, dy, dv, dtheta, intent]` + **`[rel_long, rel_lat]`** (Ego 坐标系下的纵向/横向相对位置)
- **49-144: LiDAR 感知**：96 线激光雷达距离数据

---

## 📸 场景展示

<table>
  <tr>
    <td align="center" width="50%">
      <img src="core/cpp/assets/cross.png" alt="Cross Intersection" width="100%" />
      <br />Intersection
    </td>
    <td align="center" width="50%">
      <img src="core/cpp/assets/T.png" alt="T Intersection" width="100%" />
      <br />T
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="core/cpp/assets/roundabout.png" alt="Roundabout" width="100%" />
      <br />Roundabout
    </td>
    <td align="center" width="50%">
      <img src="core/cpp/assets/highway.png" alt="Highway" width="100%" />
      <br />Highway
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="core/cpp/assets/onramp_merge.png" alt="Onramp Merge" width="100%" />
      <br />Onramp Merge
    </td>
    <td align="center" width="50%">
      <img src="core/cpp/assets/bottleneck.png" alt="Lane Bottleneck" width="100%" />
      <br />Lane Bottleneck
    </td>
  </tr>
</table>

---

## 📂 文件结构

### 核心文件
- `core/env.py`：Python 侧环境封装（对外 API：`ScenarioEnv`），负责参数配置、调用 C++ 后端、组织 obs/reward/info
- `core/cpp_backend.py`：Python ↔ C++ 后端桥接
- `core/cpp/`：C++ 后端源码（pybind11 扩展模块），包含仿真、渲染（OpenGL/GLFW）与传感器/交通流逻辑
- `core/utils.py`：路线映射、lane layout 等辅助
- `scenarios/`：场景资源目录。每个场景文件夹（如 `cross_2lane/`）包含：
  - `drivable.png`：可行驶区域掩码
  - `yellowline.png`：黄线/实线掩码（用于压线碰撞/惩罚）
  - `lane_dashes.png`：虚线渲染图
  - `lane_id.png`：车道 ID 图（用于车道索引/调试）

### 测试文件
- `test.py`：键盘控制测试脚本（会调用 C++ 渲染窗口）

---

## 🚀 快速开始

### 安装

#### 构建依赖

- `CMake >= 3.18`
- 支持 `C++17` 的编译器（Linux: GCC/Clang；Windows: MSVC）
- `OpenGL`
- `GLFW`（Linux 下 CMake 会 `find_package(glfw3 3.3 REQUIRED)`）
- `PyTorch`（用于提供 C++ 侧 `LibTorch`，CMake 会通过 Python 自动定位 Torch 的 CMake 配置）

#### 1. 编译 C++ 后端

```bash
cd core/cpp
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
# Linux
make -j$(nproc)
# Windows (MSVC)
cmake --build . --config Release
```

#### 2. pip 安装本环境 (推荐开发模式)

在项目根目录执行：

```bash
pip install -e .
```

也可以仅安装 Python 依赖：

```bash
pip install -r requirements.txt
```

#### 3. 运行测试

```bash
python test.py
```

---

## 🎮 交互/快捷键

在渲染窗口中可使用键盘快捷键进行交互：

- **V**：切换渲染模式（2D 顶视 / 3D 跟随 / 3D 轨道视角切换）。
- **C**：开启/关闭 **Connections** 连线可视化（显示智能体间的感知关联，为深红色连线）。
- **TAB**：在存活的 Ego 智能体之间**切换视角**（多智能体模式专用）。
- **L**：绑定为开关 LiDAR 可视化。
---

## 🎮 使用方法

安装完成后，你可以在任何地方通过 `drivesimx` 导入并使用环境：

```python
from drivesimx import ScenarioEnv
import numpy as np

# 1. 准备配置
config = {
    'scenario_name': 'cross_2lane',  # 必填：匹配 scenarios/ 下的目录名
    'traffic_flow': True,            # True=单智能体+交通流, False=多智能体
    'traffic_density': 0.5,          # 交通密度
    'traffic_mode': 'stochastic'     # 交通流模式 stochastic为随机模式，constant为固定模式
    'render_mode': 'human',          # 'human' 或 None
    'show_lidar': False,
    'show_lane_ids': False,
    'max_steps': 2000,
}

# 2. 创建环境
env = ScenarioEnv(config)

# 3. 运行循环
obs, info = env.reset()
for _ in range(1000):
    action = np.array([0.5, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
```

---

## ⚙️ 环境配置

### 单智能体模式（带交通流）

```python
config = {
    'scenario_name': 'cross_2lane',
    'traffic_flow': True,  # 启用交通流
    'traffic_density': 0.5,  # 交通密度
    'traffic_mode': 'stochastic', # 交通流模式 
    'render_mode': 'human',
    'max_steps': 2000,
}
```

### 多智能体模式（无交通流）

```python
config = {
    'scenario_name': 'cross_2lane',
    'traffic_flow': False,  # 禁用交通流
    'num_agents': 4,  # 智能体数量
    'use_team_reward': True,  # 是否启用团队奖励混合（默认 False，建议多智能体时按需开启）
    'render_mode': 'human',
    'max_steps': 2000,
}
```

---

## 📈 评测指标 (Metrics)

环境在评测/Benchmark 时可选开启指标统计（默认关闭）。

### 1) 核心指标

我们统计每个 episode 的以下指标：

- **成功率 (Success Rate)**：到达终点的智能体比例
- **碰撞率 (Collision Rate)**：发生碰撞的智能体比例
- **平均到达时间 (Avg Time to Success)**：成功智能体的平均到达时间（秒）

### 2) 状态与事件定义

每步 `env.step()` 返回的 `info` 中包含：

- `info["status"]`: `List[str]`，每个 agent 一个状态，常见值：
  - `"SUCCESS"`：到达终点
  - `"CRASH_CAR"`：与车辆碰撞
  - `"CRASH_WALL"`：撞墙/冲出道路
  - `"ALIVE"`：正常行驶
  - `"ON_LINE"`：压线（非终止，仅惩罚）
- `info["agent_ids"]`: `List[int]`，与 `status` 同索引对齐，用于唯一标识“实际参与过的智能体”

### 3) 变量含义

对单个 episode，定义：

- $\\mathcal{A}$：本 episode 内**实际参与过**的智能体集合
- $N = |\\mathcal{A}|$：实际参与过的智能体数量
- $\\mathcal{S} = \\{a \\in \\mathcal{A} \\mid a\\ \text{曾出现 } status=SUCCESS\\}$：成功到达的智能体集合
- $\\mathcal{C} = \\{a \\in \\mathcal{A} \\mid a\\ \text{曾出现 } status\\in\\{CRASH\\_CAR,CRASH\\_WALL\\}\\}$：发生碰撞的智能体集合
- $t_a$：智能体 $a$ **首次**到达终点（首次 `SUCCESS`）时刻（单位：秒，环境内部用 `dt` 累加）

### 4) 计算公式

- **成功率**：

$
SuccessRate = \\frac{|\\mathcal{S}|}{|\\mathcal{A}|}
$

- **碰撞率**：

$
CollisionRate = \\frac{|\\mathcal{C}|}{|\\mathcal{A}|}
$

- **平均到达时间**（只对成功智能体统计）：

$
AvgTimeToSuccess = \\frac{1}{|\\mathcal{S}|}\\sum_{a\\in\\mathcal{S}} t_a
$

当 $|\\mathcal{S}|=0$ 时，`AvgTimeToSuccess=None`。

### 5) 如何开启与获取

在创建环境时显式开启：

```python
from drivesimx import ScenarioEnv

env = ScenarioEnv({
    "scenario_name": "cross_2lane",
    "traffic_flow": False,
    "num_agents": 6,

    # 推荐：评测阶段关闭重生
    "respawn_enabled": False,

    # 开启 metrics（默认 False）
    "metrics_enabled": True,
})

obs, info = env.reset()

done = False
while not done:
    # ... 你的算法产生 actions ...
    obs, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated

# 最近一局指标
print(env.last_metrics())

# 按 scenario 聚合汇总（可跑多局后再取）
print(env.metrics_summary())
```

## 🎯 奖励函数配置

奖励函数已集成在 `core/env.py` 中，可以通过 `reward_config` 参数自定义：

```python
from core.env import DEFAULT_REWARD_CONFIG

# 1. 使用默认配置
config = {
    'reward_config': DEFAULT_REWARD_CONFIG['reward_config']
}

# 2. 自定义奖励配置
custom_reward_config = {
    'progress_scale': 24.0,              # 前进进度奖励系数（越大越鼓励向目标前进）
    'stuck_speed_threshold': 1.0,        # 判定“卡住”的速度阈值（m/s）
    'stuck_penalty': -0.001,             # 低于卡住阈值时的惩罚
    'crash_vehicle_penalty': -70.0,      # 与其他车辆碰撞惩罚
    'crash_wall_penalty': -30.0,         # 偏离道路/撞墙惩罚
    'crash_line_penalty': -1.0,          # 越过黄线惩罚（比撞墙更轻）
    'success_reward': 70.0,              # 到达目标成功奖励
    'action_smoothness_scale': -0.02,    # 动作平滑项系数（抑制突变控制）
    'team_alpha': 0.2,                   # 团队奖励混合权重（个体/团队折中）
}

config = {
    'reward_config': custom_reward_config
}
```

### 奖励组成
**1）基础个体奖励**：
```
r_i^ind(t) = r_prog(t) + r_stuck(t) + r_crashV(t) + 
             r_crashW(t) + r_crashL(t) + r_succ(t) + r_smooth(t)
```

其中各项对应配置键：
- `r_prog`  ↔ `progress_scale`：前进进度奖励缩放
- `r_stuck` ↔ `stuck_speed_threshold` + `stuck_penalty`：低速卡住惩罚
- `r_crashV` ↔ `crash_vehicle_penalty`：车辆碰撞惩罚
- `r_crashW` ↔ `crash_wall_penalty`：撞墙/离开道路惩罚
- `r_crashL` ↔ `crash_line_penalty`：压线惩罚
- `r_succ` ↔ `success_reward`：到达目标奖励
- `r_smooth` ↔ `action_smoothness_scale`：动作平滑项

**2）团队奖励混合（可选）**：
当 `use_team_reward=True` 且为多智能体模式时：
```
r_i^mix(t) = (1 - α) * r_i^ind(t) + α * r̄^ind(t)
```
其中 `α` 对应 `team_alpha`。

**3）环境附加惩罚**：
- `max_steps_penalty_no_respawn`：无重生且因步数上限截断时惩罚
- `respawn_penalty`：启用重生时，碰撞后重生惩罚
- `no_progress_penalty`：窗口内进度不足惩罚

---

## 🚗 交通流设置 (Traffic Flow)

环境支持两种交通流模式，通过 `traffic_mode` 参数配置：

### 1. 随机模式 (`stochastic`) - 默认
- **行为**：基于 `traffic_density` (到达率) 随机生成 NPC。NPC 到达目的地或发生碰撞后会被移除（`erase`）。
- **适用场景**：常规强化学习训练，追求高随机性和真实交通分布。

### 2. 恒定模式 (`constant`)
- **行为**：根据 `traffic_density` 和 `traffic_kmax` 确定固定数量的 NPC 槽位（$K = \text{round}(\text{density} \times \text{kmax})$）。
- **关键特性**：
    - **长度恒定**：NPC 死亡后不会被移除，而是标记为 `alive=false` 并传送到屏外，保证 `traffic_cars` 数组长度不变。
    - **可冻结性**：支持 `env.freeze_traffic(True)`，冻结后死亡槽位不再补齐，确保搜索过程的确定性。
- **适用场景**：MCTS 规划、确定性状态回滚（Snapshot）、追求稳定交通压力的训练。

### 配置示例

```python
config = {
    'traffic_flow': True,
    'traffic_mode': 'constant',  
    'traffic_density': 0.5,      
    'traffic_kmax': 20,     
}     
```

### NPC 车辆行为

NPC 车辆通过 C++ 后端驱动：
- **横向控制**：基于路径的 Pure Pursuit 增强型航向跟踪。
- **纵向控制**：具备自动避障的巡航控制。
- **生命周期**：
    - `stochastic`：移除并释放内存。
    - `constant`：重置状态并等待补齐（Refill）。

---

## 📝 TODO

- [x] 集成奖励计算到环境
- [x] 集成交通流生成到环境
- [x] 支持单智能体和多智能体模式
- [x] 支持多地图测试

---

## 📄 许可证

本项目遵循 MIT License。
