#pragma once
#include <string>
#include <vector>

struct RewardConfig {
    float k_prog{20.0f};
    float v_min_ms{1.0f};
    float k_stuck{-0.001f};
    float k_cv{-100.0f};
    float k_cw{-50.0f};    // Crash Wall/Off-road
    float k_cl{-1.0f};    // Crash Line (Yellow line)
    float k_succ{100.0f};
    float k_sm{-0.02f};
    float alpha{0.2f};
};

struct StepResult {
    std::vector<std::vector<float>> obs; // (N,135)
    std::vector<float> rewards;          // (N)
    std::vector<int> done;               // (N)
    std::vector<std::string> status;     // (N)

    // Reward decomposition (C++ side, before Python-side shaping)
    std::vector<float> r_progress;       // progress reward
    std::vector<float> r_stuck;          // low-speed penalty
    std::vector<float> r_smooth;         // action smoothness penalty
    std::vector<float> r_line;           // yellow-line penalty
    std::vector<float> r_crash_vehicle;  // crash with vehicle penalty
    std::vector<float> r_crash_wall;     // crash wall/offroad penalty
    std::vector<float> r_success;        // success bonus
    std::vector<float> r_team_mix;       // delta introduced by C++ team-mix

    // Info parity with Scenario/env.py
    std::vector<long long> agent_ids;    // stable per-agent ids (C++ side)
    int agents_alive{0};

    bool terminated{false};
    bool truncated{false};
    int step{0};
};
