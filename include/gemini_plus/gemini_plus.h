#ifndef NETTLM_DSE_GEMINI_PLUS_GEMINI_PLUS_H
#define NETTLM_DSE_GEMINI_PLUS_GEMINI_PLUS_H

#include <array>
#include <bitset>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "gemini_plus/mapper.h"

class gemini_plus
{
public:
    static constexpr int NUM_OPS = 20;

    gemini_plus();
    ~gemini_plus();

    void init();
    unsigned long long get_perf_recursive();
    unsigned long long get_perf_loop();
    unsigned long long calc_effBW();
    std::pair<unsigned long long, unsigned long long> get_perf_both();
    unsigned long long calc_upper();

    DataFlow *df = nullptr;
    std::vector<std::string> op_names;
    std::array<OpType, NUM_OPS> op_type{};
    std::array<int, NUM_OPS> pipe_depth{};
    std::array<TensorDim, NUM_OPS> ifm_dim{};
    std::array<TensorDim, NUM_OPS> wgt_dim{};
    std::array<TensorDim, NUM_OPS> ofm_dim{};
    std::array<std::vector<std::string>, NUM_OPS> input_op{};
    std::array<std::pair<int, int>, NUM_OPS> part{};
    std::bitset<NUM_OPS> has_Wgt;

    // These mappings are supplied by the spatial-placement search before init().
    std::map<int, std::vector<int>> assignedDpath;
    std::map<int, std::vector<int>> data_dram;
    std::vector<DRAMData> allDRAMs;
};

#endif
