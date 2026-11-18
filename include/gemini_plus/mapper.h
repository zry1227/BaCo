#ifndef NETTLM_DSE_GEMINI_PLUS_MAPPER_H
#define NETTLM_DSE_GEMINI_PLUS_MAPPER_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

enum class OpType { GEMM, ELEMENT_WISE, TRANSPOSE, MERGE, ATTENTION, ERRTYPE };
enum class FLOW_CTRL { CT, WH };
enum class NOC_DIR { L, N, E, S, W };

using coord_t = std::pair<int, int>;
using noc_pos_t = std::pair<coord_t, int>;
using link_t = std::pair<noc_pos_t, noc_pos_t>;
using flow_idx_t = std::pair<int, std::pair<int, int>>;
using part_t = std::array<int, 4>;

struct TensorDim {
    int n_r = 1;
    int rows = 0;
    int n_c = 1;
    int cols = 0;
};

struct DataPartition {
    int dpathID = -1;
    int num_row = 1;
    int rowStart = 0;
    int rowEnd = -1;
    int num_col = 1;
    int colStart = 0;
    int colEnd = -1;
};

struct DataSegment {
    std::string dataType;
    std::string op_name;
    DataPartition partition;
};

struct DRAMData {
    int dramID = -1;
    std::string type;
    std::vector<DataSegment> segments;
};

struct Operation {
    std::string name;
    OpType type = OpType::ERRTYPE;
    bool hasWeight = false;
    int pipe_depth = 0;
    TensorDim ifmDim;
    TensorDim wgtDim;
    TensorDim ofmDim;
    int ifmDRAM = -1;
    int wgtDRAM = -1;
    int ofmDRAM = -1;
    std::vector<DataPartition> ifmPartitions;
    std::vector<DataPartition> wgtPartitions;
    std::vector<DataPartition> ofmPartitions;
    std::vector<std::string> inputOps;
};

struct Connection {
    std::string srcOp;
    std::string dstOp;
    bool fromDRAM = false;
    bool toDRAM = false;
    int dramID = -1;
    std::string note;

    Connection() = default;
    Connection(const std::string& src, const std::string& dst,
               bool from_dram, bool to_dram, int dram_id,
               const std::string& description)
        : srcOp(src), dstOp(dst), fromDRAM(from_dram), toDRAM(to_dram),
          dramID(dram_id), note(description) {}
};

struct _workload {
    unsigned int h = 0;
    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int k = 0;
};

class DataFlow {
public:
    std::vector<Operation> ops;
    std::vector<Connection> conns;

    std::map<link_t, unsigned long long> hop_cnt;
    std::map<link_t, std::set<int>> src_cnt;
    std::map<link_t, std::map<int, unsigned long long>> from_src;
    std::map<link_t, bool> is_mcast_link;

    std::map<flow_idx_t, std::map<link_t, double>> E_f_s_i;
    std::map<flow_idx_t, std::map<link_t, double>> eff_E_f_s_i;
    std::map<link_t, std::set<flow_idx_t>> tbl_f;
    std::map<flow_idx_t, FLOW_CTRL> mode;
    std::map<flow_idx_t, std::pair<std::vector<int>, unsigned long long>> f_s_i;
    std::map<link_t, double> tbl_bw;
    std::map<link_t, std::set<flow_idx_t>> tbl_ct;
    std::map<link_t, std::set<flow_idx_t>> eff_tbl_ct;
    std::map<link_t, std::set<flow_idx_t>> tbl_wh;
    std::map<link_t, std::set<flow_idx_t>> eff_tbl_wh;
    std::set<flow_idx_t> f_wh_only;

    unsigned long long max_comp_dly = 0;
    unsigned int num_max_mcast_port = 0;

    DataFlow();
    void addOp(const Operation& op);
    void addConn(const Connection& connection);
    void calc_unicast(int src, int dst, int size, int root);
    void calc_multicast_loop(int src, std::vector<int>& destinations,
                             int size, int root, int flow_idx, int interval_idx);
    void calc_multicast_recursive(int src, std::vector<int>& destinations,
                                  int size, int root, int flow_idx);
    unsigned int getIfmSize(int dpathID);
    unsigned int getWgtSize(int dpathID);
    unsigned int getOfmSize(int dpathID);
    DataPartition getOfmPart(int dpathID);
    OpType getOpType(int dpathID);
    std::string getOpName(int dpathID);
    unsigned int getOpSize(int dpathID);
    _workload getWorkload(int dpathID);
    unsigned int getPipeDepth();
    void calc_NoCTraffic_recursive(const std::vector<DRAMData>& drams, int pass);
    void calc_NoCTraffic_loop(const std::vector<DRAMData>& drams, int pass);
    void calc_CompDly(int pass);
    int getDRAMID(const std::vector<DRAMData>& drams,
                  const std::string& op_name, const std::string& dtype);
    void print();
    bool isNoPlink(link_t link);
};

noc_pos_t rid2pos(int rid, int level = 0);
int cmpntid2rid(int component_id);
int dpathid2rid(int dpath_id);
void autoGeneratePartitions(Operation& op);
unsigned int getPartSize(part_t part);
NOC_DIR get_dir(link_t link);
int get_num_hop(noc_pos_t src, noc_pos_t dst);

#define PRINT_POS(...) do {} while (false)
#define PRINT_FLOW(...) do {} while (false)
#define PRINT_LINK(...) do {} while (false)

#endif
