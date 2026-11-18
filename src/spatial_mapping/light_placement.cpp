#include "spatial_mapping/light_placement.h"
#include <ctime>
#include "cluster.h"
#include "schnode.h"
#include "layerengine.h"
#include "noc.h"
#include <cassert>
#include <random>
#include <stdexcept>
#include "util.h"
#include "log.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>



FastLayerEngine *Light_placement::fast_buffer = nullptr;
double Light_placement::dram_balance_threshold = 0.10;

void Light_placement::set_dram_balance_threshold(double threshold)
{
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0)
        throw std::invalid_argument("CAO1 DRAM balance threshold must be in [0, 1]");
    dram_balance_threshold = threshold;
}

double Light_placement::get_dram_balance_threshold()
{
    return dram_balance_threshold;
}

Light_partition::Light_partition()
{
}

void Light_partition::init(const fmap_shape &shape, len_t _batch, vol_t _layerno)
{
    PartSch partSch;
    batch = _batch;
    auto partIter = partEngine.init(siz, batch, network->getNode(_layerno), partSch, 0);
    allpartition.clear();
    do
    {
        if (partSch.B <= batch && partSch.K <= shape.c && partSch.H <= shape.h && partSch.W <= shape.w)
        {
            std::vector<len_t> fac = {partSch.B, partSch.K, partSch.H, partSch.W};
            allpartition.push_back(fac);
        }
    } while (partIter.nextPart());
    assert(!allpartition.empty());
}

void Light_partition::re_init(const fmap_shape &shape, vol_t size, PartSch part)
{
    PartSch partSch;
    siz = size;
    b = part.B;
    c = part.K;
    h = part.H;
    w = part.W;
    auto partIter = partEngine.init(siz, batch, network->getNode(layerno), partSch, 0);
    allpartition.clear();
    do
    {
        if (partSch.B <= batch && partSch.K <= shape.c && partSch.H <= shape.h && partSch.W <= shape.w)
        {
            std::vector<len_t> fac = {partSch.B, partSch.K, partSch.H, partSch.W};
            allpartition.push_back(fac);
        }
    } while (partIter.nextPart());
    assert(!allpartition.empty());
}

Light_partition::Light_partition(const fmap_shape &shape, cidx_t _siz, len_t _batch, vol_t _layerno)
{
    PartSch partSch;
    siz = _siz;
    layerno = _layerno;
    batch = _batch;
    auto partIter = partEngine.init(siz, batch, network->getNode(_layerno), partSch, 0);
    do
    {
        if (partSch.B <= batch && partSch.K <= shape.c && partSch.H <= shape.h && partSch.W <= shape.w)
        {
            if (_layerno == 21 || _layerno == 20)
            {
                if (partSch.K == 1)
                {
                    std::vector<len_t> fac = {partSch.B, partSch.K, partSch.H, partSch.W};
                    allpartition.push_back(fac);
                }
            }
            else
            {
                // int num1 = 0;
                // if(partSch.B == 1)
                //     num1++;
                // if(partSch.K == 1)
                //     num1++;
                // if(partSch.H == 1)
                //     num1++;
                // if(partSch.W == 1)
                //     num1++;
                // if(num1 >= 3)
                // {
                //     std::vector<len_t> fac={partSch.B, partSch.K, partSch.H, partSch.W};
                //     allpartition.push_back(fac);
                // }
                std::vector<len_t> fac = {partSch.B, partSch.K, partSch.H, partSch.W};
                allpartition.push_back(fac);
            }
        }
    } while (partIter.nextPart());

    if (!allpartition.empty())
        change();
}

Light_partition::Light_partition(vol_t _b, vol_t _c, vol_t _h, vol_t _w, vol_t _siz, vol_t _layerno) : b(_b), c(_c), h(_h), w(_w), siz(_siz), layerno(_layerno)
{
}

Light_placement::Layer_partition::Layer_partition()
    : layerno(static_cast<lid_t>(-1)), partno(static_cast<cidx_t>(-1))
{
}

Light_placement::Layer_partition::Layer_partition(lid_t _layerno, cidx_t _partno) : layerno(_layerno), partno(_partno) {}

Light_placement::Light_placement() {}

/*Light_placement::Light_placement(SchNode* segment){
    assert(segment->get_type()==SchNode::NodeType::S);
    auto pt = dynamic_cast<SCut*>(segment);
}*/

Light_placement::Light_placement(LTreeNode *_node)
{
    node = _node;
    const auto &cnodes = node->get_children();
    layer_num = 0;
    cidx_t cnum = static_cast<cidx_t>(cnodes.size());
    assert(cnum > 0);
    utime_t *tlist = new utime_t[cnum];
    utime_t *cur_item = tlist;
    for (auto child : cnodes)
    {
        *(cur_item++) = child->get_utime();
    }
    Cluster cluster(0, Cluster::xlen * Cluster::ylen);
    auto allocRes = cluster.try_alloc(tlist, cnum);
    delete[] tlist;

    if (!allocRes) {
        throw std::runtime_error("Cannot allocate at least one core to every layer in the segment");
    }

    int allocidx = 0;
    layers = node->layers();
    FOR_BITSET(layerno, node->layers())
    {
        for (int j = 0; j < allocRes[allocidx + 1] - allocRes[allocidx]; ++j)
        {
            placement.emplace_back(layerno, j);
            layer_scheme[layerno].insert(allocRes[allocidx] + j);
        }
        partition.emplace_back(network->getNode(layerno).layer().ofmap_shape(), allocRes[allocidx + 1] - allocRes[allocidx], node->get_bgrp_size(), layerno);
        ++allocidx;
        ++layer_num;
    }
}

const std::vector<Light_partition> &Light_placement::get_partition() const
{
    return partition;
}

const std::unordered_map<lid_t, std::unordered_set<cidx_t>> &Light_placement::get_layer_scheme() const
{
    return layer_scheme;
}
const std::unordered_map<lid_t, FetchSch> &Light_placement::get_fetch_scheme() const
{
    return fetch_scheme;
}

const std::vector<Light_placement::Layer_partition> &Light_placement::get_placement() const
{
    return placement;
}

bool Light_placement::swap(cidx_t x, cidx_t y)
{
    bool isSameLayer = false;
    // LOG2("SWAP" << x << " <-> " << y);
    if (placement[x].layerno != placement[y].layerno)
    {
        // std::cout << "OP3: " << x << "<->" << y << std::endl;
        layer_scheme[placement[x].layerno].erase(x);
        layer_scheme[placement[x].layerno].insert(y);
        layer_scheme[placement[y].layerno].erase(y);
        layer_scheme[placement[y].layerno].insert(x);
    }
    else
    {
        isSameLayer = true;
        // std::cout << "OP2" << x << "<->" << y << std::endl;
    }

    std::swap(placement[x], placement[y]);

    return isSameLayer;
}

void Light_partition::change()
{
    static std::mt19937_64 gen(233);
    int ch = gen() % allpartition.size();
    vol_t temp_b = b;
    vol_t temp_c = c;
    vol_t temp_h = h;
    vol_t temp_w = w;
    if (layerno == 20 || layerno == 21)
    {
        while (allpartition[ch][1] != 1)
        {
            ch = gen() % allpartition.size();
        }
    }
    // else
    // {
    //     int num1 = 0;
    //     do
    //     {
    //         num1 = 0;
    //         for(int i = 0; i < 4; i++)
    //         {
    //             if(allpartition[ch][i] == 1)
    //                 ++num1;
    //         }
    //         ch = gen() % allpartition.size();
    //     } while(num1 <= 2);
    // }
    b = allpartition[ch][0];
    c = allpartition[ch][1];
    h = allpartition[ch][2];
    w = allpartition[ch][3];
}
// void Light_placement::change_core(int cnt)
// {
//     if (cnt == 16)
//     {
//         return; // exit
//     }
//     srand(time(NULL));
//     static std::mt19937_64 gen(rand());
//     cidx_t layer1, layer2, layer1_size, layer2_size;
//     cidx_t temp_layer1, temp_layer2;
//     int cnt1 = 0;
//     do
//     {
//         temp_layer1 = gen() % layer_num;
//         temp_layer2 = gen() % layer_num;
//         layer1 = partition[temp_layer1].layerno;
//         layer2 = partition[temp_layer2].layerno;
//         cnt1++;
//         if (cnt1 == 16)
//         {
//             mutate(Cluster::xlen * Cluster::ylen, 0, 0, 0, 0);
//             return;
//         }
//     } while (layer1 == layer2 || layer_scheme[layer1].size() == 1);

//     // layer1_size = layer_scheme[layer1].size();//core number
//     // layer2_size = layer_scheme[layer2].size();
//     // std::unordered_set<cidx_t>::iterator iter1 = layer_scheme[layer1].begin();
//     // cidx_t temp1 = *iter1;
//     // cidx_t rand_core_id = gen() % layer1_size;
//     // for (cidx_t i = 0; i < rand_core_id; ++i) {
//     //     temp1 = *(++iter1);
//     // }
//     // layer1_size--;
//     // layer2_size++;

//     // Cluster c1(0, layer1_size);
//     // Cluster c2(0, layer2_size);
//     // PartSch part1 = fast_buffer->fast_get_part(c1, network->getNode(layer1), partition[temp_layer1].batch);
//     // if (part1.size() == 0) {
//     //     change_core(++cnt);
//     //     return;
//     // }
//     // PartSch part2 = fast_buffer->fast_get_part(c2, network->getNode(layer2), partition[temp_layer2].batch);
//     // if (part2.size() == 0) {
//     //     change_core(++cnt);
//     //     return;
//     // }
//     // layer_scheme[layer1].erase(temp1);
//     // placement[temp1].layerno = layer2;
//     // partition[temp_layer1].re_init(network->getNode(layer1).layer().ofmap_shape(), layer1_size, part1);
//     // partition[temp_layer2].re_init(network->getNode(layer2).layer().ofmap_shape(), layer2_size, part2);
//     // for (auto core_id : layer_scheme[layer1]) {
//     //     if (placement[core_id].partno > placement[temp1].partno) {
//     //         placement[core_id].partno--;
//     //     }
//     // }
//     // cidx_t partnp_temp = gen() % layer2_size;
//     // for (auto core_id : layer_scheme[layer2]) {
//     //     if (placement[core_id].partno >= partnp_temp) {
//     //         placement[core_id].partno++;
//     //     }
//     // }
//     // layer_scheme[layer2].insert(temp1);
//     // placement[temp1].partno = partnp_temp;
// }

bool Light_placement::change_core()
{
    if (fast_buffer == nullptr)
        return false;

    if (partition.size() < 2 || layer_scheme.size() < 2)
        return false;

    struct TransferCandidate {
        size_t donor_index = 0;
        size_t receiver_index = 0;

        lid_t donor_layer = 0;
        lid_t receiver_layer = 0;

        cidx_t donor_cores = 0;
        cidx_t receiver_cores = 0;

        PartSch donor_part;
        PartSch receiver_part;

        double predicted_max_time =
            std::numeric_limits<double>::infinity();

        bool valid = false;
    };

    auto estimated_time =
        [this](const Light_partition &layer_partition,
               cidx_t core_count) -> double {
            if (core_count <= 0)
                return std::numeric_limits<double>::infinity();

            const lid_t layer_id = layer_partition.layerno;

            const double operations = static_cast<double>(
                network->getNode(layer_id).layer().get_utime());

            const double batch =
                static_cast<double>(layer_partition.batch);

            return operations * batch /
                   static_cast<double>(core_count);
        };

    // 计算修改前整个 segment 中最慢 layer 的时间。
    double current_max_time = 0.0;

    for (const Light_partition &layer_partition : partition) {
        const auto scheme_it =
            layer_scheme.find(layer_partition.layerno);

        if (scheme_it == layer_scheme.end() ||
            scheme_it->second.empty()) {
            return false;
        }

        const cidx_t core_count =
            static_cast<cidx_t>(scheme_it->second.size());

        current_max_time = std::max(
            current_max_time,
            estimated_time(layer_partition, core_count));
    }

    TransferCandidate best;

    // 枚举所有 donor -> receiver 单核转移。
    for (size_t donor_index = 0;
         donor_index < partition.size();
         ++donor_index) {
        const Light_partition &donor =
            partition[donor_index];

        const lid_t donor_layer = donor.layerno;

        const auto donor_scheme_it =
            layer_scheme.find(donor_layer);

        if (donor_scheme_it == layer_scheme.end())
            continue;

        const cidx_t donor_cores =
            static_cast<cidx_t>(
                donor_scheme_it->second.size());

        // 必须为 donor 至少保留一个核。
        if (donor_cores <= 1)
            continue;

        for (size_t receiver_index = 0;
             receiver_index < partition.size();
             ++receiver_index) {
            if (receiver_index == donor_index)
                continue;

            const Light_partition &receiver =
                partition[receiver_index];

            const lid_t receiver_layer =
                receiver.layerno;

            const auto receiver_scheme_it =
                layer_scheme.find(receiver_layer);

            if (receiver_scheme_it == layer_scheme.end())
                continue;

            const cidx_t receiver_cores =
                static_cast<cidx_t>(
                    receiver_scheme_it->second.size());

            const cidx_t new_donor_cores =
                donor_cores - 1;

            const cidx_t new_receiver_cores =
                receiver_cores + 1;

            // 检查新核数下是否存在合法的 partition。
            Cluster donor_cluster(0, new_donor_cores);
            Cluster receiver_cluster(0, new_receiver_cores);

            PartSch donor_part =
                fast_buffer->fast_get_part(
                    donor_cluster,
                    network->getNode(donor_layer),
                    donor.batch);

            if (donor_part.size() == 0)
                continue;

            PartSch receiver_part =
                fast_buffer->fast_get_part(
                    receiver_cluster,
                    network->getNode(receiver_layer),
                    receiver.batch);

            if (receiver_part.size() == 0)
                continue;

            // 预测本次核转移后整个 segment 的最大层时间。
            double candidate_max_time = 0.0;

            for (size_t i = 0;
                 i < partition.size();
                 ++i) {
                const Light_partition &current =
                    partition[i];

                const auto current_scheme_it =
                    layer_scheme.find(current.layerno);

                if (current_scheme_it ==
                    layer_scheme.end()) {
                    candidate_max_time =
                        std::numeric_limits<double>::infinity();
                    break;
                }

                cidx_t candidate_core_count =
                    static_cast<cidx_t>(
                        current_scheme_it->second.size());

                if (i == donor_index)
                    candidate_core_count--;
                else if (i == receiver_index)
                    candidate_core_count++;

                candidate_max_time = std::max(
                    candidate_max_time,
                    estimated_time(
                        current,
                        candidate_core_count));
            }

            // 只记录能够降低最大计算时间的方案。
            if (candidate_max_time + 1e-9 >=
                current_max_time) {
                continue;
            }

            if (!best.valid ||
                candidate_max_time <
                    best.predicted_max_time) {
                best.valid = true;

                best.donor_index = donor_index;
                best.receiver_index = receiver_index;

                best.donor_layer = donor_layer;
                best.receiver_layer = receiver_layer;

                best.donor_cores = donor_cores;
                best.receiver_cores = receiver_cores;

                best.donor_part = donor_part;
                best.receiver_part = receiver_part;

                best.predicted_max_time =
                    candidate_max_time;
            }
        }
    }

    // 没有找到预计能够改善计算瓶颈的核转移。
    if (!best.valid)
        return false;

    /*
     * 从 donor 中选择要转移的物理核。
     *
     * 优先选择距离 receiver 当前核集合最近的 donor 核，
     * 避免转移后 receiver 的核过于分散。
     */
    const auto &donor_set =
        layer_scheme.at(best.donor_layer);

    const auto &receiver_set =
        layer_scheme.at(best.receiver_layer);

    cidx_t moved_core = -1;
    cidx_t best_distance =
        std::numeric_limits<cidx_t>::max();

    for (const cidx_t donor_core : donor_set) {
        cidx_t nearest_distance =
            std::numeric_limits<cidx_t>::max();

        for (const cidx_t receiver_core :
             receiver_set) {
            nearest_distance = std::min(
                nearest_distance,
                dis(Cluster::get_pos(donor_core),
                    Cluster::get_pos(receiver_core)));
        }

        if (moved_core < 0 ||
            nearest_distance < best_distance) {
            moved_core = donor_core;
            best_distance = nearest_distance;
        }
    }

    if (moved_core < 0)
        return false;

    /*
     * 记录 moved_core 原来在 donor 中的 partition 编号。
     * 删除后，需要压缩 donor 其余核的编号。
     */
    const cidx_t removed_partno =
        placement[moved_core].partno;

    layer_scheme[best.donor_layer].erase(
        moved_core);

    for (const cidx_t core_id :
         layer_scheme[best.donor_layer]) {
        if (placement[core_id].partno >
            removed_partno) {
            placement[core_id].partno--;
        }
    }

    /*
     * 将 moved_core 添加到 receiver。
     *
     * 这里直接把它作为 receiver 的最后一个 partition，
     * 可避免插入中间位置后对所有编号进行移位。
     */
    const cidx_t new_receiver_partno =
        static_cast<cidx_t>(
            layer_scheme[best.receiver_layer].size());

    placement[moved_core].layerno =
        best.receiver_layer;

    placement[moved_core].partno =
        new_receiver_partno;

    layer_scheme[best.receiver_layer].insert(
        moved_core);

    // 根据新的核数量重新初始化两层的 partition。
    partition[best.donor_index].re_init(
        network->getNode(best.donor_layer)
            .layer()
            .ofmap_shape(),
        best.donor_cores - 1,
        best.donor_part);

    partition[best.receiver_index].re_init(
        network->getNode(best.receiver_layer)
            .layer()
            .ofmap_shape(),
        best.receiver_cores + 1,
        best.receiver_part);

    return true;
}

void Light_placement::change_DRAM()
{
    // srand(time(NULL));
    static std::mt19937_64 gen(rand_seed);
    cidx_t layer1;
    cidx_t temp_layer1;
    mlen_t new_ddr = -1;
    int data = -1;
    if (partition.empty() || NoC::DRAM_num <= 0)
        return;

    int attempts = 0;
    do
    {
        temp_layer1 = gen() % layer_num;
        layer1 = partition[temp_layer1].layerno;
        auto &dram = layer_DRAM[layer1];
        if (dram.size() < 3)
            dram.resize(3, -1);
        data = gen() % 3;
        // data = gen() % 2;
        // new_ddr = gen() % (NoC::DRAM_num+1);  //
        new_ddr = gen() % NoC::DRAM_num;
        if (new_ddr == NoC::DRAM_num)
        {
            new_ddr = -2; // -2: all dram
        }
    } while (++attempts < 128 &&
             (new_ddr == layer_DRAM[layer1][data] || layer_DRAM[layer1][data] == -1));
    if (attempts >= 128)
        return;
    layer_DRAM[layer1][data] = new_ddr;
}

int Light_placement::mutate(cidx_t dist, int min_idx, int max_idx, int delta,
                            int last_op, const NoC* noc_feedback)
{
    if (placement.size() < 2)
        return 0;

    static std::mt19937_64 gen(rand_seed);
    int type = gen() % 7;

    int op;
    
    // if(type < 3)
    // {
    //     op = 6;
    // }
    // else if (type < 6)
    if (type < 1)
    {
        op = 3;
    }
    else if(type < 2)
    {
        op = 2;
    }
    else if(type < 3)
    {
        // change_DRAM();
        op = 5;
    }
    else if(type < 4)
    {
        op = 7;
    }
    else if(type < 5)
    {
        op = 8;
    }
    else
    {
        op = 9;
    }

    switch (op)
    {
    case 2:
    {
        // std::cout << "OP2" << std::endl;
        cidx_t x = 0, y = 0;
        bool found = false;
        for (int attempt = 0; attempt < 512; ++attempt)
        {
            x = gen() % placement.size();
            y = gen() % placement.size();
            if (x != y && dis(Cluster::get_pos(x), Cluster::get_pos(y)) <= dist &&
                placement[x].layerno == placement[y].layerno) {
                found = true;
                break;
            }
        }
        if (!found)
            return 0;
        swap(x, y);
        break;
    }
    case 3:
    {
        // std::cout << "OP3" << std::endl;
        cidx_t x = 0, y = 0;
        bool found = false;
        for (int attempt = 0; attempt < 512; ++attempt)
        {
            x = gen() % placement.size();
            y = gen() % placement.size();
            if (x != y && dis(Cluster::get_pos(x), Cluster::get_pos(y)) <= dist &&
                placement[x].layerno != placement[y].layerno) {
                found = true;
                break;
            }
        }
        if (!found)
            return 0;
        // } while (placement[x].layerno != 19 && placement[y].layerno != 19);
        swap(x, y);
        break;
    }
    case 5:
        change_DRAM();
        break;
    case 6:
        if (min_idx < 0 || max_idx < 0 ||
            static_cast<size_t>(min_idx) >= placement.size() ||
            static_cast<size_t>(max_idx) >= placement.size())
            return 0;
        swap(min_idx, max_idx);
        break;
    case 7:
        // std::cout << "OP7" << std::endl;
        //dram分配均衡
        if (!change_OA_DRAM())
            return 0;
        break;
    case 8:
        // std::cout << "OP8" << std::endl;
        //对单播的拥塞进行处理
        if(noc_feedback == nullptr || !change_MK_row(*noc_feedback))
            return 0;
        break;
    case 9:
        // std::cout << "OP9" << std::endl;
        //对多播的拥塞进行处理
        if(noc_feedback == nullptr || !spread_WE(*noc_feedback))
            return 0;
        break;
    default:
        break;
    }
    return op;
}

int Light_placement::adaptive_mutate(bool communication_limited, cidx_t dist,
                                     const NoC* noc_feedback)
{
    if (placement.size() < 2)
        return 0;

    // Advance the per-solution seed so copied SA states remain reproducible,
    // while consecutive mutations do not restart from the same random value.
    std::mt19937_64 gen(rand_seed++);

    // if (!communication_limited) {
    //     // Compute-bound: first try another legal B/K/H/W decomposition.  The
    //     // number of cores owned by the layer is unchanged, so this specifically
    //     // explores mappings that can reduce per-core compute/loop overhead.
    //     //修改B/K/H/W
    //     if (!partition.empty() && (gen() & 1U)) {
    //         const size_t first = gen() % partition.size();
    //         for (size_t offset = 0; offset < partition.size(); ++offset) {
    //             Light_partition &candidate = partition[(first + offset) % partition.size()];
    //             if (!candidate.allpartition.empty()) {
    //                 candidate.change();
    //                 return 4;
    //             }
    //         }
    //     }

    //     // Preserve NetTLM's same-layer partition swap in the compute-bound
    //     // region.  It changes which physical core executes each partition.
    //     for (int attempt = 0; attempt < 512; ++attempt) {
    //         const cidx_t x = static_cast<cidx_t>(gen() % placement.size());
    //         const cidx_t y = static_cast<cidx_t>(gen() % placement.size());
    //         if (x != y && placement[x].layerno == placement[y].layerno &&
    //             dis(Cluster::get_pos(x), Cluster::get_pos(y)) <= dist) {
    //             swap(x, y);
    //             return 2;
    //         }
    //     }

    //     if (!partition.empty()) {
    //         partition[gen() % partition.size()].change();
    //         return 4;
    //     }
    //     if(change_core())
    //         return 10;
    //     return 0;
    // }



    if (!communication_limited) {
        /*
        * Compute-bound operator pool:
        *
        * 2  - swap two partitions belonging to the same layer;
        * 4  - change one layer's B/K/H/W decomposition;
        * 10 - transfer one core from a non-bottleneck layer to a bottleneck layer.
        */
        std::vector<int> ops = {2, 4, 10};

        // 每次随机排列操作顺序。
        std::shuffle(ops.begin(), ops.end(), gen);

        for (const int op : ops) {
            switch (op) {
            case 2: {
                /*
                * 在同一个 layer 内交换两个 partition 所在的物理核。
                *
                * 该操作不改变：
                * - layer 的核数量；
                * - B/K/H/W；
                * - segment；
                *
                * 只改变 partition 的物理位置。
                */
                if (placement.size() < 2)
                    break;

                for (int attempt = 0;
                    attempt < 512;
                    ++attempt) {
                    const cidx_t x =
                        static_cast<cidx_t>(
                            gen() % placement.size());

                    const cidx_t y =
                        static_cast<cidx_t>(
                            gen() % placement.size());

                    if (x == y)
                        continue;

                    if (placement[x].layerno !=
                        placement[y].layerno) {
                        continue;
                    }

                    if (dis(Cluster::get_pos(x),
                            Cluster::get_pos(y)) > dist) {
                        continue;
                    }

                    /*
                    * 如果 partno 相同，交换后不会产生变化。
                    * 正常情况下同一个 layer 的不同核应该对应不同
                    * partition，但这里仍显式检查。
                    */
                    if (placement[x].partno ==
                        placement[y].partno) {
                        continue;
                    }

                    swap(x, y);
                    return 2;
                }

                // 没找到合法核对，继续尝试其他操作。
                break;
            }

            case 4: {
                /*
                * 随机选择一个 layer，修改它的 B/K/H/W 划分方式。
                * 核数量保持不变。
                */
                if (partition.empty())
                    break;

                const size_t first =
                    static_cast<size_t>(
                        gen() % partition.size());

                for (size_t offset = 0;
                    offset < partition.size();
                    ++offset) {
                    Light_partition &candidate =
                        partition[
                            (first + offset) %
                            partition.size()
                        ];

                    if (candidate.allpartition.empty())
                        continue;

                    // 保存修改前的 partition 方案。
                    const vol_t old_b = candidate.b;
                    const vol_t old_c = candidate.c;
                    const vol_t old_h = candidate.h;
                    const vol_t old_w = candidate.w;

                    candidate.change();

                    // 只有 B/K/H/W 真正变化才认为操作成功。
                    if (candidate.b != old_b ||
                        candidate.c != old_c ||
                        candidate.h != old_h ||
                        candidate.w != old_w) {
                        return 4;
                    }
                }

                /*
                * 所有 layer 都没有产生新的 partition 方案，
                * 继续尝试其他操作。
                */
                break;
            }

            case 10: {
                /*
                * 将一个核从存在算力余量的 layer 转移到
                * 当前计算瓶颈 layer。
                *
                * change_core() 应返回：
                * true  - 核数量确实发生变化；
                * false - 没有找到有收益且合法的转移。
                */
                if (change_core())
                    return 10;

                break;
            }

            default:
                break;
            }
        }

        // 所有计算受限操作都无法产生合法变化。
        return 0;
    }

    // Communication-bound: retain the original cross-layer swap and DRAM
    // mutation, and place CAO1/CAO2/CAO3 in the same operator pool.
    std::vector<int> ops = {3, 5, 7, 8, 9};
    std::shuffle(ops.begin(), ops.end(), gen);
    for (const int op : ops) {
        switch (op) {
        case 3:
            for (int attempt = 0; attempt < 512; ++attempt) {
                const cidx_t x = static_cast<cidx_t>(gen() % placement.size());
                const cidx_t y = static_cast<cidx_t>(gen() % placement.size());
                if (x != y && placement[x].layerno != placement[y].layerno &&
                    dis(Cluster::get_pos(x), Cluster::get_pos(y)) <= dist) {
                    swap(x, y);
                    return 3;
                }
            }
            break;
        case 5: {
            const auto before = layer_DRAM;
            change_DRAM();
            if (layer_DRAM != before)
                return 5;
            break;
        }
        case 7:
            if (change_OA_DRAM())
                return 7;
            break;
        case 8:
            if (noc_feedback != nullptr && change_MK_row(*noc_feedback))
                return 8;
            break;
        case 9:
            if (noc_feedback != nullptr && spread_WE(*noc_feedback))
                return 9;
            break;
        default:
            break;
        }
    }
    return 0;
}


void Light_placement::crossover(const Light_placement &other)
{
    static std::mt19937_64 gen(277);
    if (gen() & 1)
    {
        placement = other.get_placement();
    }
    for (int i = 0; i < partition.size(); ++i)
    {
        if (gen() & 1)
        {
            partition[i] = other.partition[i];
        }
    }
}

bool Light_placement::change_OA_DRAM()
{
    if (NoC::DRAM_num < 2 || network == nullptr)
        return false;
    const size_t dram_count = static_cast<size_t>(NoC::DRAM_num);

    // One Light_partition exists for each layer in this segment and carries
    // the batch used to compute the amount of OFM data written to DRAM.
    std::unordered_map<lid_t, len_t> layer_batch;
    for (const auto &part : partition)
        layer_batch[static_cast<lid_t>(part.layerno)] = part.batch;

    struct LayerWrite {
        lid_t layer;
        long double volume;
    };

    std::vector<long double> load(dram_count, 0.0L);
    std::vector<LayerWrite> movable;
    for (const auto &entry : layer_DRAM) {
        const lid_t layer = entry.first;
        const auto &dram = entry.second;
        if (dram.size() < 3)
            continue;

        auto batch_it = layer_batch.find(layer);
        if (batch_it == layer_batch.end())
            continue;

        const long double volume = static_cast<long double>(
            network->getNode(layer).layer().ofmap_shape().tot_size(batch_it->second));
        const mlen_t destination = dram[2];
        if (destination >= 0 && static_cast<size_t>(destination) < dram_count) {
            load[static_cast<size_t>(destination)] += volume;
            movable.push_back({layer, volume});
        } else if (destination == -2) {
            // NoC::toRemoteMem already interleaves -2 traffic evenly over all
            // DRAMs.  Count that background traffic, but preserve the special
            // mapping instead of replacing it with a single destination.
            const long double share = std::floor(volume / dram_count);
            for (auto &dram_load : load)
                dram_load += share;
        }
    }

    if (movable.empty())
        return false;

    struct Move {
        size_t movable_index;
        size_t source;
        size_t destination;
    };

    static std::mt19937_64 gen(rand_seed);
    bool changed = false;
    // Every accepted move strictly reduces the global load range.  The cap is
    // an additional safeguard against floating-point or malformed-input loops.
    const size_t max_moves = movable.size() * dram_count * 4 + 1;
    for (size_t iteration = 0; iteration < max_moves; ++iteration) {
        const auto minmax = std::minmax_element(load.begin(), load.end());
        const long double min_load = *minmax.first;
        const long double max_load = *minmax.second;
        if (max_load <= 0.0L ||
            (max_load - min_load) / max_load <= dram_balance_threshold)
            break;

        std::vector<size_t> hottest;
        for (size_t dram = 0; dram < dram_count; ++dram) {
            if (load[dram] == max_load)
                hottest.push_back(dram);
        }
        const size_t source = hottest[gen() % hottest.size()];

        std::vector<Move> improving_moves;
        const long double current_range = max_load - min_load;
        for (size_t index = 0; index < movable.size(); ++index) {
            auto dram_it = layer_DRAM.find(movable[index].layer);
            if (dram_it == layer_DRAM.end() || dram_it->second.size() < 3 ||
                dram_it->second[2] != static_cast<mlen_t>(source))
                continue;

            for (size_t destination = 0; destination < dram_count; ++destination) {
                if (destination == source)
                    continue;
                std::vector<long double> candidate_load = load;
                candidate_load[source] -= movable[index].volume;
                candidate_load[destination] += movable[index].volume;
                const auto candidate_minmax =
                    std::minmax_element(candidate_load.begin(), candidate_load.end());
                const long double new_range = *candidate_minmax.second - *candidate_minmax.first;

                if (new_range < current_range)
                    improving_moves.push_back({index, source, destination});
            }
        }

        // Whole-layer dataflows are indivisible.  If no migration reduces the
        // imbalance, this is the best balance attainable by CAO1.
        if (improving_moves.empty())
            break;

        const Move &move = improving_moves[gen() % improving_moves.size()];
        const auto &write = movable[move.movable_index];
        layer_DRAM[write.layer][2] = static_cast<mlen_t>(move.destination);
        load[move.source] -= write.volume;
        load[move.destination] += write.volume;
        changed = true;
    }

    return changed;
}

bool Light_placement::change_MK_row(const NoC& noc_feedback)
{
    // A directed NoC link is represented by its two endpoints.  Direction is
    // significant because opposite directions have independent link entries.
    using RouteLink = std::tuple<int, int, int, int>;

    const auto is_core = [](const pos_t &p) {
        return p.x >= 0 && p.x < Cluster::xlen &&
               p.y >= 0 && p.y < Cluster::ylen;
    };

    // This is the same X-then-Y deterministic routing order used by
    // NoC::unicastCalc().
    const auto xy_route = [](pos_t source, const pos_t &destination) {
        std::set<RouteLink> route;
        while (source.x != destination.x) {
            const int next_x = source.x + (destination.x > source.x ? 1 : -1);
            route.emplace(source.x, source.y, next_x, source.y);
            source.x = static_cast<mlen_t>(next_x);
        }
        while (source.y != destination.y) {
            const int next_y = source.y + (destination.y > source.y ? 1 : -1);
            route.emplace(source.x, source.y, source.x, next_y);
            source.y = static_cast<mlen_t>(next_y);
        }
        return route;
    };

    const auto overlaps = [](const std::set<RouteLink> &lhs,
                             const std::set<RouteLink> &rhs) {
        auto left = lhs.begin();
        auto right = rhs.begin();
        while (left != lhs.end() && right != rhs.end()) {
            if (*left == *right)
                return true;
            if (*left < *right)
                ++left;
            else
                ++right;
        }
        return false;
    };

    const auto flows = noc_feedback.get_unicast_info();
    if (flows.size() < 2)
        return false;

    // Step 1: get the highest-volume core-to-core unicast pair.  DRAM router
    // coordinates are excluded because CAO2 must be able to swap either end.
    const NoC::unicast_info *hottest = nullptr;
    std::set<RouteLink> hottest_route;
    for (const auto &flow : flows) {
        if (!is_core(flow.source) || !is_core(flow.destination))
            continue;
        auto route = xy_route(flow.source, flow.destination);
        if (!route.empty()) {
            hottest = &flow;
            hottest_route = std::move(route);
            break;
        }
    }
    if (hottest == nullptr)
        return false;

    // Step 2: collect all other source-destination pairs that use at least
    // one directed link of the hottest pair.  Also mark their endpoint cores;
    // replacement cores are selected outside this set.
    std::vector<const NoC::unicast_info*> contending;
    std::set<std::pair<int, int>> cores_using_hottest_route;
    cores_using_hottest_route.insert({hottest->source.x, hottest->source.y});
    cores_using_hottest_route.insert({hottest->destination.x, hottest->destination.y});
    for (const auto &flow : flows) {
        if (&flow == hottest || !is_core(flow.source) || !is_core(flow.destination))
            continue;
        const auto route = xy_route(flow.source, flow.destination);
        if (overlaps(route, hottest_route)) {
            contending.push_back(&flow);
            cores_using_hottest_route.insert({flow.source.x, flow.source.y});
            cores_using_hottest_route.insert({flow.destination.x, flow.destination.y});
        }
    }
    if (contending.empty())
        return false;

    static std::mt19937_64 gen(rand_seed);

    // Step 3: choose a contending pair and then one of its endpoints.
    const auto &selected_flow = *contending[gen() % contending.size()];
    const bool move_source = (gen() & 1U) != 0;
    const pos_t selected_position =
        move_source ? selected_flow.source : selected_flow.destination;

    const auto find_core_id = [&](const pos_t &position) -> cidx_t {
        const size_t core_count = std::min(
            placement.size(),
            static_cast<size_t>(Cluster::xlen) * static_cast<size_t>(Cluster::ylen));
        for (size_t core = 0; core < core_count; ++core) {
            if (Cluster::get_pos(static_cast<cidx_t>(core)) == position)
                return static_cast<cidx_t>(core);
        }
        return static_cast<cidx_t>(-1);
    };

    const cidx_t selected_core = find_core_id(selected_position);
    if (selected_core < 0)
        return false;

    // Step 4: choose a mapped core that is not an endpoint of any flow using
    // the hottest route.  The moved pair's new XY route must also avoid it.
    std::vector<cidx_t> replacements;
    const size_t mapped_core_count = std::min(
        placement.size(),
        static_cast<size_t>(Cluster::xlen) * static_cast<size_t>(Cluster::ylen));
    for (size_t core = 0; core < mapped_core_count; ++core) {
        const pos_t candidate = Cluster::get_pos(static_cast<cidx_t>(core));
        if (static_cast<cidx_t>(core) == selected_core ||
            cores_using_hottest_route.count({candidate.x, candidate.y}) != 0)
            continue;

        const auto moved_route = move_source
            ? xy_route(candidate, selected_flow.destination)
            : xy_route(selected_flow.source, candidate);
        if (!moved_route.empty() && !overlaps(moved_route, hottest_route))
            replacements.push_back(static_cast<cidx_t>(core));
    }
    if (replacements.empty())
        return false;

    const cidx_t replacement = replacements[gen() % replacements.size()];
    return swap(selected_core, replacement);
}

bool Light_placement::spread_WE(const NoC& noc_feedback)
{
    if (placement.size() < 2)
        return false;

    const auto is_core = [](const pos_t &p) {
        return p.x >= 0 && p.x < Cluster::xlen &&
               p.y >= 0 && p.y < Cluster::ylen;
    };

    // Keep one entry per physical multicast source.  DRAM-originated
    // multicast is excluded because a DRAM router cannot be core-swapped.
    std::vector<pos_t> sources;
    for (const auto &source : noc_feedback.get_multicast_source_info()) {
        if (is_core(source.source))
            sources.push_back(source.source);
    }
    if (sources.size() < 2)
        return false;

    static std::mt19937_64 gen(rand_seed);
    const size_t mapped_core_count = std::min(
        placement.size(),
        static_cast<size_t>(Cluster::xlen) * static_cast<size_t>(Cluster::ylen));

    const auto find_core_id = [&](const pos_t &position) -> cidx_t {
        for (size_t core = 0; core < mapped_core_count; ++core) {
            if (Cluster::get_pos(static_cast<cidx_t>(core)) == position)
                return static_cast<cidx_t>(core);
        }
        return static_cast<cidx_t>(-1);
    };

    bool changed = false;
    // Each successful move removes the selected source's current row/column
    // conflict.  The cap protects malformed mappings where a swap cannot make
    // monotonic progress.
    const size_t max_moves = sources.size() * mapped_core_count + 1;
    for (size_t iteration = 0; iteration < max_moves; ++iteration) {
        // Step 1: identify multicast sources sharing a row or a column.
        std::vector<size_t> overlapping_sources;
        for (size_t i = 0; i < sources.size(); ++i) {
            bool overlaps = false;
            for (size_t j = 0; j < sources.size(); ++j) {
                if (i == j)
                    continue;
                if (sources[i].x == sources[j].x ||
                    sources[i].y == sources[j].y) {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps)
                overlapping_sources.push_back(i);
        }
        if (overlapping_sources.empty())
            break;

        // Steps 2 and 3: visit the overlapping sources in random order and use
        // the first one that has at least one legal destination.  This keeps a
        // temporarily immovable source from hiding another movable conflict.
        std::shuffle(overlapping_sources.begin(), overlapping_sources.end(), gen);
        size_t selected_index = sources.size();
        cidx_t selected_core = static_cast<cidx_t>(-1);
        std::vector<cidx_t> replacements;
        for (const size_t source_to_move : overlapping_sources) {
            const cidx_t candidate_source_core = find_core_id(sources[source_to_move]);
            if (candidate_source_core < 0)
                continue;

            std::vector<cidx_t> candidate_replacements;
            for (size_t core = 0; core < mapped_core_count; ++core) {
                if (static_cast<cidx_t>(core) == candidate_source_core)
                    continue;
                const pos_t candidate = Cluster::get_pos(static_cast<cidx_t>(core));
                bool conflicts = false;
                for (size_t source_index = 0;
                     source_index < sources.size(); ++source_index) {
                    if (source_index == source_to_move)
                        continue;
                    if (candidate.x == sources[source_index].x ||
                        candidate.y == sources[source_index].y) {
                        conflicts = true;
                        break;
                    }
                }
                if (!conflicts)
                    candidate_replacements.push_back(static_cast<cidx_t>(core));
            }

            if (!candidate_replacements.empty()) {
                selected_index = source_to_move;
                selected_core = candidate_source_core;
                replacements = std::move(candidate_replacements);
                break;
            }
        }
        if (selected_index == sources.size())
            break;

        const cidx_t replacement = replacements[gen() % replacements.size()];
        const pos_t replacement_position = Cluster::get_pos(replacement);
        if (!swap(selected_core, replacement))
            break;

        // The multicast-producing partition follows the selected mapping entry
        // to its new core.  Other multicast sources remain unchanged because a
        // multicast-source core is never a valid replacement.
        sources[selected_index] = replacement_position;
        changed = true;
    }

    return changed;
}

void Light_placement::print()
{
    // 逐 Core 输出位置、层和分区绑定
    std::cout << "Core binding details:" << std::endl;

    for (std::size_t i = 0; i < placement.size(); ++i)
    {
        const cidx_t core_id = static_cast<cidx_t>(i);
        const auto pos = Cluster::get_pos(core_id);
        const auto& part = placement[i];

        std::cout
            << "Core " << core_id
            << ", Pos(" << pos.x << "," << pos.y << ")"
            << ", Layer " << part.layerno
            << ", Partition " << part.partno
            << std::endl;
    }

    // 保留原来的二维矩阵输出
    std::cout << "Placement matrix (Layer, Partition):"
              << std::endl;

    std::vector<std::vector<Layer_partition>> matrix(
        Cluster::ylen,
        std::vector<Layer_partition>(Cluster::xlen)
    );

    cidx_t core_id = 0;

    for (const auto& part : placement)
    {
        const auto pos = Cluster::get_pos(core_id);
        matrix[pos.y][pos.x] = part;
        ++core_id;
    }

    for (cidx_t y = Cluster::ylen - 1; y >= 0; --y)
    {
        for (cidx_t x = 0; x < Cluster::xlen; ++x)
        {
            std::cout
                << "("
                << matrix[y][x].layerno
                << ","
                << matrix[y][x].partno
                << "),";
        }

        std::cout << std::endl;
    }
}

std::ostream &operator<<(std::ostream &os, const Light_placement &Light_placement)
{
    std::vector<Light_placement::Layer_partition> cores(Light_placement.placement.size());
    // std::vector<std::vector<Light_placement::Layer_partition> > matrix(Cluster::ylen, std::vector<Light_placement::Layer_partition>(Cluster::xlen));
    cidx_t core_id = 0;

    for (auto &place : Light_placement.placement)
    {
        cores[core_id] = place;
        // auto pos = Cluster::get_pos(core_id);
        // os << pos << "," << part.layerno << " / ";
        core_id++;
    }
    // os << "\n";
    std::map<cidx_t, cidx_t> tmp_part;
    for (auto &part : Light_placement.partition)
    {
        //os << part.layerno << ": " << "b=" << part.b << ",c=" << part.c << ",h=" << part.h << ",w=" << part.w << " / ";
        // os << "part[" << Light_placement.rrnd << "][" << part.layerno << "]={" << part.h << "," << part.c << "};\n";
        
        // os << "part[" << part.layerno << "]={" << part.h << "," << part.c << "};\n";
        core_id = 0;
        tmp_part.clear();
        for (auto &place : cores)
        {

            if (part.layerno == place.layerno)
            {
                tmp_part[place.partno] = core_id;
                // os << core_id << Cluster::get_pos(core_id) << " / ";
            }
            core_id++;
        }

        // os << "assignedDpath[" << Light_placement.rrnd << "][" << part.layerno << "]={";
        // os << "assignedDpath[" << part.layerno << "]={";
        // os << Light_placement.rrnd << ",";
        for (auto &tmp : tmp_part)
        {
            os << tmp.second;
            if(tmp.first != tmp_part.size() - 1)
                os << ",";

            //os << tmp.first << "(" << tmp.second << Cluster::get_pos(tmp.second) << ") / ";
        }
        // os << "};\n";
        os << "\n";
    }
    
    std::map<lid_t, std::vector<mlen_t>> data_dram;
    for (auto &l : Light_placement.layer_DRAM)
    {
        if(l.first < 20)
            data_dram[l.first] = l.second;
    }

    for (auto &l : data_dram)
    {
        data_dram[l.first] = l.second;
        // os << "data_dram[" << Light_placement.rrnd << "][" << l.first << "]={";
        // os << "data_dram[" << l.first << "]={";
        // os << Light_placement.rrnd <<",";
        const int ifm = l.second.size() > 0 ? l.second[0] : -1;
        const int wgt = l.second.size() > 1 ? l.second[1] : -1;
        const int ofm = l.second.size() > 2 ? l.second[2] : -1;
        os << ifm << "," << wgt << "," << ofm << "\n";
        // os << l.second[2] << "};";
        
    }

    // for(auto &part: Light_placement.placement){
    //     auto pos = Cluster::get_pos(core_id);
    //     matrix[pos.y][pos.x] = part;
    //     ++core_id;
    // }
    // for(cidx_t i=Cluster::ylen-1;i>=0;--i){
    //     for(cidx_t j=0;j<Cluster::xlen;++j){
    //         os << "(" << matrix[i][j].layerno << ", " << matrix[i][j].partno << "), ";
    //     }
    // }
    return os;
}

// void Light_placement::compare()
// {
    
//     std::vector<Light_placement::Layer_partition> cores(36);
//     cidx_t core_id = 0;

//     for (auto &place : placement)
//     {
//         cores[core_id] = place;
//         core_id++;
//     }

//     std::map<cidx_t, cidx_t> tmp_part;
//     for (auto &part : partition)
//     {
//         gem_p.part[part.layerno] = {part.h, part.c};
        
//         core_id = 0;
//         tmp_part.clear();
//         for (auto &place : cores)
//         {
//             if (part.layerno == place.layerno)
//             {
//                 tmp_part[place.partno] = core_id;
//             }
//             core_id++;
//         }

//         for (auto &tmp : tmp_part)
//         {
//             gem_p.assignedDpath[part.layerno].push_back(tmp.second);
//         }
//     }

//     std::map<lid_t, std::vector<mlen_t>> data_dram;
//     for (auto &l : layer_DRAM)
//     {
//         if(l.first < 20)
//             data_dram[l.first] = l.second;
//     }

//     for (auto &l : data_dram)
//     {
//         for(auto &d : l.second)
//             gem_p.data_dram[l.first].push_back(d);
//     }

//     gem_p.run();
// }
