#ifndef LIGHT_PLACEMENT_H
#define LIGHT_PLACEMENT_H
//This file corresponds to the optimization of spatial mapping, and the corresponding operation operators are in the public functions of Light_placement.
#include "ltreenode.h"
#include "partition.h"
#include <unordered_map>
#include <unordered_set>
#include <iosfwd>

class FastLayerEngine;

class SchNode;
class NoC;

class Light_partition{//per layer
public:
    vol_t b,c,h,w;
    vol_t siz,batch;
    lid_t layerno;
    std::vector<std::vector<len_t> > allpartition;
    void init(const fmap_shape& shape, len_t batch, vol_t _layerno);
    void re_init(const fmap_shape& shape,  vol_t size, PartSch part);
    Light_partition();
    Light_partition(const fmap_shape& shape, cidx_t _siz, len_t batch, vol_t _layerno);
    Light_partition(vol_t _b, vol_t _c, vol_t _h, vol_t _w, vol_t _siz, vol_t _layerno);
    void change();
};



class Light_placement{// per segment
public:
    static FastLayerEngine* fast_buffer;
    // CAO1 stops when (maximum DRAM write traffic - minimum traffic) /
    // maximum traffic is no larger than this value.  0 requests the
    // tightest balance attainable with whole-layer moves.
    static double dram_balance_threshold;
    static void set_dram_balance_threshold(double threshold);
    static double get_dram_balance_threshold();
    typedef uint16_t lid_t;
    struct Layer_partition{
        lid_t layerno;
        cidx_t partno;
        Layer_partition();
        Layer_partition(lid_t _layerno, cidx_t _partno);
    };
private:
    std::vector<Layer_partition> placement;// num = core number, layer? partition?
    std::vector<Light_partition> partition;
    std::unordered_map<lid_t, std::unordered_set<cidx_t>> layer_scheme;
    std::unordered_map<lid_t, FetchSch> fetch_scheme;
    
    cidx_t layer_num;
    Bitset layers;
    LTreeNode* node;
public:
    std::unordered_map<lid_t, std::vector<mlen_t>> layer_DRAM;// vector<ifmap_source,weight_source,ofmap_dst> ifmap_source is generally default
    unsigned int rand_seed = 233;
    Light_placement();
    Light_placement(LTreeNode* segment);
    //Light_placement(SchNode::sn_ptr segment);
    bool swap(cidx_t x, cidx_t y);
    int mutate(cidx_t dist, int min_idx, int max_idx, int delta, int last_op,
               const NoC* noc_feedback = nullptr);
    // Apply one bottleneck-aware mapping mutation.  Compute-bound searches
    // keep the layer allocation fixed and optimize the intra-layer mapping;
    // communication-bound searches use the original placement/DRAM operators
    // together with CAO1/CAO2/CAO3.
    int adaptive_mutate(bool communication_limited, cidx_t dist,
                        const NoC* noc_feedback = nullptr);
    bool change_core();//control the max iter num
    void change_DRAM();
    bool change_OA_DRAM();
    bool change_MK_row(const NoC& noc_feedback);
    bool spread_WE(const NoC& noc_feedback);
    void crossover(const Light_placement &other);
    const std::vector<Layer_partition>& get_placement() const;
    const std::vector<Light_partition>& get_partition() const;
    const std::unordered_map<lid_t, FetchSch>& get_fetch_scheme() const;
    const std::unordered_map<lid_t, std::unordered_set<cidx_t>>& get_layer_scheme() const;
    friend void init(Light_placement& place, SchNode* root);
    friend std::ostream& operator<<(std::ostream& os, const Light_placement& placement);
    void print();
};

std::ostream& operator<<(std::ostream& os, const Light_placement& placement);

#endif //LIGHT_PLACEMENT_H
