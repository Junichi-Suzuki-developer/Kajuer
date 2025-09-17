#pragma once

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace Kajuer::Common
{
    using std::pair;
    using std::size_t;
    using std::vector;

    using elec_t = std::double_t;
    using pixel_t = std::double_t;
    using real_time_t = std::double_t;
    using step_time_t = int;
    using general_index_t = int;

    using input_current_array_t = elec_t*;
    using neuron_array_t = elec_t*;
    using neuron_index_array_t = general_index_t*;
    using neuron_index_array_for_gen_t = vector<general_index_t>;
    using neuron_mask_array_t = bool*;
    using neuron_mask_index_array_t = vector<general_index_t>;
    using neuron_pair_array_t = vector<pair<general_index_t, general_index_t>>;
    using real_time_array_t = real_time_t*;
    using sensory_input_t = elec_t*;
    using synapse_array_t = elec_t*;
    using synapse_array_for_gen_t = vector<elec_t>;
    using synapse_mask_index_array_t = vector<general_index_t>;
    using time_record_array_t = step_time_t*;
    using time_record_index_array_t = vector<step_time_t>;
};
