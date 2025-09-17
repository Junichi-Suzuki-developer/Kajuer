#pragma once
#include <cstdint>
#include <random>

#include "CommonNet/Common.h"
#include "CommonNet/ConnectionLayer.h"
#include "CommonNet/NeuronLayer.h"

namespace Kajuer::Param
{
    using std::move;
    using std::mt19937;
    using std::size_t;
    using std::uint32_t;
    using std::uint_fast32_t;
    using std::vector;

    using Kajuer::Common::elec_t;
    using Kajuer::Common::general_index_t;
    using Kajuer::Common::step_time_t;
    using Kajuer::Common::ConnectionLayer;
    using Kajuer::Common::ConnectionLayerParameters;
    using Kajuer::Common::NeuronLayer;
    using Kajuer::Common::NeuronLayerParameters;
    using Kajuer::Common::NeuronLayerType;

    struct ParamList
    {
        //random seed used for generating network connections.
        const vector<uint_fast32_t> network_seed_gen{0,1,2,};

        //random seed used for generating pseudo-poisson spikes.
        const vector<uint_fast32_t> spike_seed_gen{0,1,2,};

        //random seed used for generating conduction delays.
        const vector<uint_fast32_t> delay_seed_gen{0,1,2,};

        const vector<general_index_t> num_iterations_for_each_image
        {
            250,
            250,
        };
        //[0] = for learning, [1] = for validation.

        const vector<general_index_t> num_images
        {
            60000,
            10000+1,
        };
        const vector<step_time_t> time_end
        {
            num_images[0]*num_iterations_for_each_image[0],
            (num_images[1]-1)*num_iterations_for_each_image[1],
        };
        //[0] = for learning, [1] = for validation.

        const elec_t input_bias{0.25};
        //input bias for pseudo-poisson spike train generator.

        const vector<elec_t> weight_boundary{-1.0, 1.0,};
        //[0] = for lower_bound, [1] = for upper_bound.

        const vector<elec_t> current_boundary{-100.0, 155.10,};
        //[0] = for lower_bound, [1] = for upper_bound.

        const vector<step_time_t> common_max_delay
        {
            1,
            2,
            1,
            1,
        };
        //max conduction delay.
        //[0] = for from input neuron layer, psudo-poisson spike train layer, to sensory neuron layer,
        //[1] = for from sensory neuron layer to output neuron layer.
        //notice: current implementation treat the value 1 as no delay, so please do not use 0 here.

        const vector<elec_t> common_input_gain
        {
            155.10,
            30.18,
            3000.50,
        };
        //synaptic current gain.
        //[0] = for from input neuron layer, psudo-poisson spike train layer, to sensory neuron layer,
        //[1] = for from sensory neuron layer to output neuron layer.

        const vector<vector<general_index_t>> num_neurons
        {
            {28, 28,},
            {20, 20,},
            {20, 20,},
        };
        //number of neurons.
        //[0] = for sensory neuron layer,
        //[1] = for output neuron layer.

        const uint32_t num_labels{10};

        const general_index_t num_epochs{3};

        const vector<NeuronLayerParameters> neuron_layer_params
        {
            //sensory neuron layer.
            {
                .A_e = 0.02,
                .B_e = 0.2,
                .C_e = -65.0,
                .D_e = 8.0,
                .Tau_current = 100.0,
                .Input_gain = common_input_gain[0],
                .v_init = -65.0,
                .u_init = -13.0,
                .type = NeuronLayerType::Sensory,
                .num_neurons = num_neurons[0][0] * num_neurons[0][1],
                .max_delay_msec = common_max_delay[0],
                .current_boundary = current_boundary,
                .num_labels = num_labels,
            },

            //output neuron layer.
            {
                .A_e = 0.0000001,
                .B_e = 0.00002,
                .C_e = -100.0,
                .D_e = 5.0,
                .Tau_current = 100.0,
                .Input_gain = common_input_gain[1],
                .v_init = -100.0,
                .u_init = -20.0,
                .type = NeuronLayerType::Output,
                .num_neurons = num_neurons[1][0] * num_neurons[1][1],
                .max_delay_msec = common_max_delay[1],
                .current_boundary = current_boundary,
                .num_labels = num_labels,
            },

            //inhibition neuron layer.
            {
                .A_e = 0.1,
                .B_e = 0.2,
                .C_e = -65.0,
                .D_e = 2.0,
                .Tau_current = 1.0,
                .Input_gain = common_input_gain[2],
                .v_init = -65.0,
                .u_init = -13.0,
                .type = NeuronLayerType::Inhibition,
                .num_neurons = num_neurons[2][0] * num_neurons[2][1],
                .max_delay_msec = common_max_delay[2],
                .current_boundary = current_boundary,
                .num_labels = num_labels,
            },
        };

        //define connection layer paramters.
        const vector<ConnectionLayerParameters> connection_layer_params
        {
            //senory to output neuron layer connection.
            {
                .max_delay_msec = common_max_delay[1],
                .weight_boundary = weight_boundary,
                .num_labels = num_labels,
                .C_PLUS = 1.25,
                .C_MINUS = 0.25,
                .ETA = 0.01,
                .INV_TAU_PLUS = 1.0/100.0,
                .INV_TAU_MINUS = 1.0/100.0,
                .DECAY_INV_TAU_W_0 = 1.0/600.0,
                .DECAY_ALPHA = 1.0,
                .DECAY_BETA = -2.0,
                .DECAY_W_INH = -5.0,
                .initial_weight_lower_bound = 0.0,
                .initial_weight_upper_bound = 0.3,
                .weight_factor = 108.4,
                .delay_seed_gen = delay_seed_gen[0],
            },

            //output to inhibition neuron layer connection.
            {
                .max_delay_msec = common_max_delay[2],
                .weight_boundary = weight_boundary,
                .num_labels = num_labels,
                .C_PLUS = 0.0,
                .C_MINUS = 0.0,
                .ETA = 0.0,
                .INV_TAU_PLUS = 0.0,
                .INV_TAU_MINUS = 0.0,
                .DECAY_INV_TAU_W_0 = 0.0,
                .DECAY_ALPHA = 0.0,
                .DECAY_BETA = 0.0,
                .DECAY_W_INH = 0.0,
                .initial_weight_lower_bound = 1.0,
                .initial_weight_upper_bound = 1.0,
                .weight_factor = 1.0,
                .delay_seed_gen = delay_seed_gen[1],
            },

            //inhibition to output neuron layer connection.
            {
                .max_delay_msec = common_max_delay[3],
                .weight_boundary = weight_boundary,
                .num_labels = num_labels,
                .C_PLUS = 0.0,
                .C_MINUS = 0.0,
                .ETA = 0.0,
                .INV_TAU_PLUS = 0.0,
                .INV_TAU_MINUS = 0.0,
                .DECAY_INV_TAU_W_0 = 0.0,
                .DECAY_ALPHA = 0.0,
                .DECAY_BETA = 0.0,
                .DECAY_W_INH = 0.0,
                .initial_weight_lower_bound = 1.0,
                .initial_weight_upper_bound = 1.0,
                .weight_factor = 1.0,
                .delay_seed_gen = delay_seed_gen[2],
            },
        };
    };
};
