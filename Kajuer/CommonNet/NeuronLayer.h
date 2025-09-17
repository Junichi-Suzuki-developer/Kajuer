#pragma once

//import std;

#include <mkl.h>

#include <algorithm>
#include <cstddef>
#include <execution>
#include <fstream>
#include <iostream>
#include <random>

#include "CommonNet/Common.h"

namespace Kajuer::Common
{
    using std::begin;
    using std::cerr;
    using std::clamp;
    using std::count;
    using std::cout;
    using std::end;
    using std::endl;
    using std::fill;
    using std::fill_n;
    using std::ignore;
    using std::int64_t;
    using std::int_fast64_t;
    using std::mt19937;
    using std::noboolalpha;
    using std::ofstream;
    using std::string;
    using std::to_string;
    using std::transform;
    using std::uint32_t;
    using std::uniform_real_distribution;
    using std::vector;

    using Kajuer::Common::step_time_t;
    using Kajuer::Common::general_index_t;

    //neuron layer type.
    enum NeuronLayerType
    {
        Sensory,
        Output,
        Inhibition,
    };

    struct NeuronLayerParameters
    {
        const elec_t A_e;
        const elec_t B_e;
        const elec_t C_e;
        const elec_t D_e;
        const elec_t Tau_current;
        const elec_t Input_gain;
        const elec_t v_init;
        const elec_t u_init;
        const int type;
        const general_index_t num_neurons;
        const step_time_t max_delay_msec;
        const vector<elec_t> current_boundary; //[0] = for lower_bound, [1] = for upper_bound.
        const uint32_t num_labels;
    };

    //This class defines neurons in single neuron layer.
    class NeuronLayer
    {
    private:
        const size_t alignment; //memory alignment.

    private:

        const elec_t A_e; //neuron parameter.
        const elec_t B_e; //neuron parameter.
        const elec_t C_e; //neuron parameter.
        const elec_t D_e; //neuron parameter.

        const elec_t Tau_current; //time constant for decaying current input.
        const elec_t Input_gain; //input gain for current input.

        const elec_t v_init; //initial v value.
        const elec_t u_init; //initial u value.

        const int type; //neuron layer type.

        const vector<elec_t> current_boundary; //[0] for lower bound. [1] for upper bound.

        input_current_array_t i_buffer; //buffer for i parameter.

    private:
        const general_index_t num_neurons; //num of this NeuronLayer's neurons.
        const step_time_t max_delay_msec; //max delay of synapses to this SerialNet.

        uint32_t label; //current indicated label which is utilized in learning phase.
        const uint32_t num_labels; //number  of labels.

        bool gate_with_label; //input gate depends on label.

        neuron_array_t v; //membrane potentials.
        neuron_array_t u; //hidden parameters. see [https://www.izhikevich.org/publications/spikes.pdf].

        input_current_array_t i; //input current to neurons.

        input_current_array_t bias; //bias current to neurons.

		input_current_array_t external_input; //external current input.

        neuron_mask_index_array_t fired_index_list; //fired neuron index list.

    private:
        //replace external_input pointer with arg_external_input.
        void apply_external_input(const input_current_array_t& arg_external_input)
        {
            external_input = arg_external_input;
        }

        //currently unimplemented.
        void apply_tonic_input(void)
        {
            return;
        }

    private:

        //utility function for copy constructor.
        void copy(const NeuronLayer& src)
        {
		    external_input = src.external_input;

            i_buffer = static_cast<input_current_array_t>(mkl_malloc(sizeof(elec_t) * src.num_neurons, src.alignment));

            v = static_cast<neuron_array_t>(mkl_malloc(sizeof(elec_t) * src.num_neurons, src.alignment));
            u = static_cast<neuron_array_t>(mkl_malloc(sizeof(elec_t) * src.num_neurons, src.alignment));
            i = static_cast<input_current_array_t>(mkl_malloc(sizeof(elec_t) * src.num_neurons, src.alignment));
            bias = static_cast<input_current_array_t>(mkl_malloc(sizeof(elec_t) * src.num_neurons, src.alignment));

            for(general_index_t j=0; j<src.num_neurons; j+=1)
            {
                i_buffer[j] = src.i_buffer[j];

                v[j] = src.v[j];
                u[j] = src.u[j];
                i[j] = src.i[j];
                bias[j] = src.bias[j];
            }
        }

        //judge if we can skip update synaptic weights.
        bool need_skip(const general_index_t j) const
        {
            bool ret_val{false};

            if(gate_with_label)
            {
                if((j*num_labels/num_neurons) != label)
                {
                    ret_val = true;
                }
            }

            return ret_val;
        }

    public:
        //How to use this class:
        // 1st: call apply_current_input(...).
        // 2nd: call update_neurons(...).

        //applying external current input to this layer.
        void apply_current_input(const input_current_array_t& arg_external_input)
        {
            apply_external_input(arg_external_input);
            apply_tonic_input();
        }

        //disabling label selector.
        void disable_label_selector(void)
        {
            gate_with_label = false;
        }

        //enabling label selector.
        void enable_label_selector(void)
        {
            gate_with_label = true;
        }

        //This fucntion returns fired(spiked) neuron indices.
        const neuron_mask_index_array_t& get_is_fired(void) const
        {
            return fired_index_list;
        }

        //This function returns this layer's number of neurons.
        const general_index_t get_num_neurons(void) const
        {
            return num_neurons;
        }

        //This function initializes parameters without u.
        void initialize_buffers(void)
        {
            for(general_index_t j=0; j<num_neurons; j+=1)
            {
                v[j] = v_init;
                i[j] = 0.0;
                i_buffer[j] = 0.0;
            }
        }

        //This function initializes all parameters.
        void initialize_parameters(const step_time_t current_time, const bool need_randomize=false)
        {
            uniform_real_distribution<elec_t> dist
            (
//                +0.0,
                +0.75,
                +1.0
            );

            mt19937 u_gen_engine(current_time);

            for(general_index_t j=0; j<num_neurons; j+=1)
            {
                v[j] = v_init;
                u[j] = u_init;
                if(need_randomize)
                {
                    u[j] *= dist(u_gen_engine);
                }
                i[j] = 0.0;
                i_buffer[j] = 0.0;
            }
        }

        //This function sets arg_label as label.
        void switch_label(const uint32_t arg_label)
        {
            label = arg_label;
        }

        //see the equation in [https://www.izhikevich.org/publications/spikes.pdf]
        void update_neurons(const bool debug_print=false)
        {
		    input_current_array_t& current_i{i};

            if(NeuronLayerType::Inhibition==type)
            {
                const int inc_x{1};
                const int inc_y{1};

                fill_n(&current_i[0], num_neurons, 0.0);
                daxpy(&num_neurons, &Input_gain, external_input, &inc_x, current_i, &inc_y);

                for(general_index_t j=0; j<num_neurons; j+=1)
                {
                    bias[j] = 0.0;
                }
            }
            if(NeuronLayerType::Sensory==type)
            {
                const elec_t a0{-1.0};
                const elec_t a1{1.0/Tau_current};
                const int inc_x{1};
                const int inc_y{1};

                dcopy(&num_neurons, current_i, &inc_x, i_buffer, &inc_y);
                dscal(&num_neurons, &a0, i_buffer, &inc_x);
                daxpy(&num_neurons, &Input_gain, external_input, &inc_x, i_buffer, &inc_y);
                dscal(&num_neurons, &a1, i_buffer, &inc_x);
                vdAdd(num_neurons, i_buffer, current_i, current_i);

                for(general_index_t j=0; j<num_neurons; j+=1)
                {
                    bias[j] = 0.0;
                }
            }
            else
            {
                const elec_t a0{-1.0};
                const elec_t a1{1.0/Tau_current};
                const int inc_x{1};
                const int inc_y{1};

                dcopy(&num_neurons, current_i, &inc_x, i_buffer, &inc_y);
                dscal(&num_neurons, &a0, i_buffer, &inc_x);
                daxpy(&num_neurons, &Input_gain, external_input, &inc_x, i_buffer, &inc_y);
                dscal(&num_neurons, &a1, i_buffer, &inc_x);
                vdAdd(num_neurons, i_buffer, current_i, current_i);

                for(general_index_t j=0; j<num_neurons; j+=1)
                {
                    bias[j] = -30.0;
                }
            }

            transform
            (
                std::execution::unseq,
                &current_i[0],
                &current_i[num_neurons],
                &current_i[0],
                [this](elec_t x) { return clamp(x, current_boundary[0], current_boundary[1]); }
            );

            {
                for(general_index_t j=0; j<num_neurons; j+=1)
                {
                    if(need_skip(j))
                    {
                        continue;
                    }
                    elec_t current_sum{-u[j]+current_i[j]+bias[j]};

                    current_sum = clamp(current_sum, -40.0, 135.0);
                    v[j]+=0.5*((0.04*v[j]+5.0)*v[j]+140.0+current_sum);
                    v[j]+=0.5*((0.04*v[j]+5.0)*v[j]+140.0+current_sum);
                }
            }

            //extract spiked(fired) neuron indices.
            {
                fired_index_list.clear();

                elec_t* result = std::find_if(&v[0], &v[num_neurons], [](elec_t x) { return (x>=30.0); });

                while(&v[num_neurons] != result)
                {
                    long index{result - &v[0]};
                    fired_index_list.emplace_back(index);

                    result += 1;

                    result = std::find_if(result, &v[num_neurons], [](elec_t x) { return (x>=30.0); });
                }
            }

            const general_index_t fired_index_list_size{static_cast<general_index_t>(fired_index_list.size())};

            for(general_index_t j=0; j<fired_index_list_size; j+=1)
            {
                v[fired_index_list[j]] = C_e;
            }

            {
                for(general_index_t j=0; j<num_neurons; j+=1)
                {
                    if(need_skip(j))
                    {
                        continue;
                    }
                    const elec_t du{A_e*(B_e*v[j] - u[j])};
                    u[j] += du;

                    //limit u.
                    if(u[j]>100.0)
                    {
                        u[j] = 100.0;
                    }
                }
            }

            for(general_index_t j=0; j<fired_index_list_size; j+=1)
            {
                u[fired_index_list[j]] += D_e;
            }
        }

    public:

        explicit NeuronLayer(const NeuronLayerParameters& neuron_params, const size_t arg_alignment=64) :
            alignment(arg_alignment),
            A_e(neuron_params.A_e),
            B_e(neuron_params.B_e),
            C_e(neuron_params.C_e),
            D_e(neuron_params.D_e),

            Tau_current(neuron_params.Tau_current),
            Input_gain(neuron_params.Input_gain),

            v_init(neuron_params.v_init),
            u_init(neuron_params.u_init),

            type(neuron_params.type),
            current_boundary(neuron_params.current_boundary),

            i_buffer(static_cast<input_current_array_t>(mkl_malloc(sizeof(elec_t) * neuron_params.num_neurons, alignment))),

            num_neurons(neuron_params.num_neurons),

            max_delay_msec(neuron_params.max_delay_msec),
            label(0),
            num_labels(neuron_params.num_labels),
            gate_with_label(false),

            v(static_cast<neuron_array_t>(mkl_malloc(sizeof(elec_t) * neuron_params.num_neurons, alignment))),
            u(static_cast<neuron_array_t>(mkl_malloc(sizeof(elec_t) * neuron_params.num_neurons, alignment))),
            i(static_cast<input_current_array_t>(mkl_malloc(sizeof(elec_t) * neuron_params.num_neurons, alignment))),
            bias(static_cast<input_current_array_t>(mkl_malloc(sizeof(elec_t) * neuron_params.num_neurons, alignment)))
        {
            initialize_buffers();
            initialize_parameters(0, true);

            ignore = max_delay_msec;
        }

        explicit NeuronLayer(const NeuronLayer& src) :
            alignment(src.alignment),
            A_e(src.A_e),
            B_e(src.B_e),
            C_e(src.C_e),
            D_e(src.D_e),

            Tau_current(src.Tau_current),
            Input_gain(src.Input_gain),

            v_init(src.v_init),
            u_init(src.u_init),

            type(src.type),
            current_boundary(src.current_boundary),
            num_neurons(src.num_neurons),

            max_delay_msec(src.max_delay_msec),

            label(src.label),
            num_labels(src.num_labels),
            gate_with_label(src.gate_with_label),

            fired_index_list(src.fired_index_list)
        {
            copy(src);
        }

        NeuronLayer& operator= (const NeuronLayer& src) = delete;
        NeuronLayer(NeuronLayer&&) = delete;
        NeuronLayer& operator=(NeuronLayer&&) = delete;

        virtual ~NeuronLayer(void)
        {
            mkl_free(i_buffer);

            mkl_free(v);
            mkl_free(u);
            mkl_free(i);
            mkl_free(bias);
        }
    };
};
