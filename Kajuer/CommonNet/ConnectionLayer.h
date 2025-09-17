#pragma once

#include <mkl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <functional>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <ranges>
#include <vector>

#include "CommonNet/Common.h"

namespace Kajuer::Common
{
    using std::abs;
    using std::array;
    using std::cerr;
    using std::clamp;
    using std::count;
    using std::cout;
    using std::endl;
    using std::fill_n;
    using std::function;
    using std::mt19937;
    using std::ofstream;
    using std::pair;
    using std::ref;
    using std::size_t;
    using std::transform;
    using std::uint32_t;
    using std::uint_fast32_t;
    using std::uniform_int_distribution;
    using std::uniform_real_distribution;
    using std::vector;

    using coefficient_array_t = elec_t*;
    using Kajuer::Common::neuron_array_t;
    using Kajuer::Common::neuron_index_array_t;
    using Kajuer::Common::neuron_index_array_for_gen_t;

    //This struct is utlized in Param.h.
    struct ConnectionLayerParameters
    {
        const step_time_t max_delay_msec;
        const vector<elec_t> weight_boundary;
        const uint32_t num_labels;

        const elec_t C_PLUS;
        const elec_t C_MINUS;

        const elec_t ETA;

        const elec_t INV_TAU_PLUS;
        const elec_t INV_TAU_MINUS;

        const elec_t DECAY_INV_TAU_W_0;
        const elec_t DECAY_ALPHA;
        const elec_t DECAY_BETA;
        const elec_t DECAY_W_INH;

        const elec_t initial_weight_lower_bound;
        const elec_t initial_weight_upper_bound;

        const elec_t weight_factor;

        const uint_fast32_t delay_seed_gen;
    };

    //ConnectionLayer defines all synapses between 2 neuron layers.
    class ConnectionLayer
    {
    private:
        //To optimize STDP calculation, we ignore too subtle changes, almost 0,
        //to synaptic weights induced by STDP function.
        //Since we adopt this strategy we can define Lookup table of STDP calculation stored in stdp_table class member.
        constexpr static inline general_index_t MAX_STDP_TIME_SPAN = 100;

        //Initial value of fire(spike) time for simplicity of calculation.
        constexpr static inline step_time_t INITIAL_FIRE_TIME = -99999;

    private:
        //Memory alignment for array.
        const size_t alignment;

    private:
        //This controls number of stocked spike timings.
        const general_index_t num_fire_stocks;

        //This keeps each synapses' prior neuron's spike timings.
        vector<general_index_t> pre_fire_stock_indices;

        //This keeps each synapses' subsequent neuron's spike timings.
        vector<general_index_t> post_fire_stock_indices;

    private:
        //number of synapses which this connection handles.
        general_index_t num_synapses;

        //number of neurons which the neuron layer one layer prior to this connection layer contains.
        general_index_t num_pre_neurons;

        //number of neurons which the neuron layer one layer subsequent to this connection layer contains.
        general_index_t num_post_neurons;

        //max conduction delay of synapses in this layer.
        const step_time_t max_delay_msec;

        //[0] = for lower_bound, [1] = for upper_bound.
        const vector<elec_t> weight_boundary;

        //current indicated label which is utilized in learning phase.
        uint32_t label;

        //number of labels.
        const uint32_t num_labels;

        //input gate depends on label.
        bool gate_with_label;

        //STDP parameter. Strength of STDP modification.
        const elec_t C_PLUS;

        //STDP parameter. Strength of STDP modification.
        const elec_t C_MINUS;

        //learning rate.
        const elec_t ETA;

        //STDP parameter. Inverse of time constant which defines exp function's tail length.
        const elec_t INV_TAU_PLUS;

        //STDP parameter. Inverse of time constant which defines exp function's tail length.
        const elec_t INV_TAU_MINUS;

        //Parameters of supervised STDP learning with weight decay.
        ///[https://ipsj.ixsq.nii.ac.jp/record/215041/files/IPSJ-Z83-4R-05.pdf]
        const elec_t DECAY_INV_TAU_W_0;
        const elec_t DECAY_ALPHA;
        const elec_t DECAY_BETA;
        const elec_t DECAY_W_INH;

        //L1-norm value which is utilized when normalize synaptic weights.
        const elec_t weight_factor;

        //random seed of delay time generator.
        const uint_fast32_t delay_seed_gen;

    private:
        //synaptic weights.
        synapse_array_t w;

        //conduction delays.
        vector<step_time_t> d;

        //last fire(spike) time of pre_neurons.
        vector<vector<step_time_t>> pre_last_fire_time_list;

        //last fire(spike) time of post_neurons.
        vector<vector<step_time_t>> post_last_fire_time_list;

    private:
        //temporary buffer for updating ConnectionLayer's w.
        //This struct is utilized for vector calculation using oneMKL.
        struct update_parameter_list_t
        {
            synapse_array_t w;
            coefficient_array_t f;
            real_time_array_t tau;

            synapse_mask_index_array_t target_synapses;
            step_time_t* diff_time;

            real_time_array_t buffer0;
            real_time_array_t buffer1;
            real_time_array_t buffer2;
            real_time_array_t buffer3;
        };

        //Temporary buffer for long-time-potentiation(LTP) calculation.
        update_parameter_list_t ltp_params;

        //Temporary buffer for long-time-depression(LTD) calculation.
        update_parameter_list_t ltd_params;

    private:
        //Lookup table for stdp calculation. Time range is (-MAX_STDP_TIME_SPAN, MAX_STDP_TIME_SPAN).
        array<elec_t,2*MAX_STDP_TIME_SPAN> stdp_table;

    private:
        //If we call this function, then it returns the indeices of fired(spiked) prior neurons.
        function<const neuron_mask_index_array_t&(void)> get_pre_is_fired;

        //If we call this function, then it returns the indeices of fired(spiked) subsequent neurons.
        function<const neuron_mask_index_array_t&(void)> get_post_is_fired;

    private:

        //update ltp_params.
        void calc_ltp(void)
        {
            general_index_t update_target_size{static_cast<general_index_t>(ltp_params.target_synapses.size())};

            calc_stdp(ltp_params, update_target_size);
        }

        //update ltd_params.
        void calc_ltd(void)
        {
            general_index_t update_target_size{static_cast<general_index_t>(ltd_params.target_synapses.size())};

            calc_stdp(ltd_params, update_target_size);
        }

        //Apply stdp function to update synaptic weights.
        void calc_stdp(const update_parameter_list_t& params, const general_index_t update_target_size)
        {
            const int inc_x{1};
            const int inc_y{1};

            if(0 >= update_target_size)
            {
                return;
            }

            transform
            (
                std::execution::unseq,
                &params.diff_time[0],
                &params.diff_time[update_target_size],
                &params.buffer3[0],
                [this, &params](step_time_t dt)
                {
                    elec_t dw{0.0};

                    if((-MAX_STDP_TIME_SPAN<dt)&&(MAX_STDP_TIME_SPAN>dt))
                    {
                        dw = stdp_table[MAX_STDP_TIME_SPAN+dt];
                    }
                    return dw;
                }
            );

            vdAdd(update_target_size, params.w, params.buffer3, params.w);
        }

        void clamp_weight(void)
        {
            const general_index_t target_range{num_synapses};

            if(0 >= target_range)
            {
                return;
            }

            for(general_index_t i=0; i<num_post_neurons; i+=1)
            {
                if(need_skip(i))
                {
                    continue;
                }

                for(general_index_t j=0; j<num_pre_neurons; j+=1)
                {
                    const general_index_t index{j*num_post_neurons+i};

                    w[index] = clamp(w[index], weight_boundary[0], weight_boundary[1]);
                }
            }

        }

        //Gather params for updating ltd_params utilized for calc_ltd().
        void do_gather_ltd_params(const general_index_t y, const step_time_t current_time, general_index_t& serial_index)
        {
            for(general_index_t i=0; i<num_post_neurons; i+=1)
            {
                const general_index_t index{i+num_post_neurons*y};
                const step_time_t delay{d[index]};

                step_time_t dt{0};

                for(general_index_t j=0; j<num_fire_stocks; j+=1)
                {
                    general_index_t fire_stock_index{post_fire_stock_indices[index] - (j+1)};

                    if(0 > fire_stock_index)
                    {
                        fire_stock_index += num_fire_stocks;
                    }

                    const step_time_t post_last_fire_time_candidate
                        {post_last_fire_time_list[index][fire_stock_index]};
                    const step_time_t dt_candidate
                        {((current_time) + delay) - post_last_fire_time_candidate};

                    if(dt_candidate > 0)
                    {
                        dt = dt_candidate;
                        break;
                    }
                }

                if(MAX_STDP_TIME_SPAN > dt)
                {
                    ltd_params.w[serial_index] = w[index];
                    ltd_params.target_synapses.emplace_back(index);
                    ltd_params.diff_time[serial_index] = dt;

                    serial_index += 1;
                }
            }
        }

        //Gather params for updating ltp_params utilized for calc_ltp().
        void do_gather_ltp_params(const general_index_t x, const step_time_t current_time, general_index_t& serial_index)
        {
            for(general_index_t j=0; j<num_pre_neurons; j+=1)
            {
                const general_index_t index{x+num_post_neurons*j};
                const step_time_t delay{d[index]};

                step_time_t dt{0};

                for(general_index_t j=0; j<num_fire_stocks; j+=1)
                {
                    general_index_t fire_stock_index{pre_fire_stock_indices[index] - (j+1)};

                    if(0 > fire_stock_index)
                    {
                        fire_stock_index += num_fire_stocks;
                    }

                    const step_time_t pre_last_fire_time_candidate
                        {pre_last_fire_time_list[index][fire_stock_index]};
                    const step_time_t dt_candidate
                        {pre_last_fire_time_candidate - (current_time - delay)};

                    if(dt_candidate < 0)
                    {
                        dt = dt_candidate;
                        break;
                    }
                }

                if(-MAX_STDP_TIME_SPAN < dt)
                {
                    ltp_params.w[serial_index] = w[index];
                    ltp_params.target_synapses.emplace_back(index);
                    ltp_params.diff_time[serial_index] = dt;

                    serial_index += 1;
                }
            }
        }

        //Apply synaptic current to current_input if prior neuron fires and delay time elapsed.
        void do_reserve_fires
        (
            const step_time_t current_time,
            const general_index_t y,
            vector<input_current_array_t>& current_input
        ) const
        {
            for(general_index_t i=0; i<num_post_neurons; i+=1)
            {
                if(need_skip(i))
                {
                    continue;
                }
                const general_index_t connection_serial_index{i+num_post_neurons*y};
                const step_time_t delay{d[connection_serial_index]};
                const elec_t synaptic_current{w[connection_serial_index]};
                const general_index_t target_slot{(current_time+(delay+1))%max_delay_msec};

                current_input[target_slot][i] += synaptic_current;
            }
        }

        //Update last fire(spike) time list of prior neurons.
        void do_update_pre_last_fire_time_list(const general_index_t y, const step_time_t current_time)
        {
            for(general_index_t i=0; i<num_post_neurons; i+=1)
            {
                const general_index_t connection_serial_index{i+num_post_neurons*y};

                general_index_t current_fire_stock_index{pre_fire_stock_indices[connection_serial_index]};

                pre_last_fire_time_list
                    [connection_serial_index][current_fire_stock_index] = current_time;

                current_fire_stock_index += 1;
                if(num_fire_stocks <= current_fire_stock_index)
                {
                    current_fire_stock_index = 0;
                }

                pre_fire_stock_indices[connection_serial_index] = current_fire_stock_index;
            }
        }

        //Update last fire(spike) time list of subsequent neurons.
        void do_update_post_last_fire_time_list(const general_index_t x, const step_time_t current_time)
        {
            for(general_index_t j=0; j<num_pre_neurons; j+=1)
            {
                const general_index_t connection_serial_index{x+num_post_neurons*j};

                general_index_t current_fire_stock_index{post_fire_stock_indices[connection_serial_index]};

                post_last_fire_time_list
                    [connection_serial_index][current_fire_stock_index] = current_time;

                current_fire_stock_index += 1;
                if(num_fire_stocks <= current_fire_stock_index)
                {
                    current_fire_stock_index = 0;
                }

                post_fire_stock_indices[connection_serial_index] = current_fire_stock_index;
            }
        }

        //weight decay part of supervised STDP learning with weight decay.
        ///[https://ipsj.ixsq.nii.ac.jp/record/215041/files/IPSJ-Z83-4R-05.pdf]
        void decay_weight(const update_parameter_list_t& params)
        {

            const int inc_x{1};
            const int inc_y{1};

            general_index_t update_target_size{static_cast<general_index_t>(params.target_synapses.size())};

            if(0 >= update_target_size)
            {
                return;
            }

            fill_n(&params.buffer0[0], update_target_size, DECAY_W_INH);
            fill_n(&params.buffer1[0], update_target_size, DECAY_BETA);
            fill_n(&params.buffer2[0], update_target_size, DECAY_ALPHA);

            vdSub(update_target_size, params.buffer0, params.w, params.buffer0);

            vdAdd(update_target_size, params.w, params.buffer1, params.buffer1);
            vdMul(update_target_size, params.buffer1, params.buffer2, params.buffer2);
            vdExp(update_target_size, params.buffer2, params.buffer2);
            vdMul(update_target_size, params.buffer2, params.buffer0, params.buffer2);

            daxpy(&update_target_size, &DECAY_INV_TAU_W_0, params.buffer2, &inc_x, params.w, &inc_y);
        }

        //free up buffers of params.
        void free_update_parameter_list(update_parameter_list_t& params)
        {
            mkl_free(params.w);
            mkl_free(params.diff_time);
            mkl_free(params.buffer0);
            mkl_free(params.buffer1);
            mkl_free(params.buffer2);

            mkl_free(params.buffer3);
        }

        //Gather params for updating ltd_params utilized for calc_ltd().
        void gather_ltd_params(const step_time_t current_time)
        {
            const neuron_mask_index_array_t& pre_is_fired{get_pre_is_fired()};

            const general_index_t is_fired_size{static_cast<general_index_t>(pre_is_fired.size())};

            general_index_t serial_index{0};

            for(general_index_t j=0; j<is_fired_size; j+=1)
            {
                general_index_t y{pre_is_fired[j]};

                do_gather_ltd_params(y, current_time, serial_index);
            }
        }

        //Gather params for updating ltp_params utilized for calc_ltp().
        void gather_ltp_params(const step_time_t current_time)
        {
            const neuron_mask_index_array_t& post_is_fired{get_post_is_fired()};

            const general_index_t is_fired_size{static_cast<general_index_t>(post_is_fired.size())};

            general_index_t serial_index{0};

            for(general_index_t i=0; i<is_fired_size; i+=1)
            {
                general_index_t x{post_is_fired[i]};

                do_gather_ltp_params(x, current_time, serial_index);
            }
        }

        //Initialize delay parameter.
        void initialize_d(void)
        {
            mt19937 engine(delay_seed_gen);

            uniform_int_distribution<step_time_t> dist
            (
                0,
                max_delay_msec-1
            );

            for(general_index_t i=0; i<num_synapses; i+=1)
            {
                d[i] = dist(engine);
            }
        }

        //Initialize ConnectionLayer parameters.
        void initialize_root_parameter_list(const synapse_array_t src_w, synapse_array_t dst_w)
        {
            initialize_w(src_w, dst_w);
            initialize_last_fire_time_list();
            initialize_d();
        }

        //Initialize stdp variable lookup table.
        void initialize_stdp_table(void)
        {
            const general_index_t array_offset{MAX_STDP_TIME_SPAN};

            stdp_table[0] = 0;

            for(general_index_t i=0; i<array_offset; i+=1)
            {
                if(0==i)
                {
                    stdp_table[array_offset] = 0;
                }
                else
                {
                     const elec_t x0{static_cast<elec_t>(+i) * INV_TAU_PLUS};
                     const elec_t x1{static_cast<elec_t>(+i) * INV_TAU_MINUS};

                     stdp_table[array_offset+i] = ETA * (C_PLUS * exp(x0));
                     stdp_table[array_offset-i] = ETA * (-C_MINUS * exp(x1));
                }
            }
        }

        //Initialize dst_w with src_w.
        void initialize_w(const synapse_array_t src_w, synapse_array_t dst_w)
        {
            for(general_index_t i=0; i<num_synapses; i+=1)
            {
                //Here we don't use dcopy since possibly src_w is unaligned.
                dst_w[i] = src_w[i];
            }
        }

        //Allocate root parameters' dynamic memory with mkl_malloc() since we use oneMKL functions.
        void allocate_root_parameter_list(const general_index_t num_synapses)
        {
            w = static_cast<synapse_array_t>(mkl_malloc(sizeof(elec_t) * num_synapses, alignment));
            d = vector<step_time_t>(num_synapses, 0);
            pre_last_fire_time_list =
                vector<vector<step_time_t>>(num_synapses, vector<step_time_t>(num_fire_stocks, INITIAL_FIRE_TIME));
            pre_fire_stock_indices = vector<general_index_t>(num_synapses, 0);
            post_last_fire_time_list =
                vector<vector<step_time_t>>(num_synapses, vector<step_time_t>(num_fire_stocks, INITIAL_FIRE_TIME));
            post_fire_stock_indices = vector<general_index_t>(num_synapses, 0);
        }

        //Allocate params' dynamic memory with mkl_malloc() since we use oneMKL functions.
        void allocate_update_parameter_list(update_parameter_list_t& params, const general_index_t num_synapses)
        {
            params.w = static_cast<synapse_array_t>(mkl_malloc(sizeof(elec_t) * num_synapses, alignment));
            params.diff_time = static_cast<step_time_t*>(mkl_malloc(sizeof(step_time_t) * num_synapses, alignment));
            params.buffer0 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * num_synapses, alignment));
            params.buffer1 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * num_synapses, alignment));
            params.buffer2 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * num_synapses, alignment));

            params.buffer3 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * num_synapses, alignment));
        }

        //judge if we can skip update synaptic weights.
        bool need_skip(const general_index_t i) const
        {
            bool ret_val{false};

            if(gate_with_label)
            {
                if((i*num_labels/num_post_neurons) != label)
                {
                    ret_val = true;
                }
            }

            return ret_val;
        }

        //Normalizing weights with L1-norm and weight decay.
        void normalize_weight(const step_time_t current_time)
        {
            const neuron_mask_index_array_t& pre_is_fired{get_pre_is_fired()};
            const general_index_t is_pre_fired_size{static_cast<general_index_t>(pre_is_fired.size())};

            const neuron_mask_index_array_t& post_is_fired{get_post_is_fired()};
            const general_index_t is_post_fired_size{static_cast<general_index_t>(post_is_fired.size())};

            clamp_weight();

            for(general_index_t i=0; i<num_post_neurons; i+=1)
            {
                if(need_skip(i))
                {
                    continue;
                }

                elec_t sum{0.0};

                for(general_index_t j=0; j<num_pre_neurons; j+=1)
                {
                    const general_index_t index{j*num_post_neurons+i};

                    if(w[index]<0.0)
                    {
                        sum -= w[index];
                    }
                    else
                    {
                        sum += w[index];
                    }
                }

                if(0.0 == sum)
                {
                    sum = 1.0;
                }

                for(general_index_t j=0; j<num_pre_neurons; j+=1)
                {
                    const general_index_t index{j*num_post_neurons+i};

                    w[index] *= weight_factor / sum;
                }
            }


            for(general_index_t i=0; i<num_post_neurons; i+=1)
            {
                if(need_skip(i))
                {
                    continue;
                }

                for(general_index_t j=0; j<num_pre_neurons; j+=1)
                {
                    const general_index_t index{j*num_post_neurons+i};

                    if(0 < count(&post_is_fired[0], &post_is_fired[is_post_fired_size], i))
                    {
                        const elec_t weight{w[index]};

                        elec_t dw = (-weight + DECAY_W_INH) * (exp(DECAY_ALPHA * (weight + DECAY_BETA))) * DECAY_INV_TAU_W_0;

                        if(-1.0 > dw)
                        {
                            dw = -1.0;
                        }
                        else if(1.0 <= dw)
                        {
                            dw = 1.0;
                        }
                        w[index] += dw;
                    }
                }
            }

            clamp_weight();
        }

        //after calc_ltp()/calc_ltd() we restore params' weights to w.
        void restore_params(const update_parameter_list_t& params)
        {
            const general_index_t target_synapses_size{static_cast<general_index_t>(params.target_synapses.size())};

            for(general_index_t i=0; i<target_synapses_size; i+=1)
            {
                w[params.target_synapses[i]] = params.w[i];
            }
        }

    private:
        //Utilized for copy constructor.
        void copy(const ConnectionLayer& src)
        {
            w = static_cast<synapse_array_t>(mkl_malloc(sizeof(elec_t) * src.num_synapses, src.alignment));

            ltp_params.w = static_cast<synapse_array_t>(mkl_malloc(sizeof(elec_t) * src.num_synapses, src.alignment));
            ltp_params.diff_time = static_cast<step_time_t*>(mkl_malloc(sizeof(step_time_t) * src.num_synapses, src.alignment));
            ltp_params.buffer0 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * src.num_synapses, src.alignment));
            ltp_params.buffer1 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * src.num_synapses, src.alignment));
            ltp_params.buffer2 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * src.num_synapses, src.alignment));
            ltp_params.buffer3 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * src.num_synapses, src.alignment));

            for(general_index_t j=0; j<src.num_synapses; j+=1)
            {
                ltp_params.w[j] = src.ltp_params.w[j];
                ltp_params.diff_time[j] = src.ltp_params.diff_time[j];
                ltp_params.buffer0[j] = src.ltp_params.buffer0[j];
                ltp_params.buffer1[j] = src.ltp_params.buffer1[j];
                ltp_params.buffer2[j] = src.ltp_params.buffer2[j];
                ltp_params.buffer3[j] = src.ltp_params.buffer3[j];
            }

            ltd_params.w = static_cast<synapse_array_t>(mkl_malloc(sizeof(elec_t) * src.num_synapses, src.alignment));
            ltd_params.diff_time = static_cast<step_time_t*>(mkl_malloc(sizeof(step_time_t) * src.num_synapses, src.alignment));
            ltd_params.buffer0 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * src.num_synapses, src.alignment));
            ltd_params.buffer1 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * src.num_synapses, src.alignment));
            ltd_params.buffer2 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * src.num_synapses, src.alignment));
            ltd_params.buffer3 = static_cast<real_time_array_t>(mkl_malloc(sizeof(real_time_t) * src.num_synapses, src.alignment));

            for(general_index_t j=0; j<src.num_synapses; j+=1)
            {
                ltd_params.w[j] = src.ltd_params.w[j];
                ltd_params.diff_time[j] = src.ltd_params.diff_time[j];
                ltd_params.buffer0[j] = src.ltd_params.buffer0[j];
                ltd_params.buffer1[j] = src.ltd_params.buffer1[j];
                ltd_params.buffer2[j] = src.ltd_params.buffer2[j];
                ltd_params.buffer3[j] = src.ltd_params.buffer3[j];
            }
        }

    public:
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

        //generate synaptic connections corresponding to connection_list.
        void generate_connection
        (
            const neuron_pair_array_t& connection_list,
            const general_index_t arg_num_pre_neurons,
            const general_index_t arg_num_post_neurons,
            const synapse_array_t& arg_w
        )
        {
            cerr << arg_num_pre_neurons << ' ' << arg_num_post_neurons << endl;

            const general_index_t tmp_num_pre_neurons{arg_num_pre_neurons};
            const general_index_t tmp_num_post_neurons{arg_num_post_neurons};
            const general_index_t tmp_num_connections{static_cast<general_index_t>(connection_list.size())};

            num_synapses = tmp_num_connections;
            num_pre_neurons = tmp_num_pre_neurons;
            num_post_neurons = tmp_num_post_neurons;

            allocate_root_parameter_list(tmp_num_connections);
            initialize_root_parameter_list(arg_w, w);

            allocate_update_parameter_list(ltp_params, tmp_num_connections);
            allocate_update_parameter_list(ltd_params, tmp_num_connections);
        }

        //Dump connections.
        void get_connections
        (
            synapse_array_t& arg_synapse_array,
            general_index_t& arg_num_connections
        ) const
        {
            arg_synapse_array = ref(w);
            arg_num_connections = num_synapses;
        }

        //Force update connections.
        void set_connections
        (
            const synapse_array_t& arg_synapse_array,
            const general_index_t& arg_num_connections
        )
        {
            for(general_index_t i=0; i<arg_num_connections; i+=1)
            {
                w[i] = arg_synapse_array[i];
            }
            num_synapses = arg_num_connections;
        }

        //Initialize last_fire_time_list with big negative value.
        void initialize_last_fire_time_list(void)
        {
            general_index_t update_target_size{num_synapses};

            for(general_index_t i=0; i<num_synapses; i+=1)
            {
                fill_n(pre_last_fire_time_list[i].begin(), num_fire_stocks, INITIAL_FIRE_TIME);
                fill_n(post_last_fire_time_list[i].begin(), num_fire_stocks, INITIAL_FIRE_TIME);
            }
        }

        //Store synaptic current to current_input.
        //Prerequisits: current_input is already allocated as vector before entering this function.
        void reserve_fires
        (
            const step_time_t current_time,
            vector<input_current_array_t>& current_input
        ) const
        {
            const neuron_mask_index_array_t& pre_is_fired{get_pre_is_fired()};
            const general_index_t is_fired_size{static_cast<general_index_t>(pre_is_fired.size())};

            for(general_index_t j=0; j<is_fired_size; j+=1)
            {
                general_index_t y{pre_is_fired[j]};

                do_reserve_fires(current_time, y, current_input);
            }
        }

        //switching label.
        void switch_label(const uint32_t arg_label)
        {
            label = arg_label;
        }

        //Updating w.
        void update_connections(const step_time_t current_time, const bool need_normalize=false)
        {
            update_pre_connections(current_time);

            update_post_connections(current_time);

            if(need_normalize)
            {
                normalize_weight(current_time);

                clamp_weight();
            }

            update_post_last_fire_time_list(current_time);
            update_pre_last_fire_time_list(current_time);
        }

        //Updating w with LTP.
        void update_pre_connections(const step_time_t current_time)
        {
            ltp_params.target_synapses.clear();

            gather_ltp_params(current_time);

            calc_ltp();

            restore_params(ltp_params);
        }

        //Updating pre_is_fired function.
        void update_pre_is_fired(function<const neuron_mask_index_array_t&(void)> arg_get_pre_is_fired)
        {
            get_pre_is_fired = arg_get_pre_is_fired;
        }

        //Updating pre_last_fire_time_list with obtained fire(spike) information.
        void update_pre_last_fire_time_list(const step_time_t current_time)
        {
            const neuron_mask_index_array_t& pre_is_fired{get_pre_is_fired()};

            const general_index_t is_fired_size{static_cast<general_index_t>(pre_is_fired.size())};

            for(general_index_t j=0; j<is_fired_size; j+=1)
            {
                general_index_t y{pre_is_fired[j]};

                do_update_pre_last_fire_time_list(y, current_time);
            }
        }

        //Updating w with LTD.
        void update_post_connections(const step_time_t current_time)
        {
            ltd_params.target_synapses.clear();

            gather_ltd_params(current_time);

            calc_ltd();

            restore_params(ltd_params);
        }

        //Updating post_is_fired function.
        void update_post_is_fired(function<const neuron_mask_index_array_t&(void)> arg_get_post_is_fired)
        {
            get_post_is_fired = arg_get_post_is_fired;
        }

        //Updating post_last_fire_time_list with obtained fire(spike) information.
        void update_post_last_fire_time_list(step_time_t const current_time)
        {
            const neuron_mask_index_array_t& post_is_fired{get_post_is_fired()};

            const general_index_t is_fired_size{static_cast<general_index_t>(post_is_fired.size())};

            for(general_index_t i=0; i<is_fired_size; i+=1)
            {
                general_index_t x{post_is_fired[i]};

                do_update_post_last_fire_time_list(x, current_time);
            }
        }

    public:

        explicit ConnectionLayer
        (
            const ConnectionLayerParameters& connection_params,
            const size_t arg_alignment=64
        ) :
            num_fire_stocks(connection_params.max_delay_msec), //max_delay_msec is enough but seems too much with my experience.
            alignment(arg_alignment),
            max_delay_msec(connection_params.max_delay_msec),
            weight_boundary(connection_params.weight_boundary),
            label(0),
            num_labels(connection_params.num_labels),
            gate_with_label(true),
            C_PLUS(connection_params.C_PLUS),
            C_MINUS(connection_params.C_MINUS),
            ETA(connection_params.ETA),
            INV_TAU_PLUS(connection_params.INV_TAU_PLUS),
            INV_TAU_MINUS(connection_params.INV_TAU_MINUS),
            DECAY_INV_TAU_W_0(connection_params.DECAY_INV_TAU_W_0),
            DECAY_ALPHA(connection_params.DECAY_ALPHA),
            DECAY_BETA(connection_params.DECAY_BETA),
            DECAY_W_INH(connection_params.DECAY_W_INH),
            weight_factor(connection_params.weight_factor),
            delay_seed_gen(connection_params.delay_seed_gen)
        {
            initialize_stdp_table();
        }

        ConnectionLayer(const ConnectionLayer& src) :
            num_fire_stocks(src.num_fire_stocks),
            pre_fire_stock_indices(src.pre_fire_stock_indices),
            post_fire_stock_indices(src.post_fire_stock_indices),
            alignment(src.alignment),
            num_synapses(src.num_synapses),
            num_pre_neurons(src.num_pre_neurons),
            num_post_neurons(src.num_post_neurons),
            max_delay_msec(src.max_delay_msec),
            weight_boundary(src.weight_boundary),
            label(src.label),
            num_labels(src.num_labels),
            gate_with_label(src.gate_with_label),
            C_PLUS(src.C_PLUS),
            C_MINUS(src.C_MINUS),
            ETA(src.ETA),
            INV_TAU_PLUS(src.INV_TAU_PLUS),
            INV_TAU_MINUS(src.INV_TAU_MINUS),
            DECAY_INV_TAU_W_0(src.DECAY_INV_TAU_W_0),
            DECAY_ALPHA(src.DECAY_ALPHA),
            DECAY_BETA(src.DECAY_BETA),
            DECAY_W_INH(src.DECAY_W_INH),
            weight_factor(src.weight_factor),
            delay_seed_gen(src.delay_seed_gen),
            get_pre_is_fired(src.get_pre_is_fired),
            get_post_is_fired(src.get_post_is_fired)
        {
            copy(src);
        }

        ConnectionLayer& operator= (const ConnectionLayer& src) = delete;
        ConnectionLayer(ConnectionLayer&&) = delete;
        ConnectionLayer& operator=(ConnectionLayer&&) = delete;

        virtual ~ConnectionLayer(void)
        {
            mkl_free(w);

            free_update_parameter_list(ltp_params);
            free_update_parameter_list(ltd_params);
        }
    };
};
