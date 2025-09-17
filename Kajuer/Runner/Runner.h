#pragma once

#include <mkl.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include "CommonNet/Common.h"
#include "CommonNet/ConnectionGenerator.h"
#include "CommonNet/ConnectionLayer.h"
#include "CommonNet/NeuronLayer.h"
#include "Param/Param.h"
#include "Sensor/VisualSensor.h"

namespace Kajuer::Runner
{
    using std::array;
    using std::async;
    using std::bind;
    using std::boolalpha;
    using std::cerr;
    using std::cout;
    using std::distance;
    using std::endl;
    using std::fill;
    using std::fill_n;
    using std::flush;
    using std::future;
    using std::getline;
    using std::ifstream;
    using std::ignore;
    using std::istringstream;
    using std::launch;
    using std::lock_guard;
    using std::make_pair;
    using std::mt19937;
    using std::mutex;
    using std::ofstream;
    using std::pair;
    using std::ref;
    using std::size_t;
    using std::stoull;
    using std::string;
    using std::to_string;
    using std::uniform_real_distribution;
    using std::unique_ptr;
    using std::uint_fast32_t;
    using std::uint32_t;
    using std::vector;

    using Kajuer::Common::elec_t;
    using Kajuer::Common::general_index_t;
    using Kajuer::Common::input_current_array_t;
    using Kajuer::Common::neuron_mask_array_t;
    using Kajuer::Common::neuron_mask_index_array_t;
    using Kajuer::Common::sensory_input_t;
    using Kajuer::Common::step_time_t;
    using Kajuer::Common::synapse_array_t;
    using Kajuer::Common::ConnectionGenerator;
    using Kajuer::Common::ConnectionLayer;
    using Kajuer::Common::ConnectionLayerParameters;
    using Kajuer::Common::NeuronLayer;
    using Kajuer::Common::NeuronLayerParameters;

    using Kajuer::Param::ParamList;

    using Kajuer::Sensor::VisualSensor;

    //save connection data to binary file.
    template<class T>
    static void save_data_to_bin_file
    (
        const string& file_name,
        const general_index_t num_connections,
        T& data,
        const ofstream::openmode open_mode=ofstream::binary
    )
    {
        ofstream fout(file_name, open_mode);

        fout.write(reinterpret_cast<char*>(&data[0]), num_connections*sizeof(data[0]));

        fout << flush;
    }

    //load connection data from binary file.
    template<class T>
    static void load_data_from_bin_file
    (
        const string& file_name,
        const general_index_t num_connections,
        T& data,
        const ifstream::openmode open_mode=ifstream::binary
    )
    {
        ifstream fin(file_name, open_mode);

        fin.read(reinterpret_cast<char*>(&data[0]), num_connections*sizeof(data[0]));
    }

    //This class coordinates and runs simulations.
    class Runner
    {
    private:
        //memory alignment.
        static inline constexpr size_t alignment{64};

        //number of data sets.
        static inline constexpr general_index_t num_data_sets{2};

        //number of nueron layers.
        static inline constexpr general_index_t num_neuron_layer{3};

        //number of connection layers.
        static inline constexpr general_index_t num_connection_layer{3};

        //number of visual sensor inputs.(=2 means for training and validation).
        static inline constexpr general_index_t num_visual_sensor{2};

        //side of 28x28 MNIST digit.
        static inline constexpr general_index_t length_visual_sensor{28};

        //flag of debug print enabled.
        const bool is_debug_print_enabled;

        //Entity of parameter list.
        unique_ptr<ParamList> param_list;

        //Entity of neuron layers.
        array<array<NeuronLayer, num_neuron_layer>, num_data_sets> neuron_layer_array;

        //Entity of connection layers.
        array<array<ConnectionLayer, num_connection_layer>, num_data_sets> connection_layer_array;

        //pairs of neuron layer indices.
        //e.g. src_neuron_layer:1 dst_neuron_layer:2 then connection_list contains (1,2).
        const vector<pair<general_index_t, general_index_t>> connection_list;

        //Entity of visual sensors.
        array<VisualSensor, num_visual_sensor> visual_sensor;

        //number of pixels for each visual input.(=28x28 for MNIST digits)
        general_index_t visual_input_size;

        //current visual input.
        array<sensory_input_t, num_data_sets> visual_input;

        //copy of visual input.
        array<sensory_input_t, num_data_sets> visual_input_copy;

        //label of MNIST digit.
        array<uint32_t, num_data_sets> label;

        //count up fires(spikes) for specific step of time.
        vector<general_index_t> fire_counter;

        //list of synaptic currents which will be passed to next neuron layer.
        array<vector<vector<input_current_array_t>>, num_data_sets> current_input_list;

        //random number generator for generating network structure.
        array<mt19937, num_data_sets> engine;

        //external current input.
        array<input_current_array_t, num_data_sets> external_input;

    private:

        //aggregates fire counts of specific step(current_time).
        void aggregate_fire_counts(void)
        {
            const neuron_mask_index_array_t& is_fired{neuron_layer_array[1][1].get_is_fired()};
            const general_index_t is_fired_size{static_cast<general_index_t>(is_fired.size())};

            for(general_index_t i=0; i<is_fired_size; i+=1)
            {
                fire_counter[is_fired[i]] += 1;
            }
        }

        //record spike history to file.
        void record_spikes
        (
            const string record_type,
            const step_time_t current_time,
            const general_index_t index,
            const bool is_record_enabled=false
        ) const
        {
            if(not is_record_enabled)
            {
                return;
            }

            const general_index_t target_neuron_layer{1};

            const neuron_mask_index_array_t& is_fired{neuron_layer_array[index][target_neuron_layer].get_is_fired()};
            const general_index_t is_fired_size{static_cast<general_index_t>(is_fired.size())};

            ofstream fout
            (
                "spike_records/" + record_type + "_spike_record.txt." + to_string(current_time/1000),
                ofstream::out|ofstream::app
            );

            for(general_index_t i=0; i<is_fired_size; i+=1)
            {
                fout << current_time << ' ' << is_fired[i] << endl;
            }
        }

        //clear neuron layer state variables(except for u) when time condition is satisfied.
        void clear_neuron_layer_buffers_if_necessary(const step_time_t t, const general_index_t index)
        {
            bool time_condition{0==(t%(param_list->num_iterations_for_each_image[index]))};

            if(time_condition)
            {
                for(NeuronLayer& neuron_layer : neuron_layer_array[index])
                {
                    neuron_layer.initialize_buffers();
                }
            }
        }

        //clear neuron layer state variables when time condition is satisfied.
        void clear_neuron_layer_parameters_if_necessary(const step_time_t t, const general_index_t index)
        {
            const bool time_condition{0==(t%(param_list->num_iterations_for_each_image[index]))};
            const general_index_t randomize_target_neuron_layer{1};

            if(time_condition)
            {
                for(general_index_t i=0; NeuronLayer& neuron_layer : neuron_layer_array[index])
                {
                    bool need_randomize{false};

                    if((0==index) && (randomize_target_neuron_layer==i))
                    {
                        need_randomize = true;
                    }

                    neuron_layer.initialize_parameters(t, need_randomize);
                    i += 1;
                }
            }
        }

        //clear whole current input list.
        void clear_current_input_list(const general_index_t i, const general_index_t index)
        {
            const step_time_t max_delay_msec{param_list->connection_layer_params[i].max_delay_msec};

            for(step_time_t j=0; j<max_delay_msec; j+=1)
            {
                const general_index_t num_neurons{param_list->neuron_layer_params[1].num_neurons};

                for(general_index_t k=0; k<num_neurons; k+=1)
                {
                    current_input_list[index][i][j][k] = 0.0;
                }
            }
        }

        //clear whole current input list if time condition or force_clear condition met.
        void clear_current_input_list_if_necessary(const step_time_t t, const general_index_t index, const bool force_clear=false)
        {
            bool time_condition{0==(t%(param_list->num_iterations_for_each_image[index]))};

            if(time_condition || force_clear)
            {
                for(general_index_t i=0; i<num_neuron_layer; i+=1)
                {
                    const step_time_t max_delay_msec{param_list->connection_layer_params[i].max_delay_msec};

                    for(step_time_t j=0; j<max_delay_msec; j+=1)
                    {
                        const general_index_t num_neurons{param_list->neuron_layer_params[1].num_neurons};

                        for(general_index_t k=0; k<num_neurons; k+=1)
                        {
                            current_input_list[index][i][j][k] = 0.0;
                        }
                    }
                }
            }
        }

        //clear the current input of step_time t.
        void clear_each_current_input_list(const step_time_t t, const general_index_t index)
        {
            for(general_index_t i=0; i<num_neuron_layer; i+=1)
            {
                const step_time_t max_delay_msec{param_list->connection_layer_params[i].max_delay_msec};

                const step_time_t j{t%max_delay_msec};

                const general_index_t num_neurons{param_list->neuron_layer_params[1].num_neurons};

                for(general_index_t k=0; k<num_neurons; k+=1)
                {
                    current_input_list[index][i][j][k] = 0.0;
                }
            }
        }

        //clear external input list.
        void clear_external_input_list(const general_index_t index)
        {
            fill_n(&external_input[index][0], param_list->neuron_layer_params[1].num_neurons, 0.0);
        }

        //clear fire counter if time condition met.
        void clear_fire_counter_if_necessary(const step_time_t t, const general_index_t index)
        {
            bool time_condition{0==(t%(param_list->num_iterations_for_each_image[index]))};

            if(time_condition)
            {
                fill(begin(fire_counter), end(fire_counter), 0);
            }
        }

        //clear last fire time list if time condition met.
        void clear_last_fire_time_list_if_necessary(const step_time_t t, const general_index_t index)
        {
            bool time_condition{0==(t%(param_list->num_iterations_for_each_image[index]))};

            if(time_condition)
            {
                for(ConnectionLayer& connection_layer : connection_layer_array[index])
                {
                    connection_layer.initialize_last_fire_time_list();
                }
            }
        }

        //main entity of run training.
        void do_run_training(array<step_time_t, 2>& current_time)
        {
            future<void> future_result;
            bool is_first_time{true};

            bool is_success_generating_image{false};

            neuron_layer_array[0][0].disable_label_selector();
            neuron_layer_array[0][1].enable_label_selector();
            neuron_layer_array[0][2].disable_label_selector();

            for(ConnectionLayer& connection_layer : connection_layer_array[0])
            {
                connection_layer.enable_label_selector();
            }

            clear_neuron_layer_parameters_if_necessary(0, 0);

            const step_time_t time_end{param_list->time_end[0]};

            for(step_time_t t=0; t<time_end; t+=1)
            {
                //for debugging purpose.
                if(0==(t%10000))
                {
                    cerr << "time0: " << current_time[0] << endl;
                }

                //for debugging purpose.
                if(0==(t%5000))
                {
                    synapse_array_t synapse_array;
                    general_index_t num_connections;

                    connection_layer_array[0][0].get_connections
                    (
                        synapse_array,
                        num_connections
                    );

                    save_data_to_bin_file
                    (
                        string("weight_map_dir/weight_map.bin.") + to_string(current_time[0]),
                        num_connections,
                        synapse_array
                    );
                }

                const bool time_condition0{t!=0};
                const bool time_condition1{(t%(param_list->num_iterations_for_each_image[0]*1000))==0};

                if(time_condition0 && time_condition1)
                {
                    if(not is_first_time)
                    {
                        future_result.wait();
                    }
                    is_first_time = false;

                    //copy weights.
                    for(general_index_t i=0; i<num_connection_layer; i+=1)
                    {
                        synapse_array_t synapse_array;
                        general_index_t num_connections;
                        connection_layer_array[0][i].get_connections(synapse_array, num_connections);
                        connection_layer_array[1][i].set_connections(synapse_array, num_connections);
                    }

                    visual_sensor[1].reset_data_index();

                    future_result = async(launch::async, &Runner::do_run_validation, this, ref(current_time[1]));
                }

                const array<step_time_t, 3> ranged_current_time =
                {
                    static_cast<step_time_t>(t%(param_list->connection_layer_params[0].max_delay_msec)),
                    static_cast<step_time_t>(t%(param_list->connection_layer_params[1].max_delay_msec)),
                    static_cast<step_time_t>(t%(param_list->connection_layer_params[2].max_delay_msec)),
                };

                clear_neuron_layer_buffers_if_necessary(t, 0);

                is_success_generating_image = generate_new_image_if_necessary(t, 0);
                clear_last_fire_time_list_if_necessary(t, 0);
                clear_current_input_list_if_necessary(t, 0, false);

                //case neuron_layer_array[0].
                {
                    const int inc_x{1};
                    const int inc_y{1};
                    const int array_size{static_cast<int>(visual_input_size)};
                    dcopy(&array_size, visual_input[0], &inc_x, visual_input_copy[0], &inc_y);

                    judge_overthreshold(0);

                    neuron_layer_array[0][0].apply_current_input(visual_input_copy[0]);

                }

                //prepare external input.
                {
                    clear_current_input_list(0, 0);

                    connection_layer_array[0][0].reserve_fires(t, current_input_list[0][0]);

                    clear_external_input_list(0);

                    merge_inputs_to_external_input
                    (
                        &current_input_list[0][0][ranged_current_time[0]][0],
                        &current_input_list[0][2][ranged_current_time[2]][0],
                        0
                    );

                    neuron_layer_array[0][0].update_neurons(is_debug_print_enabled);
                }

                //case neuron_layer_array[1].
                {
                    const input_current_array_t input = external_input[0];

                    neuron_layer_array[0][1].apply_current_input(input);

                    neuron_layer_array[0][1].update_neurons(is_debug_print_enabled);
                }

                //prepare current_input_list[1].
                {
                    clear_current_input_list(1, 0);

                    connection_layer_array[0][1].reserve_fires(t, current_input_list[0][1]);
                }

                //case neuron_layer_array[2].
                {
                    input_current_array_t input = &current_input_list[0][1][ranged_current_time[1]][0];

                    neuron_layer_array[0][2].apply_current_input(input);

                    neuron_layer_array[0][2].update_neurons(is_debug_print_enabled);
                }

                //update connections.
                {
                    connection_layer_array[0][0].update_connections(t, true);
                }

                //prepare current_input_list[2].
                {
                    clear_current_input_list(2, 0);

                    connection_layer_array[0][2].reserve_fires(t, current_input_list[0][2]);
                }

                //for debugging purpose.
                record_spikes("training", current_time[0], 0, true);

                current_time[0] += 1;

                if(not is_success_generating_image)
                {
                    break;
                }
            }

            future_result.wait();
        }

        //main entity of run validation.
        void do_run_validation(step_time_t& current_time)
        {
            bool is_success_generating_image{false};

            neuron_layer_array[1][0].disable_label_selector();
            neuron_layer_array[1][1].disable_label_selector();
            neuron_layer_array[1][2].disable_label_selector();

            for(ConnectionLayer& connection_layer : connection_layer_array[1])
            {
                connection_layer.disable_label_selector();
            }

            const step_time_t time_end{param_list->time_end[1]+1};

            for(step_time_t t=0; t<time_end; t+=1)
            {
                const array<step_time_t,3> ranged_current_time =
                {
                    static_cast<step_time_t>(t%(param_list->connection_layer_params[0].max_delay_msec)),
                    static_cast<step_time_t>(t%(param_list->connection_layer_params[1].max_delay_msec)),
                    static_cast<step_time_t>(t%(param_list->connection_layer_params[2].max_delay_msec)),
                };

                judge_answer_if_necessary(t, 1);
                clear_fire_counter_if_necessary(t, 1);
                clear_neuron_layer_parameters_if_necessary(t, 1);
                clear_last_fire_time_list_if_necessary(t, 1);
                clear_current_input_list_if_necessary(t, 1, false);
                is_success_generating_image = generate_new_image_if_necessary(t, 1);

                //case neuron_layer_array[0].
                {
                    const int inc_x{1};
                    const int inc_y{1};
                    const int array_size{static_cast<int>(visual_input_size)};
                    dcopy(&array_size, visual_input[1], &inc_x, visual_input_copy[1], &inc_y);

                    judge_overthreshold(1);

                    neuron_layer_array[1][0].apply_current_input(visual_input_copy[1]);

                }

                //prepare external input.
                {
                    clear_current_input_list(0, 1);

                    connection_layer_array[1][0].reserve_fires(t, current_input_list[1][0]);

                    clear_external_input_list(1);

                    merge_inputs_to_external_input
                    (
                        &current_input_list[1][0][ranged_current_time[0]][0],
                        &current_input_list[1][2][ranged_current_time[2]][0],
                        1
                    );

                    neuron_layer_array[1][0].update_neurons(is_debug_print_enabled);
                }

                //case neuron_layer_array[1].
                {
                    const input_current_array_t input = external_input[1];

                    neuron_layer_array[1][1].apply_current_input(input);

                    neuron_layer_array[1][1].update_neurons(is_debug_print_enabled);
                }

                //prepare current_input_list[1].
                {
                    clear_current_input_list(1, 1);

                    connection_layer_array[1][1].reserve_fires(t, current_input_list[1][1]);
                }

                //case neuron_layer_array[2].
                {
                    const input_current_array_t input = &current_input_list[1][1][ranged_current_time[1]][0];

                    neuron_layer_array[1][2].apply_current_input(input);

                    neuron_layer_array[1][2].update_neurons(is_debug_print_enabled);
                }

                //prepare current_input_list[2].
                {
                    clear_current_input_list(2, 1);

                    connection_layer_array[1][2].reserve_fires(t, current_input_list[1][2]);
                }

                aggregate_fire_counts();

                record_spikes("validation", current_time, 1, true);

                current_time += 1;

                if(not is_success_generating_image)
                {
                    break;
                }
            }
        }

        //generates connections.
        void generate_connections(void)
        {
            for(general_index_t i=0; i<num_data_sets; i+=1)
            {
                uniform_real_distribution<elec_t> dist
                (
                    param_list->connection_layer_params[0].initial_weight_lower_bound,
                    param_list->connection_layer_params[0].initial_weight_upper_bound
                );
                mt19937 network_gen_engine(param_list->network_seed_gen[0]);

                ConnectionGenerator::generate_full_connection
                (
                    neuron_layer_array[i][0],
                    neuron_layer_array[i][1],
                    connection_layer_array[i][0],
                    param_list->network_seed_gen[0],
                    dist,
                    network_gen_engine
                );
            }

            for(general_index_t i=0; i<num_data_sets; i+=1)
            {
                uniform_real_distribution<elec_t> dist
                (
                    param_list->connection_layer_params[1].initial_weight_lower_bound,
                    param_list->connection_layer_params[1].initial_weight_upper_bound
                );
                mt19937 network_gen_engine(param_list->network_seed_gen[1]);

                ConnectionGenerator::generate_one_by_one_neuron_connection
                (
                    neuron_layer_array[i][1],
                    neuron_layer_array[i][2],
                    connection_layer_array[i][1],
                    dist,
                    network_gen_engine
                );
            }

            for(general_index_t i=0; i<num_data_sets; i+=1)
            {
                uniform_real_distribution<elec_t> dist
                (
                    param_list->connection_layer_params[2].initial_weight_lower_bound,
                    param_list->connection_layer_params[2].initial_weight_upper_bound
                );
                mt19937 network_gen_engine(param_list->network_seed_gen[2]);

                ConnectionGenerator::generate_dst_other_neuron_connection
                (
                    neuron_layer_array[i][2],
                    neuron_layer_array[i][1],
                    connection_layer_array[i][2],
                    dist,
                    network_gen_engine
                );
            }
        }

        //generate new image and set it tp visual_input if time condition met.
        bool generate_new_image_if_necessary(const step_time_t t, const general_index_t index)
        {
            const bool time_condition{0==(t%(param_list->num_iterations_for_each_image[index]))};

            if(time_condition)
            {
                const bool is_image_generated
                {
                    visual_sensor[index].generate_next_visual_input(visual_input[index], visual_input_size, label[index])
                };

                if(not is_image_generated)
                {
                    cerr << "Error: Image generation failed." << endl;
                    return false;
                }

                neuron_layer_array[index][0].switch_label(label[index]);
                neuron_layer_array[index][1].switch_label(label[index]);
                neuron_layer_array[index][2].switch_label(label[index]);

                connection_layer_array[index][0].switch_label(label[index]);
                connection_layer_array[index][1].switch_label(label[index]);
                connection_layer_array[index][2].switch_label(label[index]);
            }
            return true;
        }

        //allocate the contents of current input list vector.
        void allocate_current_input_list(void)
        {
            for(general_index_t index=0; index<num_data_sets; index+=1)
            {
                vector<vector<synapse_array_t>> sub_current_input_list;

                for(general_index_t i=0; i<num_neuron_layer; i+=1)
                {
                    vector<synapse_array_t> each_current_input_list;

                    const step_time_t max_delay_msec{param_list->connection_layer_params[i].max_delay_msec};

                    for(step_time_t j=0; j<max_delay_msec; j+=1)
                    {
                        input_current_array_t buffer
                        {
                            static_cast<input_current_array_t>
                            (
                                mkl_malloc(sizeof(elec_t) * param_list->neuron_layer_params[1].num_neurons, alignment)
                            )
                        };

                        each_current_input_list.emplace_back(buffer);
                    }

                    sub_current_input_list.emplace_back(each_current_input_list);
                }

                current_input_list[index] = sub_current_input_list;
            }
        }

        //allocate the external input.
        void allocate_external_input(void)
        {
            external_input =
            {
                static_cast<input_current_array_t>
                (
                    mkl_malloc(sizeof(elec_t) * param_list->neuron_layer_params[1].num_neurons, alignment)
                ),
                static_cast<input_current_array_t>
                (
                    mkl_malloc(sizeof(elec_t) * param_list->neuron_layer_params[1].num_neurons, alignment)
                ),
            };
        }

        //initialize visual sensor.
        void initialize_visual_sensor(void)
        {
            visual_sensor[0].load_mnist
            (
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                static_cast<uint32_t>(param_list->num_images[0])
            );

            visual_sensor[1].load_mnist
            (
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte",
                static_cast<uint32_t>(param_list->num_images[1])
            );
        }

        //compare correct label and answer label given by trained output neurons if time condition met.
        void judge_answer_if_necessary(const step_time_t t, const general_index_t index, const bool is_debug_enabled=false)
        {
            const bool time_condition0{0!=t};
            const bool time_condition1{0==(t%(param_list->num_iterations_for_each_image[index]))};

            if(time_condition0 && time_condition1)
            {
                const uint32_t old_label{label[index]};

                const general_index_t num_neurons{param_list->neuron_layer_params[1].num_neurons};
                const uint32_t num_labels{param_list->num_labels};

                if(is_debug_enabled)
                {
                    for(general_index_t i=0; i<num_neurons; i+=1)
                    {
                        if(0 == (i%10)) cout << "answer: ";
                        if(0 != (i%10)) cout << ' ';
                        cout << fire_counter[i];
                        if(9==(i%10)) cout << endl;
                    }
                }

                vector<general_index_t> fire_sum(num_labels, 0);

                for(general_index_t i=0; general_index_t fire_count : fire_counter)
                {
                    fire_sum[i/((param_list->neuron_layer_params[1].num_neurons)/(param_list->num_labels))] += fire_count;
                    i += 1;
                }

                const auto answer_label
                {
                    distance
                    (
                        begin(fire_sum),
                        max_element(begin(fire_sum), end(fire_sum))
                    )
                };

                ofstream fout("fire_record.txt", ofstream::out|ofstream::app);
                fout << "label_answer: " << answer_label << ' ' << old_label << ' '
                     << boolalpha << (static_cast<uint32_t>(answer_label) == old_label) << endl;
            }
        }

        //alightly modify visual input with which we intend to get spike traces.
        //assumption: value range of visual_input is [0.0, 1.0].
        void judge_overthreshold(const general_index_t index)
        {
            const elec_t bias{param_list->input_bias};
            const elec_t lower_bound{0.0+bias};
            const elec_t upper_bound{1.0+bias};

            uniform_real_distribution<elec_t> dist(lower_bound, upper_bound);

            for(general_index_t i=0; i<visual_input_size; i+=1)
            {
                const elec_t threshold{dist(engine[index])};

                if(visual_input_copy[index][i] <= threshold)
                {
                    visual_input_copy[index][i] = 0.0;
                }
            }
        }

        //mer ge buffer0 and buffer1 inputs into single external input.
        void merge_inputs_to_external_input(input_current_array_t buffer0, input_current_array_t buffer1, const general_index_t index)
        {
            const general_index_t num_neurons{param_list->neuron_layer_params[1].num_neurons};
            const double a0{2000.0}; //Constant gain.
            const int inc_x{1};

            dscal(&num_neurons, &a0, buffer1, &inc_x);
            vdSub(param_list->neuron_layer_params[1].num_neurons, buffer0, buffer1, external_input[index]);
        }

        //update is_fired dunction for each connection layer.
        void update_is_fired_for_connection_layer(void)
        {
            for(general_index_t j=0; j<num_data_sets; j+=1)
            {
                for(general_index_t i=0; const pair<general_index_t, general_index_t>& connection : connection_list)
                {
                    connection_layer_array[j][i].update_pre_is_fired
                    (
                        bind(&NeuronLayer::get_is_fired, &neuron_layer_array[j][connection.first])
                    );
                    connection_layer_array[j][i].update_post_is_fired
                    (
                        bind(&NeuronLayer::get_is_fired, &neuron_layer_array[j][connection.second])
                    );
                    i += 1;
                }
            }
        }

    public:
        //run simulation.
        void run(int argc, char *argv[])
        {
            cout << "Simulation begins." << endl;

            ignore = argc;
            ignore = argv;

            array<step_time_t, 2> current_time{0, 0,};
            const general_index_t num_epochs{param_list->num_epochs};
            const general_index_t initial_epoch{0};

            for(general_index_t i=initial_epoch; i<num_epochs; i+=1)
            {
                visual_sensor[0].reset_data_index();
                do_run_training(current_time);
            }

            cout << "Simulation ends." << endl;
        }

        Runner(void) :
            is_debug_print_enabled(false),
            param_list(new ParamList()),
            neuron_layer_array
            {
                {
                    NeuronLayer(param_list->neuron_layer_params[0], alignment),
                    NeuronLayer(param_list->neuron_layer_params[1], alignment),
                    NeuronLayer(param_list->neuron_layer_params[2], alignment),

                    NeuronLayer(param_list->neuron_layer_params[0], alignment),
                    NeuronLayer(param_list->neuron_layer_params[1], alignment),
                    NeuronLayer(param_list->neuron_layer_params[2], alignment),
                },
            },
            connection_layer_array
            {
                {
                    ConnectionLayer(param_list->connection_layer_params[0], alignment),
                    ConnectionLayer(param_list->connection_layer_params[1], alignment),
                    ConnectionLayer(param_list->connection_layer_params[2], alignment),

                    ConnectionLayer(param_list->connection_layer_params[0], alignment),
                    ConnectionLayer(param_list->connection_layer_params[1], alignment),
                    ConnectionLayer(param_list->connection_layer_params[2], alignment),
                },
            },
            connection_list
            {
                make_pair(0, 1), make_pair(1, 2), make_pair(2, 1),
            },
            visual_input_size(length_visual_sensor * length_visual_sensor),
            visual_input
            (
                {
                    static_cast<sensory_input_t>(mkl_malloc(sizeof(elec_t) * visual_input_size, alignment)),
                    static_cast<sensory_input_t>(mkl_malloc(sizeof(elec_t) * visual_input_size, alignment)),
                }
            ),
            visual_input_copy
            (
                {
                    static_cast<sensory_input_t>(mkl_malloc(sizeof(elec_t) * visual_input_size, alignment)),
                    static_cast<sensory_input_t>(mkl_malloc(sizeof(elec_t) * visual_input_size, alignment)),
                }
            ),
            label({0, 0,}),
            fire_counter(param_list->neuron_layer_params[1].num_neurons, 0),
            engine({mt19937(param_list->spike_seed_gen[0]), mt19937(param_list->spike_seed_gen[0]),})
        {
            update_is_fired_for_connection_layer();
            generate_connections();
            initialize_visual_sensor();
            allocate_current_input_list();
            allocate_external_input();
        }

        virtual ~Runner(void)
        {
            for(general_index_t index=0; index<num_data_sets; index+=1)
            {
                mkl_free(external_input[index]);

                const general_index_t size0{static_cast<general_index_t>(current_input_list[index].size())};
                for(general_index_t i=0; i<size0; i+=1)
                {
                    const general_index_t size1{static_cast<general_index_t>(current_input_list[index][i].size())};
                    for(general_index_t j=0; j<size1; j+=1)
                    {
                        input_current_array_t buffer{current_input_list[index][i][j]};
                        mkl_free(buffer);
                    }
                }
                current_input_list[index].clear();
                mkl_free(visual_input_copy[index]);
                mkl_free(visual_input[index]);
            }
        }
    };
};
