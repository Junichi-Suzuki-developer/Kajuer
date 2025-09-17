#pragma once

//import std;

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

#include "CommonNet/Common.h"
#include "CommonNet/ConnectionLayer.h"
#include "CommonNet/NeuronLayer.h"

namespace Kajuer::Common
{
    using std::back_inserter;
    using std::begin;
    using std::cerr;
    using std::double_t;
    using std::end;
    using std::endl;
    using std::iota;
    using std::make_pair;
    using std::mt19937;
    using std::random_device;
    using std::sample;
    using std::size_t;
    using std::uint_fast32_t;

    using Kajuer::Common::general_index_t;
    using Kajuer::Common::neuron_index_array_t;

    //generating connections between 2 NeuronLayers and pass it to ConnectionLayer.
    class ConnectionGenerator
    {
    private:
        //This function handles actual generation of connections.
        template<class T, class U>
        inline static void do_generate_connection
        (
            const NeuronLayer& src_neuron_layer,
            const NeuronLayer& dst_neuron_layer,
            const double_t connection_ratio,
            ConnectionLayer& connection_layer,
            const uint_fast32_t seed_gen,
            T& dist,
            U& weight_engine
        )
        {
            mt19937 connection_engine{seed_gen};

            general_index_t num_src_neuron{src_neuron_layer.get_num_neurons()};
            general_index_t num_dst_neuron_original{dst_neuron_layer.get_num_neurons()};
            general_index_t num_dst_neuron{static_cast<general_index_t>(num_dst_neuron_original * connection_ratio)};

            synapse_array_for_gen_t weight(num_src_neuron * num_dst_neuron);

            neuron_index_array_for_gen_t src_neuron_index_list(num_src_neuron);
            iota(begin(src_neuron_index_list), end(src_neuron_index_list), 0);

            neuron_index_array_for_gen_t dst_neuron_index_list_seed(num_dst_neuron_original);
            iota(begin(dst_neuron_index_list_seed), end(dst_neuron_index_list_seed), 0);

            neuron_index_array_for_gen_t dst_neuron_index_list;

            sample
            (
                begin(dst_neuron_index_list_seed),
                end(dst_neuron_index_list_seed),
                back_inserter(dst_neuron_index_list),
                num_dst_neuron,
                connection_engine
            );

            neuron_pair_array_t connection_list;

            const general_index_t src_neuron_index_list_size{static_cast<general_index_t>(src_neuron_index_list.size())};
            const general_index_t dst_neuron_index_list_size{static_cast<general_index_t>(dst_neuron_index_list.size())};

            for(general_index_t i=0; i<src_neuron_index_list_size; i+=1)
            {
                for(general_index_t j=0; j<dst_neuron_index_list_size; j+=1)
                {
                    connection_list.emplace_back(make_pair(src_neuron_index_list[i], dst_neuron_index_list[j]));
                    weight[i*dst_neuron_index_list_size+j] = dist(weight_engine);
                }
            }

            connection_layer.generate_connection
            (
                connection_list,
                src_neuron_index_list_size,
                dst_neuron_index_list_size,
                weight.data()
            );
        }

    public:

        //This function generates full connection between src and dst neuron layers.
        template<class T, class U>
        static void generate_full_connection
        (
            const NeuronLayer& src_neuron_layer,
            const NeuronLayer& dst_neuron_layer,
            ConnectionLayer& connection_layer,
            const uint_fast32_t seed_gen,
            T& dist,
            U& engine
        )
        {
            do_generate_connection
            (
                src_neuron_layer,
                dst_neuron_layer,
                1.0, //100% connection.
                connection_layer,
                seed_gen,
                dist,
                engine
            );
        }

        //This function generates 1 by 1 connections between src and dst neuron layers.
        //e.g. src = (0, 1, 2) and dst = (0, 1, 2) then connections = (src:0->dst:0, src:1->dst:1, src:2->dst:2).
        template<class T, class U>
        static void generate_one_by_one_neuron_connection
        (
            const NeuronLayer& src_neuron_layer,
            const NeuronLayer& dst_neuron_layer,
            ConnectionLayer& connection_layer,
            T& dist,
            U& engine
        )
        {
            general_index_t num_src_neuron{src_neuron_layer.get_num_neurons()};
            general_index_t num_dst_neuron{dst_neuron_layer.get_num_neurons()};

            synapse_array_for_gen_t weight(num_src_neuron * num_dst_neuron);

            neuron_pair_array_t connection_list;

            for(general_index_t i=0; i<num_src_neuron; i+=1)
            {
                for(general_index_t j=0; j<num_dst_neuron; j+=1)
                {
                    connection_list.emplace_back(make_pair(i, j));

                    if(i==j)
                    {
                        weight[i*num_dst_neuron+j] = dist(engine);
                    }
                    else
                    {
                        weight[i*num_dst_neuron+j] = 0.0;
                    }
                }
            }

            connection_layer.generate_connection
            (
                connection_list,
                num_src_neuron,
                num_dst_neuron,
                weight.data()
            );
        }

       //This function generates 1 to other indices connections between src and dst neuron layers.
       //e.g. src = (0, 1, 2) and dst = (0, 1, 2) then
       //connections = (src:0->dst:1, src:0->dst:2, src:1->dst:0, src:1->dst:2, src:2->dst:0, src:2->dst:1).
       template<class T, class U>
        static void generate_dst_other_neuron_connection
        (
            const NeuronLayer& src_neuron_layer,
            const NeuronLayer& dst_neuron_layer,
            ConnectionLayer& connection_layer,
            T& dist,
            U& engine
        )
        {
            general_index_t num_src_neuron{src_neuron_layer.get_num_neurons()};
            general_index_t num_dst_neuron{dst_neuron_layer.get_num_neurons()};

            synapse_array_for_gen_t weight(num_src_neuron * num_dst_neuron);

            neuron_pair_array_t connection_list;

            for(general_index_t i=0; i<num_src_neuron; i+=1)
            {
                for(general_index_t j=0; j<num_dst_neuron; j+=1)
                {
                    connection_list.emplace_back(make_pair(i, j));

                    if(i==j)
                    {
                        weight[i*num_dst_neuron+j] = 0.0;
                    }
                    else
                    {
                        weight[i*num_dst_neuron+j] = dist(engine);
                    }
                }
            }

            connection_layer.generate_connection
            (
                connection_list,
                num_src_neuron,
                num_dst_neuron,
                weight.data()
            );
        }

        template<class T, class U>
        static void generate_dst_same_group_neuron_connection
        (
            const NeuronLayer& src_neuron_layer,
            const NeuronLayer& dst_neuron_layer,
            ConnectionLayer& connection_layer,
            const size_t num_labels,
            T& dist,
            U& engine
        )
        {
            general_index_t num_src_neuron{src_neuron_layer.get_num_neurons()};
            general_index_t num_dst_neuron{dst_neuron_layer.get_num_neurons()};

            synapse_array_for_gen_t weight(num_src_neuron * num_dst_neuron);

            neuron_pair_array_t connection_list;

            for(general_index_t i=0; i<num_src_neuron; i+=1)
            {
                for(general_index_t j=0; j<num_dst_neuron; j+=1)
                {
                    connection_list.emplace_back(make_pair(i, j));

                    if((i/num_labels) == (j/num_labels))
                    {
                        weight[i*num_dst_neuron+j] = dist(engine);
                    }
                    else
                    {
                        weight[i*num_dst_neuron+j] = 0.0;
                    }
                }
            }

            connection_layer.generate_connection
            (
                connection_list,
                num_src_neuron,
                num_dst_neuron,
                weight.data()
            );
        }
    };
};
