#pragma once

//import std;

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "CommonNet/Common.h"

namespace Kajuer::Sensor
{
    using std::begin;
    using std::cerr;
    using std::cout;
    using std::end;
    using std::endl;
    using std::double_t;
    using std::ifstream;
    using std::min;
    using std::size_t;
    using std::string;
    using std::uint8_t;
    using std::uint32_t;
    using std::vector;

    using Kajuer::Common::general_index_t;
    using Kajuer::Common::sensory_input_t;
    using Kajuer::Common::pixel_t;

    using data_t = vector<pixel_t>;
    using label_data_t = vector<uint8_t>;

    //This class handles visual sensory input especially of MNIST digits.
    class VisualSensor
    {
    private:
        vector<data_t> data_list;
        label_data_t label_list;
        general_index_t current_data_index;

    private:
		static uint32_t unpack(char arg[])
		{
			return (((arg[0] << 24) & 0xFF000000ULL) | ((arg[1] << 16) & 0x00FF0000ULL) | ((arg[2] << 8) & 0x0000FF00ULL) | (arg[3] & 0x000000FFULL));
		}

    public:
        // load num_images of MNIST digit images from data_file_name to data_list
        // and also labels from label_file_name to label_list.
        // if normalize==true then all data will be normalized into the range of [0.0, 1.0] and stored
        // and if normalize==false then raw pixel values will be stored.
		void load_mnist
		(
			const string& data_file_name,
			const string& label_file_name,
			const uint32_t num_images,
			const bool normalize=true
		)
		{
			uint32_t mnd;
			uint32_t nid;
			uint32_t r;
			uint32_t c;

			uint32_t mnl;
			uint32_t nil;

			{
				ifstream ifs_data{ data_file_name, ifstream::in | ifstream::binary };

				char tmp[4];

				ifs_data.read(tmp, sizeof(tmp));
				mnd = unpack(tmp);
				ifs_data.read(tmp, sizeof(tmp));
				nid = unpack(tmp);
				ifs_data.read(tmp, sizeof(tmp));
				r = unpack(tmp);
				ifs_data.read(tmp, sizeof(tmp));
				c = unpack(tmp);

                nid = min(nid, num_images);
				cout << "data file: " << mnd << ' ' << nid << ' ' << r << ' ' << c << endl;

				data_list.reserve(nid);

				for (uint32_t i = 0; i < nid; i += 1) {
					char byte;

					data_list.emplace_back(vector<pixel_t>());

					for (uint32_t row = 0; row < r; row += 1)
					{
						for (uint32_t col = 0; col < c; col += 1)
						{
							ifs_data.read(&byte, sizeof(byte));
							pixel_t datum_to_store(static_cast<uint8_t>(byte));
							if(normalize)
							{
								datum_to_store /= 255.0;
							}
							data_list[i].emplace_back(datum_to_store);
						}
					}
				}
			}

			{
				ifstream ifs_label{ label_file_name, ifstream::in | ifstream::binary };

				char tmp[4];

				ifs_label.read(tmp, sizeof(tmp));
				mnl = unpack(tmp);
				ifs_label.read(tmp, sizeof(tmp));
				nil = unpack(tmp);

                nil = min(nil, num_images);
				cout << "label: " << mnl << ' ' << nil << endl;

				{
					char byte;

					label_list.reserve(nil);

					for (uint32_t i = 0; i < nil; i += 1)
					{
						ifs_label.read(&byte, sizeof(byte));
						label_list.emplace_back(static_cast<uint8_t>(byte));
					}
				}
			}
		}

        // Store MNIST digit image to which current_data_index points to sensory_input.
        // We assume sensory_input is operator[] operable here.
        // Number of sensory_input pixels will be stored in sensory_input_size.
        // The label corresponding to sensory_input will be stored in label.
        bool generate_next_visual_input(sensory_input_t& sensory_input, general_index_t& sensory_input_size, uint32_t& label)
        {
            //validate current data_list.
            if(data_list.size() <= current_data_index)
            {
                current_data_index = 0;
            }

            const data_t& data{data_list[current_data_index]};
            const general_index_t data_size{static_cast<general_index_t>(data.size())};

            for(general_index_t i=0; i<data_size; i+=1)
            {
                sensory_input[i] = data[i];
            }

            sensory_input_size = data_size;

            label = label_list[current_data_index];

            current_data_index += 1;

            return true;
        }

        // Reset internal current_data_index to 0.
        void reset_data_index(void)
        {
            current_data_index = 0;
        }

    public:
        VisualSensor(void) :
            current_data_index(0)
        { }

        virtual ~VisualSensor(void)
        { }
    };
};
