# Kajuer
MNIST digit recognition using Spiking Neural Network (SNN)

## The basis
This simulation is based on the 3 great works.

### Neuron model
Izhikevich neuron model is simple and have the great potential to solve many types of problems.

And this is embedded in this simultaion.

See the reference https://www.izhikevich.org/publications/spikes.pdf

### Network model
Diehl & Cook Spiking Neural Network model is one of the most powerful solution to MNIST digit recognition problems.

This simulation embedds the essence of Diehl & Cook's work's network characteristics.

See the reference https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full

### Supervised STDP model
Supervised STDP model is charactered as the potentially both positive and negative weighed synapses.

This can improve the digit recognition accuracy.

See the reference (written in Japanese)　https://ipsj.ixsq.nii.ac.jp/record/215041/files/IPSJ-Z83-4R-05.pdf

# Preparation
## for Manjaro Linux
1. Install intel-oneapi-basekit package.

Use below command to install the dependencies.

```
pamac install intel-oneapi-basekit
echo -e 'source /opt/intel/oneapi/setvars.sh' >> ~/.zshrc
source ~/.zshrc
```

2. Chaeck out repository.

Check out this repository.

```
git clone https://github.com/Junichi-Suzuki-developer/Kajuer.git
```

# Running a simulation
1. Run a simulation.

Use below command to run a simulation.

```
cd Kajuer
./exec.sh
```

Depending on the machine power, it will take about 2 or more hours to complete the simulation.

# Simulation parameters
Modifiable parameters are stored in the file `Kajuer/Param/Param.h`.

If you would like to try another parameter set, modify it and just run `exec.sh` again.

Almost all of parameters are with comments as you can interpret what it is.

# How to interpret results
`exec.sh` outputs 3 types of log files in `Kajuer` directory.

## fire_record.txt
It will contain MNIST digit recognition results.

For example, it will contain such like below many times.

```
label_answer: 9 7 false
```

- "label_answer:" is just a label so this can be ignorable here.

- "9" is the Answer of Spiking Neural Network. What they recognize the digit as is described.

- "7" is the correct answer label, so in this example Spiking Neural Network failed to recognize the digit.

- "false" means recoginition result is not the same as the answer label.

This infomration can help visualize the recognition accuracy progress.

<img width="1920" height="1440" alt="accuracy_history" src="https://github.com/user-attachments/assets/2e783e28-c2e3-4c57-bdc7-834b68cf5fb7" />

## spike_records
### training_spike_record.txt.*
This contains the spike information during the training phase.

For example, it will contain such like below many times.

```
108 55
```

- "108" is the simulation time elapsed in milliseconds.

- "55" is the spiked output neuron index.

This infomration can help visualize the spike history.

<img width="1920" height="1440" alt="raster training_spike_record txt 44967" src="https://github.com/user-attachments/assets/9d2405ae-d3f0-4cf6-80e9-3f2d6fe264d0" />

### validation_spike_record.txt.*
This contains the spike information during the validation phase.

Contents are the same format as training_spike_record.txt.*.

## weight_map_dir
### weight_map.bin.*
This contains the raw weight values, binary format, of Simulations' output neuron layer at the time.

For example, "weight_map.bin.50000" will contain the weight map when the simulation time elapsed in 50000 milliseconds.

This information can help visualize the weight change history.

<img width="3000" height="3000" alt="test 48" src="https://github.com/user-attachments/assets/f32daf59-43b3-44e2-a30c-8d942d026ad5" />
