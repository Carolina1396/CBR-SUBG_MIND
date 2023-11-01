# Explaining Drug Repositioning: A Case-Based Reasoning Graph Neural Network Approach

This repository is a fork of CBR-SUBG from [Das el at 2022](https://github.com/rajarshd/CBR-SUBG) work. Here we implement CBR-SUBG on Mechanistic Repositioning Network with Indications (MIND), a biomedical knowledge graph that integrates two biomedical resources: [MechRepoNet](https://github.com/SuLab/MechRepoNet) and [DrugCentral](https://drugcentral.org/). Results of our implementation are reported here: 

## Installation

## Dataset

## Training

## Results 
The commands to reproduce CBR-SUBG on MIND dataset are listed below: 
```
runner.py --output_dir 01_results/
          --data_dir
          --data_name
          --paths_file_dir
          --train_batch_size
          --num_neighbors_train
          --eval_batch_size
          --num_neighbors_eval
          --gcn_dim_init
          --hidden_channels_gcn
          --drop_gcn
          --conv_layers
          --transform_input
          --use_wandb
          --dist_metric
          --dist_aggr1
          --dist_aggr2
          --sampling_loss
          --temperature
          --learning_rate
          --warmup_step
          --weight_decay
          --num_train_epochs
          --gradient_accumulation_steps
          --check_steps
```
Note: Because PyTorch does not ensure perfect reproducibility, even when using the same random seed (as explained in the PyTorch documentation at https://pytorch.org/docs/stable/notes/randomness.html), there may be slight deviations in the results compared to those reported in the paper.

The pre-trained model can be found here. To analyze results run as below:
```
runner.py
```

## Acknowledgments
This project builds upon work from [Das el at 2022](https://github.com/rajarshd/CBR-SUBG)

## Contact us 
Please open an issue or contact agonzalez@scripps.edu if you have any questions.
