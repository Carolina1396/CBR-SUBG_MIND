##  ⚙️ We are currently building this repo ⚙️

# Explaining Drug Repositioning: A Case-Based Reasoning Graph Neural Network Approach


This repository is a fork of CBR-SUBG from [Das el at 2022](https://github.com/rajarshd/CBR-SUBG) work. Here we implement CBR-SUBG on Mechanistic Repositioning Network with Indications (MIND), a biomedical knowledge graph that integrates two biomedical resources: [MechRepoNet](https://github.com/SuLab/MechRepoNet) and [DrugCentral](https://drugcentral.org/). Results of our implementation are reported here: 

## Installation

## Dataset

## Training

## Results 
The commands to reproduce CBR-SUBG on MIND dataset are listed below: 
```
runner.py --output_dir 01_results/
          --data_dir rc/00_data/
          --data_name MIND
          --paths_file_dir MIND_cbr_subgraph_knn-5_branch-200.pkl
          --train_batch_size 1
          --num_neighbors_train 5
          --eval_batch_size 1
          --num_neighbors_eval 10
          --gcn_dim_init 64
          --hidden_channels_gcn 128
          --drop_gcn 0.796562802302663
          --conv_layers 1
          --transform_input 0
          --use_wandb 1
          --dist_metric l2
          --dist_aggr1 mean
          --dist_aggr2 mean
          --sampling_loss 0.33912131320071265
          --temperature 0.10531080354774307
          --learning_rate 0.1
          --warmup_step 300
          --weight_decay 0.01
          --num_train_epochs 100
          --gradient_accumulation_steps 2
          --check_steps 10
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
