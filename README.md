##  ⚙️ We are currently building this repo ⚙️

# Explaining Drug Repositioning: A Case-Based Reasoning Graph Neural Network Approach


This repository is a fork of CBR-SUBG from [Das el at 2022](https://github.com/rajarshd/CBR-SUBG) work. Here we implement CBR-SUBG on Mechanistic Repositioning Network with Indications (MIND), a biomedical knowledge graph that integrates two biomedical resources: [MechRepoNet](https://github.com/SuLab/MechRepoNet) and [DrugCentral](https://drugcentral.org/). Results of our implementation are reported here: 

## Installation

## Download MIND Dataset
- MechRepoNet with DrugCentral Indications Knowledge Graph and Train/Test/Dev data can de dowloaded from here: [here](https://www.dropbox.com/scl/fo/53x3iul9kh1ndhpky4s52/h?rlkey=0by2m3yo4bryabvbtzp6wn7kf&dl=0)
- You can download our collected train/test/dev subgraphs here: [here](https://www.dropbox.com/scl/fo/53x3iul9kh1ndhpky4s52/h?rlkey=0by2m3yo4bryabvbtzp6wn7kf&dl=0)

## Collecting youw own subgraphs 
Run your own subgraph collection procedure running: 

1-Collect chains around train drug queries, joining the drug query entitiy to the disease answer.
````
python src/01_collect_subgraphs/01_find_paths.py --data_name <dataset_name>
                            --data_dir_name <path_to_train_graph_files>
                            --output_dir <path_to_save_output_file>
                            --cutoff 3
                            --paths_to_collect 1000 
````

2-For a given query in train/dev/test set, we retrieve its K-nearest neighbor (KNN) queries from the training set. We then gather the collected paths from step 1 and traverse the KG. The following code snippet specifies that we consider 5 KNN queries and explore up to 100 nodes at each traversal step.
```
python src/01_collect_subgraphs/02_graph_collection.py --data_name <dataset_name>
                                  --data_dir_name <path_to_train_test_dev_files>
                                  --knn 5
                                  --collected_chains_name <path_to_collected_chains_file_in_step_1>
                                  --branch_size 100
                                  --output_dir <path_to_save_output_file>
```

## Training
The ```runner.py``` file is the main file that is needed to run the code. The commands to reproduce CBR-SUBG on MIND dataset are listed below: 
```
python runner.py --output_dir 01_results/
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


## Acknowledgments
This project builds upon work from [Das el at 2022](https://github.com/rajarshd/CBR-SUBG)

## Contact us 
Please open an issue or contact agonzalez@scripps.edu if you have any questions.
