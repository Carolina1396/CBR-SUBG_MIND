
from src.utils.loss import TXent 
from src.utils.neighbors import query_knn
from src.utils.knn_subgraphs import NNSubgraphs
from transformers import get_linear_schedule_with_warmup
from src.utils.dist_scr import L2Dist, CosineDist
import pandas as pd

import torch
import numpy as np
from tqdm import tqdm
import logging
import os
import json
from collections import defaultdict
import os

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


dist_fn = {'l2': L2Dist,'cosine': CosineDist}
           

class cbrTrainer: 

    def __init__(self, rgcn_model, dataset_obj, model_args, train_args, cbr_args, device):
        
        self.model =rgcn_model #RGCN Model
        self.dataset_obj=dataset_obj #Train test dev data
        self.model_args=model_args #Model arguments
        self.train_args=train_args #training arguments
        self.cbr_args =cbr_args
        self.device=device 
        self.data_name = self.cbr_args.data_name
        self.data_dir = self.cbr_args.data_dir
        self.out_dir = self.train_args.output_dir
        self.out_name = self.train_args.res_name

        #KNN and subgraphs setup
        self.neighbors = query_knn(self.dataset_obj, self.device) #search knn 
        self.subgraphs = NNSubgraphs(self.dataset_obj) #build query + KNN subgraphs batch
        
        #scores
        self.dist_fn = dist_fn[self.train_args.dist_metric](stage1_aggr=self.train_args.dist_aggr1,
                                                            stage2_aggr=self.train_args.dist_aggr2)
        self.loss = TXent(self.train_args.temperature)  
        
        #train parameters 
        self.trainable_params = list(self.model.parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                               'weight_decay': self.train_args.weight_decay, 
                               'lr': self.train_args.learning_rate}]
        

        #optimizer
        self.optimizer = torch.optim.AdamW(grouped_parameters, 
                                      lr=self.train_args.learning_rate,
                                       weight_decay=self.train_args.weight_decay)
        
        #calculate total training steps for linear scheduler 
        total_num_steps = int(self.train_args.num_train_epochs * 
                           (len(self.dataset_obj.train_dataloader) / self.train_args.gradient_accumulation_steps)) 
        
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.train_args.warmup_step, total_num_steps)
        
        
        #load train, test, dev 
        self.train00 = pd.read_csv(os.path.join(self.data_dir, self.data_name, 'train00.txt'),
                                               sep="\t", names=['drug', 'rel', 'disease'])
        
        self.test = pd.read_csv(os.path.join(self.data_dir, self.data_name, 'test.txt'),
                                               sep="\t", names=['drug', 'rel', 'disease'])
        
        self.dev= pd.read_csv(os.path.join(self.data_dir, self.data_name, 'valid.txt'),
                                               sep="\t", names=['drug', 'rel', 'disease'])
        
        
    def ranking_eval (self, nn_batch, pred_ranks, data_name):
        """
        Get ranking of expected answers 

        """
        predictions = defaultdict(list)
        results ={}

        gold_answers = nn_batch[0].answers
        drug = nn_batch[0].query[0]
        
        if data_name == "train":
            other_answers=self.test[self.test['drug']==drug]['disease'].to_list() + self.dev[self.dev['drug']==drug]['disease'].to_list()
        
        if data_name == "dev":
            other_answers=self.test[self.test['drug']==drug]['disease'].to_list() + self.train00[self.train00['drug']==drug]['disease'].to_list()
            
        if data_name == "test":
            other_answers=self.dev[self.dev['drug']==drug]['disease'].to_list() + self.train00[self.train00['drug']==drug]['disease'].to_list()
       
        for gold_answer in gold_answers:
            filtered_answers = []

            for pred in pred_ranks:
                pred = self.dataset_obj.id2ent[pred]

                if pred not in other_answers:#make sure answers are not in other set
                    if pred in gold_answers and pred != gold_answer: # remove all other gold answers from prediction
                        continue
                    else:
                        filtered_answers.append(pred)

            rank = None
            predictions[nn_batch[0].query[0]+"_"+gold_answer].append(filtered_answers[:200])
            
            for i, e_to_check in enumerate(filtered_answers):
                if gold_answer == e_to_check:

                    rank = i + 1
                    break
            results['count'] = 1 + results.get('count', 0.0)

            if rank is not None:
                if rank <= 10:
                    results["avg_hits@10"] = 1 + results.get("avg_hits@10", 0.0)
                    if rank <= 5:
                        results["avg_hits@5"] = 1 + results.get("avg_hits@5", 0.0)
                        if rank <= 3:
                            results["avg_hits@3"] = 1 + results.get("avg_hits@3", 0.0)
                            if rank <= 1:
                                results["avg_hits@1"] = 1 + results.get("avg_hits@1", 0.0)
                results["avg_rr"] = (1.0 / rank) + results.get("avg_rr", 0.0)
        
                        
        if data_name == "test": 
            return results, predictions
        else:
            return results
    
    def train(self): 
        """
        Train over each subgraph batch  
        return: loss score and ranking
        """
        
        results = {}
        self.model.train()
        local_step =0
        losses = []
        self.check_steps = self.train_args.check_steps 
        data_name = "train"

        for batch_ctr, batch in enumerate(tqdm(self.dataset_obj.train_dataloader, desc=f"[Train]", position=0, leave=True)):

            #[query + KNN entities] #nn_slices: position of the queries
            nn_list, nn_slices = self.neighbors(batch, k=self.cbr_args.num_neighbors_train)
            #eliminating queries (or all its KNNs) with no answers in subgraph
            nn_batch, nn_slices = self.subgraphs(query_and_knn_list=nn_list, nn_slices=nn_slices)

            #Query subgraph (or KNN) is not arriving to expected answer
            if nn_batch is None:
                logger.info("The current batch was returned with no answers")
                continue

            #batch information 
            new_batch_len = len(nn_slices) - 1
            nn_batch.x = nn_batch.x.to(self.device) # #entities around seed_n
            nn_batch.edge_index = nn_batch.edge_index.to(self.device) ##2D array [head], [tail]
            nn_batch.edge_attr = nn_batch.edge_attr.to(self.device) ##edge connecting [head], [tail]

            #Batch representation
            sub_batch_repr = self.model(nn_batch.x,  #entities around seed_n
                                   nn_batch.edge_index, ##2D array [head], [tail]
                                   nn_batch.edge_attr) ##edge connecting [head], [tail] #edge type

            loss_value = None
            for i in range(new_batch_len):
                #Subgraph nodes of query
                repr_s = sub_batch_repr.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 0],
                                                       nn_batch.__slices__['x'][nn_slices[i] + 1] -
                                                       nn_batch.__slices__['x'][nn_slices[i] + 0]) 

                #True,False vector. True index represet answer nodes
                labels_s = nn_batch.label_node_ids.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 0],
                                                                  nn_batch.__slices__['x'][nn_slices[i] + 1] -
                                                                  nn_batch.__slices__['x'][nn_slices[i] + 0]) 

                #These are the nodes of KNN entities. 
                repr_t = sub_batch_repr.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                               nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                               nn_batch.__slices__['x'][nn_slices[i] + 1]) 

                #True,False vector. True index represet answer nodes (KNN)
                labels_t = nn_batch.label_node_ids.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                                          nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                                          nn_batch.__slices__['x'][nn_slices[i] + 1]) 


                #Tensor of shape [n_pos] identifying which target belongs to which neighbor
                label_identifiers = nn_batch.x_batch.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                                            nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                                            nn_batch.__slices__['x'][nn_slices[i] + 1])[labels_t] 

                assert label_identifiers.min() >= 0
                label_identifiers = label_identifiers - label_identifiers.min()
                label_identifiers = label_identifiers.to(repr_s.device)

                #Distance between query node subgraph and KNN answer nodes
                dists =self.dist_fn(repr_s, 
                                    repr_t[labels_t], 
                                    target_identifiers=label_identifiers) 


                #ranking
                pred_ranks = torch.argsort(dists).cpu().numpy() #sort predictions
                pred_ranks = nn_batch[nn_slices[i]].x[pred_ranks].cpu().numpy() #get the ids of the predictions

                ranking_results = self.ranking_eval(nn_batch, pred_ranks, data_name)
#                 print(ranking_results)
                for key_ in ranking_results:
                    results[key_] = ranking_results.get(key_,0) + results.get(key_,0)
                    
                #loss
                mask = ((labels_s == 1.0) + (torch.FloatTensor(len(dists)).uniform_() < self.train_args.sampling_loss)).to(self.device)
                contrast_loss = self.loss(dists[mask], labels_s[mask]) / new_batch_len

                if loss_value is None:
                    loss_value = contrast_loss
                else:
                    loss_value += contrast_loss

            if self.train_args.gradient_accumulation_steps > 1:
                loss_value = loss_value / self.train_args.gradient_accumulation_steps
            loss_value.backward()
            local_step += 1
            losses.append(loss_value.item())

            if batch_ctr % self.check_steps == 0:                    
                mrr_train = results["avg_rr"]/results['count']
                logger.info('[Batch Loss:{:.4} Batch MRR:{:.4}'.format(np.mean(losses), mrr_train))



            if local_step % self.train_args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 1)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
        

        final_results = {}
        normalizer = results.pop('count')
        for k, v in results.items():
            if k.startswith('avg'):
                final_results[k] = v / normalizer
            else:
                assert isinstance(v, list)
                final_results[k] = np.asarray(v)

        return np.mean(losses), final_results
    
    def run_evaluate(self, data_name, dataloader):
        """
        Evaluate test/dev subgraph batches  
        return: Ranking results
        """
        
        results = {}

        self.model.eval()
        for batch_ctr, batch in enumerate(tqdm(dataloader, desc=f"{data_name}", position=0, leave=True)):
            
            
            #[query + KNN entities] #nn_slices: position of the queries
            nn_list, nn_slices = self.neighbors(batch, k=self.cbr_args.num_neighbors_train)
            #eliminating queries (or all its KNNs) with no answers in subgraph
            nn_batch, nn_slices = self.subgraphs(query_and_knn_list=nn_list, nn_slices=nn_slices)

            #Query subgraph (or KNN) is not arriving to expected answer
            if nn_batch is None:
                logger.info("The current batch was returned with no answers") 
                continue

            #batch information 
            new_batch_len = len(nn_slices) - 1
            nn_batch.x = nn_batch.x.to(self.device) # #entities around seed_n
            nn_batch.edge_index = nn_batch.edge_index.to(self.device) ##2D array [head], [tail]
            nn_batch.edge_attr = nn_batch.edge_attr.to(self.device) ##edge connecting [head], [tail]

            #Batch representation
            sub_batch_repr = self.model(nn_batch.x,  #entities around seed_n
                                   nn_batch.edge_index, ##2D array [head], [tail]
                                   nn_batch.edge_attr) ##edge connecting [head], [tail] #edge type

            for i in range(new_batch_len):
                #Subgraph nodes of query
                repr_s = sub_batch_repr.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 0],
                                                       nn_batch.__slices__['x'][nn_slices[i] + 1] -
                                                       nn_batch.__slices__['x'][nn_slices[i] + 0]) 

                #True,False vector. True index represet answer nodes
                labels_s = nn_batch.label_node_ids.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 0],
                                                                  nn_batch.__slices__['x'][nn_slices[i] + 1] -
                                                                  nn_batch.__slices__['x'][nn_slices[i] + 0]) 

                #These are the nodes of KNN entities. 
                repr_t = sub_batch_repr.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                               nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                               nn_batch.__slices__['x'][nn_slices[i] + 1]) 

                #True,False vector. True index represet answer nodes (KNN)
                labels_t = nn_batch.label_node_ids.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                                          nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                                          nn_batch.__slices__['x'][nn_slices[i] + 1]) 


                #Tensor of shape [n_pos] identifying which target belongs to which neighbor
                label_identifiers = nn_batch.x_batch.narrow(0, nn_batch.__slices__['x'][nn_slices[i] + 1],
                                                            nn_batch.__slices__['x'][nn_slices[i + 1]] -
                                                            nn_batch.__slices__['x'][nn_slices[i] + 1])[labels_t] 

                assert label_identifiers.min() >= 0
                label_identifiers = label_identifiers - label_identifiers.min()
                label_identifiers = label_identifiers.to(repr_s.device)

                #Distance between query node subgraph and KNN answer nodes
                dists =self.dist_fn(repr_s, 
                                    repr_t[labels_t], 
                                    target_identifiers=label_identifiers) 


                #ranking
                pred_ranks = torch.argsort(dists).cpu().numpy() #sort predictions
                pred_ranks = nn_batch[nn_slices[i]].x[pred_ranks].cpu().numpy() #get the ids of the predictions
                
                
                if data_name == 'dev':
                    ranking_results = self.ranking_eval(nn_batch, pred_ranks, data_name)
                    
                if data_name == 'test':
                    ranking_results, predictions_test = self.ranking_eval(nn_batch, pred_ranks, data_name)
                    
                    # Save predictions
                    os.makedirs(self.out_dir, exist_ok=True)
                    pred_file = os.path.join(self.out_dir,  f"{self.out_name}_{self.cbr_args.formatted_datetime}.json")
                    
                    if os.path.exists(pred_file): 
                        with open(f'{pred_file}', 'r') as file:
                            data = json.load(file)
                              
                        data.update(predictions_test) 

                        with open(f'{pred_file}', 'w') as file:
                            json.dump(data, file)

                    else: 
                        with open(f'{pred_file}', 'w') as file:
                            json.dump(predictions_test, file)
                    
                
                for key_ in ranking_results:
                    results[key_] = ranking_results.get(key_,0) + results.get(key_,0)
        
        final_results = {}
        normalizer = results.pop('count')
        for k, v in results.items():
            if k.startswith('avg'):
                final_results[k] = v / normalizer
            else:
                assert isinstance(v, list)
                final_results[k] = np.asarray(v)

        return final_results

