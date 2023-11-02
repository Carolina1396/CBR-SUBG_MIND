import os
from copy import deepcopy
import torch
from torch_geometric.data import Data
import pickle
from torch_geometric.data import DataListLoader
from collections import defaultdict
from tqdm import tqdm
import logging

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class GraphData(Data):
    def __init__(self, 
                x = None, 
                edge_index = None, 
                edge_attr = None,
                seed_nodes_mask = None, 
                label_nodes_mask = None, 
                split = None, 
                ex_id = None, 
                query= None,
                query_str= None,
                answers = None, 
                knn_ids= None, 
                penalty = torch.LongTensor([0])):
    
        super(GraphData, self).__init__(x, edge_index, edge_attr)

        self.seed_node_ids = seed_nodes_mask 
        self.label_node_ids = label_nodes_mask
        self.split = split
        self.ex_id = ex_id
        self.query= query
        self.query_str= query_str
        self.answers = answers
        self.knn_ids= knn_ids
        self.penalty = penalty

class DataLoader:
    def __init__(self, data_dir, data_name, paths_file_dir, train_batch_size, eval_batch_size):
        self.data_name = data_name
        self.data_dir = data_dir
        self.train_batch_size=train_batch_size
        self.eval_batch_size= eval_batch_size
            
        graph_pkl = os.path.join(self.data_dir, "subgraphs", paths_file_dir)
        
        logger.info("=====Loading Subgraphs=====")
        with open(graph_pkl, "rb") as fin:
            self.all_paths = pickle.load(fin)

        self.load_dataset()
        
        
    def add_inv_edges_to_adj(self, adj_map):
        full_adj_map = deepcopy(adj_map)
        for e1, re2_map in adj_map.items():
            for r, e2_list in re2_map.items():
                # r_inv = r + '_inv'
                r_inv = r + "_inv" if not r.endswith("_inv") else r[:-4]
                for e2 in e2_list:
                    if e2 not in full_adj_map: full_adj_map[e2] = {}
                    if r_inv not in full_adj_map[e2]: full_adj_map[e2][r_inv] = []
                    full_adj_map[e2][r_inv].append(e1)
        for e1, re2_map in full_adj_map.items():
            for r in re2_map:
                re2_map[r] = sorted(set(re2_map[r]))
        return full_adj_map


    def load_dataset(self):
        self.ent2id = dict() #{"entity": "id"}
        self.id2ent= dict() #{"id": "entity"}
        self.rel2id= dict() #{"relation": "id"}
        self.id2rel = dict() #{"id": "relation"}

        self.full_adj_map = {}
        self.full_edge_index = [[], []] #[head][tail]
        self.full_edge_attr = [] #relations connection head-tail
        
        # vocab created from all splits, but graph should not have test/dev edges
        for split in ['train00', 'test', 'valid', 'graph']:
            for line in open(os.path.join(self.data_dir, self.data_name, f'{split}.txt')): 
                e1, r, e2 = line.strip().split('\t')
                if e1 not in self.ent2id:
                    self.ent2id[e1] = len(self.ent2id)
                
                if e2 not in self.ent2id:
                    self.ent2id[e2] = len(self.ent2id)
                
                if r not in self.rel2id:
                    self.rel2id[r] = len(self.rel2id)
                    r_inv = r +"_inv"
                    self.rel2id[r_inv] = len(self.rel2id)

                self.full_adj_map.setdefault(e1, {}).setdefault(r, []).append(e2)
                self.full_edge_index[0].append(self.ent2id[e1]) #head
                self.full_edge_index[1].append(self.ent2id[e2]) #tail
                self.full_edge_attr.append(self.rel2id[r])   #relation
              
        self.full_adj_map = self.add_inv_edges_to_adj(self.full_adj_map)
        self.n_entities = len(self.ent2id) #number of entities
        self.n_relations = len(self.rel2id) #number of relations|

        self.id2ent = {v: k for k, v in self.ent2id.items()}  #{"id": "entity"}
        self.id2rel = {v: k for k, v in self.rel2id.items()} #{"id": "relation"}

        #Edge index tensor
        self.full_edge_index = torch.LongTensor(self.full_edge_index) #[head][tail]
        self.full_edge_attr = torch.LongTensor(self.full_edge_attr) #relations connection head-ta        
                
                
        #convert rawdat to CBR format 
        self.raw_train_data_map = defaultdict(list)
        self.raw_dev_data_map =  defaultdict(list)
        self.raw_test_data_map = defaultdict(list)

        #train
        for line in open(os.path.join(self.data_dir, self.data_name, 'train00.txt')): ###
            e1, r, e2 = line.strip().split('\t')
            self.raw_train_data_map[(e1, r)].append(e2)
#             self.raw_train_data_map[(e2, r + "_inv")].append(e1)
        
        #validation
        for line in open(os.path.join(self.data_dir, self.data_name, 'valid.txt')):
            e1, r, e2 = line.strip().split('\t')

            self.raw_dev_data_map[(e1, r)].append(e2)
#             self.raw_dev_data_map[(e2, r + "_inv")].append(e1)
        
        #test
        for line in open(os.path.join(self.data_dir, self.data_name, 'test.txt')):
            e1, r, e2 = line.strip().split('\t')
            self.raw_test_data_map[(e1, r)].append(e2)
#             self.raw_test_data_map[(e2, r + "_inv")].append(e1)
       
    
    
        #Create maping of queries
        
        self.train_idmap, self.dev_idmap, self.test_idmap = {}, {}, {} #map of queries (head,relation)
        self.raw_train_data, self.raw_dev_data, self.raw_test_data = [], [], []

        #train
        for ctr, ((e1, r), e2_list) in enumerate(self.raw_train_data_map.items()):
            self.raw_train_data.append({"id": (e1, r), "question": r, "seed_entities": [e1], "answer": e2_list})
            self.train_idmap[(e1, r)] = ctr

        #dev
        for ctr, ((e1, r), e2_list) in enumerate(self.raw_dev_data_map.items()):
            self.raw_dev_data.append({"id": (e1, r), "question": r, "seed_entities": [e1], "answer": e2_list})
            self.dev_idmap[(e1, r)] = ctr

        #test
        for ctr, ((e1, r), e2_list) in enumerate(self.raw_test_data_map.items()):
            self.raw_test_data.append({"id": (e1, r), "question": r, "seed_entities": [e1], "answer": e2_list})
            self.test_idmap[(e1, r)] = ctr
        
        
        #train
        self.train_dataset = []

        for item in tqdm(self.raw_train_data):
            self.train_dataset.append(self.convert_rawdata_to_cbr(item, 'train'))
        logger.info("Train data loaded")
        
        #test
        self.test_dataset = []

        for item in tqdm(self.raw_test_data):
            self.test_dataset.append(self.convert_rawdata_to_cbr(item, 'test'))
        logger.info("Test data loaded")
        
        #dev
        self.dev_dataset = []

        for item in tqdm(self.raw_dev_data):
            self.dev_dataset.append(self.convert_rawdata_to_cbr(item, 'dev'))
        logger.info("Dev data loaded")
        

        self.train_dataloader = DataListLoader(self.train_dataset, self.train_batch_size, shuffle=True)
        self.dev_dataloader = DataListLoader(self.dev_dataset, self.eval_batch_size, shuffle=False )
        self.test_dataloader = DataListLoader(self.test_dataset, self.eval_batch_size, shuffle=False)
        
        
        
    def convert_rawdata_to_cbr(self, raw_data: dict, split: str, inplace_obj:GraphData = None):
        
        
        knn_ids = None
        
        #query id
        if split == 'train':
            ex_id = self.train_idmap[raw_data["id"]]
            
        elif split == 'dev':
            ex_id = self.dev_idmap[raw_data["id"]]
            
        elif split == 'test':
            ex_id = self.test_idmap[raw_data["id"]]
            
        
        ques_str = raw_data["id"] #query (e1, r)
        raw_data["answer"] = list(set(raw_data["answer"])) #answers of query
        
        
        #Get entities around each query
        sub_nodes, sub_edge_index, seed_ent_loc, sub_edge_attr = self.get_cached_k_hop_subgraph(raw_data)
        
        x = sub_nodes
        
        seed_nodes_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        seed_nodes_mask[seed_ent_loc] = 1
        label_nodes_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        ans_node_idx = [self.ent2id[e_] for e_ in raw_data["answer"]]
        
        penalty = 0
        for aid in ans_node_idx:
            if not torch.any(sub_nodes == aid):
                penalty += 1
            label_nodes_mask[sub_nodes == aid] = 1

        assert len(ans_node_idx) == penalty + label_nodes_mask.sum()
        penalty = torch.LongTensor([penalty])
        
        return GraphData(x, #entities around seed_n
                         sub_edge_index, #2D array [head], [tail]
                         sub_edge_attr, #edge connecting [head], [tail]
                         seed_nodes_mask, #Tensor that mask the seed node
                         label_nodes_mask,#Tensor that mask the answer nodes among all nodes around seed node
                         split, #name of data (train, test dev)
                         ex_id, #query id
                         raw_data["id"], #seed node
                         ques_str,  #seed_node, relation
                         raw_data["answer"], #answers id
                         knn_ids, #this is not calculated here (???)
                         penalty) #If answer Id is not among the nodes in subgraph        
        

    def get_cached_k_hop_subgraph(self, query):
    
        e1 = query['seed_entities'][0] #query head
        assert self.all_paths is not None

        paths_e1 = self.all_paths[e1][0]

        all_entities = set()
        sub_edge_indx = [[], []] #[head],[tail] 
        sub_edge_attr = []
        sub_nodes = []
        local_vocab = {}

        all_entities.add(e1)
        for path in paths_e1:

            for rel, ent in path:
                all_entities.add(ent)


        for i in range(len(self.id2ent)):
            if self.id2ent[i] in all_entities:
                local_vocab[self.id2ent[i]] = len(local_vocab)
                sub_nodes.append(i) 


        #Lets create 2D array (sub_edge_indx)
        #in the local subgraph first

        seen_edges = set()
        for path in paths_e1:
            curr_ent = e1
            for rel, ent in path:
                if (curr_ent, rel, ent) not in seen_edges:
                    if rel == "indication": 
                        continue     
                    sub_edge_indx[0].append(local_vocab[curr_ent])
                    sub_edge_indx[1].append(local_vocab[ent])                        
                    sub_edge_attr.append(self.rel2id[rel])
                    seen_edges.add((curr_ent, rel, ent))


                    #reverse edges
                    rel_inv = rel + "_inv"
                    inv_r = self.rel2id[rel_inv]

                    if (ent, inv_r, curr_ent) not in seen_edges:
                        sub_edge_indx[0].append(local_vocab[ent])
                        sub_edge_indx[1].append(local_vocab[curr_ent])
                        sub_edge_attr.append(inv_r)
                        seen_edges.add((ent, inv_r, curr_ent))

                curr_ent = ent


        seed_ent_loc = []
        for i, ent in enumerate(sub_nodes):
            if ent == self.ent2id[e1]:
                seed_ent_loc.append(i)
                break


        return torch.LongTensor(sub_nodes), torch.LongTensor(sub_edge_indx), \
               torch.LongTensor(seed_ent_loc), torch.LongTensor(sub_edge_attr)