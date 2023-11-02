import numpy as np
import torch
import torch.nn as nn

class query_knn(nn.Module):
    def __init__(self, dataset_obj, device):
        super(query_knn, self).__init__()
        self.dataset_obj = dataset_obj
        self.ent2id = dataset_obj.ent2id
        self.rel2id = dataset_obj.rel2id
        self.id2ent = dataset_obj.id2ent
        self.device = device
        self.entity_vectors = self.calculate_entity_vectors(dataset_obj).to(self.device)

    def calculate_entity_vectors(self, dataset_obj):
        # Create a matrix for entity vectors
        entity_vectors = np.zeros((len(self.ent2id), len(self.rel2id)))

        # Populate the entity_vectors matrix
        for edge_counter, source_entity in enumerate(dataset_obj.full_edge_index[0]):
            entity_vectors[source_entity, dataset_obj.full_edge_attr[edge_counter]] = 1

        # Calculate square root and L2 normalization
        entity_vectors = np.sqrt(entity_vectors)
        l2norm = np.linalg.norm(entity_vectors, axis=-1)
        l2norm += np.finfo(np.float).eps
        entity_vectors /= l2norm.reshape(l2norm.shape[0], 1)

        return torch.Tensor(entity_vectors)

    def calculate_similarity(self, query_entities):
        """
        Dot product calculation between given query and all other entities 
        """
        # Select query entities' vectors
        query_entities_vec = self.entity_vectors[query_entities]

        # Calculate similarity matrix
        similarity = torch.matmul(query_entities_vec, self.entity_vectors.T)
        return similarity

    def forward(self, query_list, k=None, **kwargs):
        """
        query_list: queries of interest
        k: Number of nearest neighbor to consider
        return: list of knn
        """
        
        neighbor_list, neighbor_slices = [], [0]
        query_entity_indices = torch.LongTensor([self.ent2id[query.query[0]] for query in query_list]).to(self.device)
        similarity_matrix = self.calculate_similarity(query_entity_indices)

        #Sort nearest neighbors
        nearest_neighbors_1_hop = torch.argsort(-similarity_matrix, dim=-1).cpu()

        all_knn_ids = []
        for query_index in range(len(query_list)):
            knn_ids_with_relation = []
            knn_indices = nearest_neighbors_1_hop[query_index]

            query_entity, relation = query_list[query_index].query

            #make sure knn entities are on training 
            for index in knn_indices:
                if self.id2ent[index.item()] != query_entity:
                    query_relation = [item for item in self.dataset_obj.train_idmap if item[0] == str(self.id2ent[index.item()])]

                    if query_relation: #make sure knn have relation of interest
                        knn_ids_with_relation.append(self.dataset_obj.train_idmap[(self.id2ent[index.item()], relation)])

                        if len(knn_ids_with_relation) >= (k + 5):
                            break

            all_knn_ids.append(knn_ids_with_relation)

        # Choose the top-K neighbors
        for query_index, query_item in enumerate(query_list):
            neighbor_list.extend(
                    [query_item] + [self.dataset_obj.train_dataset[knn_id] for knn_id in all_knn_ids[query_index][:k]])
            neighbor_slices.append(len(neighbor_list))

        assert neighbor_slices[-1] == len(neighbor_list)
        return neighbor_list, neighbor_slices
