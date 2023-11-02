import torch
from torch.nn import Module
from torch_geometric.data.batch import Batch


class NNSubgraphs(Module):
    def __init__(self, dataset_obj, **kwargs):
        super(NNSubgraphs, self).__init__()
        self.dataset_obj = dataset_obj

    def forward(self, query_and_knn_list, nn_slices, **kwargs):
        """
        query_and_knn_list: List of items, each containing a query and its KNN.
        nn_slices: List of indices specifying the slices within the query_and_knn_list.
        Returns a Batch object and a list of new nn_slices.
        """
        
        for ctr, val in enumerate(query_and_knn_list):
            if val.x is None:
                # Get the corresponding item from the dataset based on its split.
                if val.split == 'train':
                    item = self.dataset_obj.raw_train_data[val.ex_id] 
                elif val.split == 'dev':
                    item = self.dataset_obj.raw_dev_data[val.ex_id]
                else:
                    item = self.dataset_obj.raw_test_data[val.ex_id]

        # Logic for eliminating queries (or all their KNNs) that don't have an answer
        if query_and_knn_list[0].split == "train":
            new_query_and_knn_list, new_nn_slices = [], [0]

            for ctr, idx in enumerate(nn_slices[:-1]):
                if query_and_knn_list[idx].label_node_ids.sum().item() == 0:
                    # Query does not capture an answer, so skip it.
                    continue
                
                total = 0
                for val in query_and_knn_list[idx + 1:nn_slices[ctr + 1]]:
                    # Calculate the total number of answer nodes in the query and its KNNs.
                    total += val.label_node_ids.sum().item()
                
                if total == 0:
                    # None of the KNNs have a label node with an answer.
                    continue
                
                # Include the query and its KNNs in the new lists.
                new_query_and_knn_list.extend(query_and_knn_list[idx:nn_slices[ctr + 1]])
                new_nn_slices.append(len(new_query_and_knn_list))
            
            query_and_knn_list = new_query_and_knn_list
            nn_slices = new_nn_slices
            assert nn_slices[-1] == len(query_and_knn_list)
            
            if len(query_and_knn_list) == 0:
                # None of the queries in the batch have an answer. Return None.
                return None, None
            
        # Create a Batch object from the filtered query and KNN data and return it with the new nn_slices.
        return Batch.from_data_list(query_and_knn_list, follow_batch=['x', 'edge_attr']), nn_slices