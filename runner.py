from transformers import TrainingArguments, HfArgumentParser
from src.data_loader.query_dataloader import DataLoader
from src.cbr_trainer.cbrTrainer import cbrTrainer
from tqdm import tqdm, trange

from dataclasses import dataclass, field,asdict
import torch
from src.models.rgcn_model import RGCN
import datetime
import wandb
import logging

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

     
def main():
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("=====Parsing Arguments=====")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CBRArguments))
    model_args, train_args, cbr_args = parser.parse_args_into_dataclasses()
    fileHandler = logging.FileHandler("{0}/{1}".format(train_args.output_dir, "log.txt"))
    model_args.device = device 
    
    
    #WandB arguments
    if train_args.use_wandb:
        config = dict()
        args = [ model_args, train_args, cbr_args ]

        for object_attributes in args: 
            args_dict = asdict(object_attributes)
            
            for attribute_name, attribute_value in args_dict.items(): 
                config[attribute_name] = attribute_value
        
        wandb.init(project=cbr_args.data_name, config = config)
       
        # Format the current date and time
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        wandb.run.name = f"{formatted_datetime}_{cbr_args.data_name}" #Run #Name
        
        
    #Load data 
    logger.info("=====Loading Data=====")
    dataset_obj = DataLoader(cbr_args.data_dir, 
                            cbr_args.data_name,
                            cbr_args.paths_file_dir,
                            cbr_args.train_batch_size,
                            cbr_args.eval_batch_size)

    #Set RGCN model 
    logger.info("=====Loading Model=====")
    rgcn_model = RGCN(n_entities =dataset_obj.n_entities, 
                      n_relations= dataset_obj.n_relations, 
                      params = model_args).to(device)
    
   
    #train 
    logger.info("=====Setting Training=====")
    trainer = cbrTrainer(rgcn_model, 
                        dataset_obj, 
                        model_args,
                        train_args, 
                        cbr_args, 
                        device)
    
    for epoch in trange(train_args.num_train_epochs, desc=f"[Full Loop]"):
        train_loss, results_train = trainer.train() #train
        results_dev = trainer.run_evaluate("dev", dataset_obj.dev_dataloader) #evaluaationß
        
        logger.info('[Epoch:{}]:  Training Loss:{:.4} Training MRR:{:.4}'.format(epoch, 
                                                                                 train_loss, 
                                                                                 results_train['avg_rr']))
        if train_args.use_wandb:
            # tracks avg_rr on current batch
            wandb.log({'Loss Epoch':train_loss,
                  "MRR Train": results_train['avg_rr'],
                  "MRR Dev":results_dev['avg_rr'],
                   "Hits@1 Dev":results_dev.get('avg_hits@1', 0),
                   "Hits@3 Dev":results_dev.get('avg_hits@3', 0),
                   "Hits@5 Dev":results_dev.get('avg_hits@5', 0),
                   "Hits@10 Dev":results_dev.get('avg_hits@10', 0)})
@dataclass
class ModelArguments: 
    gcn_dim_init: int = field(default=32, metadata={"help": "Intial GCN layer dimensionality"})
    hidden_channels_gcn: int = field(default=32, metadata={"help": "Hidden GCN layer dimensionality"})
    drop_gcn: float = field(default=0.0, metadata={"help": "Dropout probability for RGCN model"})
    conv_layers: int = field(default=1, metadata={"help": "Number of GCN layers"})
    transform_input: int = field(default = 0, metadata = {"help":"Linear transformation over to input model"})

@dataclass
class DataTrainingArguments(TrainingArguments): 
    use_wandb: int = field(default=0, metadata={"help": "use wandb if 1"})
    dist_metric: str = field(default='l2', metadata={"help": "Options [l2, cosine]"})
    dist_aggr1: str = field(default='mean', metadata={"help": "Distance aggregation function at each neighbor query. "
                                                              "Options: [none (no aggr), mean, sum]"})
    dist_aggr2: str = field(default='mean', metadata={"help": "Distance aggregation function across all neighbor "
                                                              "queries. Options: [mean, sum]"}) 
    sampling_loss: float = field(default=1.0, metadata={"help": "Fraction of negative samples used"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for temperature scaled cross-entropy loss"})
    learning_rate: float = field(default=0.001, metadata={"help": "Starting learning rate"})
    warmup_step: int = (field(default=0, metadata={"help": "scheduler warm up steps"}))
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW"})
    num_train_epochs: int = field(default=5, metadata={"help": "Total number of training epochs to perform."})
    gradient_accumulation_steps: int = field( default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    check_steps: float = field(default=5.0, metadata={"help": "Steps to check training"})
    #     output_dir: str = field(default = "01_results/", metadata ={"help": "Path to directory to save results"})


@dataclass
class CBRArguments: 
    data_name: str = field(default = "MIND", metadata = {"help": "KG dataset"})
    data_dir: str = field(default= "src/00_data/", metadata={"help": "Path to data directory (contains train, test, dev)"})
    paths_file_dir: str = field(default = 'MIND_cbr_subgraph_knn-5_branch-200.pkl', metadata = {"help": "Paths file name"})
    train_batch_size: int = field(default = 1, metadata = {"help": "Training batch size"})
    num_neighbors_train: int = field(default = 5, metadata = {"help": "Number of near-neighbor entities for training"})           
    eval_batch_size: int = field(default = 1, metadata = {"help": "Test/Dev batch size"})
    num_neighbors_eval: int = field(default = 5, metadata = {"help": "Number of near-neighbor entities for test"})            

if __name__ == '__main__':
    main()

