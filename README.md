#  Digital Twin Mobility Profiling: A Spatio-Temporal Graph Learning Approach
This is the implementation of traffic prediction code in DTMP based on PyTorch. 

## structure of the code:  

- `data` folder: storing PEMSD4 and PEMSD8 dataset. You may refer to [ASTGCN](https://github.com/Davidham3/ASTGCN/tree/master/data) for more details; 
- `lib` folder: some methods for data loading and processing from [AGCRN](https://github.com/LeiBAI/AGCRN); 
- `utils.py`: method of loading adjacency graph;  
- `model.py`: implementation of Anet;  
- `train.py`, `run.py`: train and run the model.   
 
You can use `python run.py --dataset PEMSD4 --num_nodes 307` command to run the code.

