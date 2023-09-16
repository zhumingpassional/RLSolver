# Maxcut using machine learning

- **RL tricks** such as learn to optimize, and curriculum learning.
- **Massively parallel sampling** on GPU, using thousands of CUDA cores and tensor cores.

## File Structure

```
Maxcut
└──data
└──mcmc_sim
└──result
└──learn_to_anneal_x.py (ours)
└──gurobi.py
└──scip.py
└──random_walk.py
└──README.md
└──greedy.py
└──simulated_annealing.py
└──utils.py


```

## Datasets
 
1) [Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Standford university, and is stored in the "data" folder of this repo. The number of nodes is from 800 to 10000. 
  
2) __Syn__ is the synthetic data obtained by calling the function generate_write in utils.py. The number of nodes is from 10 to 50000. The (partial) synthetic data is stored in the "data" folder of this repo. If users need all the synthetic data, please refer to [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE hojh for China users). 
  
Take gset_14.txt as an example,

800 4694 # the number of nodes is 800, and the number of edges is 4694

1 7 1 # the edge connects node 1 and 7, and its weight is 1

1 10 1 # the edge connects node 1 and 10, and its weight is 1

...

### Generate synthetic data

If users want to generate a graph with n nodes and m edges, please use the function __generate_write__ in utils.py. It returns an adjacency matrix and a [networkx](https://networkx.org/documentation/stable/reference/introduction.html) graph, and the graph will be written to a file "syn_n_m.txt" in the folder "data". 

### Read data

We use the function __read_txt__ in utils.py to read the data, which returns a [networkx](https://networkx.org/documentation/stable/reference/introduction.html) graph. We can access the nodes and edges by graph.nodes and graph.edges, respectively. 


## Run algorithms

Format:
```
python learn_to_anneal.py  # xxx.py is the file name of the algorithm
```

## Solvers to Compare with

[Gurobi](https://www.gurobi.com/) is the state-of-the-art solver. The license is required, and professors/students at universities can obtain the __academic license for free__. We recommend to use Gurobi if users have licenses, since its performance is the best.

[SCIP](https://www.scipopt.org/index.php#welcome) is a well-known open-source solver, and its simplex is commonly used in "learn to branch/cut". If users do not have Gurobi licenses, SCIP is a good choice since it is __open-source and free__. Although its performance is not as good as Gurobi, we recommend to use SCIP if users do not have Gurobi licenses. 

## Store Results 

Results will be written to a file result.txt in the folder "result". The first column is the node, and the second column is the label of classified set. For example, 

1 2  # node 1 in set 2

2 1  # node 2 in set 1

3 2  # node 3 in set 2

4 1  # node 4 in set 1

5 2  # node 5 in set 2

The partial results are stored in the folder "result" in this repo. All the results are stored in [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE: hojh for China users).  

## Performance
In the following experiments, we use GPU during training by default. 

When use solvers, "gap" is calculated based on the objective of its solution and the best bound. When we use our method, "gap_best" is calculated based on the objective of our solution and the best one over other methods. To distinguish them, we use "gap_best" for our method. gap_best = $\frac{obj - obj*} { obj*}$, where $obj$ is the objective value of our method, and $obj*$ is the best objective value over all comparison methods. Therefore, we see that the solution of solvers may be better than ours, but the "gap" of solvers is larger than "gap_best" of our method, which is caused by different calculations.

1) __Gset__

[Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Stanford university. 

| graph | #nodes| #edges | BLS | DSDP | KHLWG | RUN-CSP | PI-GNN | Gurobi (0.5 h) | Gap | Gurobi (1 h) |Gap | Gurobi (10 h) |Gap | Ours | Gap_best | 
|---|----------|----|---|-----|-----|--------|----------|------| ---| ---| ----|----| ---| ----|----|
|G14 | 800 | 4694 | __3064__| | 2922 | 3061 | 2943  |3034 | 4.15%|3042| 3.61\%|3046|3.22\%| 3029 | 1.14\%|
|G15 | 800 | 4661 | __3050__ | 2938 | __3050__ | 2928 | 2990  | 3016| 4.31%|3033|3.33\%| 3034| 3.07\%| 2995 | 1.80\% | 
|G22 | 2000 | 19990 |__13359__ | 12960 | __13359__ | 13028 | 13181  |13062 |37.90%|13129| 28.94\%|13159| 21.83\%| 13167 |  1.44\% | 
|G49 | 3000 | 6000 | __6000__ | __6000__ | __6000__ | __6000__ | 5918  |__6000__ |0|__6000__ |0| __6000__ |0 | 5790|  3.50\% | 
|G50 | 3000 | 6000 | __5880__ | __5880__ | __5880__ | __5880__ | 5820  |__5880__|0|__5880__|0| __5880__|0 | 5720|  2.72\% | 
|G55 | 5000 | 12468 | __10294__ | 9960 | 10236 | 10116 | 10138  | 10103 | 15.39\%|10103| 11.92\%|10103 | 10.69\%  |10017 |  2.69\% | 
|G70 | 10000 | 9999 |__9541__ | 9456 | 9458 | - | 9421  | 9489 | 2.41\% |9490|2.26\%| 9580| 0.96\% |9358 | 1.92\% | 

2) __Syn__ 

We use the whole synthetic data. For graphs with n nodes, there are 5 datasets, and we run once for each dataset, and finally calcualte the average and standard deviation for the objective values. 

In the following table, the first row illustrates the limited time for solvers. We see that, when the number of nodes is not larger than 100, the optimal solutions are obtained, and the average running duraton is much less than 0.5 hour. The inference time of our method is less than 0.001 second.
 
|Datasets |Gurobi (0.5 h)| Gap |Gurobi (1 h) | Gap |Gurobi (1.5 h) |Gap |Ours|Gap_best |
|-------|------|----| ---- |------|----| ---- |---- |--|
|syn_10   | 17.40 $\pm$ 0.80 (0.004s) | 0| 17.40 $\pm$ 0.80 (0.004s)| 0 | 17.40 $\pm$ 0.80 (0.004s)| 0| $\pm$  | |  
|syn_50   | 134.20 $\pm$ 2.04 (0.30s)  | 0| 134.20 $\pm$ 2.04 (0.30s)| 0  | 134.20 $\pm$ 2.04 (0.30s)| 0|  $\pm$   |  |  
|syn_100  |  337.20 $\pm$ 2.71 (289.99s) |0 | 337.20 $\pm$ 2.71 (289.99s)| 0 | 337.20 $\pm$ 2.71 (289.99s)| 0|  $\pm$ |   |  
|syn_300   | 1403.80 $\pm$ 7.44 (1800s)  | 9.58\%| 1404.00 $\pm$ 7.54 (3600s) | 9.18\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_500   |  2474.00 $\pm$ 13.89 (1800s)  | 13.93\%| 2475.40 $\pm$ 15.00 (3600s)| 13.86\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_700   |  2849.60 $\pm$ 14.08 (1800s)  | 13.55\%| 2852.2 $\pm$ 14.30 (3600s) | 13.26\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_900   |  3622.20 $\pm$ 11.84 (1800s)  | 14.32\% | 3624.00 $\pm$ 9.86 (3600s) | 13.88\%   | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_1000  |  4435.80 $\pm$ 18.14 (1800s)  | 15.95%| 4437.8 $\pm$ 16.85 (3600s) |  15.59\%   | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_3000  |  17111.00 $\pm$ 16.70 (1800s)  | 36.49\% | 17145.00 $\pm$ 33.60 (3600s) | 32.73\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_5000  |  30376.60 $\pm$ 243.14 (1800s)  | 54.83\% | 30500.80 $\pm$ 223.32 (3600s) | 52.17\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_7000  |  46978.60 $\pm$ 746.83 (1800s)  | 61.00\% | 47460.00 $\pm$ 473.76 (3600s) |  56.87\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_9000  |  57730.20 $\pm$ 502.51 (1800s)  | 60.30\% | 57730.20 $\pm$ 502.51 (3600s) | 60.00\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_10000 |  60768.40 $\pm$ 585.41 (1800s)  | 59.54\% | 60768.40 $\pm$ 585.41 (3600s) |  58.67\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  


