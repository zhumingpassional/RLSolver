# RLCO: High-performance RL-based Solvers for combinatorial optimization (CO) Problems Using Massively Parallel Simulations

We aim to showcase that reinforcement learning (RL) or machine learning (ML) with GPUs delivers the best benchmark performance for large-scale combinatorial optimization (CO) problems. RL with the help of GPU computing can obtain high-quality solutions within short time. 

# Overview
<a target="\_blank">
	<div align="center">
		<img src=fig/RLCO_framework.png width="80%"/>
	</div>
</a>  

# Key Technologies
- **Massively parallel environments** of Markov chain Monte Carlo (MCMC) simulations on GPU using thousands of CUDA cores and tensor cores.

# Key References

- Mazyavkina, Nina, et al. "Reinforcement learning for combinatorial optimization: A survey." Computers & Operations Research 134 (2021): 105400.

- Bengio, Yoshua, Andrea Lodi, and Antoine Prouvost. "Machine learning for combinatorial optimization: a methodological tour d’horizon." European Journal of Operational Research 290.2 (2021): 405-421.

- Peng, Yun, Byron Choi, and Jianliang Xu. "Graph learning for combinatorial optimization: a survey of state-of-the-art." Data Science and Engineering 6, no. 2 (2021): 119-141.

- Nair, Vinod, et al. "Solving mixed integer programs using neural networks." arXiv preprint arXiv:2012.13349 (2020).

- Makoviychuk, Viktor, et al. "Isaac Gym: High performance GPU based physics simulation for robot learning." Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). 2021.
  


## File Structure

```
RLCO
└──baseline
    └──greedy.py
    └──gurobi.py
    └──mcpg.py
    └──random_walk.py
    └──scip.py
    └──simulated_annealing.py
    └──S2V_DQN
    └──DIMES
    └──iSCO
    └──PI-GNN
└──data
└──result
└──config.py
└──evaluator.py
└──network.py
└──main.py
└──l2a_xx.py (ours)
└──util.py
└──util_results.py
└──README.md


```

## Structure
<a target="\_blank">
	<div align="center">
		<img src=fig/RLSolver_structure.png width="70%"/>
	</div>
</a>  

## Datasets
Link: https://pan.baidu.com/s/1Qg-VEMvrAB_cUpYLMBubiw (CODE: gc8y)

1) Graph
   
Most data is graph, such as graph maxcut, graph partitioning, TSP.

- [Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Standford university, and is stored in the "data" folder of this repo. The number of nodes is from 800 to 10000. 
  
- __Syn__ is the synthetic data. The number of nodes is from 10 to 50000. The (partial) synthetic data is stored in the "data" folder of this repo. If users need all the synthetic data, please refer to [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE hojh for China users).
  
2) Non-graph

The data is not graph, such as the set cover problem, knapsack problem, and binary integer linear programming (BILP).

  
## Results

Link: https://pan.baidu.com/s/1Qg-VEMvrAB_cUpYLMBubiw (CODE: gc8y)

## Run algorithms

Process 1 (select problem):

config.py
```
PROBLEM = Problem.maxcut
```
We can select the problems including maxcut, graph partitioning, maximum independent set, set cover, TSP, etc. 

Process 2 (run algorithm):

```
python baseline/greedy.py  # run greedy
python baseline/gurobiy.py  # run gurobi
python baseline/mcpg.py  # run mcpg
python baseline/simulated_annealing.py  # run simulated_annealing
python baseline/S2V_DQN.py  # run S2V_DQN
python baseline/DIMES.py  # run DIMES
```
```
python l2a.py  # our algorithm
```


## Solvers to Compare with

[Gurobi](https://www.gurobi.com/)

[SCIP](https://www.scipopt.org/index.php#welcome)

## Benchmarks


* Learning to branch
  
[code](https://github.com/cwfparsonson/retro_branching/tree/master) 2023 AAAI Reinforcement Learning for Branch-and-Bound Optimisation using Retrospective Trajectories 

[code](https://github.com/ds4dm/branch-search-trees) 2021 AAAI Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies

* Learning to cut

[code](https://github.com/Wenbo11/learntocut) 2020 ICML Reinforcement learning for integer programming: Learning to cut


* RL/ML-based heuristic
  
[code](https://github.com/Hanjun-Dai/graph_comb_opt)  (greedy) 2017 NeurIPS Learning Combinatorial Optimization Algorithms over Graphs

[code](https://github.com/optsuite/MCPG) (local search) 2023, A Monte Carlo Policy Gradient Method with Local Search for Binary Optimization

[code](https://github.com/JHL-HUST/VSR-LKH) (LKH for TSP) 2021 AAAI Combining reinforcement learning with Lin-Kernighan-Helsgaun algorithm for the traveling salesman problem 

* Variational annealing

[code](https://github.com/zhumingpassional/Maxcut/tree/master/baseline/variational_classical_annealing_RNN) (VCA_RNN) 2023 Machine_Learning Supplementing recurrent neural networks with annealing to solve combinatorial optimization problems

[code](https://github.com/zhumingpassional/Maxcut/tree/master/baseline/variational_neural_annealing) (VNA) 2021 Nature_Machine_Intelligence Variational neural annealing

* Classical methods
  - [Random walk](https://github.com/zhumingpassional/Maxcut/blob/master/baseline/random_walk.py)
  - [Greedy](https://github.com/zhumingpassional/Maxcut/blob/master/baseline/greedy.py)
  - [Simulated annealing](https://github.com/zhumingpassional/Maxcut/blob/master/baseline/simulated_annealing.py)
  - Local search
  - Beam search
  - Tabu search
  - Branch-and-bound
  - Cutting plane

## Store Results 

Results will be written to a file result.txt in the folder "result". The first column is the node, and the second column is the label of classified set. For example, 

1 2  # node 1 in set 2

2 1  # node 2 in set 1

3 2  # node 3 in set 2

4 1  # node 4 in set 1

5 2  # node 5 in set 2

The partial results are stored in [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing).

## Performance
In the following experiments, we use GPU during training by default. 


1) __Gset__

[Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Stanford university. 

| graph | #nodes| #edges | BLS | DSDP    | KHLWG   | RUN-CSP| PI-GNN| Gurobi (1 h)  |Gap           | Ours | Improvement | 
|--- |------|----  |---        |-----    |-----    |--------|-------| ---           | ---           | ----| ----|
|G14 | 800  | 4694 | __3064__  |         | 2922    | 3061   | 2943  |3042           | 3.61\%        | __3064__ | +0\%|
|G15 | 800  | 4661 | __3050__  | 2938    |__3050__ | 2928   | 2990  |3033           |3.33\%         | __3050__ | +0\% | 
|G22 | 2000 | 19990|__13359__  | 12960   |__13359__| 13028  | 13181 |13129          | 28.94\%       | __13359__ |  +0\% | 
|G49 | 3000 | 6000 | __6000__  | __6000__|__6000__ |__6000__| 5918  |___6000__      |0              | __6000__|  +0\% | 
|G50 | 3000 | 6000 | __5880__  | __5880__|__5880__ |__5880__| 5820  |__5880__       |0              | __5880__|  +0\% | 
|G55 | 5000 | 12468| 10294     | 9960    | 10236   | 10116  | 10138 | 10103         | 11.92\%       |__10298__ |  +0.04\% | 
|G70 | 10000| 9999 |9541       | 9456    | 9458    | -      | 9421  | 9490          |2.26\%         |__9583__ | +0.44\% | 

2) __Syn__ 

We use the whole synthetic data with 3 distributions: barabasi albert (BA), erdos renyi (ER), and powerlaw (PL). For graphs with n nodes, there are 10 datasets, and we run once for each dataset, and finally calcualte the average objective values. 

Results on the BA distribution.
|Nodes | Greedy | SDP  | SA       | GA     | Gurobi (1 h) | PI-GNN | iSCO   | MCPG   | Ours| 
|----------|-------|------| -------- |--------|--------      | ------ |------  |--------| ------ |
|100   |272.1  |272.5 | 272.3    |275.9   |__284.1__     | 273.0  |__284.1__|__284.1__| __284.1__|
|200   |546.9  |552.9 | 560.2    |562.3   |__583.0__     | 560.6  |581.5   |__583.0__| __583.0__ |
|300   | 833.2 |839.3 | 845.3    |842.6   |__880.4__     |  846.3 |877.2   |__880.4__ | __880.4__  |
|400   |1112.1 |1123.9| 1134.6   | 1132.4 |1180.4        | 1174.6 |1176.5  |1179.5| __1181.9__ |
|500   |1383.8 |1406.3| 1432.8   |1450.3  |1476.0        | 1436.8 |1471.3  |__1478.3__| __1478.3__ |
|600   |1666.7 |1701.2| 1770.3   |1768.5  |1777.0        | 1768.5 |1771.0  |1778.6| __1781.5__ |
|700   |1961.9 |1976.7| 1984.3   |1989.2  |2071.2        | 1989.4 |2070.2  |__2076.6__| __2076.6__ |  
|800   |2237.9 |2268.8| 2273.6   |2274.8  |2358.9        | 2365.9 |2366.9  |2372.9| __2377.8__ |
|900   |2518.1 |2550.3| 2554.3   |2563.2  |2658.3        | 2539.7 |2662.4  |2670.6| __2675.1__|
|1000  |2793.8 |2834.3| 2856.2   |2861.3  |2950.2        | 2846.8 |2954.0  |2968.7| __2972.3__ |



