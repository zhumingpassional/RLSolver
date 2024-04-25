# RLCO: High-performance RL-based Solvers for CO Problems Using Massively Parallel Simulations

We aim to showcase that reinforcement learning (RL) or machine learning (ML) with GPUs delivers the best benchmark performance for large-scale combinatorial optimization (CO) problems. RL with the help of GPU computing can obtain high-quality solutions within short time. 


# Key Technologies
- **RL/ML tricks** such as learn to optimize and curriculum learning.
- **OR tricks** such as local search and tabu search.
- **Massively parallel sampling** of Markov chain Monte Carlo (MCMC) simulations on GPU using thousands of CUDA cores and tensor cores.
- **Podracer scheduling** on a GPU cloud such as DGX-2 SuperPod.
- 
# Key References

- Mazyavkina, Nina, et al. "Reinforcement learning for combinatorial optimization: A survey." Computers & Operations Research 134 (2021): 105400.

- Bengio, Yoshua, Andrea Lodi, and Antoine Prouvost. "Machine learning for combinatorial optimization: a methodological tour d’horizon." European Journal of Operational Research 290.2 (2021): 405-421.

- Peng, Yun, Byron Choi, and Jianliang Xu. "Graph learning for combinatorial optimization: a survey of state-of-the-art." Data Science and Engineering 6, no. 2 (2021): 119-141.

- Nair, Vinod, et al. "Solving mixed integer programs using neural networks." arXiv preprint arXiv:2012.13349 (2020).

- Makoviychuk, Viktor, et al. "Isaac Gym: High performance GPU based physics simulation for robot learning." Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). 2021.
  

# Workflow
<a target="\_blank">
	<div align="center">
		<img src=fig/work_flow.png width="60%"/>
	</div>
</a>  

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
    └──S2V_DQN.PY
    └──DIMES.py
└──data
└──result
└──mcmc.py
└──l2a_xx.py (ours)
└──README.md
└──utils.py


```

## Structure
<a target="\_blank">
	<div align="center">
		<img src=fig/RLSolver_structure.png width="70%"/>
	</div>
</a>  

## Datasets
Link: https://pan.baidu.com/s/1Qg-VEMvrAB_cUpYLMBubiw (CODE: gc8y)
 
1) [Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Standford university, and is stored in the "data" folder of this repo. The number of nodes is from 800 to 10000. 
  
2) __Syn__ is the synthetic data. The number of nodes is from 10 to 50000. The (partial) synthetic data is stored in the "data" folder of this repo. If users need all the synthetic data, please refer to [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE hojh for China users). 
  

## Results

Link: https://pan.baidu.com/s/1Qg-VEMvrAB_cUpYLMBubiw (CODE: gc8y)

## Run algorithms

Process 1 (select problem):

config.py
```
PROBLEM = Problem.maxcut
```
We can select the problems including maxcut, graph partitioning, maximum independent set, set cover, tsp, etc. 

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

| graph | #nodes| #edges | BLS | DSDP    | KHLWG   | RUN-CSP| PI-GNN| Gurobi (0.5 h)  | Gurobi (1 h)  |Gap           | Ours | Improvement | 
|--- |------|----  |---        |-----    |-----    |--------|-------|------           | ---           | ---           | ----| ----|
|G14 | 800  | 4694 | __3064__  |         | 2922    | 3061   | 2943  |3034             |3042           | 3.61\%        | __3064__ | +0\%|
|G15 | 800  | 4661 | __3050__  | 2938    |__3050__ | 2928   | 2990  | 3016            | 3033          |3.33\%         | __3050__ | +0\% | 
|G22 | 2000 | 19990|__13359__  | 12960   |__13359__| 13028  | 13181 |13062            |13129          | 28.94\%       | __13359__ |  +0\% | 
|G49 | 3000 | 6000 | __6000__  | __6000__|__6000__ |__6000__| 5918  |__6000__         |__6000__       |0              | __6000__|  +0\% | 
|G50 | 3000 | 6000 | __5880__  | __5880__|__5880__ |__5880__| 5820  |__5880__         |__5880__       |0              | __5880__|  +0\% | 
|G55 | 5000 | 12468| 10294     | 9960    | 10236   | 10116  | 10138 | 10103           | 10103         | 11.92\%       |__10298__ |  +0.04\% | 
|G70 | 10000| 9999 |9541       | 9456    | 9458    | -      | 9421  | 9489            | 9490          |2.26\%         |__9583__ | +0.44\% | 

2) __Syn__ 

We use the whole synthetic data. For graphs with n nodes, there are 5 datasets, and we run once for each dataset, and finally calcualte the average and standard deviation for the objective values. 
 
|Datasets |Gurobi (0.5 h)| Gap |Gurobi (1 h) | Gap |Gurobi (1.5 h) |Gap |Ours|Gap |
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


