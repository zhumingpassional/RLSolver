# AI4Maxcut: High performance GPU-based solver for the Graph Maxcut Problem

We aim to showcase that AI, especially reinforcement learning (RL) or machine learning (ML) with GPUs delivers the best benchmark performance for large-scale graph maxcut problems. AI4Maxcut collects many RL/ML tricks and also operations research (OR) tricks to improve the performance. We encourage users to try AI tricks, and implement their own new ideas based on the datasets. If encountering any problems, please submit github issues, and we can talk there.

# Key Technologies
- **Massively parallel sampling** of Markov chain Monte Carlo simulations on GPU, using thousands of CUDA cores and tensor cores.
- **Podracer scheduling** on a GPU cloud, e.g., DGX-2 SuperPod.
- **OR tricks** such as local search, and tabu search.
- **AI tricks** such as learn to optimize, and curriculum learning.


# Workflow
<a target="\_blank">
	<div align="center">
		<img src=fig/work_flow.png width="90%"/>
	</div>
</a>  



## Datasets
 
1) [Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Standford university, and is stored in the "data" folder of this repo. The number of nodes is from 800 to 10000. 
  
2) __Syn__ is the synthetic data obtained by calling the function generate_write in utils.py. The number of nodes is from 10 to 50000. The (partial) synthetic data is stored in the "data" folder of this repo. If users need all the synthetic data, please refer to [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE hojh for China users). 
  


## Benchmarks


* Learning to branch
  
[code](https://github.com/cwfparsonson/retro_branching/tree/master) 2023 AAAI Reinforcement Learning for Branch-and-Bound Optimisation using Retrospective Trajectories 

[code](https://github.com/ds4dm/branch-search-trees) 2021 AAAI Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies

* Learning to cut

[code](https://github.com/Wenbo11/learntocut) 2020 ICML Reinforcement learning for integer programming: Learning to cut


* AI-based heuristic
  
[code](https://github.com/Hanjun-Dai/graph_comb_opt)  (greedy) 2017 NeurIPS Learning Combinatorial Optimization Algorithms over Graphs

[code](https://github.com/optsuite/MCPG) (local search) 2023, A Monte Carlo Policy Gradient Method with Local Search for Binary Optimization


* Classical methods
  - Random walk
  - Greedy
  - $\epsilon$-greedy
  - Simulated annealing
  - Local search
  - Beam search
  - Tabu search
  - Branch-and-bound
  - Cutting plane


## Solvers to Compare with

[Gurobi](https://www.gurobi.com/) is the state-of-the-art solver. The license is required, and professors/students at universities can obtain the __academic license for free__. We recommend to use Gurobi if users have licenses, since its performance is the best.

[SCIP](https://www.scipopt.org/index.php#welcome) is a well-known open-source solver, and its simplex is commonly used in "learn to branch/cut". If users do not have Gurobi licenses, SCIP is a good choice since it is __open-source and free__. Although its performance is not as good as Gurobi, we recommend to use SCIP if users do not have Gurobi licenses. 


## Other Solvers

[COPT](https://www.copt.de/): a mathematical optimization solver for large-scale problems.

[CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer): a high-performance mathematical programming solver for linear programming, mixed integer programming, and quadratic programming.

[Xpress](https://www.fico.com/en/products/fico-xpress-optimization): an extraordinarily powerful, field-installable Solver Engine.

[BiqMac](https://biqmac.aau.at/): a solver only for binary quadratic or maxcut. Users should upload txt file, but the response time is not guaranteed. If users use it, we recommend to [download](https://biqmac.aau.at/) the sources and run it by local computers. 


## Store Results 

Partial results are stored in the folder "result" of this repo. All the results are stored in [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE: hojh for China users). 

With respect to maxcut, please refer to this [website](https://github.com/zhumingpassional/AI4Maxcut). With respect to TSP, please refer to this [website](https://github.com/zhumingpassional/AI4TSP). 


## File Structure

```
AI4Maxcut
└──data
└──mcmc_sim
└──result
└──learn_to_anneal_x.py (ours)
└──gurobi.py
└──scip.py
└──random_walk.py
└──greedy.py
└──simulated_annealing.py
└──utils.py
└──opt_methods
└──README.md

```

## Finished
- [x] Maxcut
## TODO
- [ ] Learning greedy



## Related Websites
+ [RLSolver](https://github.com/AI4Finance-Foundation/RLSolver))
+ [Benchmarks for optimization softwre](http://plato.asu.edu/bench.html) 

