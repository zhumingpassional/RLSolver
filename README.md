# AI4Maxcut
High performance GPU-based solver for the Graph Maxcut Problem

We aim to showcase that reinforcement learning (RL) or machine learning (ML) with GPUs delivers the best benchmark performance for large-scale graph maxcut problems. When the size of these problems becomes large, it is very hard to obtain optimal or near optimal solutions. RL with the help of GPU computing can obtain high-quality solutions within short time. AI4Maxcut collects many RL/ML tricks and also operations research (OR) tricks to improve the performance. We encourage users to try RL/ML tricks, and implement their own new ideas based on the datasets. If encountering any problems, please submit github issues, and we can talk there.

# Key Technologies
- **Massively parallel sampling** of Markov chain Monte Carlo simulations on GPU, using thousands of CUDA cores and tensor cores.
- **Podracer scheduling** on a GPU cloud, e.g., DGX-2 SuperPod.
- **OR tricks** such as local search, and tabu search.
- **RL/ML tricks** such as learn to optimize, and curriculum learning.

- 

