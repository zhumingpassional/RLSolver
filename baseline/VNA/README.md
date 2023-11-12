# Variational Neural Annealing
Variational neural annealing (VNA) is a framework to variationally simulate classical and quantum annealing for the purpose of solving optimization problems using neural networks. In this paper https://www.nature.com/articles/s42256-021-00401-3 (arXiv version: https://arxiv.org/abs/2101.10154), we show that we can implement a variational version of classical annealing (VCA) and its quantum counterpart (VQA) using recurrent neural networks. We find that our implementation significantly outperforms traditional simulated annealing in the asymptotic limit on prototypical spin models, suggesting the promising potential of this route to optimization.

This repository is aimed to facilitate the reproducibilty of the results of [our paper](https://www.nature.com/articles/s42256-021-00401-3).

Our implementation is based on [RNN wave functions's code](https://github.com/mhibatallah/RNNWavefunctions).

## Content

This repository contains a source code of our implementation and tutorials under the format of jupyter notebooks for demonstration purposes.

### `src`
This section contains our source code with the following implementations:

1. `src/VNA_1DTRNN`: an implementation of VNA using 1D Tensorized RNNs to find the ground state of a random ising chains with open boundary conditions. All you need to do is run the file `src/run_VNA_randomisingchain.py`.

2. `src/VNA_2DTRNN`: an implementation of VNA using 2D Tensorized RNNs to find the ground state of the 2D Edwards-Anderson model with open boundary conditions. To execute this module, you can run the file `src/run_VNA_EdwardsAnderson.py`.

3. `src/VNA_DilatedRNN`: an implementation of VNA using Dilated RNNs to find the ground state of the Sherrington-Kirkpatrick model. To execute this implementation, you can run the python file `src/run_VNA_SherringtonKirkpatrick.py`.

To be able to run `VCA` in each one of these modules, you can set Bx0 (initial transvere magnetic field) in the hyperparameters section to zero in the execution python files. Similarly if you want to run `VQA`, you can set T0 (initial temperature) to zero. Also, if you want to run `RVQA`, you can set Bx0 and T0 to be both non-zero. Finally, if you want to run Classical-Quantum optimization `CQO`, you can set both Bx0 and T0 to zero. More details about the acronyms `VCA`, `VQA`, `RVQA` and `CQO` are provided in [our paper](https://arxiv.org/abs/2101.10154).

We note that in this code we use the `tensordot2` operation from the [TensorNetwork package](https://github.com/google/TensorNetwork) to speed up tensorized operations.

### `tools`

This section contains the tools we used to generate the random instances of the models we considered in our paper.

### `data`

This section provides the ground states of the Edwards-Anderson (EA) and the Sherrington-Kirkpatrick (SK) models that were obtained from the [spin-glass server](http://spinglass.uni-bonn.de/) for 25 different seeds. The instances were generated using the code provided in `tools'. 

### `tutorials`
In this section of the repository, we demonstrate how our source code works in simple cases through Jupyter notebooks that you can run on [Google Colaboratory](colab.research.google.com) to take advantage of GPU speed up. These tutorials will help you to become more familiar with the content of the source code. The `tutorials` module contains the following:

1. `tutorials/VNA_1DTRNNs.ipynb`: a demonstration of VNA using 1D Tensorized RNNs applied to random ising chains with open boundary conditions.
2. `tutorials/VNA_2DTRNNs.ipynb`: a demonstration of VNA using 2D Tensorized RNNs on the 2D Edwards-Anderson model with open boundary conditions.
3. `tutorials/VNA_DilatedRNNs.ipynb`: a demonstration of VNA using Dilated RNNs applied to the Sherrington-Kirkpatrick model.

For more details, you can check our manuscript on arXiv: https://arxiv.org/abs/2101.10154 or on Nature Machine Intelligence: https://www.nature.com/articles/s42256-021-00401-3 (free access at https://rdcu.be/cAIyS).

For questions or inquiries, you can reach out to this email mohamed.hibat.allah@uwaterloo.ca.

## Dependencies
This code works on Python (3.6.10) with TensorFlow (1.13.1) and NumPy (1.16.3) modules. We also note that this code runs much faster on a GPU as compared to a CPU. No installation is required providing that the dependencies are available.

## Disclaimer
This code can be freely used for academic purposes that are socially and scientifically beneficial, however it is under Vector Institute’s Intellectual Property (IP) policy for profit related activities. 

## License
This code is under the ['Attribution-NonCommercial-ShareAlike 4.0 International'](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
 
## Citing
```bibtex
@Article{VNA2021,
author={Hibat-Allah, Mohamed and Inack, Estelle M. and Wiersema, Roeland and Melko, Roger G. and Carrasquilla, Juan},
title={Variational neural annealing},
journal={Nature Machine Intelligence},
year={2021},
month={Nov},
day={01},
volume={3},
number={11},
pages={952-961},
issn={2522-5839},
doi={10.1038/s42256-021-00401-3},
url={https://doi.org/10.1038/s42256-021-00401-3}
}
```

