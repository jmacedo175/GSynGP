# Geometric Syntactic Genetic Programming (GSynGP)

This repository provides a general purpose implementation of Geometric Syntactic Genetic Programming (GSynGP) [1], along with an implementation of a standard Genetic Programming algorithm (GP). Selecting the desired algorithm and its parameters is made on the "params" dictionary in main.py. A subset of openAI's GYM problems are provided as examples of applications. The extended versions of GSynGP [2] and GP are also implemented, enabling an easy evolution of expression trees with multiple symbols per nodes. In order to do it, symply fill in the "terminal_params" and "function_params" dictionaries in main.py.

To run an experiment, you must first create a configuration file. For your convenience, a script (make_config.py) is provided that creates various configuration files, with different parameters and benchmark problems from the symbolic regression and path planning domains. After the configuration file is created, simply execute python main.py <configuration_file> to run an experiment.

If this code is used on publications, please cite [1] if the base GSynGP (or GP) is used and [2] for the extended versions. For any questions, refer to jmacedo@dei.uc.pt

[1] Macedo, J., Fonseca, C. M., & Costa, E. (2018, April). Geometric crossover in syntactic space. In European Conference on Genetic Programming (pp. 237-252). Springer, Cham.

[2] Macedo, J., Marques, L., & Costa, E. (2020, April). Locating odour sources with geometric syntactic genetic programming. In International Conference on the Applications of Evolutionary Computation (Part of EvoStar) (pp. 212-227). Springer, Cham.

