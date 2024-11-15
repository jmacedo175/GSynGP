# Geometric Syntactic Genetic Programming (GSynGP)

This repository provides a general purpose implementation of Geometric Syntactic Genetic Programming (GSynGP) [1], along with an implementation of a standard Genetic Programming algorithm (GP). Selecting the desired algorithm and its parameters is made on the "params" dictionary in main.py. A set of popular benchmark problems from the domains of Symbolic Regression and Path Planning are provided as examples of applications. The extended versions of GSynGP [2] and GP are also implemented, enabling an easy evolution of expression trees with multiple symbols per nodes. In order to do it, symply fill in the "terminal_params" and "function_params" dictionaries in main.py.

To run an experiment, you must first create a configuration file. For your convenience, a script (make_config.py) is provided that creates various configuration files, with different parameters and benchmark problems from the symbolic regression and path planning domains. After the configuration file is created, simply execute python main.py <configuration_file> to run an experiment.

