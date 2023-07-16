#!/usr/bin/env python
import os
import sys

import numpy as np
from copy import deepcopy
import GSynGP
import math
from EA import *
from LCS_numba import *
class GP_Node:
    def __init__(self, value, params, sons, parent):
        self.value = value
        self.params = params
        self.sons = sons
        self.parent = parent
        self.height = 0  ##number of levels from this node to its furthest son
        if parent == None:
            self.depth = 0  ##number of levels from this node to the root
        else:
            self.depth = parent.depth + 1
        self.executed = False

    def toStr(self):
        s = self.value + '('
        for k in self.params.keys():
            s += k + '=' + self.params[k] + ','
        if s[-1] == ',':
            s = s[:-1]
        s += ')'
        return s

    def isTerminal(self):
        return self.sons is None

    def check_height(self):
        if(self.parent is None):
            self.depth = 0
        else:
            self.depth = self.parent.depth +1

        if self.sons is None:
            self.height = 0
        else:
            self.height = max([self.sons[i].height for i in range(len(self.sons))]) + 1

    def get_subtree(self, max_height):
        if self.height > max_height:
            return random.choice(self.sons).get_subtree(max_height)

        elif self.sons is not None and random.random() < 0.8:
            return random.choice(self.sons).get_subtree(max_height)
        else:
            return self

    def get_nodes(self, max_height, nodes):
        if self.height <= max_height:
            nodes.append(self)
        if self.sons is not None:
            for s in self.sons:
                s.get_nodes(max_height, nodes)
        return nodes

    def get_subtree_equal_probs(self, max_height):
        nodes = self.get_nodes(max_height, [])
        return random.choice(nodes)


class GP_Individual(Individual):
    def __init__(self, genotype, phenotype, creation=None):
        Individual.__init__(self, genotype, phenotype, creation)
        self.current_node = self.phenotype
        self.behavioural_distances = None
        self.pbehavioural_distances = None
        self.mean_genotypic_dist = 0
        self.mean_pgenotypic_dist = 0
        self.stddev_genotypic_dist = 0
        self.stddev_pgenotypic_dist = 0
        self.mean_behavioural_dist = 0
        self.stddev_behavioural_dist = 0
        self.mean_pbehavioural_dist = 0
        self.stddev_pbehavioural_dist = 0
        self.executedNodes = []
        self.label = ''
        self.creation = -2

    def copy_indiv(self, original):
        self.genotype = deepcopy(original.genotype)
        self.phenotype = self.copy_node(original.phenotype, None)
        self.current_node = None
        self.fitness = None
        self.fitness_vals = []  # deepcopy(original.fitnesses_vals)
        self.creation = original.creation
        self.label = original.label

    def copy_self(self):
        o = GP_Individual(None, None, None)
        o.copy_indiv(self)
        return o

    def copy_node(self, o_node, parent):
        n = GP_Node(deepcopy(o_node.value), deepcopy(o_node.params), None, parent)
        if o_node.sons is not None:
            n.sons = [self.copy_node(o_node.sons[i], n) for i in range(len(o_node.sons))]
            n.height = max([n.sons[i].height for i in range(len(n.sons))]) + 1
        return n


    def stateful_interpret(self, agent, current_node, logExecutedNodes=True):
        if current_node is None:
            current_node = self.phenotype

        while True:

            current_node.executed = True
            if logExecutedNodes and current_node not in self.executedNodes:
                self.executedNodes.append(current_node)

            if current_node.sons is not None:
                if current_node.value == 'Progn':
                    current_node = current_node.sons[0]
                else:
                    if current_node.value == 'foodAhead':
                        val = agent.foodAhead()
                    else:
                        val = eval('agent.' + current_node.toStr())

                    if(val):
                        current_node = current_node.sons[0]
                    else:
                        current_node = current_node.sons[1]

            else:
                motion_terminated = eval('agent.'+current_node.toStr())
                if motion_terminated:
                    motion_terminated = True
                    ##find the next node
                    while current_node.parent is not None:
                        if current_node == current_node.parent.sons[
                            0] and current_node.parent.value == 'Progn':
                            current_node = current_node.parent.sons[1]
                            break
                        else:
                            current_node = current_node.parent
                            #depth -= 1


                break  # breaks the loop to resume the simulation
        return current_node,motion_terminated

    def arithmetic_interpret(self, current_node, observation, logExecutedNodes=True):
        if current_node is None:
            current_node = self.phenotype

        current_node.executed = True
        if logExecutedNodes and current_node not in self.executedNodes:
            self.executedNodes.append(current_node)

        val = 0
        if current_node.sons is not None:
            if(current_node.value == 'add'):
                val = self.arithmetic_interpret(current_node.sons[0], observation, logExecutedNodes) + self.arithmetic_interpret(current_node.sons[1], observation,logExecutedNodes)
            elif(current_node.value == 'sub'):
                val = self.arithmetic_interpret(current_node.sons[0], observation,logExecutedNodes) - self.arithmetic_interpret(current_node.sons[1], observation,logExecutedNodes)
            elif(current_node.value == 'mult'):
                val = self.arithmetic_interpret(current_node.sons[0], observation,logExecutedNodes) * self.arithmetic_interpret(current_node.sons[1], observation,logExecutedNodes)
            elif(current_node.value == 'div'):
                den = self.arithmetic_interpret(current_node.sons[1], observation,logExecutedNodes)
                if(den ==0):
                    val = 1
                else:
                    val = self.arithmetic_interpret(current_node.sons[0],observation, logExecutedNodes)*1.0/ den
                
            elif(current_node.value == 'sin'):
                try:
                    val = np.sin(self.arithmetic_interpret(current_node.sons[0],observation, logExecutedNodes))
                except:
                    val = 1
            elif(current_node.value == 'cos'):
                try:
                    val = np.cos(self.arithmetic_interpret(current_node.sons[0],observation, logExecutedNodes))
                except:
                    val = 1
            elif(current_node.value == 'exp'):
                try:
                    val = np.exp(self.arithmetic_interpret(current_node.sons[0],observation, logExecutedNodes))
                except:
                    val = 0##protected log according to Nicolau and Agapitos 2020 "Choosing function sets with better generalisation performance for symbolic regression models"
            elif(current_node.value == 'lnmod'):
                try:
                    val = abs(self.arithmetic_interpret(current_node.sons[0],observation, logExecutedNodes))
                    if(val !=0):
                        val = np.log(val)
                except:
                    val = 0 ##protected log according to Nicolau and Agapitos 2020 "Choosing function sets with better generalisation performance for symbolic regression models"
            elif(current_node.value == 'sqrt'):
                try:
                    val = np.sqrt(abs(self.arithmetic_interpret(current_node.sons[0],observation, logExecutedNodes)))
                except:
                    val = 1
            elif(current_node.value=='plog'):
                try:
                    val = abs(self.arithmetic_interpret(current_node.sons[0],observation, logExecutedNodes))
                    if(val !=0):
                        val = np.log10(val)
                except:
                    val = 0 ##protected log according to Nicolau and Agapitos 2020 "Choosing function sets with better generalisation performance for symbolic regression models"

            elif(current_node.value == 'ppow'):
                try:
                    x1 = self.arithmetic_interpret(current_node.sons[0],observation, logExecutedNodes)
                    x2 = self.arithmetic_interpret(current_node.sons[1],observation, logExecutedNodes)
                    if(x1!=0 or (x1==x2 and x1==0)):
                        val = np.power(abs(x1), x2)
                    else:
                        val = 0
                except:
                    val = 0
            if np.isinf(val):
                if(val >0):
                    val=sys.maxsize
                else:
                    val = -sys.maxsize
        else:
            if(isinstance(current_node.value, str)):
                if('constant' in current_node.value):
                    val = current_node.params['value']
                else:
                    i = current_node.value.split('_')[1]
                    if(i=='all'):
                        val= np.zeros(observation.shape) + observation
                    else:
                        val = observation[int(i)]
            else:
                val = current_node.value

        return val


    def toList(self, l, current_node):
        s = current_node.value + '('
        keys = list(current_node.params.keys())
        keys.sort()
        for i in range(len(keys)):
            p = keys[i]
            if i < len(keys) - 1:
                s += p + '=' + str(current_node.params[p]) + ','
            else:
                s += p + '=' + str(current_node.params[p])
        s += ')'
        l.append(s)
        if current_node.sons is not None:
            for i in range(len(current_node.sons)):
                l = self.toList(l, current_node.sons[i])
        return l

    def toTree(self, genotype, curr_ind, parent, function_set):
        val = genotype[curr_ind]
        curr_ind += 1
        value = val[:val.index('(')]
        params = val[val.index('(') + 1:]
        val = [value, params]

        n = GP_Node(val[0], {}, None, parent)

        if val[1] != ')':
            val[1] = val[1][:-1].split(',')
            for v in val[1]:
                v = [v[:v.index('=')], v[v.index('=') + 1:]]
                try:
                    n.params[v[0]] = int(v[1])
                except:
                    try:
                        n.params[v[0]] = float(v[1])
                    except:
                        n.params[v[0]] = v[1]



        if val[0] in function_set:
            n.sons = []
            [s, curr_ind] = self.toTree(genotype, curr_ind, n, function_set)
            n.sons.append(s)
            [s, curr_ind] = self.toTree(genotype, curr_ind, n, function_set)
            n.sons.append(s)

        n.check_height()
        return [n, curr_ind]

    def get_terminals(self, l, current_node):
        if current_node.sons is None:
            l.append(current_node)
        else:
            for s in current_node.sons:
                self.get_terminals(l, s)

    def get_functions(self, l, current_node):
        if current_node.sons is not None:
            l.append(current_node)
            for s in current_node.sons:
                if s.sons is not None:
                    self.get_functions(l, s)

    def make_genotype(self):
        self.genotype = []
        self.genotype = self.toList(self.genotype, self.phenotype)
        self.genotype = np.array(self.genotype, dtype='<U100')



class GP(EA):
    def __init__(self, pop_size, generations, ei, ri, p_cross, p_mut, tourn_size, elite_size, evaluation_function, p_reeval, lpp, pareto, log_file, max_depth, terminal_set,
                 function_set, terminal_params, function_params, iteration, mutation_sigma, params_coefs=(1,1), maximise=False, compute_diversities=False, visualize=True, n_instances = 1, prob_type=None):
        EA.__init__(self, pop_size, generations, ei, ri, p_cross, p_mut, tourn_size, elite_size, evaluation_function, p_reeval, lpp, pareto,log_file, genotypic_distance, compute_diversities,compute_diversities,params_coefs, maximise, visualize, n_instances, prob_type)

        self.max_depth = max_depth
        self.terminal_set = terminal_set
        self.function_set = np.array(function_set)
        self.terminal_params = terminal_params
        self.function_params = function_params
        self.iteration = iteration
        self.mean_executed_nodes = 0
        self.std_executed_nodes = 0
        self.mean_percentage_executed_nodes = 0
        self.std_percentage_executed_nodes = 0
        self.mutation_sigma = mutation_sigma

    def stratToString(self, ind):
        l = []
        ind.toList(l, ind.phenotype)
        return str(l)

    '''
    def log_old(self, gen):
        # print('logging')

        if (gen <= 0):
            f = open(self.log_file, 'w')
            # f = open('logs/'+self.search_stage+'_'+str(self.params['run'])+'.txt','w')

            s = 'generation\tgenotype\tindiv_type\tfitness\tfitness_vals'
            s+='\tbestIndiv_mean_genotypic_dist\tbestIndiv_stddev_genotypic_dist\tbestIndiv_mean_pgenotypic_dist\tbestIndiv_stddev_pgenotypic_dist'
            s+='\tbestIndiv_mean_behavioural_dist\tbestIndiv_stddev_behavioural_dist\tbestIndiv_mean_pbehavioural_dist\tbestIndiv_stddev_pbehavioural_dist\tbestIndiv_size\tbestIndiv_executed_nodes'
            s+='\tmean_pop_size\tstd_pop_size\tmean_size_diff\tstd_size_diff\tmean_pgenotypic_dist\tstd_pgenotypic_dist\tmean_behavioural_dist\tstd_behavioural_dist\tmean_pbehavioural_dist\tstd_pbehavioural_dist\tmean_pop_fitness\tstd_pop_fitness\tmean_pexecuted_nodes\tstd_pexecuted_nodes\tmean_gsyngp_iterations\tstd_gsyngp_iterations'

            f.write(s + '\n')

        else:
            f = open(self.log_file, 'a')

        genotype = str(self.best_indiv.genotype.tolist())
        while '\\' in genotype:
            genotype = genotype.replace('\\', '')

        try:
            ind_iters = int(self.ind_iterations[gen])
            mean = np.mean(self.gsyngp_iterations[gen][:ind_iters])
            print('GSynGP iterations for gen '+str(gen),self.gsyngp_iterations[gen])
            std = np.std(self.gsyngp_iterations[gen][:ind_iters])
            ind_sd = int(self.ind_size_diff[gen])
            mean_size_diff = np.mean(self.size_diff[gen][:ind_sd])
            std_size_diff = np.std(self.size_diff[gen][:ind_sd])
            print('mean iters', mean, std)
            print('mean size diff', mean_size_diff, std_size_diff)

            #if len(self.gsyngp_iterations) > 0:
            #    mean = np.mean(self.gsyngp_iterations[:self.ind_gsyngp_iterations[0]])
            #    std = np.std(self.gsyngp_iterations[:self.ind_gsyngp_iterations[0]])
            #else:
            #    mean = 0
            #    std = 0
        except:
            mean = 0
            std = 0

        s = str(gen) + '\t' + genotype + '\t' + self.best_indiv.label + '\t' + str(self.best_indiv.fitness) + '\t' + str(self.best_indiv.fitness_vals.tolist())

        s+='\t' +str(self.best_indiv.mean_genotypic_dist)+'\t'+str(self.best_indiv.stddev_genotypic_dist)
        s+= '\t'+str(self.best_indiv.mean_pgenotypic_dist)+'\t'+str(self.best_indiv.stddev_pgenotypic_dist)
        s += '\t' + str(self.best_indiv.mean_behavioural_dist) + '\t' + str(self.best_indiv.stddev_behavioural_dist)
        s += '\t' + str(self.best_indiv.mean_pbehavioural_dist) + '\t' + str(self.best_indiv.stddev_pbehavioural_dist)
        s += '\t' + str(len(self.best_indiv.genotype)) + '\t' + str(len(self.best_indiv.executedNodes))
        s+= '\t' + str(self.mean_pop_size) + '\t' + str(
            self.std_pop_size) +'\t' + str(mean_size_diff) + '\t' + str(
            std_size_diff) + '\t' + str(
            self.mean_pgenotypic_dist) + '\t' + str(self.stddev_pgenotypic_dist) + '\t' + \
            str(self.mean_behavioural_diversity)+ '\t'+str(self.std_behavioural_diversity)+'\t'+str(self.mean_pbehavioural_diversity)+ '\t'+str(self.std_pbehavioural_diversity)+ '\t' + str(self.mean_fitness) + '\t' +str(self.std_fitness) +'\t'+ str(
            self.mean_percentage_executed_nodes) + '\t' + str(self.std_percentage_executed_nodes)

        #s = 'generation\tgenotype\tindiv_type\tfitness\tfitness_vals'

        #s += '\tbestIndiv_mean_genotypic_dist\tbestIndiv_stddev_genotypic_dist\tbestIndiv_mean_pgenotypic_dist\tbestIndiv_stddev_pgenotypic_dist'
        #s += '\tbestIndiv_mean_behavioural_dist\tbestIndiv_stddev_behavioural_dist\tbestIndiv_size\tbestIndiv_executed_nodes'
        #s += '\tmean_pop_size\tstd_pop_size\tmean_pgenotypic_dist\tstd_pgenotypic_dist\tmean_behavioural_dist\tstd_behavioural_dist\tmean_pop_fitness\tstd_pop_fitness\tmean_pexecuted_nodes\tstd_pexecuted_nodes\tmean_gsyngp_iterations\tstd_gsyngp_iterations'
        #len(self.population[i].executedNodes)



        s += '\t' + str(mean) + '\t' + str(std)
        f.write(s + '\n')
        f.close()
    '''

    def log(self, gen):

        if (gen <= 0):
            f = open(self.log_file, 'w')

            s = 'generation\tmean_pop_size\tstd_pop_size\tmean_size_diff\tstd_size_diff\tmean_genotypic_dist\tstd_genotypic_dist\tmean_pgenotypic_dist\tstd_pgenotypic_dist'
            s += '\tmean_behavioural_dist\tstd_behavioural_dist\tmean_pbehavioural_dist\tstd_pbehavioural_dist\tmean_pop_fitness\tstd_pop_fitness'
            s += '\tmean_executed_nodes\tstd_executed_nodes\tmean_pexecuted_nodes\tstd_pexecuted_nodes\tmean_gsyngp_iterations\tstd_gsyngp_iterations'

            headers = self.best_indiv.get_headers()
            for h in headers:
                s+='\t'+'bestIndiv_'+h

            f.write(s + '\n')

        else:
            f = open(self.log_file, 'a')

        try:

            ind_iters = int(self.ind_iterations[gen])
            mean_iterations = np.mean(self.gsyngp_iterations[gen][:ind_iters])
            std_iterations = np.std(self.gsyngp_iterations[gen][:ind_iters])
            ind_sd = int(self.ind_size_diff[gen])
            mean_size_diff = np.mean(self.size_diff[gen][:ind_sd])
            std_size_diff = np.std(self.size_diff[gen][:ind_sd])

        except Exception as e:
            print(e)
            mean_iterations = 0
            std_iterations = 0
            mean_size_diff=0
            std_size_diff=0

        s = str(gen) + '\t' + str(self.mean_pop_size) + '\t' + str(self.std_pop_size) + '\t' + str(mean_size_diff) + '\t' + str(std_size_diff) + '\t' + str(self.mean_genotypic_dist) + '\t' + str(self.stddev_genotypic_dist) + '\t' + str(self.mean_pgenotypic_dist) + '\t' + str(self.stddev_pgenotypic_dist)
        s +='\t'+str(self.mean_behavioural_diversity) + '\t' + str(self.std_behavioural_diversity) + '\t' + str(self.mean_pbehavioural_diversity) + '\t' + str(self.std_pbehavioural_diversity) + '\t' + str(self.mean_fitness) + '\t' + str(self.std_fitness)
        s += '\t' + str(self.mean_executed_nodes) + '\t' + str(self.std_executed_nodes) + '\t' + str(self.mean_percentage_executed_nodes) + '\t' + str(self.std_percentage_executed_nodes)+ '\t' +str(mean_iterations)+ '\t' +str(std_iterations)


        s += '\t' + self.best_indiv.toStringCompleteSingleLine()
        f.write(s + '\n')
        f.close()

    def ramped_half_and_half(self, max_depth):
        def pop_initialization(pop_size):
            population = []
            depths = [i + 1 for i in range(max_depth - 1)]
            depths.reverse()
            l = len(depths)

            for i in range(pop_size):
                population.append(
                    GP_Individual(None, self.random_individual(0, depths[i % l], None, random.random() < 0.5), {}))
                population[-1].make_genotype()

            return population
        return pop_initialization

    def random_individual(self, current_depth, max_depth, parent, grow):
        n = GP_Node(None, {}, None, parent)
        if current_depth == max_depth or (grow and random.random() < current_depth * 1.0 / max_depth):
            ##make a leaf node
            n.value = random.choice(self.terminal_set)
            
            if n.value in self.terminal_params.keys():
                for p in self.terminal_params[n.value].keys():
                    n.params[p] = random.choice(self.terminal_params[n.value][p])

        else:
            ##make an inner node
            n.value = random.choice(self.function_set)
            if n.value in self.function_params.keys():
                for p in self.function_params[n.value].keys():
                    n.params[p] = random.choice(self.function_params[n.value][p])
            n.sons = [self.random_individual(current_depth + 1, max_depth, n, grow) for i in range(2)]
            n.height = max([n.sons[0].height, n.sons[1].height]) + 1
        return n

    def subtree_crossover(self, indA, indB, curr_offsprings, max_offsprings, gen):
        indA = indA.copy_self()
        indB = indB.copy_self()        

        ##biased cut points
        funcs, terms = [],[]
        indA.get_functions(funcs,indA.phenotype)
        indA.get_terminals(terms, indA.phenotype)
        if(random.random()<0.9):
            subA = random.choice(funcs)
        else:
            subA = random.choice(terms)

        ##bloat control
        funcs, terms = [],[]
        indB.get_functions(funcs,indB.phenotype)
        indB.get_terminals(terms, indB.phenotype)

        if(random.random()<0.9):
            cands = []
            for f in funcs:
                if(f.height<=self.max_depth - subA.depth):
                    cands.append(f)
            if(len(cands)>0):
                subB = random.choice(cands)
            else:
                subB = random.choice(terms)
        else:
            subB = random.choice(terms)


        subBnodes = []
        subB.get_nodes(subB.height, subBnodes)
        subAnodes = []
        subA.get_nodes(subA.height, subAnodes)

        if subA.parent is None:
            return [indB],[0],[0]
        elif subA is subA.parent.sons[0]:
            subA.parent.sons[0] = subB
        else:
            subA.parent.sons[1] = subB

        subB.parent = subA.parent
        subB.depth = subA.depth
        p = subB.parent
        while p is not None:
            p.check_height()
            p = p.parent

        return [indA], [0], [len(subBnodes) - len(subAnodes)]


    def GSynGP_crossover(self, indA, indB, curr_offsprings, max_offsprings, gen):
        a, iterations_performed, size_diff = GSynGP.crossover3(indA.genotype, indB.genotype, self.function_set, self.iteration, curr_offsprings, max_offsprings, gen)
        a = GP_Individual(a, None, None)
        [phenotype, curr_ind] = a.toTree(a.genotype, 0, None, self.function_set)
        a.phenotype = phenotype
        return [a],[iterations_performed],[size_diff]


    def mutate_node(self,n, is_terminal):
        if(len(n.params.keys())==0):
            if(is_terminal):
                mutated = self.change_node_symbol(n, self.terminal_set, self.terminal_params)
            else:
                mutated = self.change_node_symbol(n, self.function_set, self.function_params)
        else:
            ##has parameters
            if(is_terminal and len(self.terminal_set)>1):
                ##there is more than one terminal
                if(random.random()>0.5): 
                    mutated=self.change_node_params(n,self.terminal_params)
                else:
                    mutated=self.change_node_symbol(n, self.terminal_set, self.terminal_params)
                
            elif(is_terminal):
                mutated=self.change_node_params(n,self.terminal_params)
                
            elif(not is_terminal and len(self.function_set)>1):
                if(random.random()>0.5):
                    mutated=self.change_node_params(n,self.function_params)
                else:
                    mutated=self.change_node_symbol(n, self.function_set, self.function_params)
                
            elif(not is_terminal):
                mutated=self.change_node_params(n,self.function_params)
        return n, mutated


    def uniform_node_mutation(self, ind):
        ##a combination of symbol and param mutations. Chooses a node randomly (with no preference
        #for terminals or non-terminals). If it has parameters, then it randomly 
        #chooses to either replace the symbol or modify its parameters. Otherwise, it just replaces the symbol
        
        nodes=[]
        ind.get_terminals(nodes, ind.phenotype)    
        ind.get_functions(nodes, ind.phenotype)
        mutated = False
        while (len(nodes) > 0 and not mutated):
            n = random.choice(nodes)
            nodes.remove(n)

            is_terminal = (n.value in self.terminal_set)
            n, mutated = self.mutate_node(n, is_terminal)
        return ind


    def biased_node_mutation(self, ind):
        ##a combination of symbol and param mutations. Chooses a node randomly (with no preference
        #for terminals or non-terminals). If it has parameters, then it randomly 
        #chooses to either replace the symbol or modify its parameters. Otherwise, it just replaces the symbol
        terminals, functions=[],[]
        ind.get_terminals(terminals, ind.phenotype)    
        ind.get_functions(functions, ind.phenotype)
        mutated = False
        
        while ((len(terminals) > 0 or len(functions) > 0) and not mutated):
            if(len(functions) > 0 and random.random()<0.9):
                nodes = functions
                is_terminal = False
            else:
                nodes = terminals
                is_terminal = True
            n = random.choice(nodes)
            nodes.remove(n)
            n, mutated = self.mutate_node(n, is_terminal)
        return ind

    def change_node_params(self, node, params):
        # changes 1 param of a single node
        mutated = False
        if (len(node.params.keys()) == 0):
            return mutated

        p_cands = list(node.params.keys())
        random.shuffle(p_cands)
        i=0
        while(not mutated and i<len(p_cands)): 
            p = p_cands[i]

            s = node.params[p]
            if isinstance(s, str):
                try:
                    s = eval(s)
                    vals = [eval(k) for k in params[node.value][p]]
                    high = max(vals)
                    low = min(vals)
                    domain_width = high-low
                    
                    if isinstance(s, float):
                        node.params[p] = str(min(high, max(low, round(node.params[p] + random.gauss( 0.0, self.mutation_sigma*domain_width), 3))))
                    elif isinstance(s, int):
                        node.params[p] = str(min(high, max(low, int(round(node.params[p] + random.gauss( 0.0, self.mutation_sigma*domain_width))))))
                    mutated = True
                    break
                except Exception as e:
                    mutated = False

            else:                
                high = max(params[node.value][p])
                low = min(params[node.value][p])
                domain_width = high-low
                if isinstance(s, float):
                    node.params[p] = min(high, max(low, round(node.params[p] + random.gauss( 0.0, self.mutation_sigma*domain_width), 3)))
                    mutated = True
                    break
                elif isinstance(s, int):
                    node.params[p] = min(high, max(low, int(round(node.params[p] + random.gauss( 0.0, self.mutation_sigma*domain_width)))))
                    mutated = True
                    break
            
            if not mutated:
                v_cands = list(params[node.value][p])
                random.shuffle(v_cands)

                for v in v_cands:
                    if (v != node.params[p]):
                        node.params[p] = v
                        mutated = True
                        break

            i+=1
        return mutated




    def change_node_symbol(self, node, symbols, params):
        if len(symbols) == 1:
            return False

        mutated = False
        s_cands = symbols
        random.shuffle(s_cands)
        for i in range(len(s_cands)):
            v=s_cands[i]

            if v != node.value:
                node.value = v
                nkeys = list(node.params.keys())
                for p in nkeys:
                    if v not in params.keys() or p not in params[v].keys():
                        del node.params[p]

                if v in params.keys():
                    for p in params[v].keys():
                        if p not in node.params.keys() or node.params[p] not in params[v][p]:
                            node.params[p] = random.choice(params[v][p])

                mutated = True
                break
        return mutated

@njit
def genotypic_distance(A, B, float_vals=False):
    ##compute the LCS between A and B
    C = LCS(A, B)

    [ma, mb, na, nb] = LCS_MASKS(A, B, C)
    return na, nb
