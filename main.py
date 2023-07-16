import os
import GP
from copy import deepcopy
from EvaluationFunctions import *
import sys


def work(params):
    random.seed(params['seeds'][params['run']])
    np.random.seed(params['seeds'][params['run']])

    if params['compute_diversities']:
        params['log_directory']+='_diversities'
    if not os.path.exists(params['log_directory']):
        try:
            os.makedirs(params['log_directory'])
        except:
            pass

    if not os.path.exists(params['log_directory'] + '/logs_' + params['strat_name']):
        try:
            os.makedirs(params['log_directory'] + '/logs_' + params['strat_name'])
        except:
            pass

    if not params['restore']:
        log_file = params['log_directory'] + '/logs_' + params['strat_name'] + '/' + params['experiment'] + '_' + str(
            params['run']) + '.txt'

    else:
        log_file = ''

    gp = GP.GP(params['pop_size'], params['generations'], params['ei'], params['ri'], params['p_cross'],
               params['p_mut'], params['tourn_size'], params['elite_size'],
               params['evaluation_function'], params['p_reeval'], params['lpp'], params['pareto'], log_file,
               params['max_depth'],
               params['terminal_set'], params['function_set'], params['terminal_params'], params['function_params'],
               params['iteration'], params['mutation_sigma'], params['params_coefs'], params['maximise'],
               params['compute_diversities'], False, params['n_instances'], params['prob_type'])

    if params['restore']:
        ind = GP.GP_Individual(genotype=None, phenotype=None, creation=-1)
        ind.phenotype = ind.toTree(params['genotype'], 0, None, params['function_set'])[0]

        l = []
        ind.genotype = ind.toList(l, ind.phenotype)

        if not params['optimize_strat']:
            fitness = eval(params['evaluation_function'])(ind)
            print(fitness)
        else:
            pass

    else:
        data = gp.restoreState(True, params['function_set'])

        if (data == None):
            gp.pop_based_evolution(gp.ramped_half_and_half(params['initial_max_depth']), eval(params['crossovers']),
                                   params['crossover_odds'], eval(params['mutations']),
                                   params['mutation_odds'], eval(params['parent_selection']),
                                   eval(params['survivors_selection']), gp.log)
        else:
            gen = data[0]
            randomState = data[1]
            pop = data[2]
            metrics = data[3]
            for i in range(len(pop)):

                ind = GP.GP_Individual(genotype=None, phenotype=None, creation=-2)
                for k in pop[i].keys():
                    if pop[i][k] is not None:
                        ind.__dict__[k] = pop[i][k]

                ind.phenotype = ind.toTree(ind.genotype, 0, None, params['function_set'])[0]
                ind.make_genotype()

                pop[i] = ind
            ###implement this based on pop_based_evolution
            gp.pop_based_evolution(gp.ramped_half_and_half(params['initial_max_depth']), eval(params['crossovers']),
                                   params['crossover_odds'], eval(params['mutations']),
                                   params['mutation_odds'], eval(params['parent_selection']),
                                   eval(params['survivors_selection']), gp.log, pop, gen + 1, randomState, metrics)


def read_config(filename):
    f = open(filename)
    line = f.readline()
    params = {}
    while (line != ''):
        if ('\n' in line):
            line = line[:-1]
        line = line.split('\t')
        params[line[0]] = eval(line[1])

        line = f.readline()
    f.close()
    return params


if __name__ == '__main__':
    if(len(sys.argv)!=2):
        print('USAGE: python main.py <config_file>')
        exit()
    fname = sys.argv[1]
    params = read_config(fname)

    
    sys.setrecursionlimit(sys.getrecursionlimit()*3) ## use this to evolve large individuals

    params['ei'] = int(params['ei'] * params['pop_size'])
    params['ri'] = int(params['ri'] * params['pop_size'])

    params['experiment'] = fname  
    if ('/' in params['experiment']):
        params['experiment'] = params['experiment'].split('/')[-1]
    if '.txt' in params['experiment']:
        params['experiment'] = params['experiment'].split('.txt')[0]


    if (params['prob_type'] == 'SR'):
        ev = params['problem'] + '(' + str(params['domain']) + ',' + str(params['npoints']) + ',' + str(
            params['ndimensions']) + ',' + str(params['visualize']) + ')'
        eval_func = eval(ev)

        params['evaluation_function'] = ev
        params['function_params'] = {}

        if ('var' in params['terminal_set']):
            params['terminal_set'].remove('var')
            params['terminal_set'].extend(eval_func.variables)


        for k in params['terminal_params']:
            for ki in params['terminal_params'][k]:
                if (isinstance(params['terminal_params'][k][ki], str)):
                    try:
                        params['terminal_params'][k][ki] = eval(params['terminal_params'][k][ki])
                    except:
                        pass

        if params['strat_name'] == 'SGP':
            params['crossovers'] = '[gp.subtree_crossover]'

        else:
            if (not params['manyoff']):
                params['crossovers'] = '[gp.GSynGP_crossover]'
            else:
                params['crossovers'] = '[gp.GSynGP_manyoff_crossover]'
        params['crossover_odds'] = [1.1]

        params['mutation_sigma'] = 0.1
        params['mutations'] = "[gp.uniform_node_mutation]"
        params['mutation_odds'] = [1.1]

        if params['survivors_selection'] == 'elitist':
            params['survivors_selection'] = "gp.elitist_selection"
        else:
            params['survivors_selection'] = "gp.merge_selection"


    elif (params['prob_type'] == 'AgentController'):

        ev = params['problem'] + '(' + str(params['visualize']) + ')'
        eval_func = eval(ev)
        params['evaluation_function'] = ev
        params['function_params'] = {}

        if ('var' in params['terminal_set']):
            params['terminal_set'].remove('var')
            params['terminal_set'].extend(eval_func.variables)

        # params['terminal_params'] = {} ##terminal_params and function_params are used to implicitly create nodes with multiple symbols. Simply add the terminal/function symbol as key, followed by a dictionary of param:list of possible values

        for k in params['terminal_params']:
            for ki in params['terminal_params'][k]:
                if (isinstance(params['terminal_params'][k][ki], str)):
                    try:
                        params['terminal_params'][k][ki] = eval(params['terminal_params'][k][ki])
                    except:
                        pass

        if params['strat_name'] == 'SGP':
            params['crossovers'] = '[gp.subtree_crossover]'

        else:
            params['crossovers'] = '[gp.GSynGP_crossover]'

        params['crossover_odds'] = [1.1]

        params['mutation_sigma'] = 0.1

        params['mutations'] = "[gp.uniform_node_mutation]"  
        params['mutation_odds'] = [1.1]


        if params['survivors_selection'] == 'elitist':
            params['survivors_selection'] = "gp.elitist_selection"
        else:
            params['survivors_selection'] = "gp.merge_selection"


    runs = []
    for i in range(0, params['n_runs']):
        params = deepcopy(params)
        params['run'] = i
        runs.append(params)

    for run in runs:
        work(run)
