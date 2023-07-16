
def read_config(filename):
    f = open(filename)
    line = f.readline()
    params={}
    while(line!=''):
        #print(line)
        if('\n' in line):
            line = line[:-1]
        line = line.split('\t')
        #print(line, len(line))
        params[line[0]] = eval(line[1])

        line = f.readline()
    f.close()
    return params


def write_config_file(configs, suffix):
    sel = "tourn"+str(configs["tourn_size"])
    if("rolette_wheel_selection" in configs['parent_selection']):
        sel = "roulette"

    fname = configs['problem'] + '_' + configs['strat_name'] + '_' + configs['iteration'] + '_' + str(configs['ri'])+'ri'+str(configs['ei'])+'ei'+'_'+str(configs['p_cross'])+'pcross'+str(configs['p_mut'])+'pmut_'+sel+suffix+'.txt'
    f = open(fname,'w')
    for k in configs.keys():
        #print(k)
        val = configs[k]
        if (isinstance(val, str)):
            val = '\"'+val+'\"'
        else:
            val = str(val)
        f.write(k+'\t'+val+'\n')

    f.close()

def make_configs_sr(arithmetic, consts, parent_selection):
    algs = ['GSynGP', 'SGP']
    problems = ['Koza1','Paige1', 'Keijzer12', 'Koza3']
    iterations = ['1', 'random', 'half']
    immigrants = [(0.07,0.03)]
    variation_operators = [(0.7,0.3)]



    if(arithmetic):    
        function_set = ['add', 'sub', 'mult', 'div']
    else:
        function_set = ['add', 'sub', 'mult', 'div', 'sin', 'cos', 'exp', 'lnmod']


    if(consts):
        terminal_set = ['var','uniform_constant']
    else:
        terminal_set = ['var']
    
    suffix = ''
    if(arithmetic):
        suffix += '_arithmetic'
    if(consts):
        suffix+='_consts'

    log_directory = 'logs'+suffix

    prob_type = 'SR'

    configs = {'n_runs': 30, 'n_instances': 5, 'strat_name': 'GSynGP', 'debug': False, 'visualize': False,
    'restore': False, 'optimize_strat': False, 'genotype': ['mult()', 'div()', 'exp()', 'x_0()', 'x_0()', 'cos()', 'x_0()', 'x_0()', 'mult()', 'sin()', 'x_0()', 'x_0()', 'div()', 'x_0()', 'x_0()'], 
    'p_cross': 0.7, 'p_mut': 0.3, 'tourn_size': 2, 'mutation_size': 1, 'initial_max_depth':6,'max_depth': 17,
    'continuous_evolution': False, 'p_reeval': 0.0, 'n_evaluations': 1, 'params_coefs': (1, 1), 
    'survivors_selection': 'elitist', 'elite_size': 3, 'iteration': 'half', 'pop_size': 500, 'generations': 1000,
    'ri': 0.0, 'ei':0.0, 'lpp': False, 'pareto': False, 
    'seeds': [985, 570, 815, 356, 782, 148, 998, 188, 306, 783, 245, 521, 264, 121, 772, 635, 485, 754, 583, 740, 581, 926, 272, 855, 667, 742, 605, 148, 163, 353], 
    'log_directory': log_directory, 'maximise': False, 'compute_diversities': True, 'problem': 'Koza1', 'prob_type': prob_type, 
    'function_set': function_set, 'terminal_set': terminal_set,
    'terminal_params':{'uniform_constant':{'value':"np.linspace(-1,1,21)"}}, 'manyoff':False, 'parent_selection': parent_selection}




    for prob in problems:
        configs['problem'] = prob
        if('Koza' in prob or 'Nguyen5' in prob):
            ndimensions=1
            domain = [[-1,1] for i in range(ndimensions)]
            npoints = 20

        elif(prob == 'Paige1'):
            ndimensions=2
            domain = [[-5,5] for i in range(ndimensions)]
            npoints = 20
        elif(prob=='Keijzer12'):
            ndimensions=2
            domain = [[-3,3] for i in range(ndimensions)]
            npoints = 20            
        else:    
            ndimensions=2
            domain = [[-10,10] for i in range(ndimensions)]
            npoints = 20

        configs['ndimensions'] = ndimensions
        configs['domain'] = domain
        configs['npoints'] = npoints
        for alg in algs:
            configs['strat_name'] = alg
            for it in iterations:
                if(alg == 'GSynGP' or it=='1'):
                    configs['iteration'] = it
                    print(alg, it)
                    for ri, ei in immigrants:                     
                        configs['ri'] = ri 
                        configs['ei'] = ei
                        for p_cross, p_mut in variation_operators:
                            configs['p_cross'] = p_cross
                            configs['p_mut'] = p_mut

                            if((p_cross!=1.0 and p_mut!=0.0) or (ri ==0.0 and ei==0.0)):
                                write_config_file(configs,suffix)

def make_configs_sf(parent_selection):

    algs = ['GSynGP', 'SGP']
    problems = ['SantaFeAntTrailMin', 'LosAltosHillsAntTrailMin']
    iterations = ['1', 'random', 'half']
    immigrants = [(0.0,0.0),(0.07,0.03)]
    variation_operators = [(0.7,0.3)]

    domain = [[-1,1]]
    ndimensions = 1
    npoints = 20
    prob_type = 'AgentController'


    
    suffix = ''
    log_directory = 'logs'+suffix

    configs = {'n_runs': 30, 'n_instances': 5, 'strat_name': 'GSynGP', 'debug': False, 'visualize': False,
    'restore': False, 'optimize_strat': False, 'genotype': ['mult()', 'div()', 'exp()', 'x_0()', 'x_0()', 'cos()', 'x_0()', 'x_0()', 'mult()', 'sin()', 'x_0()', 'x_0()', 'div()', 'x_0()', 'x_0()'], 
    'p_cross': 0.7, 'p_mut': 0.3, 'tourn_size': 2, 'mutation_size': 1, 'initial_max_depth':6,'max_depth': 17,
    'continuous_evolution': False, 'p_reeval': 0.0, 'n_evaluations': 1, 'params_coefs': (1, 1), 
    'survivors_selection': 'elitist', 'elite_size': 3, 'iteration': 'half', 'pop_size': 500, 'generations': 1000,
    'ri': 0.0, 'ei':0.0, 'lpp': False, 'pareto': False, 
    'seeds': [985, 570, 815, 356, 782, 148, 998, 188, 306, 783, 245, 521, 264, 121, 772, 635, 485, 754, 583, 740, 581, 926, 272, 855, 667, 742, 605, 148, 163, 353], 
    'log_directory': log_directory, 'maximise': True, 'compute_diversities': True, 'problem': 'Koza1', 'prob_type': prob_type, 
    'domain': domain, 'ndimensions': ndimensions, 'npoints': npoints, 
    'function_set': ['Progn', 'foodAhead'], 'terminal_set': ['left','right','move'],
    'terminal_params':{}, 'manyoff':False, 'parent_selection': parent_selection}


    for prob in problems:
        configs['problem'] = prob
        if('Min' in prob):
            configs['maximise']=False
        else:
            configs['maximise']=True

        for alg in algs:
            configs['strat_name'] = alg
            for it in iterations:
                if(alg == 'GSynGP' or it=='1'):
                    configs['iteration'] = it
                    print(alg, it)
                    for ri, ei in immigrants:                     
                        configs['ri'] = ri 
                        configs['ei'] = ei
                        for p_cross, p_mut in variation_operators:
                            configs['p_cross'] = p_cross
                            configs['p_mut'] = p_mut

                            if((p_cross!=1.0 and p_mut!=0.0) or (ri ==0.0 and ei==0.0)):
                                write_config_file(configs, suffix)

if __name__ == '__main__':
    for parent_selection in ["gp.rolette_wheel_selection", "gp.tournament_selection"]:
        for arithmetic, consts in [(False, False), (False, True), (True, False), (True, True)]:
            make_configs_sr(arithmetic, consts, parent_selection)
            make_configs_sf(parent_selection)
