import numpy as np
from copy import deepcopy
import random
from LCS_numba import *
import time
class node:
    def __init__(self, v, s,depth):
        self.value = v
        self.sons = s
        self.depth=depth



@njit()
def count_notations(ma):
    af = 0
    at = 0
    for c in ma:
        if 'F_' in c:
            af += 1
        elif 'T_' in c:
            at += 1
    return af, at

@njit()
def compute_distance(ma, mb):
    af, at = count_notations(ma)
    bf, bt = count_notations(mb)

    ift = min(bf, bt)  ##number of insertions
    aft = min(af, at)  ##number of deletions
    sf = af - aft  # number of F replacements
    st = at - aft  # number of T replacements

    d = ift + aft + sf + st  ##the distance between two individuals is equal to the number of insertions, deletions and replacements required to turn A into B
    ##if instead of maximising deletions and insertions, the odds favour maximising the replacements, the distance is the same, as for each replacement a deletion and insertion must be deducted

    return d

def isnumber(x):
    try:
        a = float(x)
        return True
    except:
        return False

def checker(function_set, s, index, ntabs):
    if index == len(s):
        return None, 0
    val = s[index]
    if('(' in val):
        val= val[:val.index('(')]

    if val in function_set:
        if index < len(s) - 2:
            v = s[index]
            [s1, index] = checker(function_set, s, index + 1, ntabs + 1)
            [s2, index] = checker(function_set, s, index + 1, ntabs + 1)
            if s1 is None or s2 is None:
                return None, 0
            sons = [s1, s2]
            d = max([sons[0].depth, sons[1].depth]) + 1
            n = node(v, sons, d)
            return n, index
        else:
            print('error', s)
            exit(0)
    else:
        if isnumber(s[index]):
            return node(float(s[index]), None, 0), index
        else:
            return node(s[index], None, 0), index

#@njit()
def print_tree_preorder(root,s):
    if isnumber(root.value):
        s.append(str(root.value))
    else:
        s.append(root.value)

    if root.sons is not None:
        print_tree_preorder(root.sons[0],s)
        print_tree_preorder(root.sons[1],s)
    return s

#@njit()
def check_indiv(ind, function_set):
    t, i = checker(function_set, ind, 0, 0)

    if t != None:
        s = np.array(print_tree_preorder(t, []))
        if not different(s, ind):
            return True
    return False

#@njit()
def concat(exA):
    s = []
    for i in range(len(exA)):
        if 'T_' in exA[i] or 'F_' in exA[i]:
            s.append(exA[i][exA[i].index('_') + 1:])
        elif ' ' not in exA[i]:
            s.append(exA[i])

    return np.array(s, dtype=exA.dtype)

@njit()
def get_params(s_node):
    params = {}
    s_params = s_node[s_node.index('(')+1:-1]
    if len(s_params)>0:
        s_params = s_params.split(',')
        for s in s_params:
            p = s[:s.index('=')]
            v = s[s.index('=')+1:]
            params[p] = v
    return params

@njit()
def concat_node(node_val, params):
    s=node_val+'('
    for k in params.keys():
        s+=k+'='+str(params[k])+','
    if ',' in s:
        s=s[:-1]
    s+=')'
    return s

#@njit()
def merge_params(p1,p2,nFrom2):
    k1 = list(p1.keys())
    k2=list(p2.keys())

    common_keys = []
    for k in k1:
        if k in k2:
            common_keys.append(k)
        else:
            ##remove this to keep the params as dorment genes
            del p1[k]

    #add genes that only exist in p2
    added = 0
    for k in k2:
        if k not in common_keys:
            p1[k] = p2[k]
            added+=1

    random.shuffle(common_keys)
    i=0
    c=0
    while i<len(common_keys):
        k = common_keys[i]
        if k== 'iterations' or k== 'intervals':
            p1[k] = int(round((float(p1[k])+float(p2[k]))*0.5))
        elif k== 'c_direction':
            if c<nFrom2-added:
                p1[k] = p2[k]
                c+=1
        else:
            try:
                p1[k] = round((float(p1[k])+float(p2[k]))*0.5,1)
            except:
                if c<nFrom2-added:
                    p1[k] = p2[k]
                    c+=1
        i+=1
    ##the remaining params are already in p1
    return p1

#@njit()
def replaceT(exA,exB,function_set):
    indA=[]
    indB=[]

    for i in range(len(exA)):
        if 'T_' in exA[i]:
            indA.append(i)
        if 'T_' in exB[i]:
            indB.append(i)

    random.shuffle(indB)
    for iB in indB:
        oldIBA = exA[iB]
        oldIBB = exB[iB]
        exB[iB] = exB[iB][2:]
        if exA[iB]== '   ':
            exA[iB] = exB[iB]
            random.shuffle(indA)

            for iT in indA:
                oldIT = exA[iT]
                exA[iT]= '   '

                ind = concat(exA)
                if check_indiv(ind, function_set):
                    return ind
                exA[iT] = oldIT

        elif 'T_' in exA[iB]:
            node_val = exB[iB]
            pA = get_params(exA[iB])
            pB = get_params(node_val)

            pA = merge_params(pA,pB, 1)
            exA[iB] = concat_node(node_val[:node_val.index('(')],pA)
            return concat(exA)
        else:
            random.shuffle(indA)
            for iT in indA:
                oldIT = exA[iT]
                exA[iT]= exB[iB]

                ind = concat(exA)
                if check_indiv(ind, function_set):
                    return ind
                exA[iT] = oldIT
        exA[iB] = oldIBA
        exB[iB] = oldIBB
    return None

#@njit()
def deleteFT(exA, exB, function_set):
    indF = []
    indT = []

    for i in range(len(exA)):
        if 'F_' in exA[i]:
            indF.append(i)
        elif 'T_' in exA[i]:
            indT.append(i)

    random.shuffle(indF)
    for iF in indF:
        oldIFA = exA[iF]
        oldIFB = exB[iF]
        exA[iF] = '   '
        exB[iF] = exB[iF][2:]

        random.shuffle(indT)
        for iT in indT:
            oldITA = exA[iT]
            exA[iT] = '   '
            oldITB = exB[iT]
            exB[iT] = exB[iT][2:]
            ind = concat(exA)
            if check_indiv(ind, function_set):
                return ind
            exA[iT] = oldITA
            exB[iT] = oldITB

        exA[iF] = oldIFA
        exB[iF] = oldIFB
    return None

#@njit()
def insertFT(exA, exB, function_set):
    indBF = []
    indBT=[]

    for i in range(len(exA)):
        if 'F_' in exB[i] and '   ' in exA[i]:
            indBF.append(i)
        elif 'T_' in exB[i] and '   ' in exA[i]:
            indBT.append(i)

    random.shuffle(indBF)
    for iF in indBF:
        if exA[iF] == '   ':
            oldIFA = exA[iF]
            oldIFB = exB[iF]
            exA[iF] = exB[iF][exB[iF].index('_') + 1:]
            exB[iF] = exB[iF][2:]

            random.shuffle(indBT)
            for iT in indBT:
                if exA[iT] == '   ':
                    oldITA = exA[iT]
                    oldITB = exB[iT]
                    exA[iT] = exB[iT][exB[iT].index('_') + 1:]
                    exB[iT] = exB[iT][2:]
                    ind = concat(exA)
                    if check_indiv(ind, function_set):
                        return ind
                    exA[iT] = oldITA
                    exB[iT] = oldITB
            exA[iF] = oldIFA
            exB[iF] = oldIFB
    return None



#@njit()
def replaceF(exA, exB, function_set):
    indA = []
    indB = []

    for i in range(len(exA)):
        if 'F_' in exA[i]:
            indA.append(i)
        if 'F_' in exB[i]:
            indB.append(i)

    random.shuffle(indB)

    for iB in indB:
        oldIBA = exA[iB]
        oldIBB = exB[iB]
        exB[iB] = exB[iB][2:]
        if exA[iB] == '   ':
            exA[iB] = exB[iB]
            random.shuffle(indA)
            for iT in indA:
                oldIT = exA[iT]
                exA[iT] = '   '
                ind = concat(exA)
                if check_indiv(ind, function_set):
                    return ind
                exA[iT] = oldIT

        elif 'F_' in exA[iB]:
            node_val = exB[iB]
            pA = get_params(exA[iB])
            pB = get_params(node_val)

            pA = merge_params(pA, pB, 1)
            exA[iB] = concat_node(node_val[:node_val.index('(')], pA)
            return concat(exA)
        else:
            random.shuffle(indA)
            for iT in indA:
                oldIT = exA[iT]
                exA[iT] = exB[iB]
                ind = concat(exA)
                if check_indiv(ind, function_set):
                    return ind
                exA[iT] = oldIT
        exA[iB] = oldIBA
        exB[iB] = oldIBB
    return None




@njit()
def different(A,B):
    if(len(A)!=len(B)):
        return True
    for i in range(len(A)):
        if(A[i]!=B[i]):
            return True
    return False

@njit()
def compute_masks(A, B, function_set):
    rA, rB = toStructure(A, function_set), toStructure(B, function_set)
    C = LCS(A, B)
    [ma, mb, na, nb] = LCS_MASKS(A, B, C)
    [exA, exB] = expanded_masks(ma, mb, A, B, rA,rB,function_set)
    return rA, rB, C, ma,mb, exA, exB


def existances(exM):
    T_, F_ = False, False
    for s in exM:
        if len(s)>2:
            if(s[:2] == 'T_'):
                T_ = True
            elif(s[:2] == 'F_'):
                F_ = True
    return F_, T_

def crossover3(A, B, function_set, iteration, curr_offsprings, max_offsprings, gen):
    it = 0
    orig_size = len(A)
    if(not different(A, B)):
        return A, 0, 0

    rA, rB, C, ma, mb, exA, exB = compute_masks(A, B, function_set)

    if iteration == 'random':
        times = compute_distance(exA, exB) * 0.5 * random.random() 
    elif iteration == 'half':
        times = compute_distance(exA, exB) * 0.5 
    else:
        times = eval(iteration)

    while it < times and different(A,B):
        options = []

        aF, aT = existances(exA)
        bF, bT = existances(exB)

        if(aF and aT):
            options.append(deleteFT)
        if(aT and bT):
            options.append(replaceT)
        if(bF and bT):
            options.append(insertFT)
        if(bF and aF):
            options.append(replaceF)


        random.shuffle(options)

        for choice in options:

            off = choice(exA, exB, function_set)
            if(off is not None):
                A = off
                break

        it += 1
    return A, it, len(A) - orig_size



if __name__=='__main__':
    '''
    A = ['add()', 'mult()', 'uniform_constant(value=0.5)', 'x_1()', 'x_0()']
    #B = np.array(['add()', 'mult()', 'uniform_constant(value=0.25)', 'x_0()', 'x_0()'])
    B = ['div()','x_1()','add()', 'mult()', 'uniform_constant(value=0.25)', 'x_0()', 'x_0()']
    function_set = np.array(['add', 'sub', 'mult', 'div', 'sin', 'cos', 'exp', 'lnmod'])
    terminal_set = ['var', 'uniform_constant']
    terminal_params = {'uniform_constant': {'value': 'np.linspace(-1,1,21)'}}
    #A = np.array(['mult()', 'x_0()', 'sub()', 'exp()', 'div()', 'x_0()', 'uniform_constant(value=0.7000000000000002)', 'x_0()', 'sin()', 'add()', 'div()', 'uniform_constant(value=-0.09999999999999998)', 'div()', 'sub()', 'uniform_constant(value=-0.5)', 'mult()', 'uniform_constant(value=0.7000000000000002)', 'exp()', 'sin()', 'div()', 'lnmod()', 'x_0()', 'uniform_constant(value=0.068)', 'x_0()', 'lnmod()', 'uniform_constant(value=0.7000000000000002)', 'uniform_constant(value=0.32)', 'uniform_constant(value=0.982)', 'x_0()', 'uniform_constant(value=0.10000000000000009)', 'uniform_constant(value=-0.09999999999999998)'])
    #B = np.array(['exp()', 'lnmod()', 'mult()', 'sin()', 'x_0()', 'x_0()', 'add()', 'uniform_constant(value=-1.0)', 'add()', 'exp()', 'x_0()', 'x_0()', 'add()', 'x_0()', 'exp()', 'div()', 'add()', 'x_0()', 'uniform_constant(value=-0.64)', 'mult()', 'sub()', 'cos()', 'div()', 'exp()', 'cos()', 'x_0()', 'add()', 'x_0()', 'uniform_constant(value=-1.0)', 'uniform_constant(value=-0.914)', 'exp()', 'x_0()', 'uniform_constant(value=-0.7)', 'x_0()', 'div()', 'exp()', 'exp()', 'mult()', 'uniform_constant(value=0.6000000000000001)', 'x_0()', 'lnmod()', 'sin()', 'x_0()', 'uniform_constant(value=-0.009)', 'x_0()', 'x_0()', 'x_0()', 'uniform_constant(value=-0.09999999999999998)', 'uniform_constant(value=0.10000000000000009)', 'uniform_constant(value=1.0)', 'x_0()'])

    A = np.array(A, dtype='<U100')
    B = np.array(B, dtype='<U100')
    iteration = '10'
    l_iterations = [0 for i in range(1000)]
    size_diff = [0 for i in range(1000)]
    ind_iter = [0]
    rA, rB, C, ma, mb, exA, exB = compute_masks(A, B)
    print(A.tolist())
    print(B.tolist())
    print('rA', rA.tolist())
    print('rB', rB.tolist())
    print('ma',ma.tolist())
    print('mb',mb.tolist())
    print('exA',exA.tolist())
    print('exB', exB.tolist())
    exA = np.array(['   ', '   ', 'add()', 'mult()', 'uniform_constant(value=0.25)', 'x_0()','x_0()'])
    exB = np.array(['F_div()', 'T_x_1()', 'add()', 'mult()', 'uniform_constant(value=0.25)','x_0()', 'x_0()'])
    #print('After terminal replacement (uniform constant)')
    #print('exA',exA.tolist())
    #print('exB', exB.tolist())
    #print('ima',ima.tolist())
    #print('imb',imb.tolist())

    #exit()

    off, it, size_diff  = crossover3(A, B, function_set, iteration, 0, 1, 0)#recursive_crossover(A, B, function_set,0, iteration, l_iterations, size_diff, ind_iter)
    print(off, it, size_diff)
    #print('A', A)
    #print('B',B)
    #print('n offs',n_offs)
    #for o in offs:
    #    print(o)


    '''


    A = ['ifFoodAhead()', 'move()', 'Progn()', 'left()', 'move()']
    B = ['Progn()','Progn()','move()', 'move()', 'right()']

    function_set = np.array(['ifFoodAhead', 'Progn'])
    terminal_set = ['left', 'right', 'move']
    terminal_params = {}

    A = np.array(A, dtype='<U100')
    B = np.array(B, dtype='<U100')
    iteration = '1'

    i=0
    while(list(A)!=list(B)):
        print('\n\n---iteration '+str(i)+'-----')
        rA, rB, C, ma, mb, exA, exB = compute_masks(A, B, function_set)
        i+=1
        print(A.tolist())
        print(B.tolist())
        print('rA', rA.tolist())
        print('rB', rB.tolist())
        print('ma',ma.tolist())
        print('mb',mb.tolist())
        print('exA',exA.tolist())
        print('exB', exB.tolist())
        #print('After terminal replacement (uniform constant)')
        #print('exA',exA.tolist())
        #print('exB', exB.tolist())
        #print('ima',ima.tolist())
        #print('imb',imb.tolist())

        #exit()

        off, it, size_diff  = crossover3(A, B, function_set, iteration, 0, 1, 0)#recursive_crossover(A, B, function_set,0, iteration, l_iterations, size_diff, ind_iter)
        print(off, it, size_diff)
        A = off
