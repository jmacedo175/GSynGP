from copy import deepcopy
import numpy as np
from numba import njit

@njit
def contains(vec, symbol):
    for i in vec:
        if i==symbol:
            return True
    return False
@njit
def toStructure(s, function_set):
    r = np.zeros(len(s), dtype = s.dtype)
    for i in range(len(s)):
        si = s[i].split('(')[0]
        if(contains(function_set, si)):
            r[i] = 'f'
        else:
            r[i] = 't'
    return r


@njit()
def LCS(A, B):
    lA = len(A)
    lB = len(B)
    C = np.zeros((lA + 1, lB + 1))
    i = 1

    while i <= lA:
        j = 1
        while j <= lB:
            if A[i - 1] == B[j - 1]:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i - 1][j], C[i][j - 1])
            j += 1
        i += 1
    return C

@njit()
def LCS_MASKS(A, B, C):#, float_vals=False):
    ##no float val support
    ma = np.zeros(len(A), dtype = A.dtype)
    for i in range(len(A)):
        ma[i] = '.'
    mb = np.zeros(len(B), dtype = B.dtype)#np.copy(B)
    for i in range(len(B)):
        mb[i] = '.'
    na = i = len(A)
    nb = j = len(B)

    while i > 0 and j > 0:
        if C[i][j] != C[i - 1][j] and C[i][j] != C[i][j - 1] and C[i][j] == C[i - 1][j - 1] + 1:

            if ma[i - 1] == '.':
                na -= 1
            if mb[j - 1] == '.':
                nb -= 1
            ma[i - 1] = A[i - 1]
            mb[j - 1] = B[j - 1]
            i -= 1
            j -= 1
        elif C[i][j] == C[i][j - 1] and C[i][j] != C[i - 1][j]:
            #if (float_vals and ma[i-1]==-2) or ma[i - 1] == '.':
            if ma[i - 1] == '.':
                na -= 1
            ma[i - 1] = A[i - 1]

            j -= 1
        elif C[i][j] != C[i][j - 1] and C[i][j] == C[i - 1][j]:
            #if (float_vals and mb[j-1]==-2) or mb[j - 1] == '.':
            if mb[j - 1] == '.':
                nb -= 1
            mb[j - 1] = B[j - 1]
            i -= 1
        else:
            j -= 1

    ##na, nb are respectively the number of symbols in A, B that are not common
    return ma, mb, na, nb


@njit()
def LCS_MASKS_FLOATS(A, B, C):
    #ma = [0 for i in range(len(A)-2)]
    #mb = [0 for i in range(len(A)-2)]
    ma = np.zeros(len(A))-2
    mb = np.zeros(len(B))-2
    na = len(A)
    nb = len(B)
    i = len(A)
    j = len(B)

    while i > 0 and j > 0:
        if C[i][j] != C[i - 1][j] and C[i][j] != C[i][j - 1] and C[i][j] == C[i - 1][j - 1] + 1:

            if (ma[i-1]==-2):
                na -= 1
            if (mb[j-1]==-2):
                nb -= 1
            
            ma[i - 1] = A[i - 1]
            mb[j - 1] = B[j - 1]
            i -= 1
            j -= 1
        elif C[i][j] == C[i][j - 1] and C[i][j] != C[i - 1][j]:
            if (ma[i-1]==-2):
                na -= 1
            ma[i - 1] = A[i - 1]

            j -= 1
        elif C[i][j] != C[i][j - 1] and C[i][j] == C[i - 1][j]:
            if (mb[j-1]==-2):
                nb -= 1
            mb[j - 1] = B[j - 1]
            i -= 1
        else:
            # if(random.random()<0.5):
            j -= 1
            # else:
            #    j-=1

    ##na, nb are respectively the number of symbols in A, B that are not common
    return ma, mb, na, nb

@njit()
def INVERTED_LCS_MASKS(A, B, C):
    ma = np.copy(A)
    mb = np.copy(B)


    na = len(A)
    nb = len(B)
    i = len(A)
    j = len(B)

    while i > 0 and j > 0:
        if C[i][j] != C[i - 1][j] and C[i][j] != C[i][j - 1] and C[i][j] == C[i - 1][j - 1] + 1:
            if ma[i - 1] != '.':
                na -= 1
            if mb[j - 1] != '.':
                nb -= 1
            ma[i - 1] = '.'
            mb[j - 1] = '.'
            i -= 1
            j -= 1

        elif C[i][j] != C[i][j - 1] and C[i][j] == C[i - 1][j]:
            if mb[j - 1] != '.':
                nb -= 1
            mb[j - 1] = '.'
            i -= 1
        elif C[i][j] == C[i][j - 1] and C[i][j] != C[i - 1][j]:
            if ma[i - 1] != '.':
                na -= 1
            ma[i - 1] = '.'

            j -= 1
        else:
            j -= 1

    return ma, mb, na, nb


@njit()
def expanded_masks(ma, mb, A, B,rA,rB, function_set):
    exA = np.zeros(len(A)+len(B), dtype=ma.dtype)#[]
    exB = np.zeros(len(A)+len(B), dtype=ma.dtype)#[]
    indA, indB=0, 0

    '''
    af = np.zeros(len(A), dtype=ma.dtype)#[]
    bf = np.zeros(len(B), dtype=ma.dtype)#[]
    for i in range(len(A)):
        if contains(function_set,A[i]) or ('(' in A[i] and contains(function_set,A[i].split('(')[0])):
            af[i] = 1
    for i in range(len(B)):
        if contains(function_set, B[i]) or ('(' in B[i] and contains(function_set, B[i].split('(')[0])):
            bf[i] = 1
    print('af', list(af), len(af))
    print('rA', list(rA), len(rA))
    print('bf', list(bf), len(bf))
    print('rB', list(rB), len(rB))
    '''

    i = 0
    j = 0
    while i < len(ma) or j < len(mb):
        #print(i,j)
        if i < len(ma) and ma[i] == '.':
            if rA[i] == 'f': #af[i]==1:
                exA[indA] = 'F_' + A[i]
            else:
                exA[indA] = 'T_' + A[i]
            indA+=1
            i += 1
            if j < len(mb) and mb[j] == '.':
                if rB[j] =='f': #bf[j]==1:
                    exB[indB] = 'F_' + B[j]
                else:
                    exB[indB] = 'T_' + B[j]
                j += 1
            else:
                exB[indB]='   '
            indB+=1

        elif j < len(mb) and mb[j] == '.':
            if rB[j] == 'f': #bf[j]==1:
                exB[indB] = 'F_' + B[j]
            else:
                exB[indB] = 'T_' + B[j]
            indB+=1
            j += 1

            exA[indA] = '   '
            indA+=1

        elif i < len(ma) and j < len(mb) and ma[i] == mb[j]:
            exA[indA] = ma[i]
            exB[indB] = ma[i]
            indA+=1
            indB+=1
            i += 1
            j += 1
        elif i < len(ma) and j == len(mb):
            exA[indA] = ma[i]
            indA+=1
            exB[indB] = '   '
            indB+=1
            i += 1
            #print('indA', indA)
            #print('indB', indB)
        elif j < len(mb) and i == len(ma):
            exB[indB] = mb[j]
            indB+=1
            #exB.append(mb[j])
            #exA.append('   ')
            exA[indA] = '   '
            indA+=1
            #print('indA', indA)
            #print('indB', indB)
            j += 1
        else:
            #print('Entra aqui')
            #print(ma[i], mb[j])
            break
    #print(indA, indB)
    eA = np.zeros(indA, dtype=ma.dtype)
    eB = np.zeros(indB, dtype = mb.dtype)
    for i in range(indA):
        eA[i] = exA[i]
    for i in range(indB):
        eB[i] = exB[i]
    return eA,eB #exA, exB
