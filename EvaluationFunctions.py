import numpy as np
import random
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

def random_fitness(ind):
    return random.random()

class SymbolicRegression:
    def __init__(self,domains, npoints, ndimensions, visualize):
        self.type='symbolicRegression'
        self.domains = domains
        self.npoints = npoints
        self.ndimensions = ndimensions

        self.points = self.targets = None

        self.preds = np.zeros(self.npoints)
        self.visualize=visualize
        self.variables = ['x_'+str(i) for i in range(ndimensions)]

    def initialize(self):
        self.initializePoints()
        self.initializeTargets()

    def initializePoints(self):
        self.points = [[random.random() for j in range(self.ndimensions)] for i in range(self.npoints)]
        self.points.sort()
        self.points = np.array(self.points)
        for i in range(self.npoints):
            for j in range(self.ndimensions):
                self.points[i][j] = self.points[i][j]*(self.domains[j][1]-self.domains[j][0])+self.domains[j][0]

        self.targets = self.points

    def evaluate(self,ind):
        if(self.visualize):
            print('genotype', ind.genotype)
        for i in range(self.npoints):
            
            val = ind.arithmetic_interpret(None, self.points[i], logExecutedNodes=True)
            self.preds[i] = val

            if(self.visualize):
                print('x:'+str(self.points[i])+', yp:'+str(self.preds[i])+', yt:'+str(self.targets[i]))
    
        if(self.visualize):

            print('MSE',self.MSE(self.preds, self.targets))
            plt.plot(self.points, self.preds,label = 'predicted')
            plt.plot(self.points, self.targets, label='targets')
            plt.ylim(min(self.targets)-0.5*(max(self.targets)-min(self.targets)), max(self.targets)+0.5*(max(self.targets)-min(self.targets)))
            plt.legend()
            plt.show()

        val = self.MSE(self.preds, self.targets)
        if(np.isnan(val)):
            return np.inf
        return val
        
    def RMSE(self, preds, targets):
        return np.sqrt(np.mean((preds-targets)**2))

    def MSE(self, preds, targets):
        return np.mean((preds-targets)**2)
        
    def visualize(self):
        plt.figure()
        if(self.ndimensions==1):
            plt.plot(self.points, self.preds,label = 'predicted')
            plt.plot(self.points, self.targets, label='targets')
            plt.ylim(min(self.targets)-0.5*(max(self.targets)-min(self.targets)), max(self.targets)+0.5*(max(self.targets)-min(self.targets)))
            plt.legend()
            plt.show()
        elif(self.ndimensions==2):
            x = np.arange(self.domains[0][0],self.domains[0][1]+0.1,0.1)
            y = np.arange(self.domains[1][0],self.domains[1][1]+0.1,0.1)
            x,y = np.meshgrid(x,y)
            z = self.func([x,y])
            ax = plt.axes(projection='3d')
            ax.plot_surface(x,y,z, cmap='viridis')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.tight_layout()
            plt.show()
        else:
            print('unsuported dimensions')

class Koza1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**4 + points**3+points**2+points)[0]

class Koza2(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**5 - 2*points**3 + points)[0]

class Koza3(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**6 - 2*points**4 + points**2)[0]


class Nguyen1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**3 + points**2 + points)[0]

class Nguyen3(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**5 + points**4 + points**3 + points**2 + points)[0]

class Nguyen4(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (points**6 + points**5 + points**4 + points**3 + points**2 + points)[0]


class Nguyen5(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (np.sin(points**2) * np.cos(points) - 1)[0]

class Nguyen6(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (math.sin(points) * math.sin(points+points**2))[0]

class Nguyen7(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (math.log(points+1) * math.log(points**2+1))[0]

class Nguyen8(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return (np.sqrt(points))[0]

class Nguyen9(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return math.sin(points[0]) + math.sin(points[1]**2)

class Nguyen10(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return 2*math.sin(points[0]) + math.cos(points[1])

class Nguyen11(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return points[0]**points[1]

class Nguyen12(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return points[0]**4 - points[0]**3 + 0.5*points[1]**2 - points[1]


class Paige1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x, y = points[0], points[1]
        return 1.0/(1.0+x**-4) + 1.0/(1.0+y**-4)

class Korns1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)
    def initializePoints(self):
        self.domains = [[-50,50] for i in range(5)]
        self.npoints = 10000
        self.ndimensions=5
        self.points = [[random.random()*(self.domains[j][1]-self.domains[j][0])+self.domains[j][0] for j in range(self.ndimensions)] for i in range(self.npoints)]

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x, y, z, w, v = points[0], points[1], points[2], points[3],points[4]
        return 1.0/(1.0+x**-4) + 1.0/(1.0+y**-4)


class Keijzer12(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x, y  = points[0], points[1]
        return x**4 -x**3 + (y**2)/2 -y 





#the functions from this point onwards are not part of "GP needs better benchmarks"
class R1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return ((x+1.0)**3)/(x**2 - x + 1.0 )

class R2(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)
    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return (x**5 - 3*x**3 + 1.0) / (x**2 + 1)

class R3(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)
    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return (x**6 + x**5) / (x**4 + x**3 + x**2 + x + 1)

class Livemore1(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return 1.0/3 + x + math.sin(x**2)

class Livemore2(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return math.sin(x**2)*math.cos(x)-2.0

class Livemore3(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return math.sin(x**3)*math.cos(x**2)-1.0

class Livemore4(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.log10(x+1) + np.log10(x**2+1) + np.log10(x)


class Livemore5(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return x**4 - x**3 + x**2 - y


class Livemore6(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return 4*x**4 + 3*x**3 + 2*x**2 + x

class Livemore7(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.sinh(x)

class Livemore8(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.cosh(x)

class Livemore9(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return x**9 + x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x

class Livemore10(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return 6 * np.sin(x) *np.cos(y)

class Livemore11(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return (x**2 * x**2) /(x+y)

class Livemore12(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return (x**5) /(y**3)

class Livemore13(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return x**(1.0/3)

class Livemore14(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        
        return x**3 + x**2 +x + np.sin(x) + np.sin(x**2)

class Livemore15(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        
        return x**(1.0/5)

class Livemore16(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        
        return x**(2.0/5)

class Livemore17(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        y = points[1]
        return 4*np.sin(x)*np.cos(y)

class Livemore18(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.sin(x**2)*np.cos(x)-5

class Livemore19(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return x**5 + x**4 + x**2 + x

class Livemore20(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.exp(-x**2)

class Livemore21(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x


class Livemore22(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        x = points[0]
        return np.exp(-0.5*x**2)



class Schwefel(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return 418.9829*self.ndimensions - np.sum(points * np.sin(np.sqrt(abs(points))))

class Sphere(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return np.sum(points **2)    

class Rosenbrock(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        a = 0.0
        for i in range(len(points)-1):
            a+=100*(points[i+1]-points[i]**2)**2 + (points[i] -1)**2

class Rastringin(SymbolicRegression):
    def __init__(self,domain, npoints, ndimensions, visualize):
        SymbolicRegression.__init__(self,domain, npoints, ndimensions, visualize)

    def initializeTargets(self):
        self.targets = np.array([self.func(self.points[i]) for i in range(self.npoints)])

    def func(self, points):
        return 10*len(points) + np.sum(points**2 -10*np.cos(2*math.pi*points))    

    




class ArtificialAnt:
    def __init__(self, visualize, map_name):
        self.type='artificialAnt'
        self.visualize=visualize
        self.x, self.y, self.heading = 0, 0, 'r'

        self.eaten=0
        self.map_name = map_name
        self.read_grid(map_name)

        self.total_food = np.sum(self.original_grid)
        self.grid = np.zeros(self.original_grid.shape)
        self.grid += self.original_grid

    def initialize(self):
        pass
        
    def read_grid(self,fname='sf_map.txt'):

        f = open(fname)
        lines = f.readlines()
        for i in range(len(lines)):
            if('\n' in lines[i]):
                lines[i] = lines[i][:lines[i].index('\n')]
        self.original_grid = np.zeros((len(lines[0]), len(lines)))
        
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                self.original_grid[i][j] = int(lines[i][j])
        f.close()


    def plot_grid(self):
        grid = self.grid
        for i in range(grid.shape[0]):
            s=''
            for j in range(grid.shape[1]):
                if(self.x == i and self.y == j):
                    s+='['+self.heading+']'
                else:
                    if(grid[i][j]==1):
                        s+='[O]'
                    else:
                        s+='[ ]'
            print(s)

    def imshow_grid(self):
    
        grid = self.grid
        for i in range(len(grid)):
            print('{'+str(list(grid[i]))[1:-1]+'},')
        plt.imshow(grid)
        plt.show()

    def left(self):
        if(self.heading=='r'):
            self.heading = 'u'
        elif(self.heading == 'u'):
            self.heading = 'l'
        elif(self.heading == 'l'):
            self.heading = 'd'
        elif(self.heading =='d'):
            self.heading = 'r'
        return True
    def right(self):
        if(self.heading == 'l'):
            self.heading = 'u'
        elif(self.heading=='u'):
            self.heading = 'r'
        elif(self.heading =='r'):
            self.heading = 'd'        
        elif(self.heading == 'd'):
            self.heading = 'l'
        return True

    def move(self):
        if(self.heading == 'u'):
            self.x = (self.x-1)%self.grid.shape[0]
        elif(self.heading == 'd'):
            self.x = (self.x+1)%self.grid.shape[0]
        elif(self.heading == 'l'):
            self.y = (self.y-1)%self.grid.shape[1]
        elif(self.heading == 'r'):
            self.y = (self.y+1)%self.grid.shape[1]

        if(self.grid[self.x][self.y] == 1):
            self.grid[self.x][self.y] = 0
            self.eaten +=1

        return True

    def foodAhead(self):
        if(self.heading=='l'):
            x, y = self.x, (self.y-1)%self.grid.shape[1]
        elif(self.heading=='r'):
            x, y = self.x, (self.y+1)%self.grid.shape[1]
        elif(self.heading=='d'):
            x, y = (self.x+1)%self.grid.shape[0], self.y
        elif(self.heading=='u'):
            x, y = (self.x-1)%self.grid.shape[0], self.y
        return (self.grid[x][y] == 1)

    def teleoperate(self):
        print('press l,r,m to turn left, right or move; x to exit')
        choice = input('>')
        #print(type(choice))
        while(choice!='x'):
            if(choice == 'l'):
                self.left()
            elif(choice == 'r'):
                self.right()
            elif(choice =='m'):
                self.move()

            self.plot_grid()
            choice = input('>')

    def evaluate(self,ind):
        self.eaten = 0
        self.grid = self.grid*0+self.original_grid
        ind.behaviour = np.zeros((2,self.steps))
        current_node = None
        if(self.visualize):
            print('genotype', ind.genotype)

        c = 0
        for i in range(self.steps):            
            current_node, motion_terminated = ind.stateful_interpret(self, current_node, logExecutedNodes=True)
            ind.behaviour[0][i] = self.x
            ind.behaviour[1][i] = self.y
            c=i
            if(self.visualize):
                self.plot_grid()
        while(c<self.steps):
            ind.behaviour[0][c] = ind.behaviour[0][c-1]
            ind.behaviour[1][c] = ind.behaviour[1][c - 1]
            c+=1
        return self.eaten
        
class SantaFeAntTrail(ArtificialAnt):
    def __init__(self, visualize,map_name='maps/sf_map.txt'):
        ArtificialAnt.__init__(self, visualize, map_name)
        self.steps = 600

class LosAltosHillsAntTrail(ArtificialAnt):
    def __init__(self, visualize,map_name='maps/lah_map.txt'):
        ArtificialAnt.__init__(self, visualize, map_name)
        self.steps = 3000

class SantaFeAntTrailMin(ArtificialAnt):
    def __init__(self, visualize,map_name='maps/sf_map.txt'):
        ArtificialAnt.__init__(self, visualize, map_name)
        self.steps = 600

    def evaluate(self,ind):
        return 89-super().evaluate(ind)


class LosAltosHillsAntTrailMin(ArtificialAnt):
    def __init__(self, visualize,map_name='maps/lah_map.txt'):
        ArtificialAnt.__init__(self, visualize, map_name)
        self.steps = 3000

    def evaluate(self,ind):
        return 157-super().evaluate(ind)


    