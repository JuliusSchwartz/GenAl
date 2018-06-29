import numpy as np
import matplotlib.pyplot as plt

np.random.seed(44)

#############################
#### Objective function defined
##############################


def obj(x1, x2, x3, x4, x5):
    A = -(x2+47)*np.sin(np.sqrt(np.abs(x2+0.5*x1+47))) - x1*np.sin(np.sqrt(np.abs(x1 - x2 - 47)))
    B = -(x3+47)*np.sin(np.sqrt(np.abs(x3+0.5*x2+47))) - x2*np.sin(np.sqrt(np.abs(x2 - x3 - 47)))
    C = -(x4+47)*np.sin(np.sqrt(np.abs(x4+0.5*x3+47))) - x3*np.sin(np.sqrt(np.abs(x3 - x4 - 47)))
    D = -(x5+47)*np.sin(np.sqrt(np.abs(x5+0.5*x4+47))) - x4*np.sin(np.sqrt(np.abs(x4 - x5 - 47)))
    return A + B + C + D

###########################
##### Function which maps range 0->32767 to -512->512, and clips numbers that go beyond this boundary
###########################

def map(A):
    B = 1024.0*A/32767 - 512
    if B > 512:
        return 512
    elif B < -512:
        return -512
    else:
        return B



##########################
########### function mate() that breeds two parents and replaces them with their offspring
#########################

def mate(parA, parB):
    parA = '{0:b}'.format(parA) ########Parent numbers converted into binary#
    parB = '{0:b}'.format(parB) 
        
    L_s = min(len(parA), len(parB))
    
    clipA = parA[:-L_s]    #####Extra digits "clipped" so crossover happens between strings of equivalent length
    clipB = parB[:-L_s]
    
    rsA = parA[-L_s:]
    rsB = parB[-L_s:]
       
    nz = [int(rsA[i]) - int(rsB[i]) for i in range(L_s)] ###### crossover is performed on reduced surrogates
    inds = [i for i, e in enumerate(nz) if e != 0]

    try:
        beg = min(inds)
        end = max(inds)
    except:
        return [int("".join(parA), 2), int(''.join(parB), 2)]
        
    RSA = rsA[beg:end+1]
    RSB = rsB[beg:end+1]
      
    LL_s = min(len(RSA), len(RSB))
    LL_b = min(len(RSA), len(RSB)) 

    prob = [1.0/2**(LL_s-1-i) for i in range(LL_s)]   ####The further to the left the binary digit, the smaller the 
                                                      ##### probability of the break point being located there
    for i in prob:
        if np.random.rand() < i:
            place = prob.index(i)
            break

    br = LL_s - place

    chA = RSA[:-br] + RSB[-br:]
    chB = RSB[:-br] + RSA[-br:]
    
    rsA = clipA + rsA[0:beg] + chA + rsA[end+1:]
    rsB = clipB + rsB[0:beg] + chB + rsB[end+1:]
     
    return [int("".join(rsA), 2), int(''.join(rsB), 2)]    #### Numbers converted back from binary

##########################
########### function mut() which mutates input numbers
#########################


def mut(parents, rate):
    parentA = '{0:b}'.format(parents[0])    ######## Converted into binary
    parentB = '{0:b}'.format(parents[1])
    parentC = '{0:b}'.format(parents[2])
    parentD = '{0:b}'.format(parents[3])
    parentE = '{0:b}'.format(parents[4])

    midA = [(ch if np.random.rand() >= rate else str(1-int(ch))) for ch in parentA]  ###Each bit has same probability
    midB = [(ch if np.random.rand() >= rate else str(1-int(ch))) for ch in parentB] ##### of flipping
    midC = [(ch if np.random.rand() >= rate else str(1-int(ch))) for ch in parentC]
    midD = [(ch if np.random.rand() >= rate else str(1-int(ch))) for ch in parentD]
    midE = [(ch if np.random.rand() >= rate else str(1-int(ch))) for ch in parentE]
    return [int("".join(midA),2), int("".join(midB),2), int("".join(midC),2), int("".join(midD),2), int("".join(midE),2)]



##########################
########### function next_gen() builds next generation out of best performing solutions in current generation
#########################

def next_gen(current):
    
    next = []
    
    fitness = [obj(map(x[0]), map(x[1]), map(x[2]), map(x[3]), map(x[4])) for x in current] ###########Solutions are ranked 
    ranking = [fitness.index(x) for x in np.sort(fitness)]
    current = [current[x] for x in ranking] 
    
    
    
    N = len(fitness)
    S = 2.0
    selection = [((S*(N+1-2*x)+ 2*(x-1))/(N-1)) for x in range(1, N+1)]
    rem = selection - np.floor(selection)
    
    for i in range(N):
        for j in range(int(np.floor(selection)[i])):
            next.append(current[i])
        
    while len(next) < N:
        for i in range(N):
            if rem[i] > np.random.random_sample() and len(next) < N :
                next.append(current[i])
         
    return next


##########################
########### function pair_mate() randomly selects parents and cross breeds them with probability selected by "prob" argument
#########################


def pair_mate(gen, prob):
    np.random.shuffle(gen)
    h1 = gen[:len(gen)/2]
    h2 = gen[len(gen)/2:]
    
    for i in range(len(gen)/2):
        if np.random.random_sample() < prob:
            h1[i][0], h2[i][0] = mate(h1[i][0], h2[i][0])
            h1[i][1], h2[i][1] = mate(h1[i][1], h2[i][1])
            h1[i][2], h2[i][2] = mate(h1[i][2], h2[i][2])
            h1[i][3], h2[i][3] = mate(h1[i][3], h2[i][3])
            h1[i][4], h2[i][4] = mate(h1[i][4], h2[i][4])
    
    return h1 + h2

##########################
########### function iter() takes as input the current generation probability of recombination and mutation probability
########### and calls the above functions to produce a new generation of solutions
#########################


def iter(current, prob, rate):
    next = next_gen(current)
    next = pair_mate(next, prob)
    next = [mut(next[i], rate) for i in range(len(next))]
    
    return next

##########################
########### function main() takes as inputs the probability of recombination, the mutation probability and the size of
########### generation and runs the GA on the 5DEH function for 10 000 objective function evaluations.
#########################


def main(PM, PC, parents):
    
    num_it = 10000/parents
    Pm = PM
    Pc = PC
    
    res = []
    res_av = []
    X_pts = []
    Y_pts = []
    solns = []

    
    next = []                      ###### Members of first generation are randomly selected
    for i in range(0, parents):  
        next.append([np.random.randint(0, 32767), np.random.randint(0, 32767), 
                 np.random.randint(0, 32767), np.random.randint(0, 32767), np.random.randint(0, 32767)])
        
        

    for i in range(0, num_it):
        cur = [obj(map(x[0]), map(x[1]), map(x[2]), map(x[3]), map(x[4])) for x in next]
        solns.append((map(next[cur.index(min(cur))][0]), map(next[cur.index(min(cur))][1]), map(next[cur.index(min(cur))][2]), 
                      map(next[cur.index(min(cur))][3]), map(next[cur.index(min(cur))][4])))
        res.append(min(cur))
        res_av.append(np.mean(cur))
        next = iter(next, Pc, Pm)
        
    A_pts = [x[0] for x in solns]
    B_pts = [x[1] for x in solns]
    C_pts = [x[2] for x in solns]
    D_pts = [x[3] for x in solns]
    E_pts = [x[4] for x in solns]
    
    pts = [A_pts, B_pts, C_pts, D_pts, E_pts]


    best = min(res)
    co_ords = solns[res.index(best)]
    
    return res, res_av, solns, best, co_ords, pts

#################################
#### Code below gives example of GA being run on obj function 50 times, with results plotted in a histogram
#################################

ult_res = []
ult_solns = []

PPm = 0.01
PPc = 0.5

size = 500

for w in range(0, 50):
    a = main(PPm, PPc, size)
    ult_res.append(a[3])
    ult_solns.append(a[4])


    
ult_best = min(ult_res)
ult_best_id = ult_res.index(ult_best)
ult_soln = ult_solns[ult_best_id]
print ult_best, ult_soln

plt.hist(ult_res, 20, color = 'green', histtype='bar', ec='black')
plt.title('GA 5DEH, PM = 0.01, PC = 0.5, size = 500')
plt.xlabel('Objective function value of best solution')
plt.ylabel('Number of best solutions')
plt.show()

print min(ult_res)
print np.mean(ult_res)
print np.std(ult_res)
