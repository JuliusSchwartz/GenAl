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

def map(A):
    return 1024*A - 512

##########################
###########
#########################

##########################
########### function max_min() clips any numbers that go outside range -512->512
#########################


def max_min(A):
    if A > 1:
        return 1
    if A < 0:
        return 0
    else:
        return A
    
##########################
########### function update(generates new solutions (current))
#########################
    
    
def update(current, D, T):
    
    alpha = 0.1
    omega = 2.1
    

    D_new = np.zeros((5, 5))
    next = np.zeros(5)
    
    R = np.array([D[0][0]*(2*np.random.rand() - 1), D[1][1]* (2*np.random.rand() - 1), D[2][2]* (2*np.random.rand() - 1), 
                  D[3][3]*(2*np.random.rand() - 1), D[4][4]* (2*np.random.rand() - 1)])

    next[0] = max_min(current[0] + R[0])
    next[1] = max_min(current[1] + R[1])
    next[2] = max_min(current[2] + R[2])
    next[3] = max_min(current[3] + R[3])
    next[4] = max_min(current[4] + R[4])

    delta = obj(map(next[0]), map(next[1]), map(next[2]), map(next[3]), map(next[4])) - obj(map(current[0]), 
                map(current[1]), map(current[2]), map(current[3]), map(current[4]))
    

    d = np.sqrt(R[0]**2 + R[1]**2 + R[2]**2 + R[3]**2 + R[4]**2)
    
    R = np.fabs(R)
        
    D_new[0][0] = (1-alpha)*D[0][0] + alpha*omega*abs(R[0])
    D_new[1][1] = (1-alpha)*D[1][1] + alpha*omega*abs(R[1])
    D_new[2][2] = (1-alpha)*D[2][2] + alpha*omega*abs(R[2])
    D_new[3][3] = (1-alpha)*D[3][3] + alpha*omega*abs(R[3])
    D_new[4][4] = (1-alpha)*D[4][4] + alpha*omega*abs(R[4])
        
    if delta < 0:
        return next, D_new
    
    elif np.exp(-delta/(T*d)) > np.random.rand():
        return next, D_new
    
    else:
        return current, D
    
##########################
########### function init_T() initialises the temperature
#########################    

def init_T(current, D, n):
    
    alpha = 0.1
    omega = 2.1
    
    obj_list = []
    
    for i in range(0, n):
    
        D_new = np.zeros((5, 5))
        next = np.zeros(5)
    
        R = np.array([D[0][0]*(2*np.random.rand() - 1), D[1][1]* (2*np.random.rand() - 1), D[2][2]* (2*np.random.rand() - 1), 
                      D[3][3]*(2*np.random.rand() - 1), D[4][4]* (2*np.random.rand() - 1)])
    
        next[0] = max_min(current[0] + R[0])
        next[1] = max_min(current[1] + R[1])
        next[2] = max_min(current[2] + R[2])
        next[3] = max_min(current[3] + R[3])
        next[4] = max_min(current[4] + R[4]) 
        
        R = np.fabs(R)     
        
        D_new[0][0] = (1-alpha)*D[0][0] + alpha*omega*abs(R[0])
        D_new[1][1] = (1-alpha)*D[1][1] + alpha*omega*abs(R[1])
        D_new[2][2] = (1-alpha)*D[2][2] + alpha*omega*abs(R[2])
        D_new[3][3] = (1-alpha)*D[3][3] + alpha*omega*abs(R[3])
        D_new[4][4] = (1-alpha)*D[4][4] + alpha*omega*abs(R[4])
        
        f = obj(map(next[0]), map(next[1]), map(next[2]), map(next[3]), map(next[4]))
        obj_list.append(f)

        current = next 
        D = D_new
    
    T = np.std(obj_list)
    return T


##########################
########### function main() implements SA on objective function
#########################

def main(k, L_k):
    
    res_list = []
    solns = []
    
    ###### Initial solution generated randomly
    
    init = np.array([np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()])

    D_start = np.zeros((5,5))

    for i in range(0, 5):
        D_start[i][i] = init[i]
    
    T = init_T(init, D_start, 100)
    
    current, D = update(init, D_start, T)
    
    for j in range(0, k):
        eta_list = []
        for i in range(0, L_k):            
            
            X, Y = update(current, D, T)
            if np.any(current != X) == True:
                current, D = X, Y
                eta_list.append(obj(map(current[0]), map(current[1]), map(current[2]), map(current[3]), map(current[4])))
                res_list.append(obj(map(current[0]), map(current[1]), map(current[2]), map(current[3]), map(current[4])))
                solns.append((map(current[0]), map(current[1]), map(current[2]), map(current[3]), map(current[4])))

        alpha_2 = max(0.5, np.exp(-0.7*T/np.std(eta_list)))
        #alpha_2 = 0.95
        T = alpha_2*T

    best = min(res_list)
    best_id = res_list.index(best)
    co_ords = solns[best_id]

    return best, co_ords, res_list, solns



#################################
#### Code below gives example of SA being run on obj function 50 times, with results plotted in a histogram
#################################

ult_res = []
ult_solns = []

kk = 20
L_kk = 500

for w in range(0, 50):
    a = main(kk, L_kk)
    ult_res.append(a[0])
    ult_solns.append(a[1])
    
ult_best = min(ult_res)
ult_best_id = ult_res.index(ult_best)
ult_soln = ult_solns[ult_best_id]
print ult_best, ult_soln


plt.hist(ult_res, 20, color='red', histtype='bar', ec='black')
plt.title('SA 5DEH, k = 20, L_k = 500, Huang Schedule')
plt.xlabel('Objective function value of best solution')
plt.ylabel('Number of best solutions')
plt.show()

print min(ult_res)
print np.mean(ult_res)
print np.std(ult_res)
