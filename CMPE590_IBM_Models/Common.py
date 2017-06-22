import math


def is_converged(new,old,num_of_iterations):
    epsilone = 0.00000001
    if num_of_iterations > max_num_of_iterations :
        return True

    for i in range(len(new)):
        for j in range(len(new[0])):
            if math.fabs(new[i][j]- old[i][j]) > epsilone:
                return False
    return True


def nCr(n,r):
    try:
        if n-r < 0 :
            return 1
        f = math.factorial
        return f(n) / f(r) / f(n-r)
    except:
        print( "value error " + str(n) + "  " + str(r))
        raise


def factorial(n):
    if n < 0 :
        return 1
    f = math.factorial
    return f(n)


const = 0.1
num_of_train_sample = 4000
num_of_test_sample = 100
max_num_of_iterations = 3

#colors for print W  = '\033[0m'  # white (normal)
BL = '\033[30m' # black
R = '\033[31m' # red
G = '\033[32m' # green
O = '\033[33m' # orange
B = '\033[34m' # blue
P = '\033[35m' # purple