import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import math as m

TRUE_VALUE = 1

def is_sampling(n, p, q):

    '''

    Importance sampling for estimating E[X^2] of a normal / cosinus 
    distribution, using random samples from a uniform distribution at a given
    interval.

    '''

    distr = {
        "normal": 1,
        "uniform": 2,
        "cosinus": 3
    }

    estimate = 0

    # check for uniform q distribution
    if distr[q] == 2:

        # calculate importance sampling with normal distribution
        if distr[p] == 1:

            # set parameters and pdfs
            mu_1, sigma_1 = 0, 1
            a, b = -5, 5
            p_x = norm(mu_1, sigma_1)
            q_x = uniform(a,b-a)

            # draw sample
            X = np.random.uniform(a, b, n)

            # estimate expected value
            for x in X:
                estimate +=  pow(x,2) * p_x.pdf(x)/q_x.pdf(x)

        # calculate importance sampling with cosinus distribution
        elif distr[p] == 3:

            # set parameters and pdf
            a, b = -1, 1
            q_x = uniform(a,b-a)

            # draw sample
            X = np.random.uniform(a, b, n)

            # estimate expected value
            for x in X:
                estimate +=  pow(x,2) * cos_pdf(x)/q_x.pdf(x)

        else:
            print("Distribution of p must be either cosinus or normal")
            return 0

        # multiply by inverse of number of samples
        is_ratio = (1/n)*estimate

        print("-------IMPORTANCE SAMPLING WITH {} SAMPLES-------".format(n))
        print("Estimate of E[X^2] for {} pdf: {} ".format(p, is_ratio))
        print("Error: {}".format(is_ratio-TRUE_VALUE))
        print("----------------------------------------------------")

    else:
        print("Distribution of q must be uniform")
        return 0

def cos_pdf(x):
    return (1+m.cos(x * m.pi))/2

if __name__== '__main__':
    num_samples = 100000
    is_sampling(num_samples,"normal","uniform")
    is_sampling(num_samples,"cosinus","uniform")
