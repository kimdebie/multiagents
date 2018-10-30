import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import math as m

def is_sampling():

    n = 10000

    # parameters of the probability distributions
    mu_1, sigma_1 = 0, 1
    a, b = -5, 5

    # sample from the distributions
    X = np.random.uniform(a, b, n)

    # setup pdfs
    norm_pdf = norm(mu_1, sigma_1)
    uniform_pdf = uniform(a, b-a)

    estimate = 0

    for x in X:
        estimate +=  pow(x,2) * norm_pdf.pdf(x)/uniform_pdf.pdf(x)

    is_ratio = (1/n)*estimate
    print("Estimate for X2: {} ".format(is_ratio))

    true_value = 1
    print("Error: {}".format(is_ratio-true_value))

def is_sampling_cos():
    n = 10000
    a, b = -1, 1

    # sample from the distributions
    X = np.random.uniform(a, b, n)

    # setup pdf
    uniform_pdf = uniform(a, b-a)

    estimate = 0

    for x in X:
        estimate +=  pow(x,2) * cos_pdf(x)/uniform_pdf.pdf(x)

    is_ratio = (1/n)*estimate
    print("Estimate for X2 with cosinus function: {} ".format(is_ratio))

    true_value = 1
    print("Error: {}".format(is_ratio-true_value))


def cos_pdf(x):
    value = (1+m.cos(x * m.pi))/2
    return value

if __name__== '__main__':
    is_sampling()
    is_sampling_cos()
