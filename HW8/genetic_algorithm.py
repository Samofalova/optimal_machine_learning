import numpy as np
from sympy import *
import itertools
import random

def genetic_algorithm(func, restr, strength_of_mutation=1, number_of_generations=20):
    """
    Solving an optimisation problem for a function of one or more variables
    with inequality-type constraints by Genetic Algorithm.
    Returns the found point.

    Parameters
    ----------
    func : string
        Function for optimisation.
    restr : list
        List of strings with given linear restrictions.
    strength_of_mutation : float, default=1
        If the value is different from 1, then some number of genes
        will be multiplied by the specified value.
    number_of_generations : int, default=20
        The number of iterations.
        
    Examples
    --------
    >>> func = '-x1-2*x2+x2**2'
    >>> restr = ['-3*x1-2*x2 >= -6', '-x1-2*x2 >= -4']
    >>> x = genetic_algorithm(func, restr)
    >>> x
    [1.5201394553670138, 0.7196650985988775]
    """
    res = get_data(func, restr)
    func = res['func']
    restr = res['restr']
    number_of_genes = res['number_of_genes']
    c = res['c']
    symbs = res['symbs']

    population = get_zero_population(number_of_genes, restr, c, symbs)

    for generation in range(number_of_generations):
        
        evaluated = evaluate(population, func, symbs)
        population = get_new_population(evaluated, strength_of_mutation, restr, number_of_genes, c, symbs)

    evaluated = evaluate(population, func, symbs)
    return evaluated[0]

def get_data(func, restr):
    func = sympify(func)
    symbs = list(func.free_symbols)
    number_of_genes = len(symbs)
    c = []
    restrictions = []
    for i in range(len(restr)):
            restrictions.append(sympify(restr[i][:restr[i].index('>')].replace(' ', '')))
            c.append(restr[i][restr[i].index('>')+2:].replace(' ', ''))
    return {'func' : func,
            'restr' : restrictions,
            'number_of_genes': number_of_genes, 
            'c': c, 
            'symbs' : symbs}

def check_restrictions(gene, restr, c, symbs):
    flag = True
    for i in range(len(restr)):
        if sympify(restr[i]).subs(list(zip(symbs, list(gene)))) < float(c[i]):
            flag = False
    return flag

def get_zero_population(number_of_genes, restr, c, symbs):
    len_of_population = 500
    zero_population = []
    while len(zero_population) < len_of_population:
        x = np.random.uniform(-10, 10, number_of_genes)
        if check_restrictions(x, restr, c, symbs) == True:
            zero_population.append(x)
    return zero_population

def evaluate(population, func, symbs):
    finding_best = []
    for gene in population:
        value = func.subs(list(zip(symbs, list(gene))))
        finding_best.append((list(gene), value))
    finding_best.sort(key = lambda x: x[1])
    best = []
    for i in range(len(finding_best)):
        best.append(finding_best[i][0])
    return best

def recombination(parents, number_of_genes):
    positions = []
    for i in range(number_of_genes):
        c = []
        for j in range(len(parents)):
            c.append(parents[j][i])
        positions.append(c)
    children_tuple = list(itertools.product(*positions))
    children = [list(child) for child in children_tuple]
    return children

def get_new_population(evaluated, strength_of_mutation, restr, number_of_genes, c, symbs):
    if len(evaluated) > 10:
        n = int(len(evaluated)/2)
    else:
        n = len(evaluated)
    p = 2
    parents = evaluated[:p]
    children = recombination(parents, number_of_genes)

    new_population = evaluated[:n]
    new_population.extend(children)
    number_of_mutations = random.choice(list(range(len(children))))
    for i in range(number_of_mutations):
        index = random.choice(list(range(number_of_mutations)))
        new_population[index] = [gene*strength_of_mutation for gene in new_population[index]]
    population = []
    for gene in new_population:
        if check_restrictions(list(gene), restr, c, symbs) == True:
            population.append(gene)
    return population