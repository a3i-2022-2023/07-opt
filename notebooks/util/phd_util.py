#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy import stats
import pandas as pd
from skopt.space import Space
from skopt.sampler import Lhs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import models
from tensorflow.keras import activations
from sklearn import metrics
from eml.net.embed import encode
import eml.backend.ortool_backend as ortools_backend
from eml.net.reader import keras_reader
from eml.net.process import ibr_bounds
from ortools.linear_solver import pywraplp
# from livelossplot.inputs.tf_keras import PlotLossesCallback
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import time
import skopt

def SIR(y, beta, gamma):
    # Unpack the state
    S, I, R = y
    N = sum([S, I, R])
    # Compute partial derivatives
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    # Return gradient
    return np.array([dS, dI, dR])


def plot_df_cols(data, figsize=None, legend=True):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    x = data.index
    plt.xlabel(data.index.name)
    # Plot all columns
    for cname in data.columns:
        y = data[cname]
        plt.plot(x, y, label=cname)
    # Add legend
    if legend and len(data.columns) <= 10:
        plt.legend(loc='best')
    # Make it compact
    plt.tight_layout()
    # Show
    plt.show()


def simulate_SIR(S0, I0, R0, beta, gamma, tmax, steps_per_day=1):
    # Build initial state
    Z = S0 + I0 + R0
    Z = Z if Z > 0 else 1 # Handle division by zero
    y0 = np.array([S0, I0, R0]) / Z
    # Wrapper
    nabla = lambda y, t: SIR(y, beta, gamma)
    # Solve
    t = np.linspace(0, tmax, steps_per_day * tmax)
    Y = odeint(nabla, y0, t)
    # Wrap as dataframe
    data = pd.DataFrame(data=Y, index=t, columns=['S', 'I', 'R'])
    data.index.rename('time', inplace=True)
    # Return the results
    return data


def simulate_SIR_NPI(S0, I0, R0, beta, gamma, steps_per_day=1):
    # Prepare the result data structure
    S, I, R = [], [], []
    # Store the initial state
    S.append(S0)
    I.append(I0)
    R.append(R0)
    # Loop over all weeks
    for i, b in enumerate(beta):
        # Simulate one week
        wres = simulate_SIR(S[i], I[i], R[i], b, gamma, 7,
                steps_per_day=steps_per_day)
        # Store the final state
        last = wres.iloc[-1]
        S.append(last['S'])
        I.append(last['I'])
        R.append(last['R'])
    # Wrap into a dataframe
    res = pd.DataFrame(data=np.array([S, I, R]).T,
            columns=['S', 'I', 'R'])
    return res



def sample_points(ranges, n_samples, mode, seed=None):
    assert(mode in ('uniform', 'lhs', 'max_min'))
    # Build a space
    space = Space(ranges)
    # Seed the RNG
    np.random.seed(seed)
    # Sample
    if mode == 'uniform':
        X = space.rvs(n_samples)
    elif mode == 'lhs':
        lhs = Lhs(lhs_type="classic", criterion=None)
        X = lhs.generate(space.dimensions, n_samples)
    elif mode == 'max_min':
        lhs = Lhs(criterion="maximin", iterations=100)
        X = lhs.generate(space.dimensions, n_samples)
    # Convert to an array
    return np.array(X)


def generate_SIR_input(max_samples, mode='lhs',
        normalize=True, seed=None, max_beta=1):
    # Sampling space: unnormalized S, I, R, plus beta
    ranges = [(0.,1.), (0.,1.), (0.,1.), (0., float(max_beta))]
    # Generate input
    X = sample_points(ranges, max_samples, mode, seed=seed)
    # Normalize
    if normalize:
        # Compute the normalization constants
        Z = np.sum(X[:, :3], axis=1)
        Z = np.where(Z > 0, Z, 1)
        # Normalize the first three columns
        X[:, :3] = X[:, :3] / Z.reshape(-1, 1)
    # Wrap into a DataFrame
    data = pd.DataFrame(data=X, columns=['S', 'I', 'R', 'beta'])
    return data


def generate_SIR_output(sir_in, gamma, tmax, steps_per_day=1):
    # Prepare a data structure for the results
    res = []
    # Loop over all examples
    for idx, in_series in sir_in.iterrows():
        # Unpack input
        S0, I0, R0, beta = in_series
        # Simulate
        sim_data = simulate_SIR(S0, I0, R0, beta, gamma, tmax,
                steps_per_day=steps_per_day)
        # Compute output
        res.append(sim_data.values[-1, :])
    # Wrap into a dataframe
    data = pd.DataFrame(data=res, columns=['S', 'I', 'R'])
    # Return
    return data


def plot_2D_samplespace(data, figsize=None):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup axes
    plt.xlabel('x_0')
    plt.xlabel('x_1')
    # Plot points
    plt.plot(data[:, 0], data[:, 1], 'bo', label='samples', color='tab:blue')
    plt.plot(data[:, 0], data[:, 1], 'bo', markersize=60, alpha=0.3,
            color='tab:blue')
    # Make it compact
    plt.tight_layout()
    # Show
    plt.show()



def build_ml_model(input_size, output_size, hidden=[],
        output_activation='linear', name=None):
    # Build all layers
    ll = [keras.Input(input_size)]
    for h in hidden:
        ll.append(layers.Dense(h, activation='relu'))
    ll.append(layers.Dense(output_size, activation=output_activation))
    # Build the model
    model = keras.Sequential(ll, name=name)
    return model


class SimpleProgressBar(object):
    def __init__(self, epochs, width=80):
        self.epochs = epochs
        self.width = width
        self.csteps = 0

    def __call__(self, epoch, logs):
        # Compute the number of new steps
        nsteps = int(self.width * epoch / self.epochs) - self.csteps
        if nsteps > 0:
            print('=' * nsteps, end='')
        self.csteps += nsteps


def train_ml_model(model, X, y, epochs=20,
        verbose=0, patience=10, batch_size=32,
        validation_split=0.2, sample_weight=None,
        loss='mse', compile_model=True):
    # Compile the model
    if compile_model:
        model.compile(optimizer='Adam', loss=loss)
    # Build the early stop callback
    cb = []
    if validation_split > 0:
        cb += [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    # if verbose == 0:
    #     cb += [callbacks.LambdaCallback(on_epoch_end=
    #         SimpleProgressBar(epochs))]
    # Train the model
    history = model.fit(X, y, validation_split=validation_split,
                     callbacks=cb, batch_size=batch_size,
                     epochs=epochs, verbose=verbose,
                     sample_weight=sample_weight)
    return history


# def plot_training_history(history=None, figsize=None):
#     plt.figure(figsize=figsize)
#     for metric in history.history.keys():
#         plt.plot(history.history[metric], label=metric)
#     # if 'val_loss' in history.history.keys():
#     #     plt.plot(history.history['val_loss'], label='val. loss')
#     if len(history.history.keys()) > 0:
#         plt.legend()
#     plt.xlabel('epochs')
#     plt.tight_layout()
#     plt.show()


def plot_training_history(history=None,
        figsize=None, print_scores=True, restore_best_weights=True):
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.tight_layout()
    plt.show()
    if print_scores:
        s = 'Model loss:'
        if 'val_loss' in history.history and restore_best_weights:
            bidx = np.argmin(history.history["val_loss"])
            vll = history.history["val_loss"][bidx]
            trl = history.history["loss"][bidx]
            s += f' {trl:.4f} (training)'
            s += f' {vll:.4f} (validation)'
        elif 'val_loss' in history.history and not restore_best_weights:
            vll = history.history["val_loss"][-1]
            trl = history.history["loss"][-1]
            s += f' {trl:.4f} (training)'
            s += f' {vll:.4f} (validation)'
        else:
            trl = history.history["loss"][-1]
            s += f' {trl:.4f} (training)'
        print(s)


def get_ml_metrics(model, X, y):
    # Obtain the predictions
    pred = model.predict(X)
    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(y, pred))
    # Compute the MAE
    mae = metrics.mean_absolute_error(y, pred)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(y, pred)
    return r2, mae, rmse


def save_ml_model(model, name):
    model.save_weights(f'../data/{name}.h5')
    with open(f'../data/{name}.json', 'w') as f:
        f.write(model.to_json())


def load_ml_model(name):
    with open(f'../data/{name}.json') as f:
        model = models.model_from_json(f.read())
        model.load_weights(f'../data/{name}.h5')
        return model


class NPI(object):
    """Docstring for NPI. """

    def __init__(self, name, effect, cost):
        """TODO: to be defined. """
        self.name = name
        self.effect = effect
        self.cost = cost



def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def solve_sir_brute_force(
        npis : list,
        S0 : float,
        I0 : float,
        R0 : float,
        beta_base : float,
        gamma : float,
        nweeks : int = 1,
        budget : float = None):
    # Build a table with beta-confs for a week
    doms = [np.array([0, 1]) for _ in range(len(npis))]
    beta_confs = cartesian_product(doms)
    # Compute costs
    npi_costs = np.array([npi.cost for npi in npis])
    beta_costs = np.dot(beta_confs, npi_costs)
    # Compute beta values
    npi_eff = np.array([npi.effect for npi in npis])
    beta_vals = beta_confs * npi_eff
    beta_vals += (1-beta_confs)
    beta_vals = beta_base * np.product(beta_vals, axis=1)
    # Build all combinations of beta-values and costs
    bsched_cost = cartesian_product([beta_costs for _ in range(nweeks)])
    bsched_vals = cartesian_product([beta_vals for _ in range(nweeks)])
    # Filter out configurations that do not meen the budget
    mask = np.sum(bsched_cost, axis=1) <= budget
    bsched_feas = bsched_vals[mask]
    # Simulate them all
    best_S = -np.inf
    best_sched = None
    for bsched in bsched_feas:
        # Simulate
        res = simulate_SIR_NPI(S0, I0, R0, bsched, gamma, steps_per_day=1)
        last_S = res.iloc[-1]['S']
        if last_S > best_S:
            best_S = last_S
            best_sched = bsched
    return best_S, best_sched



def solve_sir_planning(
        keras_model,
        npis : list,
        S0 : float,
        I0 : float,
        R0 : float,
        beta_base : float,
        nweeks : int = 1,
        budget : float = None,
        tlim : float = None,
        init_state_csts : bool = True,
        network_csts : bool = True,
        effectiveness_csts : bool = True,
        cost_csts : bool = True,
        beta_ub_csts : bool = True,
        use_hints : bool = True):

    assert(0 <= beta_base <= 1)

    # Build a model object
    slv = pywraplp.Solver.CreateSolver('CBC')

    # Define the variables
    X = {}
    for t in range(nweeks+1):
        # Build the SIR variables
        X['S', t] = slv.NumVar(0, 1, f'S_{t}')
        X['I', t] = slv.NumVar(0, 1, f'I_{t}')
        X['R', t] = slv.NumVar(0, 1, f'R_{t}')
        if t < nweeks:
            X['b', t] = slv.NumVar(0, 1, f'b_{t}')

        # Build the NPI variables
        if t < nweeks:
            for npi in npis:
                name = npi.name
                X[name, t] = slv.IntVar(0, 1, f'{name}_{t}')

    # Build the cost variable
    maxcost = sum(npi.cost for npi in npis) * nweeks
    X['cost'] = slv.NumVar(0, maxcost, 'cost')

    # Build the initial state constraints
    if init_state_csts:
        slv.Add(X['S', 0] == S0)
        slv.Add(X['I', 0] == I0)
        slv.Add(X['R', 0] == R0)

    # Build the network constraints
    if network_csts:
        # Build a backend object
        bkd = ortools_backend.OrtoolsBackend()
        # Convert the keras model in internal format
        nn = keras_reader.read_keras_sequential(keras_model)
        # Set bounds
        nn.layer(0).update_lb(np.zeros(4))
        nn.layer(0).update_ub(np.ones(4))
        # Propagate bounds
        ibr_bounds(nn)
        # Build the encodings
        for t in range(1, nweeks+1):
            vin = [X['S',t-1], X['I',t-1], X['R',t-1], X['b',t-1]]
            vout = [X['S',t], X['I',t], X['R',t]]
            encode(bkd, nn, slv, vin, vout, f'nn_{t}')

    # Build the effectiveness constraints
    if effectiveness_csts:
        for t in range(nweeks):
            # Set base beta as the starting one
            bbeta = beta_base
            # Process all NPIs
            for i, npi in enumerate(npis):
                name = npi.name
                # For all NPIs but the last, build a temporary beta variable
                if i < len(npis)-1:
                    # Build a new variable to be used as current beta
                    cbeta = slv.NumVar(0, 1, f'b_{name}_{t}')
                    X[f'b_{name}', t] = cbeta
                else:
                    # Use the "real" beta as current
                    cbeta = X['b', t]
                # Linearize a guarded division
                slv.Add(cbeta >= npi.effect * bbeta - 1 + X[name, t])
                slv.Add(cbeta >= bbeta - X[name, t])
                # Add an upper bound, if requested
                if beta_ub_csts:
                    slv.Add(cbeta <= npi.effect * bbeta + 1 - X[name, t])
                    slv.Add(cbeta <= bbeta + X[name, t])
                # Reset base beta
                bbeta = cbeta

    # Define the cost
    if cost_csts:
        slv.Add(X['cost'] == sum(npi.cost * X[npi.name, t]
            for t in range(nweeks) for npi in npis))
        if budget is not None:
            slv.Add(X['cost'] <= budget)

    # Define the objectives
    slv.Maximize(X['S', nweeks])

    # Build a heuristic solution
    if use_hints:
        hvars, hvals = [], []
        # Sort NPIs by decreasing "efficiency"
        snpis = sorted(npis, key=lambda npi: -(1-npi.effect) / npi.cost)
        # Loop over all the NPIs
        rbudget = budget
        for npi in snpis:
            # Activate on as many weeks as possible
            for w in range(nweeks):
                if rbudget > npi.cost:
                    hvars.append(X[npi.name, w])
                    hvals.append(1)
                    rbudget -= npi.cost
                else:
                    hvars.append(X[npi.name, w])
                    hvals.append(0)
        # Set hints
        slv.SetHint(hvars, hvals)

    # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(tlim * 1000)
    # Solve the problem
    status = slv.Solve()
    # Return the result
    res = None
    closed = False
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        res = {}
        for k, x in X.items():
            res[k] = x.solution_value()
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.INFEASIBLE):
        closed = True
    return res, closed



def sol_to_dataframe(sol, npis, nweeks):
    # Define the column names
    cols = ['S', 'I', 'R', 'b']
    cols += [npi.name for npi in npis]
    # Prepare a result dataframe
    res = pd.DataFrame(index=range(nweeks+1), columns=cols, data=np.nan)
    for w in range(nweeks):
        res.loc[w, 'S'] = sol['S', w]
        res.loc[w, 'I'] = sol['I', w]
        res.loc[w, 'R'] = sol['R', w]
        res.loc[w, 'b'] = sol['b', w]
        for n in npis:
            res.loc[w, n.name] = sol[n.name, w]
    # Store the state for the final week
    res.loc[nweeks, 'S'] = sol['S', nweeks]
    res.loc[nweeks, 'I'] = sol['I', nweeks]
    res.loc[nweeks, 'R'] = sol['R', nweeks]
    # Return the result
    return res


# def generate_market_dataset(nsamples, nitems, noise=0, seed=None):
#     assert(0 <= noise <= 1)
#     # Seed the RNG
#     np.random.seed(seed)
#     # Generate costs
#     exponents = np.random.choice([0.2, 0.3, 4, 5], size=nitems)
#     base = 0.1 + 0.08 * exponents + 0.2 * np.random.rand(nitems)

#     # Generate input
#     x = np.random.rand(nsamples)
#     # Prepare a result dataset
#     res = pd.DataFrame(data=x, columns=['x'])

#     # scale = np.sort(scale)[::-1]
#     for i in range(nitems):
#         # Compute base cost
#         cost = x**exponents[i]
#         # Rebase
#         cost = cost - np.min(cost) + base[i]
#         # sx = direction[i]*speed[i]*(x+offset[i])
#         # cost = base[i] + scale[i] / (1 + np.exp(sx))
#         res[f'C{i}'] = cost
#     # Add noise
#     if noise > 0:
#         for i in range(nitems):
#             pnoise = noise * np.random.randn(nsamples)
#             res[f'C{i}'] = np.maximum(0, res[f'C{i}'] + pnoise)
#     # Reindex
#     res.set_index('x', inplace=True)
#     # Sort by index
#     res.sort_index(inplace=True)
#     # Return results
#     return res


def generate_market_dataset(nsamples, nitems, noise=0, seed=None):
    assert(0 <= noise <= 1)
    # Seed the RNG
    np.random.seed(seed)
    # Generate costs
    speed = np.random.choice([10, 13, 15], size=nitems)
    base = 0.4 * np.random.rand(nitems)
    scale = 0.4 + 1 * np.random.rand(nitems)
    offset = -0.7 * np.random.rand(nitems)

    # Generate input
    x = np.random.rand(nsamples)
    # Prepare a result dataset
    res = pd.DataFrame(data=x, columns=['x'])

    # scale = np.sort(scale)[::-1]
    for i in range(nitems):
        # Compute base cost
        cost = scale[i] / (1 + np.exp(-speed[i] * (x + offset[i])))
        # Rebase
        cost = cost - np.min(cost) + base[i]
        # sx = direction[i]*speed[i]*(x+offset[i])
        # cost = base[i] + scale[i] / (1 + np.exp(sx))
        res[f'C{i}'] = cost
    # Add noise
    if noise > 0:
        for i in range(nitems):
            pnoise = noise * np.random.randn(nsamples)
            res[f'C{i}'] = np.maximum(0, res[f'C{i}'] + pnoise)
    # Reindex
    res.set_index('x', inplace=True)
    # Sort by index
    res.sort_index(inplace=True)
    # Return results
    return res



def train_test_split(data, test_size, seed=None):
    assert(0 < test_size < 1)
    # Seed the RNG
    np.random.seed(seed)
    # Shuffle the indices
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    # Partition the indices
    sep = int(len(data) * (1-test_size))
    res_tr = data.iloc[idx[:sep]]
    res_ts = data.iloc[idx[sep:]]
    return res_tr, res_ts


def generate_market_problem(nitems, rel_req, seed=None):
    # Seed the RNG
    np.random.seed(seed)
    # Generate the item values
    # values = np.ones(nitems)
    values = 1 + 0.2*np.random.rand(nitems)
    # Generate the requirement
    req = rel_req * np.sum(values)
    # Return the results
    return MarketProblem(values, req)


class MarketProblem(object):
    """Docstring for MarketProblem. """

    def __init__(self, values, requirement):
        """TODO: to be defined. """
        # Store the problem configuration
        self.values = values
        self.requirement = requirement

    def solve(self, costs, tlim=None, print_solution=False):
        # Quick access to some useful fields
        values = self.values
        req = self.requirement
        nv = len(values)
        # Build the solver
        slv = pywraplp.Solver.CreateSolver('CBC')
        # Build the variables
        x = [slv.IntVar(0, 1, f'x_{i}') for i in range(nv)]
        # Build the requirement constraint
        rcst = slv.Add(sum(values[i] * x[i] for i in range(nv)) >= req)
        # Build the objective
        slv.Minimize(sum(costs[i] * x[i] for i in range(nv)))

        # Set a time limit, if requested
        if tlim is not None:
            slv.SetTimeLimit(1000 * tlim)
        # Solve the problem
        status = slv.Solve()
        # Prepare the results
        if status in (slv.OPTIMAL, slv.FEASIBLE):
            res = []
            # Extract the solution
            sol = [x[i].solution_value() for i in range(nv)]
            res.append(sol)
            # Determine whether the problem was closed
            if status == slv.OPTIMAL:
                res.append(True)
            else:
                res.append(False)
        else:
            # TODO I am not handling the unbounded case
            # It should never arise in the first place
            if status == slv.INFEASIBLE:
                res = [None, True]
            else:
                res = [None, False]
        # Print the solution, if requested
        if print_solution:
            print_sol(self, res[0], res[1], costs)
        return res

    def __repr__(self):
        return f'<MarketProblem: {self.values} {self.requirement}>'


def print_sol(prb, sol, closed, costs):
    # Obtain indexes of selected items
    idx = [i for i in range(len(sol)) if sol[i] > 0]
    # Obtain the corresponding values
    values = [prb.values[i] for i in idx]
    # Print selected items with values and costs
    s = ', '.join(f'{i}' for i in idx)
    print('Selected items:', s)
    s = f'Cost: {sum(costs):.2f}, '
    s += f'Value: {sum(values):.2f}, '
    s += f'Requirement: {prb.requirement:.2f}, '
    s += f'Closed: {closed}'
    print(s)


def compute_regret(problem, predictor, pred_in, true_costs, tlim=None):
    # Obtain all predictions
    costs = predictor.predict(pred_in)
    # Compute all solutions
    sols = []
    for c in costs:
        sol, _ = problem.solve(c, tlim=tlim)
        sols.append(sol)
    sols = np.array(sols)
    # Compute the true solutions
    tsols = []
    for c in true_costs:
        sol, _ = problem.solve(c, tlim=tlim)
        tsols.append(sol)
    tsols = np.array(tsols)
    # Compute true costs
    costs_with_predictions = np.sum(true_costs * sols, axis=1)
    costs_with_true_solutions = np.sum(true_costs * tsols, axis=1)
    # Compute regret
    regret = costs_with_predictions - costs_with_true_solutions
    # Return true costs
    return regret


def plot_histogram(data, label=None, bins=20, figsize=None,
        data2=None, label2=None, print_mean=False):
    # Build figure
    fig = plt.figure(figsize=figsize)
    # Setup x axis
    plt.xlabel(label)
    # Define bins
    rmin, rmax = data.min(), data.max()
    if data2 is not None:
        rmin = min(rmin, data2.min())
        rmax = max(rmax, data2.max())
    bins = np.linspace(rmin, rmax, bins)
    # Histogram
    hist, edges = np.histogram(data, bins=bins)
    hist = hist / np.sum(hist)
    plt.step(edges[:-1], hist, where='post', label=label)
    if data2 is not None:
        hist2, edges2 = np.histogram(data2, bins=bins)
        hist2 = hist2 / np.sum(hist2)
        plt.step(edges2[:-1], hist2, where='post', label=label2)
    # Make it compact
    plt.tight_layout()
    # Legend
    plt.legend()
    # Show
    plt.show()
    # Print mean, if requested
    if print_mean:
        s = f'Mean: {np.mean(data):.3f}'
        if label is not None:
            s += f' ({label})'
        if data2 is not None:
            s += f', {np.mean(data2):.3f}'
            if label2 is not None:
                s += f' ({label2})'
        print(s)


def print_ml_metrics(model, X, y, label=None):
    # Obtain the predictions
    pred = model.predict(X)
    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(y, pred))
    # Compute the MAE
    mae = metrics.mean_absolute_error(y, pred)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(y, pred)
    lbl = '' if label is None else f' ({label})'
    print(f'R2: {r2:.2f}, MAE: {mae:.2}, RMSE: {rmse:.2f}{lbl}')


class DFLModel(keras.Model):
    def __init__(self, prb, tlim=None, recompute_chance=1, **params):
        super(DFLModel, self).__init__(**params)
        # Store configuration parameters
        self.prb = prb
        self.tlim = tlim
        self.recompute_chance = recompute_chance
        # Build metrics
        self.metric_loss = keras.metrics.Mean(name="loss")
        # self.metric_regret = keras.metrics.Mean(name="regret")
        # self.metric_mae = keras.metrics.MeanAbsoluteError(name="mae")
        # Prepare a field for the solutions
        self.sol_store = None

    # def dfl_fit(self, X, y, **kwargs):
    #     # Precompute all solutions for the true costs
    #     self.sol_store = []
    #     for c in y:
    #         sol, closed = self.prb.solve(c, tlim=self.tlim)
    #         self.sol_store.append(sol)
    #     self.sol_store = np.array(self.sol_store)
    #     # Call the normal fit method
    #     return self.fit(X, y, **kwargs)


    def fit(self, X, y, **kwargs):
        # Precompute all solutions for the true costs
        self.sol_store = []
        for c in y:
            sol, closed = self.prb.solve(c, tlim=self.tlim)
            self.sol_store.append(sol)
        self.sol_store = np.array(self.sol_store)
        # Call the normal fit method
        return super(DFLModel, self).fit(X, y, **kwargs)

    def _find_best(self, costs):
        tc = np.dot(self.sol_store, costs)
        best_idx = np.argmin(tc)
        best = self.sol_store[best_idx]
        return best

    def train_step(self, data):
        # Unpack the data
        x, costs_true = data
        # Quick access to some useful fields
        prb = self.prb
        tlim = self.tlim

        # Loss computation
        with tf.GradientTape() as tape:
            # Obtain the predictions
            costs = self(x, training=True)
            # Solve all optimization problems
            sols, tsols = [], []
            for c, tc in zip(costs.numpy(), costs_true.numpy()):
                # Decide whether to recompute the solution
                if np.random.rand() < self.recompute_chance:
                    sol, closed = prb.solve(c, tlim=self.tlim)
                    # Store the solution, if needed
                    if self.recompute_chance < 1:
                        # Check if the solutions is already stored
                        if not (self.sol_store == sol).all(axis=1).any():
                            self.sol_store = np.vstack((self.sol_store, sol))
                else:
                    sol = self._find_best(c)
                # Find the best solution with the predicted costs
                sols.append(sol)
                # Find the best solution with the true costs
                tsol = self._find_best(tc)
                tsols.append(tsol)
            # Convert solutions to numpy arrays
            sols = np.array(sols)
            tsols = np.array(tsols)
            # Compute the cost difference
            cdiff = costs - costs_true
            # Compute the solution difference
            sdiff = tsols - sols
            # Compute the loss terms
            loss_terms = tf.reduce_sum(cdiff * sdiff, axis=1)
            # Compute the mean loss
            loss = tf.reduce_mean(loss_terms)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update main metrics
        self.metric_loss.update_state(loss)
        # regrets = tf.reduce_sum((sols - tsols) * costs_true, axis=1)
        # mean_regret = tf.reduce_mean(regrets)
        # self.metric_regret.update_state(mean_regret)
        # self.metric_mae.update_state(costs_true, costs)
        # Update compiled metrics
        self.compiled_metrics.update_state(costs_true, costs)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    # def test_step(self, data):
    #     # Unpack the data
    #     x, costs_true = data
    #     # Compute predictions
    #     costs = self(x, training=False)
    #     # Updates the metrics tracking the loss
    #     self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #     # Update the metrics.
    #     self.compiled_metrics.update_state(y, y_pred)
    #     # Return a dict mapping metric names to current value.
    #     # Note that it will include the loss (tracked in self.metrics).
    #     return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.metric_loss]
        # return [self.metric_loss, self.metric_regret]


def build_dfl_ml_model(input_size, output_size,
        problem, tlim=None, hidden=[], recompute_chance=1,
        output_activation='linear', name=None):
    # Build all layers
    nnin = keras.Input(input_size)
    nnout = nnin
    for h in hidden:
        nnout = layers.Dense(h, activation='relu')(nnout)
    nnout = layers.Dense(output_size, activation=output_activation)(nnout)
    # Build the model
    model = DFLModel(problem, tlim=tlim, recompute_chance=recompute_chance,
            inputs=nnin, outputs=nnout, name=name)
    return model


def train_dfl_model(model, X, y, tlim=None,
        epochs=20, verbose=0, patience=10, batch_size=32,
        validation_split=0.2):
    # Compile and train
    model.compile(optimizer='Adam', run_eagerly=True)
    if validation_split > 0:
        cb = [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    else:
        cb = None
    # history = model.dfl_fit(X, y, validation_split=validation_split,
    #                  callbacks=cb, batch_size=batch_size,
    #                  epochs=epochs, verbose=verbose)
    history = model.fit(X, y, validation_split=validation_split,
                     callbacks=cb, batch_size=batch_size,
                     epochs=epochs, verbose=verbose)
    return history




def load_cmapss_data(data_folder):
    # Read the CSV files
    fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
    cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
    datalist = []
    nmcn = 0
    for fstem in fnames:
        # Read data
        data = pd.read_csv(f'{data_folder}/{fstem}.txt', sep=' ', header=None)
        # Drop the last two columns (parsing errors)
        data.drop(columns=[26, 27], inplace=True)
        # Replace column names
        data.columns = cols
        # Add the data source
        data['src'] = fstem
        # Shift the machine numbers
        data['machine'] += nmcn
        nmcn += len(data['machine'].unique())
        # Generate RUL data
        cnts = data.groupby('machine')[['cycle']].count()
        cnts.columns = ['ftime']
        data = data.join(cnts, on='machine')
        data['rul'] = data['ftime'] - data['cycle']
        data.drop(columns=['ftime'], inplace=True)
        # Store in the list
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Put the 'src' field at the beginning and keep 'rul' at the end
    data = data[['src'] + cols + ['rul']]
    # data.columns = cols
    return data


def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res


def plot_df_heatmap(data, labels=None, vmin=-1.96, vmax=1.96,
        figsize=None, s=4, normalize='standardize'):
    # Normalize the data
    if normalize == 'standardize':
        data = data.copy()
        data = (data - data.mean()) / data.std()
    else:
        raise ValueError('Unknown normalization method')
    # Build a figure
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(np.mod(labels, 10)))
    plt.tight_layout()
    plt.show()


def partition_by_machine(data, tr_machines):
    # Separate
    tr_machines = set(tr_machines)
    tr_list, ts_list = [], []
    for mcn, gdata in data.groupby('machine'):
        if mcn in tr_machines:
            tr_list.append(gdata)
        else:
            ts_list.append(gdata)
    # Collate again
    tr_data = pd.concat(tr_list)
    ts_data = pd.concat(ts_list)
    return tr_data, ts_data


def plot_pred_scatter(y_pred, y_true, figsize=None, autoclose=True):
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, y_true, marker='.', alpha=max(0.01, 1 / len(y_pred)))
    xl, xu = plt.xlim()
    yl, yu = plt.ylim()
    l, u = min(xl, yl), max(xu, yu)
    plt.plot([l, u], [l, u], ':', c='0.3')
    plt.xlim(l, u)
    plt.ylim(l, u)
    plt.xlabel('prediction')
    plt.ylabel('target')
    plt.tight_layout()
    plt.show()


def rescale_CMAPSS(tr, ts):
    # Define input columns
    dt_in = tr.columns[3:-1]
    # Compute mean and standard deviation
    trmean = tr[dt_in].mean()
    trstd = tr[dt_in].std().replace(to_replace=0, value=1) # handle static fields
    # Rescale all inputs
    ts_s = ts.copy()
    ts_s[dt_in] = (ts_s[dt_in] - trmean) / trstd
    tr_s = tr.copy()
    tr_s[dt_in] = (tr_s[dt_in] - trmean) / trstd
    # Compute the maximum RUL
    trmaxrul = tr['rul'].max()
    # Normalize the RUL
    ts_s['rul'] = ts['rul'] / trmaxrul
    tr_s['rul'] = tr['rul'] / trmaxrul
    # Return results
    params = {'trmean': trmean, 'trstd': trstd, 'trmaxrul': trmaxrul}
    return tr_s, ts_s, params


def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=None):
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.xlabel('time')
    plt.legend()
    plt.tight_layout()
    plt.show()


class RULCostModel:
    def __init__(self, maintenance_cost, safe_interval=0):
        self.maintenance_cost = maintenance_cost
        self.safe_interval = safe_interval

    def cost(self, machine, pred, thr, return_margin=False):
        # Merge machine and prediction data
        tmp = np.array([machine, pred]).T
        tmp = pd.DataFrame(data=tmp,
                           columns=['machine', 'pred'])
        # Cost computation
        cost = 0
        nfails = 0
        slack = 0
        for mcn, gtmp in tmp.groupby('machine'):
            idx = np.nonzero(gtmp['pred'].values < thr)[0]
            if len(idx) == 0:
                cost += self.maintenance_cost
                nfails += 1
            else:
                cost -= max(0, idx[0] - self.safe_interval)
                slack += len(gtmp) - idx[0]
        if not return_margin:
            return cost
        else:
            return cost, nfails, slack


def optimize_threshold(machine, pred, th_range, cmodel,
        plot=False, figsize=None):
    # Compute the optimal threshold
    costs = [cmodel.cost(machine, pred, thr) for thr in th_range]
    opt_th = th_range[np.argmin(costs)]
    # Plot
    if plot:
        plt.figure(figsize=figsize)
        plt.plot(th_range, costs)
        plt.xlabel('threshold')
        plt.ylabel('cost')
        plt.tight_layout()
    # Return the threshold
    return opt_th


def plot_series(series=None, samples=None, std=None, target=None,
        figsize=None, s=4, alpha=0.95, xlabel=None, ylabel=None,
        samples_lbl='samples', target_lbl='target',
        samples2=None, samples2_lbl=None):
    plt.figure(figsize=figsize)
    if series is not None:
        plt.plot(series.index, series, label=ylabel)
    if target is not None:
        plt.plot(target.index, target, ':', label=target_lbl, color='0.7')
    if std is not None:
        plt.fill_between(series.index,
                series.values - std, series.values + std,
                alpha=0.3, color='tab:blue', label='+/- std')
    if samples is not None:
        plt.scatter(samples.index, samples, label=samples_lbl,
                color='tab:orange')
    if samples2 is not None:
        plt.scatter(samples2.index, samples2, label=samples2_lbl,
                color='tab:red')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if (series is not None) + (samples is not None) + \
            (target is not None) > 1:
        plt.legend()
    else:
        plt.ylabel(ylabel)
    plt.tight_layout()


def max_acq(mu, std, best_y, n_samples, ftype, alpha=0):
    # Obtain the acquisition function values
    if ftype == 'pi':
        acq = stats.norm.cdf(best_y, loc=mu, scale=std)
        acq = np.array(acq)
    elif ftype == 'lcb':
        lcb, ucb = stats.norm.interval(alpha, loc=mu, scale=std)
        acq = -lcb
    elif ftype == 'ei':
        acq = (best_y - mu) * stats.norm.cdf(best_y, loc=mu, scale=std)
        acq += std * stats.norm.pdf(best_y, loc=mu, scale=std)
        acq = np.array(acq)
    else:
        raise ValueError('Unknown acquisition function type')
    # Return the best point
    bidx = int(np.argmax(acq))
    best_x = mu.index[bidx]
    best_acq = acq[bidx]
    return best_x, best_acq


def univariate_gp_tt(kernel, x_tr, y_tr, x_eval,
        suppress_warning=False):
    assert(len(x_eval.shape) == 1 or x_eval.shape[1] == 1)
    assert(len(x_tr.shape) == 1 or x_tr.shape[1] == 1)
    # Reshape x_tr and x_eval
    x_tr = x_tr.reshape(-1, 1)
    x_eval = x_eval.ravel()
    # Build a GP model
    mdl = GaussianProcessRegressor(kernel=kernel,
                                   n_restarts_optimizer=10,
                                   normalize_y=True)
    # Traing the GP model
    with warnings.catch_warnings():
        if suppress_warning:
            warnings.simplefilter('ignore')
        mdl.fit(x_tr, y_tr)
    # Predict
    mu, std = mdl.predict(x_eval.reshape(-1, 1), return_std=True)
    # Return results
    mu = pd.Series(index=x_eval, data=mu.ravel())
    std = pd.Series(index=x_eval, data=std.ravel())
    return mu, std


def simple_univariate_BO(f, l, u, max_it=10, init_points=3, alpha=0.95,
        seed=None, n_samples_gs=10000, ftype='lcb', return_state=False,
        tol=1e-3, suppress_warnings=False):
    # Define the kernel
    kernel = RBF(1, (1e-6, 1e0)) + WhiteKernel(1, (1e-6, 1e0))
    # Reseed the RNG
    np.random.seed(seed)
    # Sample initial points
    X = np.random.uniform(l, u, size=(init_points, 1))
    y = f(X)
    t = [0] * init_points
    # Determine the current best point
    best_idx = int(np.argmin(y))
    best_x = X[best_idx][0]
    best_y = y[best_idx]
    # Init the GP model
    x = np.linspace(l, u, n_samples_gs)
    mu, std = univariate_gp_tt(kernel, X, y, x, suppress_warnings)
    # Main Loop
    for nit in range(max_it):
        # Minimize the acquisition function
        next_x, acq = max_acq(mu, std, best_y, n_samples_gs, ftype, alpha)
        # Check termination criteria
        if acq < tol: break
        # Update the best solution
        next_y = f(next_x)
        if next_y < best_y:
            best_x = next_x
            best_y = next_y
        # Udpate the pool and the GP model
        if nit < max_it - 1:
            X = np.vstack((X, next_x))
            y = np.vstack((y, f(next_x)))
            t.append(nit+1)
            mu, std = univariate_gp_tt(kernel, X, y, x, suppress_warnings)
    # Return results
    res = best_x
    if return_state:
        samples = pd.Series(index=X.ravel(), data=y.ravel())
        sopt = pd.Series(index=[best_x], data=[best_y])
        res = res, {'samples': samples, 'mu': mu, 'std': std, 't': t,
                'sopt': sopt}
    return res


def opt_classifier_policy(mdl, X, y, machines, cmodel,
        n_iter=10, init_points=5, epochs_per_it=3,
        validation_split=0.2, seed=None, verbose=0):
    # Store weights
    init_wgt = mdl.get_weights()
    # Define a data structre to store the weights for each solution
    stored_weights = {}

    # Define the cost function
    def f(x):
        # Unpack the input
        thr, c0_weight = x
        # Define new labels
        tr_lbl = (y >= thr)
        # Reset weights
        mdl.set_weights(init_wgt)
        # Define sample weigths
        sample_weight = np.where(tr_lbl, 1, c0_weight)
        # Fit
        start = time.time()
        train_ml_model(mdl, X, tr_lbl, epochs=epochs_per_it,
                validation_split=validation_split,
                sample_weight=sample_weight, loss='binary_crossentropy')
        dur = time.time() - start
        # Cost computation
        tr_pred = np.round(mdl.predict(X).ravel())
        tr_cost = cmodel.cost(machines, tr_pred, 0.5)
        # Store the model weights
        stored_weights[thr, c0_weight] = mdl.get_weights()
        # Print info
        if verbose > 0:
            s = f'thr: {thr:.3f}, w0: {c0_weight:.3f}'
            s += f', cost: {tr_cost}, time: {dur:.2f}'
            print(s)
        # Return the cost
        return tr_cost

    # Define the bouding box
    box = {'thr': (0.0, 0.05), 'c0_weight': (1., 5.)}

    # Start optimization
    res = skopt.gp_minimize(f,
            [box[k] for k in ('thr', 'c0_weight')],
            acq_func='gp_hedge',
            n_calls=n_iter,
            n_random_starts=init_points,
            noise=None,
            random_state=seed,
            verbose=False)

    # Set the best weights
    best_thr, best_c0_weight = res.x
    mdl.set_weights(stored_weights[best_thr, best_c0_weight])
    return res


def plot_ml_model(model):
    return keras.utils.plot_model(model, show_shapes=True,
            show_layer_names=True, rankdir='LR')
