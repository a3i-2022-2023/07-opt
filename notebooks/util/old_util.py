#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras import callbacks
import tensorflow as tf
from scipy.integrate import odeint

def click_through_rate(avg_ratings, num_reviews, dollar_ratings):
    dollar_rating_baseline = {"D": 3, "DD": 2, "DDD": 4, "DDDD": 4.5}
    return 1 / (1 + np.exp(
        np.array([dollar_rating_baseline[d] for d in dollar_ratings]) -
        avg_ratings * np.log1p(num_reviews) / 4))


def load_restaurant_data():
    def sample_restaurants(n):
        avg_ratings = np.random.uniform(1.0, 5.0, n)
        num_reviews = np.round(np.exp(np.random.uniform(0.0, np.log(200), n)))
        dollar_ratings = np.random.choice(["D", "DD", "DDD", "DDDD"], n)
        ctr_labels = click_through_rate(avg_ratings, num_reviews, dollar_ratings)
        return avg_ratings, num_reviews, dollar_ratings, ctr_labels


    def sample_dataset(n, testing_set):
        (avg_ratings, num_reviews, dollar_ratings, ctr_labels) = sample_restaurants(n)
        if testing_set:
            # Testing has a more uniform distribution over all restaurants.
            num_views = np.random.poisson(lam=3, size=n)
        else:
            # Training/validation datasets have more views on popular restaurants.
            num_views = np.random.poisson(lam=ctr_labels * num_reviews / 40.0, size=n)

        return pd.DataFrame({
                "avg_rating": np.repeat(avg_ratings, num_views),
                "num_reviews": np.repeat(num_reviews, num_views),
                "dollar_rating": np.repeat(dollar_ratings, num_views),
                "clicked": np.random.binomial(n=1, p=np.repeat(ctr_labels, num_views))
            })

    # Generate
    np.random.seed(42)
    data_train = sample_dataset(2000, testing_set=False)
    data_val = sample_dataset(1000, testing_set=False)
    data_test = sample_dataset(1000, testing_set=True)
    return data_train, data_val, data_test


def plot_ctr_truth(figsize=None):
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3
    avgr = np.repeat(np.linspace(0, 5, res), res)
    nrev = np.tile(np.linspace(0, 200, res), res)
    avgr_ticks = np.linspace(0, 5, nticks)
    nrev_ticks = np.linspace(0, 200, nticks)
    rticks = np.linspace(0, res, nticks)
    for i, drating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        drt = [drating] * (res*res)
        ctr = click_through_rate(avgr, nrev, drt)
        plt.subplot(1, 4, i+1)
        plt.pcolor(ctr.reshape((res, res)), vmin=0, vmax=1)
        plt.xlabel('average rating')
        if i == 0:
            plt.ylabel('num. reviews')
        plt.title(drating)
        plt.xticks(rticks, avgr_ticks, fontsize=7)
        if i == 0:
            plt.yticks(rticks, nrev_ticks, fontsize=7)
        else:
            plt.yticks([], [])
    plt.tight_layout()


def plot_ctr_distribution(data, figsize=None):
    plt.figure(figsize=figsize)
    nbins = 15
    plt.subplot(131)
    plt.hist(data['avg_rating'], density=True, bins=nbins)
    plt.xlabel('average rating')
    plt.subplot(132)
    plt.hist(data['num_reviews'], density=True, bins=nbins)
    plt.xlabel('num. reviews')
    plt.subplot(133)
    vcnt = data['dollar_rating'].value_counts()
    vcnt /= vcnt.sum()
    plt.bar([0.5, 1.5, 2.5, 3.5],
            [vcnt['D'], vcnt['DD'], vcnt['DDD'], vcnt['DDDD']])
    plt.xlabel('dollar rating')
    plt.tight_layout()


def build_nn_model(input_shape, output_shape, hidden,
        output_activation='linear', kernel_regularizers=[], scale=None):
    model_in = keras.Input(shape=input_shape, dtype='float32')
    x = model_in
    for i, h in enumerate(hidden):
        kr = kernel_reguralizers[i] if i < len(kernel_regularizers)-1 else None
        x = layers.Dense(h, activation='relu', kernel_regularizer=kr)(x)
    kr_out = kernel_regularizers[-1] if len(kernel_regularizers) > len(hidden) else None
    model_out = layers.Dense(output_shape, activation=output_activation,
            kernel_regularizer=kr_out)(x)
    if scale is not None:
        model_out *= scale
    model = keras.Model(model_in, model_out)
    return model


def train_nn_model(model, X, y, loss,
        verbose=0, patience=10,
        validation_split=0.0, **fit_params):
    # Compile the model
    model.compile(optimizer='Adam', loss=loss)
    # Build the early stop callback
    cb = []
    if validation_split > 0:
        cb += [callbacks.EarlyStopping(patience=patience,
            restore_best_weights=True)]
    # Train the model
    history = model.fit(X, y, callbacks=cb,
            validation_split=validation_split,
            verbose=verbose, **fit_params)
    return history


def plot_nn_model(model, show_layer_names=True, show_layer_activations=True, show_shapes=True):
    return keras.utils.plot_model(model, show_shapes=show_shapes,
            show_layer_names=show_layer_names, rankdir='LR',
            show_layer_activations=show_layer_activations)


def plot_training_history(history=None,
        figsize=None, print_final_scores=True):
    plt.figure(figsize=figsize)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    # if 'val_loss' in history.history.keys():
    #     plt.plot(history.history['val_loss'], label='val. loss')
    if len(history.history.keys()) > 0:
        plt.legend()
    plt.xlabel('epochs')
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()
    if print_final_scores:
        trl = history.history["loss"][-1]
        s = f'Final loss: {trl:.4f} (training)'
        if 'val_loss' in history.history:
            vll = history.history["val_loss"][-1]
            s += f', {vll:.4f} (validation)'
        print(s)


def plot_ctr_estimation(estimator, scale,
        split_input=False, one_hot_categorical=True,
        figsize=None):
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3
    avgr = np.repeat(np.linspace(0, 5, res), res).reshape(-1, 1)
    avgr = avgr / scale['avg_rating']
    nrev = np.tile(np.linspace(0, 200, res), res).reshape(-1, 1)
    nrev = nrev / scale['num_reviews']
    avgr_ticks = np.linspace(0, 5, nticks)
    nrev_ticks = np.linspace(0, 200, nticks)
    rticks = np.linspace(0, res, nticks)
    for i, drating in enumerate(['D', 'DD', 'DDD', 'DDDD']):
        if one_hot_categorical:
            # Categorical encoding for the dollar rating
            dr_cat = np.zeros((1, 4))
            dr_cat[0, i] = 1
            dr_cat = np.repeat((dr_cat), res*res, axis=0)
            # Concatenate all inputs
            x = np.hstack((avgr, nrev, dr_cat))
        else:
            # Integer encoding for the categorical attribute
            dr_cat = np.full((res*res, 1), i)
            x = np.hstack((avgr, nrev, dr_cat))
        # Split input, if requested
        if split_input:
            x = [x[:, i].reshape(-1, 1) for i in range(x.shape[1])]
        # Obtain the predictions
        ctr = estimator.predict(x, verbose=0)
        plt.subplot(1, 4, i+1)
        plt.pcolor(ctr.reshape((res, res)), vmin=0, vmax=1)
        plt.xlabel('average rating')
        if i == 0:
            plt.ylabel('num. reviews')
        plt.title(drating)
        plt.xticks(rticks, avgr_ticks, fontsize=7)
        if i == 0:
            plt.yticks(rticks, nrev_ticks, fontsize=7)
        else:
            plt.yticks([], [])
    plt.tight_layout()


def plot_ctr_calibration(calibrators, scale, figsize=None):
    plt.figure(figsize=figsize)
    res = 100
    nticks = 3

    # Average rating calibration
    avgr = np.linspace(0, 5, res).reshape(-1, 1)
    avgr = avgr / scale['avg_rating']
    avgr_cal = calibrators[0].predict(avgr, verbose=0)
    plt.subplot(131)
    plt.plot(avgr, avgr_cal)
    plt.xlabel('avg_rating')
    plt.ylabel('cal. output')
    plt.grid(linestyle=':')
    # Num. review calibration
    nrev = np.linspace(0, 200, res).reshape(-1, 1)
    nrev = nrev / scale['num_reviews']
    nrev_cal = calibrators[1].predict(nrev, verbose=0)
    plt.subplot(132)
    plt.plot(nrev, nrev_cal)
    plt.xlabel('num_reviews')
    plt.grid(linestyle=':')
    # Dollar rating calibration
    drating = np.arange(0, 4).reshape(-1, 1)
    drating_cal = calibrators[2].predict(drating, verbose=0).ravel()
    plt.subplot(133)
    xticks = np.linspace(0.5, 3.5, 4)
    plt.bar(xticks, drating_cal)
    plt.xticks(xticks, ['D', 'DD', 'DDD', 'DDDD'])
    plt.grid(linestyle=':')

    plt.tight_layout()


def load_communities_data(data_folder, nan_discard_thr=0.05):
    # Read the raw data
    fname = os.path.join(data_folder, 'CommViolPredUnnormalizedData.csv')
    data = pd.read_csv(fname, sep=';', na_values='?')
    # Discard columns
    dcols = list(data.columns[-18:-2]) # directly crime related
    dcols = dcols + list(data.columns[7:12]) # race related
    dcols = dcols + ['nonViolPerPop']
    data = data.drop(columns=dcols)
    # Use relative values
    for aname in data.columns:
        if aname.startswith('pct'):
            data[aname] = data[aname] / 100
        elif aname in ('numForeignBorn', 'persEmergShelt',
                       'persHomeless', 'officDrugUnits',
                       'policCarsAvail', 'policOperBudget', 'houseVacant'):
            data[aname] = data[aname] / (data['pop'] / 100e3)
    # Remove redundant column (a relative columns is already there)
    data = data.drop(columns=['persUrban', 'numPolice',
                              'policeField', 'policeCalls', 'gangUnit'])
    # Discard columns with too many NaN values
    thr = nan_discard_thr * len(data)
    cols = data.columns[data.isnull().sum(axis=0) >= thr]
    cols = [c for c in cols if c != 'violentPerPop']
    data = data.drop(columns=cols)
    # Remove all NaN values
    data = data.dropna()
    # Shuffle
    rng = np.random.default_rng(42)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    return data.iloc[idx]


def plot_lr_weights(weights, attributes, cap_num=None,
        figsize=None):
    plt.figure(figsize=figsize)
    # Sort attributes by decreasing absolute weights
    idx = np.argsort(np.abs(weights))[::-1]
    if cap_num is not None:
        idx = idx[:cap_num]
    fontsize = min(8, 300 / len(idx))
    x = np.linspace(0.5, 0.5+len(idx), len(idx))
    plt.bar(x, weights[idx])
    plt.xticks(x, labels=attributes[idx], rotation=45, fontsize=fontsize)
    plt.grid(linestyle=':')
    plt.tight_layout()


def plot_pred_by_protected(data, pred, protected, figsize=None):
    plt.figure(figsize=figsize)
    # Prepare the data for the boxplot
    x, lbls = [], []
    # Append the baseline
    pred = pred.ravel()
    x.append(pred)
    lbls.append('all')
    # Append the sub-datasets
    for aname, dom in protected.items():
        for val in dom:
            mask = (data[aname] == val)
            x.append(pred[mask])
            lbls.append(f'{aname}={val}')
    plt.boxplot(x, labels=lbls)
    plt.grid(linestyle=':')
    plt.tight_layout()


def DIDI_r(data, pred, protected):
    res = 0
    avg = np.mean(pred)
    for aname, dom in protected.items():
        for val in dom:
            mask = (data[aname] == val)
            res += abs(avg - np.mean(pred[mask]))
    return res


class CstDIDIModel(keras.Model):
    def __init__(self, base_pred, attributes, protected, alpha, thr):
        super(CstDIDIModel, self).__init__()
        # Store the base predictor
        self.base_pred = base_pred
        # Weight and threshold
        self.alpha = alpha
        self.thr = thr
        # Translate attribute names to indices
        self.protected = {list(attributes).index(k): dom for k, dom in protected.items()}
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def call(self, data):
        return self.base_pred(data)

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.base_pred(x, training=True)
            mse = self.compiled_loss(y_true, y_pred)
            # Compute the constraint regularization term
            ymean = tf.math.reduce_mean(y_pred)
            didi = 0
            for aidx, dom in self.protected.items():
                for val in dom:
                    mask = (x[:, aidx] == val)
                    didi += tf.math.abs(ymean - tf.math.reduce_mean(y_pred[mask]))
            cst = tf.math.maximum(0.0, didi - self.thr)
            loss = mse + self.alpha * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]


class LagDualDIDIModel(keras.Model):
    def __init__(self, base_pred, attributes, protected, thr):
        super(LagDualDIDIModel, self).__init__()
        # Store the base predictor
        self.base_pred = base_pred
        # Weight and threshold
        self.alpha = tf.Variable(0., name='alpha')
        self.thr = thr
        # Translate attribute names to indices
        self.protected = {list(attributes).index(k): dom for k, dom in protected.items()}
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def call(self, data):
        return self.base_pred(data)

    def __custom_loss(self, x, y_true, sign=1):
        y_pred = self.base_pred(x, training=True)
        # loss, mse, cst = self.__custom_loss(x, y_true, y_pred)
        mse = self.compiled_loss(y_true, y_pred)
        # Compute the constraint regularization term
        ymean = tf.math.reduce_mean(y_pred)
        didi = 0
        for aidx, dom in self.protected.items():
            for val in dom:
                mask = (x[:, aidx] == val)
                didi += tf.math.abs(ymean - tf.math.reduce_mean(y_pred[mask]))
        cst = tf.math.maximum(0.0, didi - self.thr)
        loss = mse + self.alpha * cst
        return sign*loss, mse, cst

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(x, y_true, sign=1)

        # Separate training variables
        tr_vars = self.trainable_variables
        wgt_vars = tr_vars[:-1]
        mul_vars = tr_vars[-1:]

        # Update the network weights
        grads = tape.gradient(loss, wgt_vars)
        self.optimizer.apply_gradients(zip(grads, wgt_vars))

        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(x, y_true, sign=-1)

        grads = tape.gradient(loss, mul_vars)
        self.optimizer.apply_gradients(zip(grads, mul_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]


def euler_method(f, y0, t, return_gradients=False):
    # Prepare a data structure for the results
    y = np.zeros((len(t), len(y0)))
    # Initial state
    y[0, :] = y0
    if return_gradients:
        dy = np.zeros((len(t), len(y0)))
    # Solve the ODE using Euler method
    for i in range(1, len(t)):
        # Current step and gradient
        step = t[i] - t[i-1]
        dy_l = f(y[i-1, :], t[i-1])
        # If requested, store the gradient
        if return_gradients:
            dy[i-1, :] = dy_l
        # Compute the next state
        y[i, :] = y[i-1, :] + step * dy_l
    # Return the results
    if return_gradients:
        return y, dy
    else:
        return y


def plot_euler_method(y, t, dy=None, xlabel=None, ylabel=None,
        figsize=None, horizon=2):
    plt.figure(figsize=figsize)
    plt.plot(t, y, marker='o', linestyle='')
    # Plot gradients, if available
    if dy is not None:
        for i in range(len(y)-horizon):
            ti, tf = t[i], t[i+horizon]
            plt.plot([ti, tf], [y[i], y[i] + (tf-ti) * dy[i]],
                    linestyle=':', color='0.2', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()


def simulate_RC(V0, tau, Vs, tmax, steps_per_unit=1):
    # Define the initial state, gradient function, and time vector
    y0 = np.array([V0])
    nabla = lambda y, t: 1. / tau * (Vs - y)
    t = np.linspace(0, tmax, steps_per_unit * tmax + 1)
    # Solve
    Y = odeint(nabla, y0, t)
    # Wrap as dataframe
    data = pd.DataFrame(data=Y, index=t, columns=['V'])
    data.index.rename('time', inplace=True)
    # Return the results
    return data


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
    plt.grid(linestyle=':')
    # Make it compact
    plt.tight_layout()
    # Show
    plt.show()


class RCNablaLayer(keras.layers.Layer):
    def __init__(self, tau_ref=0.1, vs_ref=0.1,
            fixed_tau=None, fixed_vs=None):
        super(RCNablaLayer, self).__init__()
        # Store the reference values/scales
        self.tau_ref = tau_ref
        self.vs_ref = vs_ref
        # Prepare an initializer
        p_init = tf.random_normal_initializer()
        # Init the tau parameter
        if fixed_tau is None:
            self.logtau = tf.Variable(
                initial_value=p_init(shape=(1, ), dtype="float32"),
                trainable=True,
            )
        else:
            val = np.log(fixed_tau / self.tau_ref, dtype='float32')
            self.logtau = tf.Variable(initial_value=val, trainable=False)
        # Init the vs parameter
        if fixed_vs is None:
            self.logvs = tf.Variable(
                initial_value=p_init(shape=(1, ), dtype="float32"),
                trainable=True,
            )
        else:
            val = np.log(fixed_vs / self.vs_ref, dtype='float32')
            self.logvs = tf.Variable(initial_value=val, trainable=False)

    def get_tau(self):
        return tf.math.exp(self.logtau) * self.tau_ref

    def get_vs(self):
        return tf.math.exp(self.logvs) * self.vs_ref

    def call(self, inputs):
        # Unpack the inputs (state and time)
        y, t = inputs
        # Compute the gradient
        dy = 1. / self.get_tau() * (self.get_vs() - y)
        return dy


class ODEEulerModel(keras.Model):
    def __init__(self, f, auxiliary_input=False, **params):
        super(ODEEulerModel, self).__init__(**params)
        # Store configuration parameters
        self.f = f
        self.auxiliary_input = auxiliary_input

    def call(self, inputs, training=False):
        # Unpack the initial state & time
        if self.auxiliary_input:
            y, T, aux = inputs
        else:
            y, T = inputs
        # Solve the ODE via Euler's method
        res = [y]
        for i in range(T.shape[1]-1):
            # Obtain vector with consecutive time steps
            t, nt = T[:, i:i+1], T[:, i+1:i+2]
            # Compute the state gradient
            if self.auxiliary_input:
                dy = self.f([y, t, aux[:, i, :]], training=training)
            else:
                dy = self.f([y, t], training=training)
            # Update the state
            y = y + (nt - t) * dy
            # Store the result
            res.append(y)
        # Concatenate all results along a new axis
        res = tf.stack(res, axis=1)
        return res

    def train_step(self, data):
        # Unpack the data
        if self.auxiliary_input:
            (y0, T, aux), yt = data
        else:
            (y0, T), yt = data
        # Loss computation
        with tf.GradientTape() as tape:
            # Integrate the ODE
            if self.auxiliary_input:
                y = self.call([y0, T, aux], training=True)
            else:
                y = self.call([y0, T], training=True)
            # Compute the loss
            mask = ~tf.math.is_nan(yt)
            # residuals = y[mask] - yt[mask]
            # loss = tf.math.reduce_mean(tf.math.square(residuals))
            loss = self.compiled_loss(yt[mask], y[mask])
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # # Update main metrics
        # self.metric_loss.update_state(loss)
        # Update compiled metrics
        self.compiled_metrics.update_state(yt[mask], y[mask])
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        if self.auxiliary_input:
            (y0, T, aux), yt = data
        else:
            (y0, T), yt = data
        # Integrate the ODE and compute the mask
        if self.auxiliary_input:
            y = self.call([y0, T, aux], training=False)
        else:
            y = self.call([y0, T], training=False)
        # Updates the metrics tracking the loss
        mask = ~tf.math.is_nan(yt)
        self.compiled_loss(yt[mask], y[mask])
        # Update the metrics
        self.compiled_metrics.update_state(yt[mask], y[mask])
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


class NPI(object):
    def __init__(self, name, effect, cost):
        self.name = name
        self.effect = effect
        self.cost = cost


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


def compute_beta(beta_base, npis, npi_schedule):
    # Build a starting vector
    ns = len(npi_schedule)
    beta = np.full((ns, ), beta_base)
    # Loop over all the NPIs
    for npi in npis:
        # Build a vector with the effects
        effect = np.where(npi_schedule[npi.name], npi.effect, 1)
        # Apply the effect
        beta *= effect
    # Pack everything in a dataframe
    res = pd.DataFrame(beta, index=npi_schedule.index, columns=['beta'])
    return res


def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def sample_NPIs(npis, nweeks, max_active_npis=None, seed=None):
    # Build the set of all possible configurations
    options = [np.array([0, 1]) for _ in range(len(npis))]
    all_conf = cartesian_product(options)
    # Filter configurations
    if max_active_npis is not None:
        all_conf = all_conf[np.sum(all_conf, axis=1) <= max_active_npis]
    # Define indices of configurations that can be sampled
    idx = np.arange(len(all_conf))
    # Seed the RNG
    np.random.seed(seed)
    # Sample some configurations at random
    conf_idx = np.random.choice(idx, size=nweeks, replace=True)
    conf = all_conf[conf_idx]
    # Pack everything in a dataframe
    res = pd.DataFrame(conf, columns=[npi.name for npi in npis])
    for j, npi in enumerate(npis):
        res[npi.name] = conf[:, j]
    return res


def simulate_SIR(S0, I0, R0, beta, gamma, tmax, steps_per_day=1):
    # Build initial state
    Z = S0 + I0 + R0
    Z = Z if Z > 0 else 1 # Handle division by zero
    y0 = np.array([S0, I0, R0]) / Z
    # Wrapper
    nabla = lambda y, t: SIR(y, beta, gamma)
    # Solve
    t = np.linspace(0, tmax, steps_per_day * tmax + 1)
    Y = odeint(nabla, y0, t)
    # Wrap as dataframe
    data = pd.DataFrame(data=Y, index=t, columns=['S', 'I', 'R'])
    data.index.rename('time', inplace=True)
    # Return the results
    return data


def simulate_SIR_NPI(S0, I0, R0, beta, gamma, steps_per_day=1):
    # Prepare the result data structure
    S, I, R = [], [], []
    # Loop over all weeks
    res = []
    S, I, R = S0, I0, R0
    for w, b in enumerate(beta):
        # Simulate one week
        wres = simulate_SIR(S, I, R, b, gamma, 7,
                steps_per_day=steps_per_day)
        # Retrieve all states
        t = np.arange(0, 7)
        wres_days = wres.loc[t]
        wres_days['week'] = w
        # Store the results
        res.append(wres_days)
        # Update the current state
        S, I, R = wres[['S', 'I', 'R']].iloc[-1]
    # Wrap into a dataframe
    res = pd.concat(res, axis=0)
    return res


def gen_SIR_NPI_dataset(S0, I0, R0,
        beta_base, gamma, npis, nweeks, steps_per_day=1, max_active_npis=None, seed=None):
    # Sample NPIs
    npi_sched = sample_NPIs(npis, nweeks, max_active_npis=max_active_npis, seed=seed)
    # Compute the corresponding beta values
    beta = compute_beta(beta_base, npis, npi_sched)
    # Simulate with the given beta schedule
    beta_sched = [b for b in beta['beta'].values]
    sir_data = simulate_SIR_NPI(S0, I0, R0, beta_sched, gamma, steps_per_day)
    # Merge NPI data
    res = sir_data.join(npi_sched, on='week')
    res = res.join(beta, on='week')
    # Reindex
    idx = np.linspace(0, nweeks*7-1, nweeks*7)
    res.set_index(idx, inplace=True)
    return res


class NPISIRNablaLayer(keras.layers.Layer):
    def __init__(self, beta_pred, gamma_ref=0.1, fixed_gamma=None):
        super(NPISIRNablaLayer, self).__init__()
        # Store the model for predicting beta
        self.beta_pred = beta_pred
        # Store the reference values for gamma
        self.gamma_ref = gamma_ref
        # Prepare an initializer
        p_init = tf.random_normal_initializer()
        # Init the gamma parameter
        if fixed_gamma is None:
            self.loggamma = tf.Variable(
                initial_value=p_init(shape=(1, ), dtype="float32"),
                trainable=True,
            )
        else:
            val = np.log(fixed_gamma / self.gamma_ref, dtype='float32')
            self.loggamma = tf.Variable(initial_value=val, trainable=False)

    def get_gamma(self):
        return tf.math.exp(self.loggamma) * self.gamma_ref

    def call(self, inputs):
        # Unpack the inputs (state, time, and NPIs)
        y, t, npis = inputs
        # Slice the state
        S, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        # Compute beta
        beta = self.beta_pred(npis)
        # Compute the gradient
        N = tf.math.reduce_sum(y, axis=1, keepdims=True)
        s2i = beta * S * I / N
        i2r = self.get_gamma() * I
        dS = - s2i
        dI = s2i - i2r
        dR = i2r
        # Concatenate
        dy = tf.concat([dS, dI, dR], axis=1)
        return dy
