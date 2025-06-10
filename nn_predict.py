import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.asarray(x)
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    elif x.ndim == 2:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    else:
        raise ValueError("Input must be 1D or 2D array.")

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Batch Normalization ===
def batch_normalization(x, gamma, beta, moving_mean, moving_variance, epsilon=1e-3):
    # 推論時使用移動平均和移動方差
    return gamma * (x - moving_mean) / np.sqrt(moving_variance + epsilon) + beta

# Infer TensorFlow h5 model using numpy
# Support Dense, Flatten, relu, softmax, BatchNormalization, Dropout
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
        elif ltype == "BatchNormalization":
            # BatchNormalization 有 4 個參數：gamma, beta, moving_mean, moving_variance
            gamma = weights[wnames[0]]        # scale
            beta = weights[wnames[1]]         # offset
            moving_mean = weights[wnames[2]]  # moving_mean
            moving_variance = weights[wnames[3]]  # moving_variance
            x = batch_normalization(x, gamma, beta, moving_mean, moving_variance)
        elif ltype == "Dropout":
            # 推論時直接跳過 Dropout，不做任何處理
            pass

    return x

# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
