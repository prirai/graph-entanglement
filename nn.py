import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from  graph_tools import *
import networkx as nx

def create_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

def train_model(X, y, input_shape, epochs=10, batch_size=32):
	flattened_matrices = np.array([matrix.flatten() for matrix in X])
	model = create_model(input_shape)
	model.fit(flattened_matrices, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
	return model

def predict_value(model, matrices):
    flattened_matrices = np.array([matrix.flatten() for matrix in matrices])
    predictions = model.predict(flattened_matrices)
    return predictions

def vec2str(a): return ''.join(str(i) for i in a)
def mat2str(A): return vec2str(vec2str(a) for a in A)

import math    
if __name__ == "__main__":
    n1 = 5
    n2 = 6
    m = max(n1, n2)**2 # max(5,6)**2 = 36
    adj_mats_a, g6codes_a = get_adj_mat(n1) # n1*n1
    adj_mats_b, g6codes_b = get_adj_mat(n2) # n2*n2
    G_s_1 = [nx.from_dict_of_lists(adj_mat_to_adj_list(i)) for i in adj_mats_a]
    G_s_2 = [nx.from_dict_of_lists(adj_mat_to_adj_list(i)) for i in adj_mats_b]

    # adj_str_a = [np.array(list(mat2str(get_parity_matrix(n1, i).astype(np.int_)).zfill(m))).astype(int) for i in G_s_1]
    # adj_str_b = [np.array(list(mat2str(get_parity_matrix(n1, i).astype(np.int_)).zfill(m))).astype(int) for i in G_s_2]
    parity_mat_a = [get_parity_matrix(n1, i).astype(np.int_) for i in G_s_1]
    print(parity_mat_a[:5])
    parity_mat_b = [get_parity_matrix(n2, i).astype(np.int_) for i in G_s_2]

    for i, e in enumerate(adj_mats_b):
         print(f'1: ', end='')
         print(e, end='')
         print(parity_mat_b[i])
    X = np.array([np.array(list(mat2str(i).zfill(m))) for i in np.array(adj_mats_a)]).astype(np.float32)
    print(X.shape)
    print(X[:10])
    y = np.array([collect_combs(i) for i in parity_mat_a]).astype('float32')
    print("y:")
    print(y[:10])
    input_shape = X.shape[1]
    model = train_model(X, y, input_shape, epochs=10)
    Z = np.array(adj_mats_b).astype('float32')
    Z_labels = np.array([collect_combs(np.array(i)) for i in parity_mat_b]).astype('float32')
    print("Z's shape:", Z.shape)
    predictions = predict_value(model, Z)
    t_c = f_c = 0
    print("predictions:", predictions)
    print("truth:", Z_labels)
    for i in range(len(predictions)):
        if np.round(predictions[i]) == Z_labels[i]:
            t_c += 1
        else:
            f_c += 1
    results = model.evaluate(Z, Z_labels, batch_size=128)
    print("test loss, test acc:", results)
    print(f'Accuracy = {(t_c)/(t_c+f_c)}')

