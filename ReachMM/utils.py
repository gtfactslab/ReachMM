import time
import numpy as np

def run_time (func, *args, **kwargs) :
    before = time.time()
    ret = func(*args, **kwargs)
    after = time.time()
    return ret, (after - before)

def uniform_disjoint (set, N) :
    probs = [s[1] - s[0] for s in set]; probs = probs / np.sum(probs)
    return np.array([np.random.choice([
        np.random.uniform(s[0], s[1]) for s in set
    ], p=probs) for _ in range(N)])

def gen_ics(RANGES, N) :
    X = np.empty((N, len(RANGES)))
    for i, range in enumerate(RANGES) :
        X[:,i] = uniform_disjoint(range, N)
    return X

def file_to_numpy (filenames) :
    with open('data/' + filenames[0] + '.npy', 'rb') as f :
        nploaded = np.load(f)
        X = nploaded['X']
        U = nploaded['U']
    for FILE in filenames[1:] :
        with open('data/' + FILE + '.npy', 'rb') as f :
            nploaded = np.load(f)
            X = np.append(X, nploaded['X'], axis=0)
            U = np.append(U, nploaded['U'], axis=0)
    return X,U

def numpy_to_file (X, U, filename) :
    with open(filename, 'wb') as f :
        np.savez(f, X=X, U=U)