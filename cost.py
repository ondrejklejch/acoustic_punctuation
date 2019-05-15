import numpy as np
from theano import tensor, shared, function
from theano.printing import Print


def create_priors(n):
    centers = np.array([
        [0.94819489, 0.42780784],
        [0.5486535, 1.],
        [1., 0.],
        [0.53411103, 0.25566054],
        [0., 0.42991347],
        [0.12370328, 0.75529003],
        [0.43735104, 0.3408001 ],
    ])

    #centers = np.array([
    #    [0.5, 1/6.0],
    #    [0.5, 3/6.0],
    #    [0.5, 5/6.0],
    #    [0.2, 2/6.0],
    #    [0.2, 4/6.0],
    #    [0.8, 2/6.0],
    #    [0.8, 4/6.0],
    #])

    A = np.arange(n*n).reshape((n,n))
    grid = np.stack([A / n, A % n], axis=2) / (n - 1.0)

    priors = np.zeros((7, n*n))
    for i in range(centers.shape[0]):
        distance = -10 * (np.sum(np.power(grid - centers[i], 2), axis=2))
        priors[i] = (np.exp(distance) / np.sum(np.exp(distance))).flatten()

    return priors


def reshape_activations(activations, size):
    new_activations = tensor.zeros_like(activations)

    activations = activations.reshape((-1, 2, (size ** 2) / 2))
    new_activations = tensor.set_subtensor(new_activations[:,0::2], activations[:,0])
    new_activations = tensor.set_subtensor(new_activations[:,1::2], activations[:,1])

    return new_activations


def compute_activations(representation, weights, size):
    representation = representation.reshape((-1, size ** 2))
    weights = tensor.sqrt(tensor.sum(tensor.power(weights, 2), axis=1))
    activations = tensor.nnet.softmax(representation * weights)

    return reshape_activations(activations, size)

def stimulation_cost(size, representation, weights, outputs, mask):
    priors = shared(create_priors(size))[outputs.flatten()]
    activations = compute_activations(representation, weights, size)
    mask = mask.flatten()

    return tensor.sum(mask * tensor.sum(priors * tensor.log(priors / activations), axis=1)) # / outputs.shape[0]



#size = 16
#representation = tensor.ftensor3()
#weights = tensor.fmatrix()
#outputs = tensor.imatrix()
#mask = tensor.imatrix()
#cost = stimulation_cost(size, representation, weights, outputs, mask)
#
#compute_cost = function([representation, weights, outputs, mask], cost, on_unused_input="ignore")
#
#
#representations = np.random.normal(size=(10,10,256)).astype(np.float32)
#weights = np.random.normal(size=(256, 256)).astype(np.float32)
#outputs = np.random.choice(np.arange(7), size=(10, 10)).astype(np.int32)
#mask = np.random.choice(np.arange(2), size=(10, 10)).astype(np.int32)
#
#print compute_cost(representations, weights, outputs, mask)



