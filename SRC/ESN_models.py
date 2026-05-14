import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.sparse import csr_matrix
from ESN_utilities import analyze_adjacency_matrix
from tqdm import tqdm

# Spectral radius scaling
def scale_spectral_radius(W, target_radius):
    """Rescale matrix W so its spectral radius equals target_radius."""
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    
    if current_radius == 0:
        return W  # avoid division by zero
    
    return (W / current_radius) * target_radius


# ER reservoir generation
def reservoir_ER(n, p=0.05, spectral_radius=0.9, random_state=42, 
directed=True, w_range=(-1,1), keep_self_loops=True, complex_valued=False, distr="uniform"):
    """
    Generate an ER (Erdős–Rényi) reservoir.
    
    n               : number of neurons
    p               : probability of edge creation
    spectral_radius : target spectral radius
    """
    # Step 1: Generate random graph
    G = nx.erdos_renyi_graph(n, p, directed=directed, seed=random_state)

    # res_type = "directed" if directed else "undirected"
    
    # print(f"Generated {res_type} ER graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Step 2: Adjacency matrix as NumPy array
    A = nx.to_numpy_array(G)

    # Step 2.5: Optionally add random self-loops keeping the same
    # density of the rest of the matrix
    if keep_self_loops:
        rng = np.random.default_rng(random_state)
        for i in range(n):
            if rng.uniform(0,1) < p:
                A[i,i] = rng.uniform(w_range[0], w_range[1])
    else:
        np.fill_diagonal(A, 0.0)    
    
    # Step 3: Replace non-zero entries with random weights
    rng = np.random.default_rng(random_state)

    if complex_valued:
        if distr == "uniform":
            raw = rng.uniform(w_range[0], w_range[1], size=(n, n)) + 1j*rng.uniform(w_range[0], w_range[1], size=(n, n))
        elif distr == "normal":
            raw = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
    else:
        if distr == "uniform":
            raw = rng.uniform(w_range[0], w_range[1], size=(n, n))
        elif distr == "normal":
            raw = rng.standard_normal((n, n))

    if not directed:
        raw = np.triu(raw, 1)
        raw = raw + raw.T

    W = A * raw    
    
    # Step 4: Scale spectral radius
    W = scale_spectral_radius(W, spectral_radius)
    
    return W


# Barabasi reservoir generation
def reservoir_BA(n, m=3, spectral_radius=0.9, random_state=42, 
directed=True, w_range=(-1,1), keep_self_loops=True, complex_valued=False, distr="uniform"):
    """
    Generate a BA (Barabási–Albert) scale-free reservoir.
    
    n               : number of neurons
    m               : number of edges for each new node (controls hubs)
    spectral_radius : target spectral radius
    """
    # Step 1: Generate BA graph (undirected)
    G = nx.barabasi_albert_graph(n, m, seed=random_state)
    
    # Step 2: Convert to directed by making edges bidirectional
    if directed:
        G = G.to_directed()

    # Compute effective edge probability p for self-loops
    total_possible_edges = n * (n - 1) if directed else n * (n - 1) / 2
    p = G.number_of_edges() / total_possible_edges

    # res_type = "directed" if directed else "undirected"
    # print(f"Generated {res_type} BA graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Step 3: Adjacency matrix
    A = nx.to_numpy_array(G)

    # Step 3.5: Optionally add random self-loops keeping the same
    # density of the rest of the matrix
    if keep_self_loops:
        rng = np.random.default_rng(random_state)
        for i in range(n):
            if rng.uniform(0,1) < p:
                A[i,i] = rng.uniform(w_range[0], w_range[1])
    else:
        np.fill_diagonal(A, 0.0)                    
    
    # Step 4: Random weights for edges
    rng = np.random.default_rng(random_state)
    
    if complex_valued:
        if distr == "uniform":
            raw = rng.uniform(w_range[0], w_range[1], size=(n, n)) + 1j*rng.uniform(w_range[0], w_range[1], size=(n, n))
        elif distr == "normal":
            raw = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
    else:
        if distr == "uniform":
            raw = rng.uniform(w_range[0], w_range[1], size=(n, n))
        elif distr == "normal":
            raw = rng.standard_normal((n, n))

    if not directed:
        raw = np.triu(raw, 1)
        raw = raw + raw.T

    W = A * raw    

    # Step 5: Scale spectral radius
    W = scale_spectral_radius(W, spectral_radius)
    
    return W


# WS reservoir generation
def reservoir_WS(n, k=6, beta=0.3, spectral_radius=0.9, random_state=42, 
                 directed=True, w_range=(-1,1), keep_self_loops=True, complex_valued=False, distr="uniform"):
    """
    Generate a WS (Watts–Strogatz) small-world reservoir.
    
    n               : number of neurons
    k               : each node is connected to k nearest neighbors
    beta            : rewiring probability (0 = regular, 1 = random)
    spectral_radius : target spectral radius
    """
    # Step 1: Generate small-world graph
    G = nx.watts_strogatz_graph(n, k, beta, seed=random_state)
    
    # Step 2: Convert to directed graph (optional)
    if directed:
        G = G.to_directed()

    # Compute effective edge probability p for self-loops
    total_possible_edges = n * (n - 1) if directed else n * (n - 1) / 2
    p = G.number_of_edges() / total_possible_edges

    # res_type = "directed" if directed else "undirected"
    # print(f"Generated {res_type} WS graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Step 3: Adjacency matrix
    A = nx.to_numpy_array(G)

    # Step 3.5: Optionally add random self-loops keeping the same
    # density of the rest of the matrix
    if keep_self_loops:
        rng = np.random.default_rng(random_state)
        for i in range(n):
            if rng.uniform(0,1) < p:
                A[i,i] = rng.uniform(w_range[0], w_range[1])
    else:
        np.fill_diagonal(A, 0.0)                    
    
    # Step 4: Random weights for edges
    rng = np.random.default_rng(random_state)
    
    if complex_valued:
        if distr == "uniform":
            raw = rng.uniform(w_range[0], w_range[1], size=(n, n)) + 1j*rng.uniform(w_range[0], w_range[1], size=(n, n))
        elif distr == "normal":
            raw = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
    else:
        if distr == "uniform":
            raw = rng.uniform(w_range[0], w_range[1], size=(n, n))
        elif distr == "normal":
            raw = rng.standard_normal((n, n))

    if not directed:
        raw = np.triu(raw, 1)
        raw = raw + raw.T

    W = A * raw    

    # Step 5: Scale spectral radius
    W = scale_spectral_radius(W, spectral_radius)
    
    return W


# Custom reservoir generation
def reservoir_CU(n, density=0.1, spectral_radius=0.9, random_state=42, 
                 directed=True, w_range=(-1,1), keep_self_loops=True, complex_valued=False, distr="uniform"):    
    # Reservoir weights (initialized randomly as a sparse matrix)
    rng = np.random.default_rng(random_state)            
    total = n * n
    k = int(np.round(total * density))
    k = max(1, min(k, total))  # ensure at least 1 and at most total positions

    # sample unique flat indices for non-zero entries
    idx = rng.choice(total, size=k, replace=False)
    rows = idx // n
    cols = idx % n
    
    if complex_valued:
        if distr == "uniform":
            raw = rng.uniform(w_range[0], w_range[1], size=k) + 1j*rng.uniform(w_range[0], w_range[1], size=k)
        elif distr == "normal":
            raw = rng.standard_normal(k) + 1j*rng.standard_normal(k)
    else:
        if distr == "uniform":
            raw = rng.uniform(w_range[0], w_range[1], size=k)
        elif distr == "normal":
            raw = rng.standard_normal(k)
    
    raw_mat = csr_matrix((raw, (rows, cols)), shape=(n, n))
    W_res = raw_mat.toarray()

    if not keep_self_loops:
        if complex_valued:
            W_res[np.diag_indices_from(W_res)] = 0.0 + 0.0j
        else:
            np.fill_diagonal(W_res, 0.0)

    if not directed:
        # make symmetric
        upper = np.triu(W_res, 1)
        W_res = upper + upper.T

    # Step 4: Scale by spectral radius              
    W_res = scale_spectral_radius(W_res, spectral_radius)

    # print(f"Generated CU graph with density={density}")
    
    return W_res


def to_numpy_safe(x, TORCH_NUMPY_WORKS):

    """
    This function safely converts a tensor to a numpy array,
    handling cases where the tensor may not be directly
    convertible to numpy (which can happen in certain
    environments where PyTorch cannot use the numpy C-API).
    If the input does not have a "detach" method, it simply
    converts it to a numpy array using np.asarray.
    """

    # Version 3: with explicit check for torch numpy compatibility, which is more efficient when available
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu()
        if TORCH_NUMPY_WORKS:
            return x.numpy()  # fast path if available
        else:
            return np.asarray(x.tolist())  # fallback
    return np.asarray(x)


class CQ_ESN: 

    def __init__(self, input_size, seq_length,
                 reservoir_size, reservoir_type="ER",
                 directed=True, w_range=(-1, 1), keep_self_loops=True,
                 distr="uniform",
                 er_p=0.03, ba_m=2, ws_k=6, ws_beta=0.2,
                 density=0.1, random_state=42, spectral_radius=0.9,
                 leak_alpha=0.8,
                 ridge_alpha=1.0e-6,
                 kernel_ridge_alpha=1.0,
                 complex_valued=True,
                 center_states=False,
                 reduce_states_rank_svd=False,
                 reduce_states_rank_umap=False,
                 umap_dist_matrix='L2',
                 add_bias=False,
                 complex_bias=False,
                 normalize_states=False,
                 denormalize_preds=False,
                 quantum_kernel_pca=False,
                 scale_norm=False,
                 scores_dim=None,
                 qpca_dim=None,
                 umap_dim=None,
                 use_einsum=False,
                 discard_transients=0,
                 select_random=False,
                 select_n_samples=None,
                 silent=False, reservoir_analysis=False,
                 torch_numpy_works=False
                 ):

        # Initialize network parameters
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.reservoir_type = reservoir_type
        self.seq_length = seq_length
        self.leak_alpha = leak_alpha
        self.ridge_alpha = ridge_alpha
        self.kernel_ridge_alpha = kernel_ridge_alpha
        self.complex_valued = complex_valued
        self.center_states = center_states
        self.reduce_states_rank_svd = reduce_states_rank_svd
        self.reduce_states_rank_umap = reduce_states_rank_umap
        self.umap_dist_matrix = umap_dist_matrix
        self.add_bias = add_bias
        self.complex_bias = complex_bias
        self.normalize_states = normalize_states
        self.denormalize_preds = denormalize_preds
        self.quantum_kernel_pca = quantum_kernel_pca
        self.scale_norm = scale_norm
        self.scores_dim = scores_dim
        self.umap_dim = umap_dim
        self.qpca_dim = qpca_dim
        self.use_einsum = use_einsum
        self.discard_transients = discard_transients
        self.select_random = select_random
        self.select_n_samples = select_n_samples
        self.silent = silent
        self.reservoir_analysis = reservoir_analysis
        self.torch_numpy_works = torch_numpy_works

        # if self.reduce_states_rank_svd and self.reduce_states_rank_umap:
        #     raise ValueError("Choose only one reduction method: reduce_states_rank_svd or reduce_states_rank_umap.")

        n = reservoir_size

        if reservoir_type == "ER":
            W_er = reservoir_ER(n, p=er_p, spectral_radius=spectral_radius,
                                random_state=random_state, directed=directed, w_range=w_range,
                                keep_self_loops=keep_self_loops, complex_valued=complex_valued, distr=distr)
            self.W_res = W_er
            if not self.silent:
                print(f"Initialized ER reservoir with p={er_p}")
        elif reservoir_type == "BA":
            W_ba = reservoir_BA(n, m=ba_m, spectral_radius=spectral_radius,
                                random_state=random_state, directed=directed, w_range=w_range,
                                keep_self_loops=keep_self_loops, complex_valued=complex_valued, distr=distr)
            self.W_res = W_ba
            if not self.silent:
                print(f"Initialized BA reservoir with m={ba_m}")
        elif reservoir_type == "WS":
            W_ws = reservoir_WS(n, k=ws_k, beta=ws_beta, spectral_radius=spectral_radius,
                                random_state=random_state, directed=directed, w_range=w_range,
                                keep_self_loops=keep_self_loops, complex_valued=complex_valued, distr=distr)
            self.W_res = W_ws
            if not self.silent:
                print(f"Initialized WS reservoir with k={ws_k}, beta={ws_beta}")

        elif reservoir_type == "CU":
            W_cu = reservoir_CU(n, density=density, spectral_radius=spectral_radius,
                                random_state=random_state, directed=directed, w_range=w_range,
                                keep_self_loops=keep_self_loops, complex_valued=complex_valued, distr=distr)
            self.W_res = W_cu
            if not self.silent:
                print(f"Initialized CU reservoir with density={density}")

        # Analyze reservoir
        if reservoir_analysis:
            analysis = analyze_adjacency_matrix(self.W_res, res_arch=reservoir_type)
            print(f"Reservoir analysis: {analysis}")

        # Reservoir bias: apply a random state for reproducibility: no normalization here
        rng = np.random.default_rng(2 * random_state)
        if self.complex_valued:
            self.b_res = rng.uniform(-1.0, 1.0, size=(reservoir_size, 1)) + 1j * rng.uniform(-1.0, 1.0, size=(reservoir_size, 1))
        else:
            self.b_res = rng.uniform(-1.0, 1.0, size=(reservoir_size, 1))

        # Input weights
        rng = np.random.default_rng(3 * random_state)
        self.W_in = (rng.uniform(-1.0, 1.0, size=(reservoir_size, input_size)) + 1j * rng.uniform(-1.0, 1.0, size=(reservoir_size, input_size))) \
            if self.complex_valued else rng.uniform(-1.0, 1.0, size=(reservoir_size, input_size))
        # Normalize the matrix by its largest singular value
        u, s, vh = np.linalg.svd(self.W_in, full_matrices=False)
        self.W_in /= s[0]

        # Input weights for einsum product
        self.W_in_einsum = np.tile(self.W_in.T, (self.seq_length, 1, 1))

        # Weights to be trained.
        self.W_out = None
        self.readout_model = None
        self.readout_target_is_complex = False
        self.readout_method = None
        self.kernel_ridge_kernel = "linear"
        self.kernel_ridge_alpha = None
        self.kernel_ridge_kwargs = {}
        self.quantum_kernel = None
        self.quantum_kernel_kwargs = {}
        self._quantum_kernel_padded_feature_dimension = None
        self.readout_train_X = None
        self.kernel_matrix_train = None
        self.last_feature_map_circuit = None
        self.last_overlap_circuit = None
        self.umap_reducer = None
        self._umap_complex_mode = False
        self._umap_real_components = None
        self.states_rank = None

    # Complex tanh
    def complex_tanh(self, z):
        if self.complex_valued:
            return np.tanh(z.real) + 1j * np.tanh(z.imag)
        else:
            return np.tanh(z)

    def _prepare_readout_features(self, X):
        X = np.asarray(X)
        if np.iscomplexobj(X):
            return np.hstack([X.real, X.imag])
        return X

    # The following method visualizes qiskit quantum circuits.    
    def _visualize_feature_map_and_overlap(self, n_qubits, padded_dim, entangler=None, reps=6, output="mpl", style=None, example_state=None):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import StatePreparation, unitary_overlap

        if example_state is None:
            # Use a non-trivial deterministic state so the decomposed plot shows gates.
            phases = np.exp(1j * 2.0 * np.pi * np.arange(padded_dim) / max(1, padded_dim))
            example_state = phases.astype(complex)
            example_state = example_state / np.linalg.norm(example_state)
        else:
            example_state = np.asarray(example_state, dtype=complex).flatten()
            if example_state.size < padded_dim:
                example_state = np.pad(example_state, (0, padded_dim - example_state.size), mode="constant")
            elif example_state.size > padded_dim:
                example_state = example_state[:padded_dim]
            nrm = np.linalg.norm(example_state)
            if np.isclose(nrm, 0.0):
                phases = np.exp(1j * 2.0 * np.pi * np.arange(padded_dim) / max(1, padded_dim))
                example_state = phases.astype(complex)
                example_state = example_state / np.linalg.norm(example_state)
            else:
                example_state = example_state / nrm

        feature_circ = QuantumCircuit(n_qubits)
        feature_circ.append(StatePreparation(example_state), range(n_qubits))
        if entangler is not None:
            feature_circ = feature_circ.compose(entangler)

        # Build a second, distinct sample circuit for a non-trivial overlap visualization.
        phase_ramp = np.exp(1j * np.linspace(0.0, np.pi / 2.0, padded_dim))
        example_state_2 = np.roll(example_state * phase_ramp, 1)
        nrm2 = np.linalg.norm(example_state_2)
        if np.isclose(nrm2, 0.0):
            example_state_2 = np.zeros(padded_dim, dtype=complex)
            example_state_2[1 % padded_dim] = 1.0 + 0.0j
        else:
            example_state_2 = example_state_2 / nrm2

        feature_circ_2 = QuantumCircuit(n_qubits)
        feature_circ_2.append(StatePreparation(example_state_2), range(n_qubits))
        if entangler is not None:
            feature_circ_2 = feature_circ_2.compose(entangler)

        overlap_circ = unitary_overlap(feature_circ, feature_circ_2)

        self.last_feature_map_circuit = feature_circ
        self.last_overlap_circuit = overlap_circ

        return feature_circ, overlap_circ

    def _visualize_feature_map_and_overlap_2(self, feature_map):
    
        from qiskit.circuit.library import unitary_overlap

        feature_circ = feature_map
        overlap_circ = unitary_overlap(feature_circ, feature_circ)

        self.last_feature_map_circuit = feature_circ
        self.last_overlap_circuit = overlap_circ

        return feature_circ, overlap_circ


    # Training function: states update for a batch of sequences
    # IMPORTANT: For states updates no need to take the adjoint of the input and
    # reservoir matrix and the bias vector. The simple transpose suffices.
    # Instead we need to take the adjoint in the ridge regression to fit the output weights,
    # since the normal equation involves the transpose of the design matrix.
    def train_from_dataloader(self, data_loader,
                              ridge_alpha=1.0,
                              leak_alpha=0.8,
                              use_einsum=False,
                              discard_transients=0,
                              readout_method="ridge",
                              kernel_ridge_kernel="linear",
                              kernel_ridge_alpha=1.0,
                              kernel_ridge_kwargs=None,
                              quantum_kernel=None,
                              custom_feature_map=None,
                              custom_feature_map_kwargs=None,
                              quantum_kernel_kwargs=None):
        
        self.readout_method = readout_method
        self.kernel_ridge_kernel = kernel_ridge_kernel
        self.kernel_ridge_kwargs = kernel_ridge_kwargs if kernel_ridge_kwargs is not None else {}
        self.kernel_ridge_alpha = kernel_ridge_alpha
        self.quantum_kernel = quantum_kernel
        self.quantum_kernel_kwargs = quantum_kernel_kwargs if quantum_kernel_kwargs is not None else {}
        self.custom_feature_map = custom_feature_map
        self.custom_feature_map_kwargs = custom_feature_map_kwargs if custom_feature_map_kwargs is not None else {}

        """Collect reservoir states from train_dl and fit a ridge, classical kernel-ridge, or quantum-kernel-ridge readout."""
        states_list = []
        y_list = []
        self.final_states = None

        # print(f'Number of batches in dataloader: {len(data_loader)}')

        for batch in data_loader:
            inputs, targets = batch

            # Helper function to convert pytorch tensors to numpy arrays, if the environment
            # does not allow pytorch to load numpy
            inputs_np = to_numpy_safe(inputs, self.torch_numpy_works)
            y_np = to_numpy_safe(targets, self.torch_numpy_works)

            # "run_reservoir_batch" generates states from each batch of sequences
            self.final_states = self.run_reservoir_batch(inputs_np, y_np, final_states=self.final_states,
                                                         leak_alpha=leak_alpha,
                                                         use_einsum=use_einsum,
                                                         )

            states_list.append(self.final_states)
            y_list.append(y_np)

        # Here we can remove one or more batches from the beginning if we want to discard
        # initial transient states. This is the "spinup" period commonly used in ESN implementations.
        # It makes sense to discard some transients only if the dataloader is not shuffling the dataset.
        if discard_transients > 0:
            states_list = states_list[discard_transients:]
            y_list = y_list[discard_transients:]

        # Concatenate all states and targets from all the batches:
        self.all_states = np.vstack(states_list)
        self.all_targets = np.vstack(y_list)

        # X is the design matrix of all states of the entire dataset,
        # and Y is the column vector of the corresponding targets.
        # We can deep copy X and Y to preserve the original states and targets for later analysis, if needed.
        # or simply recalculate the stacks
        X = np.vstack(states_list)
        Y = np.vstack(y_list)

        # Randomly select a subset of the states for training the readout, if scores_dim is set. 
        # This can speed up training and reduce memory usage.
        if self.select_random and self.select_n_samples is not None and self.select_n_samples < X.shape[0]:
            rng = np.random.default_rng()
            indices = rng.choice(X.shape[0], size=self.select_n_samples, replace=False)
            X = X[indices]
            Y = Y[indices]

        # We just center the states.
        if self.center_states:
            self.X_mean = np.mean(X, axis=0, keepdims=True)
            X = X - self.X_mean

            # The mean will be added back for predictions
            self.Y_mean = np.mean(Y, axis=0, keepdims=True)
            Y = Y - self.Y_mean

        # low-rank approximation of the state matrix using SVD.
        if self.reduce_states_rank_svd:
            u_ent, s_ent, vh_ent = np.linalg.svd(X, full_matrices=False)
            states_rank = self.scores_dim if self.scores_dim is not None else int(min(u_ent.shape[0], u_ent.shape[1]))
            # Calculate the largest power of 2 that is less than or equal to ent_rank
            states_rank = 2 ** int(np.floor(np.log2(states_rank)))
            if self.scores_dim is not None:
                states_rank = min(states_rank, self.scores_dim)
            if self.add_bias:
                states_rank -= 1

            self.states_rank = states_rank

            # We save the svd components for later reconstruction of the states during prediction.
            self.u_ent_reduced = u_ent[:, :states_rank]
            self.s_ent_reduced = s_ent[:states_rank]
            self.vh_ent_reduced = vh_ent[:states_rank, :]

            # Here we use the scores directly
            # X = (u_ent_reduced * s_ent_reduced)
            # Equivalently, we can use the loadings to calculate the scores, but we need
            # to take the adjoint to preserve the complex structure of the states
            X = X @ self.vh_ent_reduced.T.conj()

        if self.reduce_states_rank_umap:
            from umap import UMAP
            states_rank = self.umap_dim if self.umap_dim is not None else int(min(X.shape[0], X.shape[1]))
            states_rank = 2 ** int(np.floor(np.log2(states_rank)))
            if self.umap_dim is not None:
                states_rank = min(states_rank, self.umap_dim)
            if self.add_bias:
                states_rank -= 1

            self.states_rank = states_rank

            if self.complex_valued:

                # IMPORTANT: in this case we need to calculate a distance matrix,
                # since UMAP does not natively support complex-valued data, 
                # and the distance matrix allows us to preserve the complex structure of 
                # the states without having to separate real and imaginary parts. 
                # However, there are different ways to calculate a distance matrix
                # between complex vectors (L2,L1,cosine distance,mahalanobis,
                # Fubini-Study).                
        
                X_distance_matrix = np.zeros((X.shape[0], X.shape[0]))

                if self.umap_dist_matrix == 'L2':
                    # calculate distance matrix using L2 norm
                    for irange in range(X.shape[0]):
                        for jrange in range(irange,X.shape[0]):
                            d_ij = np.linalg.norm(X[irange] - X[jrange])
                            X_distance_matrix[irange, jrange] = d_ij
                            X_distance_matrix[jrange, irange] = d_ij
                            X_distance_matrix[irange, irange] = 0.0
                elif self.umap_dist_matrix == 'L1':
                    # calculate distance matrix using L1 norm
                    for irange in range(X.shape[0]):
                        for jrange in range(irange,X.shape[0]):
                            d_ij = np.linalg.norm(X[irange] - X[jrange], ord=1)
                            X_distance_matrix[irange, jrange] = d_ij
                            X_distance_matrix[jrange, irange] = d_ij
                            X_distance_matrix[irange, irange] = 0.0
                elif self.umap_dist_matrix == 'cosine':
                    # calculate distance matrix using cosine distance (1 - cosine similarity)
                    for irange in range(X.shape[0]):
                        for jrange in range(irange,X.shape[0]):
                            d_ij = 1 - np.abs(np.dot(X[irange], X[jrange].conj())) / (np.linalg.norm(X[irange]) * np.linalg.norm(X[jrange]) + 1e-10)
                            X_distance_matrix[irange, jrange] = d_ij
                            X_distance_matrix[jrange, irange] = d_ij
                            X_distance_matrix[irange, irange] = 0.0
                elif self.umap_dist_matrix == 'mahalanobis':
                    # calculate distance matrix using Mahalanobis distance
                    cov = np.cov(X, rowvar=False) + 1e-10 * np.eye(X.shape[1])  # add small regularization term to avoid singularity
                    inv_cov = np.linalg.inv(cov)
                    for irange in range(X.shape[0]):
                        for jrange in range(irange,X.shape[0]):
                            diff = X[irange] - X[jrange]
                            d_ij = np.sqrt(diff.conj().T @ inv_cov @ diff).real
                            X_distance_matrix[irange, jrange] = d_ij
                            X_distance_matrix[jrange, irange] = d_ij
                            X_distance_matrix[irange, irange] = 0.0
                elif self.umap_dist_matrix == 'fubini-study':
                    # calculate distance matrix using Fubini-Study metric
                    for irange in range(X.shape[0]):
                        for jrange in range(irange,X.shape[0]):
                            inner_product = np.abs(np.dot(X[irange], X[jrange].conj())) / (np.linalg.norm(X[irange]) * np.linalg.norm(X[jrange]) + 1e-10)
                            d_ij = np.arccos(np.clip(inner_product, -1.0, 1.0))
                            X_distance_matrix[irange, jrange] = d_ij
                            X_distance_matrix[jrange, irange] = d_ij
                            X_distance_matrix[irange, irange] = 0.0
                else:
                    raise ValueError(f"Unsupported umap_dist_matrix '{self.umap_dist_matrix}'. Use 'L2', 'L1', 'cosine', 'mahalanobis', or 'fubini-study'.")

                self.umap_reducer = UMAP(n_components=states_rank, metric='precomputed')

                X_orig = np.copy(X)
                X_pinv = np.linalg.pinv(X_orig)
                X_umap = self.umap_reducer.fit_transform(X_distance_matrix)

                self.umap_transform = X_pinv @ X_umap
                X = X_orig @ self.umap_transform
            else:
                self.umap_reducer = UMAP(n_components=states_rank, metric='euclidean')
                X = self.umap_reducer.fit_transform(X)

        # Since X is the design matrix: here we add a bias column
        if self.add_bias and not self.complex_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        # Add a bias in which both the real part and the imaginary part are equal to 1.
        elif self.add_bias and self.complex_bias:
            bias_column = np.ones((X.shape[0], 1),dtype=complex)
            bias_column *= np.exp(1j*np.pi/4) # rotate bias to have both real and imaginary parts non-zero and equal in magnitude
            X = np.hstack([X, bias_column])
        else:
            X = X

        # Normalize X along the 2nd dimension (features) to prepare for quantum amplitude feature mapping.
        if self.normalize_states:
            X_norm = np.linalg.norm(X, axis=1, keepdims=True)
            X_norm[X_norm == 0] = 1  # Avoid division by zero
            self.X_norm_training = X_norm # save the norm for scaling with the corresponding norm during predictions.
            X = X / X_norm            

            # IMPORTANT: this is the key trick here. We normalize both features and targets with 
            # the same norm here. So, both sides of the equation stay the same and it becomes possible 
            # to use normalized data as it would be required by a quantum kernel.
            Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
            Y_norm[Y_norm == 0] = 1  # Avoid division by zero
            self.Y_norm_training = Y_norm            
            Y = Y / X_norm

        # Alternative quantum kernel PCA branch with optional further 
        # dimensionality reduction before fitting the readout.
        if self.quantum_kernel_pca:
            from qiskit_machine_learning.circuit.library.raw_feature_vector import raw_feature_vector
            from qiskit_machine_learning.kernels import FidelityStatevectorKernel
            from sklearn.decomposition import KernelPCA

            # input map dimensions
            self.feature_map = raw_feature_vector(self.scores_dim)
            self.qpca_kernel = FidelityStatevectorKernel(feature_map=self.feature_map)
            matrix_train = self.qpca_kernel.evaluate(x_vec=X)
            # output quantum map dimensions after quantum kernel PCA, may be the same as input or smaller
            self.kernel_pca_q = KernelPCA(n_components=self.qpca_dim if self.qpca_dim else X.shape[1], kernel="precomputed")

            # save X features before quantum kernel PCA to be reused to evaluate the quantum pca kernel for predictions
            self.X_train_features = np.copy(X)
            X = self.kernel_pca_q.fit_transform(matrix_train)

        # Initialize some instance variables (i.e, readout model, kernel_matrix).
        self.readout_model = None
        self.readout_train_X = None
        self.kernel_matrix_train = None
        self.readout_target_is_complex = np.iscomplexobj(Y)

        if self.readout_method == "ridge":
            # Ridge closed-form of the normal equation: W = (X^T X + alpha I)^{-1} X^T Y
            I = np.eye(X.shape[1], dtype=X.dtype)
            # if self.add_bias:
            #     I[-1, -1] = 0.0   # do not regularize bias term
            A = X.T.conj() @ X + ridge_alpha * I
            B = X.T.conj() @ Y
            self.W_out = np.linalg.solve(A, B)

        elif self.readout_method == "kernel_ridge":
            from sklearn.kernel_ridge import KernelRidge

            kernel_ridge_kwargs = {} if kernel_ridge_kwargs is None else dict(kernel_ridge_kwargs)
            X_fit = self._prepare_readout_features(X)
            Y_fit = np.asarray(Y)

            # Note: unlike the closed-form ridge branch, sklearn's KernelRidge will also
            # regularize the bias column if add_bias=True.
            self.readout_model = KernelRidge(
                alpha=kernel_ridge_alpha,
                kernel=kernel_ridge_kernel,
                **kernel_ridge_kwargs,
            )
            self.readout_model.fit(X_fit, Y_fit)

            dual_coef = np.asarray(self.readout_model.dual_coef_)
            if dual_coef.ndim == 1:
                dual_coef = dual_coef[:, np.newaxis]

            kernel_name = kernel_ridge_kernel.lower() if isinstance(kernel_ridge_kernel, str) else None
            if kernel_name == "linear":
                # For linear kernels we can map back to an explicit primal readout matrix.
                self.W_out = X_fit.T @ dual_coef
            else:
                # For non-linear kernels (e.g., rbf) keep dual coefficients in self.W_out.
                self.W_out = dual_coef

        elif self.readout_method == "quantum_kernel_ridge":
            from qiskit import QuantumCircuit
            # from qiskit.circuit.library import zz_feature_map
            from qiskit.circuit.library import efficient_su2
            from sklearn.kernel_ridge import KernelRidge
            from qiskit_machine_learning.circuit.library.raw_feature_vector import raw_feature_vector
            from qiskit_machine_learning.kernels import FidelityStatevectorKernel

            # X_fit = np.asarray(X, dtype=complex)
            X_fit = np.asarray(X)
            Y_fit = np.asarray(Y) 

            self.readout_train_X = np.copy(X_fit)

            feature_map_reps = None
            feature_map_ent = None

            if self.custom_feature_map_kwargs.get("use_custom_features", None):
                feature_map_reps = int(self.custom_feature_map_kwargs.pop("feature_map_reps", 1))
                feature_map_ent = str(self.custom_feature_map_kwargs.pop("entanglement", "linear")).lower()    
                        
            # self.feature_map = raw_feature_vector(self.scores_dim)
            if self.custom_feature_map is None: 
                self.feature_map = raw_feature_vector(self.readout_train_X.shape[1])
                n_qubits = int(np.log2(self.readout_train_X.shape[1]))
            elif self.custom_feature_map == "Efficient_su2":
                self.readout_train_X = self._prepare_readout_features(self.readout_train_X)
                if feature_map_reps is None:
                    feature_map_reps = 1
                if feature_map_ent is None:
                    feature_map_ent = "linear"
                feature_per_qubit = 2*(feature_map_reps + 1) # for efficient su2, each repetition adds 2 parameters per qubit (rx and rz), plus the initial layer
                n_qubits = int(self.readout_train_X.shape[1]/feature_per_qubit) 
                self.feature_map = efficient_su2(num_qubits=n_qubits, reps=feature_map_reps, entanglement=feature_map_ent)
            else:
                raise ValueError(f"Unsupported feature_map '{self.custom_feature_map}'. Use None, 'Efficient_su2', or a custom qiskit circuit.")

            if self.quantum_kernel_kwargs.get("use_entanglement_feature_map", None):

                ent_reps = int(self.quantum_kernel_kwargs.pop("entangling_reps", 1))
                ent_gate = str(self.quantum_kernel_kwargs.pop("entangling_gate", "cz")).lower()
                ent_pairs = self.quantum_kernel_kwargs.pop("entangling_pairs", None)

                # n_qubits = int(np.log2(self.scores_dim))
                # n_qubits = int(np.log2(X_fit.shape[1]))

                if ent_pairs is None:
                    ent_pairs = [(i, i + 1) for i in range(max(0, n_qubits - 1))]
                
                self.entangler = QuantumCircuit(n_qubits)
                for _ in range(max(1, ent_reps)):
                    for q0, q1 in ent_pairs:
                        if ent_gate == "cz":
                            self.entangler.cz(int(q0), int(q1))
                        elif ent_gate == "cx":
                            self.entangler.cx(int(q0), int(q1))
                        elif ent_gate == "iswap":
                            self.entangler.iswap(int(q0), int(q1))
                        else:
                            raise ValueError("Unsupported entangling_gate. Use 'cz', 'cx', or 'iswap'.")
                        
                self.feature_map = self.feature_map.compose(self.entangler)

            self.qridge_kernel = FidelityStatevectorKernel(feature_map=self.feature_map)
            self.kernel_matrix_train = self.qridge_kernel.evaluate(x_vec=self.readout_train_X)

            self.readout_model = KernelRidge(alpha=kernel_ridge_alpha, kernel="precomputed")
            self.readout_model.fit(self.kernel_matrix_train, Y_fit)

            dual_coef = np.asarray(self.readout_model.dual_coef_)
            if dual_coef.ndim == 1:
                dual_coef = dual_coef[:, np.newaxis]
            self.W_out = dual_coef

        else:
            raise ValueError(
                f"Unsupported readout_method '{self.readout_method}'. Use 'ridge', 'kernel_ridge', or 'quantum_kernel_ridge'."
            )

        return

    # This method generates states from each batch of sequences. These states are called "final states"
    # because they can be passed as input to the next batch, allowing for stateful training across batches.
    # This is useful for time series data where we want to maintain temporal dependencies across batches.
    # The use of einsum allows for a more efficient computation of the input transformation,
    # especially for larger batch sizes and sequence lengths.
    def run_reservoir_batch(self, input_batch, target_batch, final_states=None, leak_alpha=0.8,
                            use_einsum=False):
        """
           The fact that the states can be passed in allows for "continuing" the reservoir
           from a previous state, useful for stateful training across batches. Only the first batch
           will use zero initial states. This allows some sort of spinup represented by the size
           of the first batch.
        """
        X = np.asarray(input_batch)
        Y = np.asarray(target_batch)

        # Batch size, sequence length, feature dimension
        B, T, F = X.shape

        if final_states is None:
            states = np.zeros((B, self.reservoir_size))
        else:
            # Since the final batch may be smaller than previous batches,
            # we can choose to use the first B states or the last B states from the previous batch.
            # states = final_states[:B,] # if we want to use the first B states
            states = final_states[-B:,]  # if we want to use the last B states

        # Here we need to generate the states for the entire batch of sequences.
        # The use of einsum allows us to compute the input transformation for all time
        # steps at once.
        if use_einsum:
            U = np.einsum('bij,ijk->bk', X, self.W_in_einsum)
            states = (1 - leak_alpha) * states + leak_alpha * self.complex_tanh(states @ self.W_res.T + U + self.b_res.T)
            if not self.silent:
                print('Using einsum for input transformation.')
        else:
            # Here we loop through time steps.
            for t in range(T):
                u = X[:, t]
                states = (1 - leak_alpha) * states + leak_alpha * self.complex_tanh(states @ self.W_res.T + u @ self.W_in.T + self.b_res.T)

        return states

    # This method is used only for prediction after training. It takes a single sequence
    # (not a batch of sequences) and generates predictions using the trained output weights.
    # The method also includes options for using only the final states from training,
    # or updating the states as predictions are generated moving forward.
    # While the states can be updated, the output weights are not updated. Therefore,
    # this function cannot be used for continued training, but only for prediction.
    def predict(self, input_sequence, leak_alpha=0.8, states=None, use_einsum=False,
                use_training_states=False, update_training_states=False,
                ):
        """
        Predict for a single sequence. Returns array length F.
        """

        input_sequence = to_numpy_safe(input_sequence, self.torch_numpy_works)

        # Batch size, sequence length, feature dimension
        B, T, F = input_sequence.shape

        final_states = self.final_states

        if states is None and not use_training_states:
            states = np.zeros((final_states.shape[0], self.reservoir_size))
        elif states is None and use_training_states:
            states = final_states
        else:
            pass

        if use_einsum:  # use einsum for input transformation in one shot
            U = np.einsum('bij,ijk->bk', input_sequence, self.W_in_einsum)
            states = (1 - leak_alpha) * states + leak_alpha * self.complex_tanh(states @ self.W_res.T + U + self.b_res.T)
        else:
            for t in range(T):
                u = input_sequence[:, t]
                states = (1 - leak_alpha) * states + leak_alpha * self.complex_tanh(states @ self.W_res.T + u @ self.W_in.T + self.b_res.T)

        if update_training_states:
            self.final_states = states

        X = np.copy(states)

        # Center the data.
        if self.center_states:
            X = X - self.X_mean  

        # Move states into eigenspace for predictions
        if self.reduce_states_rank_svd:
            X = X @ self.vh_ent_reduced.T.conj()  # here we take the adjoint to preserve the complex structure of the states

        # move states into umap space for predictions
        if self.reduce_states_rank_umap:
            if self.complex_valued:
                X = X @ self.umap_transform
            else:
                X = self.umap_reducer.transform(X)

        # Since X is the design matrix: here we add a bias column
        if self.add_bias and not self.complex_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        # Add a complex bias.
        elif self.add_bias and self.complex_bias:
            bias_column = np.ones((X.shape[0], 1),dtype=complex)
            bias_column *= np.exp(1j*np.pi/4) # rotate bias to have both real and imaginary parts non-zero and equal in magnitude
            X = np.hstack([X, bias_column])
        else:
            X = X

        # Here is the corresponding normalization trick for predictions.
        if self.normalize_states:
            X_norm = np.linalg.norm(X, axis=1, keepdims=True)
            X_norm[X_norm == 0] = 1  # Avoid division by zero
            X = X / X_norm

        # This section is the alternative quantum kernel PCA branch. Here we need to 
        # evaluate the quantum kernel between the test states and the training states, 
        # and then apply the kernel PCA transformation to the test kernel matrix.
        # Note we are not further reducing the dimensionality of the states in this branch, 
        # but only expressing the states as quantum state vectors.
        if self.quantum_kernel_pca:
            # from qiskit_machine_learning.kernels import FidelityStatevectorKernel
            matrix_test = self.qpca_kernel.evaluate(x_vec=X, y_vec=self.X_train_features)

            # from sklearn.decomposition import KernelPCA
            X = self.kernel_pca_q.transform(matrix_test)    

        # Predict using the prepared features.
        if self.readout_method == "ridge":
            preds = X @ self.W_out     

        if self.readout_method == "kernel_ridge":            
            X_fit = self._prepare_readout_features(X)
            preds = np.asarray(self.readout_model.predict(X_fit))

            if preds.ndim == 1:
                preds = preds[:, np.newaxis]                  

        if self.readout_method == "quantum_kernel_ridge":  
            # X_fit = np.asarray(X, dtype=complex)
            X_fit = np.asarray(X) 
            if self.custom_feature_map == "Efficient_su2":
                X_fit = self._prepare_readout_features(X_fit)           
            self.kernel_matrix_test = self.qridge_kernel.evaluate(x_vec=X_fit, y_vec=self.readout_train_X)
            preds = np.asarray(self.readout_model.predict(self.kernel_matrix_test))

            if preds.ndim == 1:
                preds = preds[:, np.newaxis]

        # Second part of the "normalization" trick.
        # We predict using the normalized states, 
        # but then we multiply the predictions by the norm to 
        # un-normalize them. This is intuitively 
        # the same as bringing the kernel out of quantum state 
        if self.normalize_states and self.denormalize_preds:
            preds *= X_norm

        if self.scale_norm:
            norm_scale = self.X_norm_training.mean() / X_norm.mean()
            preds *= norm_scale        

        n_preds = preds.shape[0]

        if n_preds > 1:
            preds = np.mean(preds, axis=0, keepdims=True)

        # The following conditional is necessary for readout method 'ridge', 
        # because the output is complex if self.complex_valued is True
        if self.complex_valued and self.readout_method == "ridge":
            preds = np.abs(preds) * np.sign(np.real(preds))

        # Add the Y mean back to the predictions
        preds += self.Y_mean

        return preds, states