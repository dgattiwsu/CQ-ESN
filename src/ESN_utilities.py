import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import random
from torch import nn


# Function to load the Berkeley TAVG dataset
def load_global_tavg(path):
    # read - skip commented lines starting with '%' and use whitespace as delimiter
    df = pd.read_csv(path,
                     comment='%',          # skip header/comment lines beginning with %
                     delim_whitespace=True,
                     header=None,
                     na_values=['NaN', 'nan'],
                     engine='python')

    # assign sensible column names: first two are Year, Month, rest are anomaly/unc pairs
    ncols = df.shape[1]
    if ncols < 3:
        raise ValueError(f"unexpected number of columns: {ncols}")

    # create generic pair names for the remaining columns
    pair_count = (ncols - 2) // 2
    col_names = ['Year', 'Month']
    for i in range(pair_count):
        col_names += [f'Anomaly_{i+1}', f'Unc_{i+1}']

    # if there is an odd extra column, add a generic name
    if len(col_names) < ncols:
        col_names += [f'Extra_{len(col_names)+1}']

    df.columns = col_names[:ncols]

    # convert year/month to a datetime (first day of month)
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

    # optional: set Date as index
    df = df.set_index('Date')

    return df


# Scaling functions
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform the training data
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform the test data
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# Inverse scaling for a forecasted value. Since scaler was fit on 
# supervised rows [lag_1, lag_2, ..., target], then X must be the 
# feature part of that same row (in the same scaled representation 
# the scaler expects). 
# You build a row = [*X, yhat] (1D), reshape to (1, n_features) 
# and call inverse_transform; the returned row's last element 
# is the unscaled forecast.
def invert_scale(scaler, X, yhat):
	new_row = [x for x in X] + [yhat]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]  


# Find earliest complete year of data
def earliest_complete_year(df):
    # Ensure Year column exists
    if 'Year' not in df.columns:
        df = df.copy()
        df['Year'] = df.index.year

    # Rows with any NaN
    nan_rows = df[df.isna().any(axis=1)]

    if nan_rows.empty:
        return int(df['Year'].min())  # all data already complete

    last_nan_year = int(nan_rows['Year'].max())
    candidate = last_nan_year + 1

    # Verify candidate span is NaN-free
    span = df[df['Year'] >= candidate]
    if span.empty:
        return None  # no data after last_nan_year
    if span.isna().any().any():
        return None  # still NaNs after candidate; no complete span
    return candidate 

# Usage:
# earliest_year = earliest_complete_year(combined_tavg_1900_to_2020_df)
# print("Earliest fully complete year:", earliest_year) 


# Generate datasets for LSTM and ESN with only one target
class TAVG_Dataset(Dataset):
    def __init__(self, data, seq_length=12):

        # Use the raw data without scaling
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index+self.seq_length,:], dtype=torch.float),
            torch.tensor(self.data[index+self.seq_length,:], dtype=torch.float),
            # if only global temperatures as target
            # torch.tensor(self.data[index+self.seq_length,0], dtype=torch.float).unsqueeze(-1),
        )

# Generate datasets for LSTM and ESN with separate targets for label and autoregression
class TAVG_Dataset_AR(Dataset):
    def __init__(self, data, seq_length=12):

        # Use the raw data without scaling
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index:index+self.seq_length,1:], dtype=torch.float),
            # Here we have two targets: global temp and full state
            torch.tensor(self.data[index+self.seq_length,0], dtype=torch.float),
            torch.tensor(self.data[index+self.seq_length,1:], dtype=torch.float)
            # if only global temperatures as target
            # torch.tensor(self.data[index+self.seq_length,0], dtype=torch.float).unsqueeze(-1),
        )
    

# Random number generator
def set_pytorch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# Plot history files
def plot_history(history):
    # train_loss, train_mae, train_mape, val_loss, val_mae, val_mape = history
    train_loss, val_loss = history

    epochs = range(1, len(train_loss) + 1)

    # fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    ax.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax.set_title('Loss')
    ax.legend()

    plt.tight_layout()
    plt.show()


# Validation metrics
def nrmse(preds, targets, method="range", axis=None, eps=1e-8):
    """
    Compute Normalized Root Mean Squared Error between two numpy arrays.

    Args:
        preds: predicted values (numpy array).
        targets: true values (numpy array), same shape as preds.
        method: normalization method, one of:
            - "range": divide by (max - min) of targets
            - "mean": divide by mean(targets)
            - "std": divide by std(targets)
            - "iqr": divide by interquartile range (75th-25th percentile) of targets
        axis: axis or axes along which to compute RMSE/normalizer (passed to numpy functions).
              Use None to treat the arrays as flattened.
        eps: small epsilon to avoid division by zero.

    Returns:
        NRMSE (float or numpy array depending on axis): rmse / normalizer
    """
    # ensure shapes match
    if preds.shape != targets.shape:
        raise ValueError("preds and targets must have the same shape")

    # compute RMSE (allow NaNs)
    mse = np.nanmean((np.asarray(preds) - np.asarray(targets))**2, axis=axis)
    rmse = np.sqrt(mse)

    # compute normalizer based on chosen method
    t = np.asarray(targets)
    if method == "range":
        tmax = np.nanmax(t, axis=axis)
        tmin = np.nanmin(t, axis=axis)
        norm = tmax - tmin
    elif method == "mean":
        norm = np.abs(np.nanmean(t, axis=axis))
    elif method == "std":
        norm = np.nanstd(t, axis=axis)
    elif method == "iqr":
        q75 = np.nanpercentile(t, 75, axis=axis)
        q25 = np.nanpercentile(t, 25, axis=axis)
        norm = q75 - q25
    else:
        raise ValueError(f"unknown method: {method}")

    # avoid division by zero
    norm = np.where(np.abs(norm) < eps, eps, norm)

    return rmse / norm


def psd_nrmse(preds, targets, fs=1.0, method="range", per_series=False, eps=1e-8):
    """
    Compute Normalized RMSE between power spectral densities of preds and targets.

    Args:
        preds: array_like, shape (T,) or (N_series, T)
        targets: array_like, same shape as preds
        fs: sampling frequency (Hz). Default 1.0
        method: normalization method for NRMSE; one of {"range","mean","std","iqr"}
        per_series: if True return array of NRMSE per series, else return scalar mean NRMSE
        eps: small value to avoid division by zero

    Returns:
        float (mean NRMSE) or np.ndarray (NRMSE per series)
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if preds.shape != targets.shape:
        raise ValueError("preds and targets must have the same shape")

    # make (N_series, T)
    if preds.ndim == 1:
        preds = preds[np.newaxis, :]
        targets = targets[np.newaxis, :]
    if preds.ndim != 2:
        raise ValueError("inputs must be 1D or 2D (series x time) arrays")

    N_series, T = preds.shape

    # remove mean (detrend) to focus PSD on variability
    preds_d = preds - np.mean(preds, axis=1, keepdims=True)
    targets_d = targets - np.mean(targets, axis=1, keepdims=True)

    # one-sided periodogram using rfft
    Pp = np.fft.rfft(preds_d, axis=1)
    Pt = np.fft.rfft(targets_d, axis=1)
    freqs = np.fft.rfftfreq(T, d=1.0/fs)

    # power spectral density (periodogram) normalization:
    # PSD = (|FFT|^2) / (fs * T), then double non-DC and non-Nyquist bins for one-sided
    Sp = (np.abs(Pp) ** 2) / (fs * T)
    St = (np.abs(Pt) ** 2) / (fs * T)

    # double energy for one-sided spectrum except DC (index 0) and Nyquist (if present)
    if T % 2 == 0:
        # even: last bin is Nyquist and should not be doubled
        Sp[:, 1:-1] *= 2.0
        St[:, 1:-1] *= 2.0
    else:
        Sp[:, 1:] *= 2.0
        St[:, 1:] *= 2.0

    # compute RMSE across frequency axis for each series
    mse = np.mean((Sp - St) ** 2, axis=1)
    rmse = np.sqrt(mse)

    # compute normalizer from true PSD per chosen method
    if method == "range":
        norm = np.nanmax(St, axis=1) - np.nanmin(St, axis=1)
    elif method == "mean":
        norm = np.abs(np.nanmean(St, axis=1))
    elif method == "std":
        norm = np.nanstd(St, axis=1)
    elif method == "iqr":
        q75 = np.nanpercentile(St, 75, axis=1)
        q25 = np.nanpercentile(St, 25, axis=1)
        norm = q75 - q25
    else:
        raise ValueError(f"unknown method: {method}")

    norm = np.where(np.abs(norm) < eps, eps, norm)
    nrmse_per_series = rmse / norm

    return nrmse_per_series if per_series else float(np.mean(nrmse_per_series))

# Analyze adjacency matrix
def analyze_adjacency_matrix(W, tol=1e-10,res_arch=None):
    """
    Analyze a (weighted) adjacency matrix W:
    - classify as directed/undirected via symmetry
    - report asymmetry ratio, node/edge counts, and density (no self-loops)
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    n = W.shape[0]

    # Directed vs undirected (numerical symmetry)
    symmetric = np.allclose(W, W.T, atol=tol)
    graph_type = "undirected" if symmetric else "directed"

    # Asymmetry ratio (as before)
    diff = np.abs(W - W.T)
    asym_edges = np.count_nonzero(diff > tol)
    total_edges = np.count_nonzero((np.abs(W) + np.abs(W.T)) > tol)
    asym_ratio = asym_edges / (total_edges if total_edges else 1)

    # Edge counting (exclude self-loops)
    nz = np.abs(W) > tol
    np.fill_diagonal(nz, False)
    
    if symmetric:
        # count each undirected edge once
        n_edges = int(np.count_nonzero(np.triu(nz, k=1)))
        max_edges = n * (n - 1) // 2
    else:
        # count all directed edges
        n_edges = int(np.count_nonzero(nz))
        max_edges = n * (n - 1)

    density = n_edges / max_edges if max_edges > 0 else 0.0

    return {
        "graph_arch": res_arch,
        "graph_type": graph_type,
        "asym_ratio": float(asym_ratio),
        "n_nodes": int(n),
        "n_edges": int(n_edges),
        "density": float(density),
    }


# Plot degree distribution with power-law fit
import matplotlib.pyplot as plt
from scipy import stats

def plot_degree_distribution_with_powerlaw(
    W,
    res_type="Unknown",
    k_min=None,
    tol=1e-10,
    show_fit=True,
    ax=None,
    plot_mode="loglog",   # "loglog" (default) or "linear"
    degree_type="out",    # "out" or "in"
    y_stat="pmf",         # "pmf" (probability) or "freq" (counts)
    plot_kind="scatter",  # "scatter" (default) or "bar"
    fontsize=10
):
    """
    Plot degree distribution of reservoir adjacency matrix and (optionally) fit power law to the tail.

    Args:
        W: (n x n) reservoir weight matrix (numpy array)
        res_type: label for the reservoir (e.g., "ER", "BA", "WS")
        k_min: minimum degree for tail fit (if None, use 95th percentile of node degrees)
        tol: threshold for non-zero edges
        show_fit: whether to overlay the power-law fit curve
        ax: matplotlib axis (created if None)
        plot_mode: "loglog" for log-log plot; "linear" for linear axes
        degree_type: "out" sums rows; "in" sums columns (for directed graphs)
        y_stat: "pmf" plots probability mass function; "freq" plots raw counts (frequency)
        plot_kind: "scatter" for scatter; "bar" for bar plot

    Returns:
        dict with fit results and data arrays.
    """
    

    # Binary adjacency
    A = (np.abs(W) > tol).astype(int)

    # Degree choice
    if degree_type == "out":
        degrees = np.sum(A, axis=0)
    elif degree_type == "in":
        degrees = np.sum(A, axis=1)
    else:
        raise ValueError("degree_type must be 'out' or 'in'")

    # Degree distribution
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    n_nodes = len(degrees)
    pmf = counts / n_nodes

    # Choose what to plot on y-axis
    if y_stat == "pmf":
        y_vals = pmf
        y_label = "P(k)"
    elif y_stat == "freq":
        y_vals = counts
        y_label = "Frequency"
    else:
        raise ValueError("y_stat must be 'pmf' or 'freq'")

    # Tail selection for fit
    if k_min is None:
        k_min = int(np.percentile(degrees, 95))
        k_min = max(k_min, 1)

    tail_mask = unique_degrees >= k_min
    k_tail = unique_degrees[tail_mask]
    p_tail = pmf[tail_mask]  # fit always on PMF tail

    # Power-law fit in log10 space: log10 P(k) = a + b log10 k
    slope = intercept = r_value = p_value = std_err = None
    gamma = np.nan
    r_squared = np.nan
    if len(k_tail) > 1 and np.all(p_tail > 0):
        log_k = np.log10(k_tail)
        log_p = np.log10(p_tail)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_p)
        gamma = -slope
        r_squared = r_value**2

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if plot_kind == "scatter":
        ax.scatter(unique_degrees, y_vals, alpha=0.6, s=50, label=f'Empirical ({y_stat})')
    elif plot_kind == "bar":
        ax.bar(unique_degrees, y_vals, width=0.8, alpha=0.7, color='tab:blue',
               edgecolor='none', align='center', label=f'Empirical ({y_stat})')
    else:
        raise ValueError("plot_kind must be 'scatter' or 'bar'")

    # Overlay fit curve
    if show_fit and slope is not None and not np.isnan(slope):
        k_fit = np.linspace(max(1, k_min), max(unique_degrees), 200)
        # model in PMF space
        p_fit = 10 ** (intercept + slope * np.log10(k_fit))
        y_fit = p_fit * n_nodes if y_stat == "freq" else p_fit
        fit_label = f'Power-law fit (γ={gamma:.2f}, R²={r_squared:.3f})'
        ax.plot(k_fit, y_fit, 'r--', linewidth=2, label=fit_label)

    # Axes scale
    if plot_mode == "loglog":
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif plot_mode == "linear":
        ax.set_xscale('linear')
        ax.set_yscale('linear')
    else:
        raise ValueError("plot_mode must be 'loglog' or 'linear'")

    ax.set_xlabel('Degree k', fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.set_title(f'{res_type} Degree Distribution (deg={degree_type}, y={y_stat})\n(k_min={k_min})',
                 fontsize=fontsize)
    ax.legend(fontsize=fontsize-2)
    ax.grid(True, alpha=0.3)

    if ax is None:
        plt.tight_layout()
        plt.show()

    return {
        'res_type': res_type,
        'k_min': k_min,
        'slope': slope,
        'gamma': gamma,
        'R_squared': r_squared,
        'degrees': degrees,
        'unique_degrees': unique_degrees,
        'counts': counts,
        'pmf': pmf,
        'plot_mode': plot_mode,
        'y_stat': y_stat,
        'degree_type': degree_type,
        'plot_kind': plot_kind,
        'ax': ax,   
        'fontsize': fontsize
    }


# Convert reservoir matrix to PyTorch Geometric graph
import torch
from torch_geometric.data import Data

def reservoir_to_pyg_graph(W_res, threshold=0.0, self_loops=False):
    """
    Convert a weighted reservoir matrix into a PyTorch Geometric graph.
    
    Args:
        W_res: (n x n) numpy array or torch tensor (reservoir weight matrix)
        threshold: float, minimum absolute weight to include an edge (default: 0.0)
        self_loops: bool, whether to include self-loops (diagonal elements) (default: False)
        
    Returns:
        data: PyTorch Geometric Data object with:
            - edge_index: COO format edge list [2, num_edges]
            - edge_attr: edge weights [num_edges, 1]
            - x: node features (rows of W_res) [num_nodes, num_features]
            - num_nodes: number of nodes
    """
    # Convert to numpy if torch tensor
    if isinstance(W_res, torch.Tensor):
        W_res = W_res.cpu().numpy()
    
    W_res = np.asarray(W_res)
    
    if W_res.ndim != 2 or W_res.shape[0] != W_res.shape[1]:
        raise ValueError("W_res must be a square matrix")
    
    n = W_res.shape[0]
    
    # Create mask for edges to include
    mask = np.abs(W_res) > threshold
    
    # Optionally remove self-loops
    if not self_loops:
        mask = mask & ~np.eye(n, dtype=bool)
    
    # Get edge indices in COO format
    row_indices, col_indices = np.where(mask)
    edge_weights = W_res[mask]
    
    # Convert to PyTorch tensors
    edge_index = torch.tensor(np.stack([row_indices, col_indices]), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    
    # Use rows of W_res as node features
    x = torch.tensor(W_res, dtype=torch.float)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n
    )
    
    return data


# Example usage:
# Assuming you have a reservoir matrix W_res
# data = reservoir_to_pyg_graph(W_res, threshold=1e-10, self_loops=False)
# print(data)
# print(f"Number of nodes: {data.num_nodes}")
# print(f"Number of edges: {data.edge_index.shape[1]}")
# print(f"Node features shape: {data.x.shape}")
# print(f"Edge attributes shape: {data.edge_attr.shape}")

# Plot ESN reservoir as directed graph
import networkx as nx

def plot_esn_reservoir(W,
                       tol=1e-10,
                       layout="spring",
                       max_nodes=None,
                       figsize=(6,6),
                       node_size_base=120,
                       node_color="lightgray",
                       edge_cmap=("tab:red","tab:blue"),
                       edge_alpha=0.7,
                       width_scale=2.0,
                       arrows=True,
                       with_labels=False,
                       seed=42,ax=None):
    """
    Visualize an ESN reservoir weight matrix as a sparse directed graph.

    Convention: W[i,j] != 0 means edge j -> i (source = column j, target = row i).

    Args:
        W              : (n x n) numpy array (reservoir matrix)
        tol            : abs(weight) threshold to keep an edge
        layout         : 'spring' | 'kamada_kawai' | 'circular' | 'shell' | 'spectral'
        max_nodes      : if set, randomly subsample nodes for large reservoirs
        figsize        : matplotlib figure size
        node_size_base : base node size (scaled by (in_degree+out_degree))
        node_color     : color of nodes
        edge_cmap      : (color_neg, color_pos) for negative / positive weights
        edge_alpha     : edge transparency
        width_scale    : scales edge width by |weight|
        arrows         : draw arrows if True
        with_labels    : show node labels
        seed           : random seed for reproducible layouts

    Returns:
        G (networkx.DiGraph)
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")

    n = W.shape[0]

    # Optional node subsampling for very large reservoirs
    if max_nodes is not None and max_nodes < n:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max_nodes, replace=False)
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep] = True
    else:
        keep_mask = np.ones(n, dtype=bool)

    # Build directed graph
    G = nx.DiGraph()
    for j in range(n):            # source
        if not keep_mask[j]:
            continue
        for i in range(n):        # target
            if not keep_mask[i]:
                continue
            w = W[i, j]
            if abs(w) > tol:
                G.add_edge(j, i, weight=w)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        raise ValueError("Unknown layout")

    # Node sizes (degree-based)
    degs = np.array([G.in_degree(v) + G.out_degree(v) for v in G.nodes()])
    node_sizes = node_size_base * (1 + degs / (degs.max() if degs.max() > 0 else 1))

    # Split edges by sign
    edges_pos = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 0]
    edges_neg = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]

    # Edge widths
    weights = np.array([abs(d["weight"]) for _, _, d in G.edges(data=True)])
    if len(weights):
        w_min, w_max = weights.min(), weights.max()
        widths = width_scale * (0.2 + (weights - w_min) / (w_max - w_min + 1e-12))
    else:
        widths = []

    # Map widths back to ordered edge list for drawing separately
    # Recompute ordered list to align with widths array
    ordered_edges = list(G.edges(data=True))
    width_map = { (u,v): widths[k] for k,(u,v,_) in enumerate(ordered_edges) }

    widths_pos = [width_map[(u,v)] for (u,v) in edges_pos]
    widths_neg = [width_map[(u,v)] for (u,v) in edges_neg]

    # plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_color,
                           linewidths=0.5,
                           edgecolors="black",
                           ax=ax)
    if edges_pos:
        nx.draw_networkx_edges(G, pos,
                               edgelist=edges_pos,
                               edge_color=edge_cmap[1],
                               width=widths_pos,
                               alpha=edge_alpha,
                               arrows=arrows,
                               arrowsize=10,
                               ax=ax)
    if edges_neg:
        nx.draw_networkx_edges(G, pos,
                               edgelist=edges_neg,
                               edge_color=edge_cmap[0],
                               width=widths_neg,
                               alpha=edge_alpha,
                               style='dashed',
                               arrows=arrows,
                               arrowsize=10,
                               ax=ax)

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)

    ax.set_title(f"Reservoir Graph (nodes={n}, edges={G.number_of_edges()})", fontsize=11)
    ax.axis("off")
    # plt.tight_layout()
    # plt.show()

    # return G
    return


# Plot reservoir weight matrix as colored dots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def plot_reservoir_matrix(W, fig, ax,
                         tol=1e-10,
                        #  figsize=(6, 6),
                         cmap='RdBu_r',
                         dot_size_scale=50,
                         show_colorbar=True,
                         title="Reservoir Weight Matrix"):
    """
    Visualize ESN reservoir weight matrix as colored dots.
    
    Args:
        W              : (n x n) numpy array (reservoir matrix)
        tol            : threshold below which weights are not shown
        figsize        : matplotlib figure size
        cmap           : colormap name ('RdBu_r' = red for negative, blue for positive)
        dot_size_scale : base size multiplier for dots (scaled by |weight|)
        show_colorbar  : whether to show colorbar
        title          : plot title
        
    Returns:
        fig, ax
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    
    n = W.shape[0]
    
    # Get non-zero entries above threshold
    mask = np.abs(W) > tol
    rows, cols = np.where(mask)
    weights = W[mask]
    
    # Normalize dot sizes by absolute weight
    if len(weights) > 0:
        abs_weights = np.abs(weights)
        w_min, w_max = abs_weights.min(), abs_weights.max()
        if w_max > w_min:
            sizes = dot_size_scale * (0.2 + 0.8 * (abs_weights - w_min) / (w_max - w_min))
        else:
            sizes = np.full_like(abs_weights, dot_size_scale)
    else:
        sizes = []
    
    # Create figure
    # fig, ax = plt.subplots(figsize=figsize)
    
    # Use diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=weights.min() if len(weights) else -1, 
                        vcenter=0, 
                        vmax=weights.max() if len(weights) else 1)
    
    # Scatter plot: row index = y, column index = x
    scatter = ax.scatter(cols, rows, 
                        c=weights, 
                        s=sizes, 
                        cmap=cmap,
                        norm=norm,
                        alpha=0.8,
                        edgecolors='none')
    
    # Formatting
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # invert y-axis so (0,0) is top-left
    ax.set_xlabel('Source Node (j)', fontsize=10)
    ax.set_ylabel('Target Node (i)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Colorbar
    # if show_colorbar:
    #     cbar = plt.colorbar(scatter, ax=ax)
    #     cbar.set_label('Weight', fontsize=10)

    if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Weight', fontsize=9)
            cbar.ax.tick_params(labelsize=8)    
    
    # Add statistics text
    stats_text = f'Nodes: {n}\n'
    stats_text += f'Edges: {len(weights)}\n'
    stats_text += f'Density: {len(weights)/(n*n):.3f}\n'
    stats_text += f'Weight range: [{weights.min():.3f}, {weights.max():.3f}]' if len(weights) else 'No edges'
    
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    # plt.tight_layout()
    # plt.show()
    
    # return fig, ax

    return


# Example usage with your ESN reservoir:
# fig, ax = plot_reservoir_matrix(esn.W_res, tol=1e-8, dot_size_scale=100)

import joblib

def save_esn_models(models, filepath):
    try:
        joblib.dump(models, filepath, compress=3)  # compress=3 reduces file size
        print(f"✓ Saved {len(models)} models to {filepath}")
    except Exception as e:
        print(f"✗ Error saving models: {e}")

def load_esn_models(filepath):
    try:
        models = joblib.load(filepath)
        print(f"✓ Loaded {len(models)} models from {filepath}")
        return models
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return None
