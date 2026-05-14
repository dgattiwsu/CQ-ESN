## CQ-ESN

CQ-ESN: Hybrid Classical-Quantum Echo State Networks for Time Series Forecasting.  

CQ-ESN was designed to facilitate separating the contribution of different effects (real *vs* complex-valued states, interference, entangling) in Quantum Echo State Networks. Standard ESNs use ridge regression (implemented *via* a closed form version of the normal equations) to learn a linear readout from the reservoir states to the target output. In CQ-ESN, we replace this **classical ridge regression** with **kernel ridge regression** or with **quantum kernel ridge regression**, which uses a quantum kernel to compute the inner products between reservoir states in a $n$-dimensional feature space. The quantum kernel is estimated using a quantum circuit that encodes the reservoir states as quantum states and measures their **overlaps**. An alternative to this approach would be to use a quantum version of the normal equation to implement the ridge regression, but this would require a quantum algorithm for matrix inversion (e.g., [HHL](https://arxiv.org/abs/2507.15537)), which is less efficient than using quantum kernels for regression. Future versions of CQ-ESN will explore this alternative approach. Other alternatives (i.e. [here](https://arxiv.org/abs/2412.07910), using quantum circuits to directly learn the readout weights) have also been described.

While the current CQ-ESN implementation represents a direct conversion of the classical ESN algorithm to a quantum method, one complication is associated with the fact that by definition quantum state vectors are normalized. Unfortunately, ESN states normalization is associated with some loss of information. This can be easily verified by running a standard closed-form ridge regression on normalized states. To overcome this problem, we multiply the predictions derived from the readout by the same norm used to normalize the states. This is a common practice in quantum machine learning when using quantum kernels derived from state overlaps, which allows us to observe the additional quantum effects from entanglement, while mitigating the information loss due to normalization. 

<div style="border:1px solid #ccc; border-radius:6px; padding:12px; background: #050094; max-width:90%; margin-bottom:20px; margin-left:20px; margin-right:20px;">

#### CQ-ESN Installation

We recommend using CQ-ESN in a virtual environment. The following are the recommended steps to generate a suitable environment using pip and conda:

```bash
conda create -n ibm_qml_311 python=3.11.13
conda activate ibm_qml_311
conda update pip
pip3 install qiskit-machine-learning
pip3 install 'qiskit-machine-learning[torch]'
pip3 install 'qiskit-machine-learning[sparse]'
conda install --channel=numba llvmlite
pip3 install nlopt
pip3 install pandas
pip3 install openpyxl
pip3 install matplotlib
pip3 install seaborn
pip3 install plotly
pip3 install ipykernel
pip3 install ipympl
pip3 install jupyter
pip3 install jupyterlab
pip3 install pyprind
pip3 install statsmodels
pip3 install xgboost
pip3 install ipywidgets
pip3 install 'qiskit[visualization]'
pip3 install pyTensorlab
pip3 install qiskit-aer
pip3 install qiskit-ibm-runtime
pip3 install torch_geometric
conda install -c conda-forge umap-learn
conda deactivate    
```

CQ-ESN adopts the `qiskit_machine_learning` library, which provides a convenient interface for computing quantum kernels using various quantum circuits and feature maps. Unfortunately, the `qiskit_machine_learning` library requires numpy 2.4, while the version of Pytorch loaded with `qiskit-machine-learning[torch]` is compatible with numpy 1.26. This results in runtime errors only in the rare occasions in which a numpy C extension is called to act on torch tensors. CQ-ESN has a workaround for this issue based on a simple patch.

</div>
