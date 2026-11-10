# Bandgap Prediction of Lithium-based Transition Metal Oxides (Li-TM-O) using CGCNN

This notebook demonstrates a complete end-to-end pipeline to predict the electronic bandgap of Lithium Transition Metal Oxides (Li-TM-O, where *TM* = Fe, Co, Ni, Mn, Ti, V, Cr) using a Crystal Graph Convolutional Neural Network (CGCNN) implemented in PyTorch Geometric.

---

## 1. Project Goal
The primary goal is to predict the **bandgap (eV)** of crystal structures from raw crystallographic information files (`.cif` format) and atomic properties. Accurately predicting bandgaps helps identify promising battery materials (e.g., cathodes, solid-state electrolytes) with optimal electronic conductivities.

---

## 2. Methodology & Process Flow

### **A. Data Loading & Exploratory Data Analysis (EDA)**
- Loaded a dataset containing **5,169 materials** mapped with their Chemical Formulas and target Bandgaps.
- Visualized the target distribution which reveals a mixed distribution:
 - **Metals** (`0.0 eV`): ~15.8%
 - **Semiconductors** (`0-3 eV`): ~68.4%
 - **Insulators** (`>3 eV`): ~15.9%
- Verified that there is a corresponding CIF structure file available for every material ID.

### **B. Feature Engineering & Crystal Graph Construction**
- **Node (Atom) Features (19-dimensional)**: Constructed using PyMatGen `Element` properties including atomic number, period, group, atomic radius, electronegativity, ionization energy, Mendeleev number, electron affinity, molar volume, electronic block (s, p, d, f), and a transition metal flag. Estimated oxidation states are added dynamically.
- **Edge (Bond) Features (41-dimensional)**: Computed by sorting neighbor distances (up to 12 nearest neighbors within an 8.0 Å cutoff) and passing them through a Gaussian expansion filter.

### **C. Efficient Dataset & Scalable Processing**
- Created a disk-based custom PyG `Dataset` (`LiTMODataset`) that processes and saves crystal graphs one-by-one to prevent RAM saturation.
- Handled data leakage prevention using **`GroupShuffleSplit`** grouped by chemical formula to ensure structures with the exact same composition do not span across train and test sets.

### **D. Model Architecture (CGCNN)**
- **Node Embedding**: Embeds raw 19D atom features into a 64D latent space.
- **Message Passing (`CGCNNConv`)**: Integrates neighbor states and bond representations using gated and update neural layers (with Sigmoid and Softplus activations).
- **Pooling**: Aggregates node features via global average pooling (`global_mean_pool`) into a single crystal vector representation.
- **Fully Connected Network**: Processes the crystal vector through multiple dense layers with Batch Normalization, Dropout (`25%`) and Softplus activation to predict a continuous bandgap value.

### **E. Training & Optimization**
- Trained for up to 200 epochs using **AdamW** with a learning rate of `0.0005` and weight decay of `1e-3`.
- Monitored loss with **MAE (L1 Loss)** on scaled targets and dynamically reduced the learning rate using `ReduceLROnPlateau`.
- Employed early stopping with a patience of 30 epochs.

---

## 3. Results Summary

The model training was terminated early at **Epoch 94** once validation loss stopped improving, successfully avoiding overfitting.

| Dataset Split | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Coefficient of Determination (R<sup>2</sup>) |
| :--- | :---: | :---: | :---: |
| **Training Set** | 0.2730 eV | 0.1881 eV | **0.8778** |
| **Test Set** | 0.4355 eV | 0.3670 eV | **0.7701** |

### **Key Findings**
* The model performs remarkably well on unseen crystal graphs, explaining **~77.0%** of the variance in the bandgaps of the test set.
* Out-of-fold generalization remains robust, showing that CGCNN convolutional filters can effectively capture atomic environments and crystal symmetries to predict complex quantum properties.
