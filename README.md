# Predicting Battery Material Properties with Machine Learning

Can a computer predict how a battery material will behave just from what it's made of and how its atoms are arranged, without running expensive lab experiments or quantum simulations? This project tests that idea on lithium battery cathode materials.

---

## 1. Aim of the project

Batteries in phones and electric cars use cathode materials made of **lithium (Li)**, **a transition metal (TM)** such as iron, cobalt or nickel, and **oxygen (O)**. Before anyone makes a new material, it helps to know two things about it:

| Property | What it tells us | Why it matters for batteries |
|---|---|---|
| **Formation energy** | How stable the material is: how much energy is released when its elements combine | An unstable material can break down while the battery is being charged and discharged |
| **Band gap** | How easily electrons can move through the material: 0 eV means it conducts like a metal, a large value means it behaves like an insulator | A cathode needs electrons to flow, so conductivity affects how fast a battery can charge |

The project set out to answer three questions:

1. Can classical machine-learning models (**Random Forest, XGBoost and CatBoost**) predict formation energy and band gap from a material's chemical makeup?
2. Band gap turned out much harder to predict than formation energy. **Does it help to give the models information about the transition metal specifically?**
3. Can a **graph neural network (CGCNN)**, which looks at the actual 3D arrangement of atoms, predict band gap better than the classical models?

---

## 2. The data

- **Source:** [Materials Project](https://materialsproject.org/), a free database of material properties computed by simulation.
- **Materials:** 5,169 compounds containing lithium, oxygen and one of seven transition metals: **Fe, Co, Ni, Mn, Ti, V or Cr**.
- **Filter:** only stable or near-stable materials (energy above hull ≤ 0.05 eV/atom) with 3 to 6 different elements.
- **Fair testing:** the data was split 90% for training and 10% for testing (4,653 and 516 materials). The split was grouped by chemical formula, so a formula in the test set never appears in training. The models are tested on chemistry they have never seen, rather than on near-copies of training examples.

<p align="center"><img src="images/bandgap_distribution.png" width="850"></p>

About 16% of the materials are metals (band gap = 0), 68% are semiconductors and 16% are insulators.

---

## 3. Approach

### Notebook 1: `Battery_Research.ipynb` (classical machine learning)

Classical models can't read a chemical formula directly, so each material first has to be turned into a list of numbers that describe it (**features**).

1. **Describe each material with numbers.**
   - *Composition features* (made with the `matminer` library's Magpie data): for 13 elemental properties such as electronegativity, atomic radius, number of valence electrons and melting point, take the average across all atoms in the formula, the largest value, and how much the values vary.
   - *Structure features*: density, volume, volume per atom, crystal system, space group and symmetry.
2. **Train three tree-based models:** Random Forest, XGBoost and CatBoost. Each model's settings were tuned automatically with Optuna over 100 trials.
3. **Predict formation energy**, then **band gap**, and measure how far the predictions are from the true values.
4. **Add transition-metal features (band gap only).** After the first band gap results were weak, four new features were added that describe *only the transition metal* in each material:
   - `tm_oxidation_state`: the charge the transition metal carries (for example, Fe²⁺ or Fe³⁺)
   - `tm_d_valence`: the number of outer "d" electrons the metal has
   - `tm_electronegativity`: how strongly the metal attracts electrons
   - `tm_atomic_radius`: the size of the metal atom
5. **Explain the models** with feature-importance plots and SHAP analysis, and **estimate uncertainty** by training 200 slightly different CatBoost models and checking how much their predictions disagree.

### Notebook 2: `CGCNN_Bandgap_Prediction.ipynb` (graph neural network)

Instead of summarising a material as a list of averages, a **Crystal Graph Convolutional Neural Network (CGCNN)** looks at the crystal itself.

1. **Turn each crystal into a graph.** Each atom becomes a point (a *node*), and each pair of neighbouring atoms within 8 Å gets a connection (an *edge*), up to 12 neighbours per atom. Each atom carries 19 numbers describing it (atomic number, electronegativity, electrons in each orbital, whether it is a transition metal, and so on), and each connection records the distance between the two atoms.
2. **Let atoms "talk" to their neighbours.** Over 3 rounds of *message passing*, each atom updates what it knows using information from the atoms around it. A gate in the network learns which neighbours matter most.
3. **Combine all the atoms** into a single summary of the crystal and predict its band gap.
4. Train for up to 200 epochs, stopping early if the model stops improving, on the same 90/10 split as the classical models so the results can be compared directly.

---

## 4. Results

**How to read the scores**

- **R²**: how much of the variation in the true values the model captures. 1.0 is perfect; 0 means the model does no better than guessing the average every time. *Higher is better.*
- **MAE (mean absolute error)**: on average, how far off a prediction is, in eV (for formation energy, eV per atom). *Lower is better.*
- **MSE (mean squared error)**: like MAE, but large mistakes count extra. *Lower is better.*

All scores below are on the **unseen test materials**.

<p align="center"><img src="images/model_comparison.png" width="850"></p>

### 4.1 Formation energy: composition alone is enough

| Model | R² | MSE | MAE (eV/atom) |
|---|---|---|---|
| Random Forest | 0.972 | 0.0050 | 0.041 |
| XGBoost | 0.980 | 0.0037 | 0.039 |
| **CatBoost** | **0.988** | **0.0021** | **0.030** |

All three models predict formation energy almost perfectly, with typical errors of only about 0.03 to 0.04 eV/atom.

**Why it works so well:** formation energy describes how strongly the atoms in a material bond together overall. That depends mostly on *which* elements are present and in what proportions, and averaging the properties of the elements captures it well. A material's overall stability behaves much like a weighted sum of its ingredients.

<p align="center"><img src="images/formation_energy_cb.png" width="650"><br><em>CatBoost feature importance for formation energy. A few composition features, mostly about the d-electrons of the metals, carry most of the prediction.</em></p>

### 4.2 Band gap without transition-metal features: a big drop

| Model | R² | MSE | MAE (eV) |
|---|---|---|---|
| Random Forest | 0.358 | 1.025 | 0.765 |
| **XGBoost** | **0.478** | **0.834** | **0.690** |
| CatBoost | 0.433 | 0.906 | 0.734 |

Using the same kind of features, the models capture less than half of what decides the band gap, and they are off by about 0.7 eV on average. That is a large error when the band gaps in this dataset only range from 0 to about 5 eV.

### 4.3 Band gap with transition-metal features: a big improvement

| Model | R² | MSE | MAE (eV) |
|---|---|---|---|
| Random Forest | 0.680 | 0.511 | 0.521 |
| XGBoost | 0.668 | 0.530 | 0.550 |
| **CatBoost** | **0.715** | **0.455** | **0.528** |

Adding just four features about the transition metal raised R² from **0.36–0.48 to 0.67–0.72**, and cut the average error from about 0.7 eV to about 0.5 eV.

### 4.4 Why the transition-metal features made such a difference

The clearest evidence is in **what the models paid attention to** before and after the change. The plots below show the 20 features each model relied on most.

<table>
<tr>
<th width="50%">Without TM features</th>
<th width="50%">With TM features</th>
</tr>
<tr>
<td><b>Random Forest</b><br><img src="images/bandgap_noTM_rf.png"></td>
<td><b>Random Forest</b><br><img src="images/bandgap_TM_rf.png"></td>
</tr>
<tr>
<td><b>XGBoost</b><br><img src="images/bandgap_noTM_xgb.png"></td>
<td><b>XGBoost</b><br><img src="images/bandgap_TM_xgb.png"></td>
</tr>
<tr>
<td><b>CatBoost</b><br><img src="images/bandgap_noTM_cb.png"></td>
<td><b>CatBoost</b><br><img src="images/bandgap_TM_cb.png"></td>
</tr>
</table>

**What the plots show**

- **Without TM features (left)**, the models leaned on indirect clues such as **density**, **energy above hull**, **atomic weight** and **magnetic moment (GSmagmom)**. No single feature stood out strongly, and the three models didn't agree on what mattered. The models were guessing from indirect evidence.
- **With TM features (right)**, all three models independently ranked **`tm_oxidation_state`** as the most important feature by a wide margin, followed by **`tm_d_valence`** and **`tm_electronegativity`**. The three models now agree, which is a strong sign they found a real physical signal rather than noise.

**The physics behind it, in plain terms**

In these lithium–metal–oxide materials, the band gap is controlled mainly by the **transition metal's outer "d" electrons**. Lithium mostly gives away its electron, and oxygen's electrons sit in a fairly predictable place. The transition metal is the part whose electrons decide whether the material conducts.

The **oxidation state** sets how many d electrons the metal has left. For example, iron as Fe²⁺ has 6 d electrons, while Fe³⁺ has 5. The number of d electrons, and how they fill the available energy levels, largely decides whether there is a gap for electrons to jump across and how big it is. The same metal can therefore give very different band gaps depending on its charge.

The original composition features **averaged across every atom in the formula**. In a material like Li₅FeO₄, only 1 of the 10 atoms is the transition metal, so its information is heavily diluted by lithium and oxygen. Averaging also can't show the oxidation state at all: it depends on how electrons are shared between atoms, not on which atoms are present. The new features **pulled the transition metal's information out on its own**, so the models could see it directly.

**Why formation energy didn't need this:** stability depends on the *whole* material's bonding, so averages describe it well. Band gap depends on the electronic behaviour of *one specific type of atom*, so averages hide the most important information.

### 4.5 Band gap with the graph neural network (CGCNN)

| Model | R² | MSE | MAE (eV) |
|---|---|---|---|
| Best classical model without TM features (XGBoost) | 0.478 | 0.834 | 0.690 |
| Best classical model with TM features (CatBoost) | 0.715 | 0.455 | 0.528 |
| **CGCNN** | **0.770** | **0.367** | **0.436** |

CGCNN gave the **best band gap predictions of all the models**. Compared with the best classical model, it raised R² from 0.715 to 0.770 and cut the average error by about 0.09 eV, a reduction of roughly 17%.

<p align="center"><img src="images/cgcnn_parity.png" width="550"><br><em>CGCNN predicted vs. actual band gap on the test set. Points on the red dashed line are perfect predictions.</em></p>

**Why CGCNN does better**

- **It sees the structure, not just the ingredients.** Two materials with the same formula can have different band gaps if their atoms are arranged differently. Averaged composition features can't tell them apart; a crystal graph can.
- **It sees each atom's local surroundings.** A transition metal's d electrons are strongly affected by how many oxygen atoms surround it and how close they are. Message passing lets each metal atom "learn" about its neighbours directly.
- **It doesn't need hand-made features.** The classical models only improved after the transition-metal features were designed by hand. CGCNN was given basic facts about each atom (including its d-electron count and a flag for whether it is a transition metal) and learned which patterns matter by itself.
- **It overfits less.** The classical models scored R² of about 0.93–0.97 on training data but about 0.7 on test data, meaning they partly memorised the training set. CGCNN scored 0.88 on training and 0.77 on test, a much smaller gap, so it generalises better to new materials.

**Where it still struggles:** the parity plot shows two problem areas. Many true metals and small-gap materials (actual band gap near 0 to 1 eV) are predicted with a gap, and some materials with real gaps are predicted as metals (the points along the bottom). The boundary between "conducts" and "doesn't conduct" remains the hardest part to get right.

---

## 5. Key takeaways

1. **Formation energy is easy.** Simple composition features let classical models predict it almost perfectly (R² ≈ 0.99).
2. **Band gap is hard, and it depends on the transition metal.** Without features that isolate the transition metal, the classical models reached R² of only 0.36–0.48.
3. **Domain knowledge pays off.** Adding four transition-metal features, especially the **oxidation state**, raised R² to about 0.72. The feature-importance plots show all three models switching to rely on this physically meaningful signal.
4. **Structure matters.** CGCNN, which reads the 3D arrangement of atoms, gave the best band gap predictions (R² = 0.77, MAE = 0.44 eV) without any hand-designed transition-metal features.

---

## 6. Repository contents

| File | What it contains |
|---|---|
| `Battery_Research.ipynb` | Data download, exploration, feature engineering, Random Forest / XGBoost / CatBoost for formation energy and band gap, SHAP analysis, uncertainty estimates |
| `CGCNN_Bandgap_Prediction.ipynb` | Converting crystal structures (CIF files) to graphs, and building, training and evaluating the CGCNN band gap model |
| `images/` | Figures used in this README |

## 7. Tools used

Python · pymatgen · matminer · Materials Project API · scikit-learn · XGBoost · CatBoost · Optuna · SHAP · PyTorch · PyTorch Geometric · Google Colab

## 8. Possible next steps

- Hold out a separate validation set for tuning and early stopping, so the test set is used only once for the final scores.
- Add the transition-metal features to the CGCNN atoms and see whether combining both approaches does better.
- Treat "metal vs. non-metal" as its own classification step before predicting the size of the gap, since this boundary causes the largest errors.
- Try newer graph models (for example, ALIGNN or MEGNet) that also account for bond angles.
