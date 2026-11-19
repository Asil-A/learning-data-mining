# DMML Cram Sheet — Lectures 1–6 (MCQ, tricky-question edition)

> ⚠️ = a likely trap. "SLIDE ERROR" = the slide itself is wrong or confusing, so know both versions.

---

## L1 — Introduction

**Definitions**
- **Data mining** = discovering patterns in large data sets. It sits where **ML, statistics and database systems** meet. ⚠️ DM is **the analysis step of KDD** (Knowledge Discovery in Databases), *not* the whole of KDD.
- **ML** = algorithms/statistical models that do a task **without explicit instructions**, relying on patterns and inference. It is a **subset of AI** and learns from "training data".
- ML is closely related to **computational statistics**. **Mathematical optimization** supplies its methods, theory and application domains.

**4 categories of ML model**
| Type | Training data |
|---|---|
| Supervised | training data provided |
| Unsupervised | **no** training data available |
| Semi-supervised | **some** training data |
| Reinforcement | ⚠️ **NOT part of this module** |

**Types of learning and their example algorithms**
| Task | Predicts | Algorithms (as listed on the slides) | Supervised? | Success measured |
|---|---|---|---|---|
| Classification | discrete class | Decision Tree, Neural Nets, **Logistic Regression** | Yes | **objectively**, on test data |
| Association | relations between features | **Apriori, FP-Growth** | no class specified | — |
| Clustering | groups | **K-means, DBScan** | **No** | ⚠️ **subjectively** |
| Regression / numeric prediction | number | Linear, Multiple Regression, **Time Series** | Yes | test data |

- ⚠️ Logistic **Regression** is listed under **classification**.
- Association vs classification: association can predict **any** attribute, and **more than one** at a time. That gives **far more** association rules than classification rules, so constraints are needed: **minimum coverage and minimum accuracy**.
- Numeric prediction is a "variant of classification where the class is numeric".

**Ethics**
- **85% of Americans** can be identified from **zip code + birth date + sex**, so anonymizing data is difficult.
- The same attribute can be unethical in one application (loans, insurance: sex, religion, race) and fine in another (**medical**). Proxy attributes are also a problem (area code ↔ race).
- "**Purely statistical arguments are never sufficient!**" Caveats must be attached to results.
- Questions to ask: who may access the data, why it was collected, what conclusions are legitimate, and whether resources are put to good use.

**KDD steps (in order):** learn the domain → create target data set (selection) → **cleaning and pre-processing (~60% of effort)** → reduction/transformation → choose DM function → choose algorithm → data mining → pattern evaluation and presentation → use the knowledge.

**CRISP-DM (IBM), 6 phases:** Business Understanding → Data Understanding → **Data Preparation (60% of time)** → Modelling → Evaluation → Deployment.
- ⚠️ **Splitting into training/test sets, sorting, deriving attributes, merging, sampling, aggregating, handling blanks** are all **Data Preparation**, *not* Modelling.
- Modelling: run several models with **default parameters**, then fine-tune, or go back to data preparation.
- Evaluation: judged against the **business success criteria** set at the start. Conclusions drawn there are called **"findings"**.
- Deployment: plan/monitor the deployment, **final report, project review**.
- It is both a **methodology** (phases, tasks, relationships) and a **process model** (the DM life cycle).
- **ASUM** = Analytics Solutions **Unified** Method (also IBM).
- Architecture (bottom to top): data sources → cleaning/integration/filtering → DB/warehouse server → DM engine (+ knowledge base) → pattern evaluation → GUI.
- Module tool: **Python on Google Colab**.

---

## L2/L3 — Instances, attributes, EDA, pre-processing

- **Instances = rows**, **attributes = columns**. Input is a single relation / **flat file**: a restricted form with **no relationships between objects**, but the most common in practice.
- **Multi-instance learning:** each example is a *set* of instances (same attributes); one or more of them may be responsible for the classification. Example: **drug activity prediction**.
- If the number of attributes varies, use an **"irrelevant value" flag**.

**Levels of measurement**
| Type | Order? | Equal units / differences? | True zero? | Allowed operations | Examples |
|---|---|---|---|---|---|
| Nominal | ✗ | ✗ | ✗ | **equality tests only**; count, **mode** | outlook (sunny/overcast/rainy) |
| Ordinal | ✓ | ✗ (no distance) | ✗ | comparison (<, >); ⚠️ no +/− | hot > mild > cool |
| Interval | ✓ | ✓ | ✗ **zero not defined** | **differences make sense; sum/product do not** | ⚠️ **°F temperature, year** |
| Ratio | ✓ | ✓ | ✓ | **all** math operations | distance, 1.5 m, 2.3 kg |

- ⚠️ The nominal/ordinal boundary is **not always clear** (e.g. "outlook").
- ⚠️ Whether there is an "inherent" zero **depends on scientific knowledge** (Fahrenheit knew no lower limit to temperature).
- "Nominal" comes from the Latin for **name**.

**Algorithm suitability**
- **Nominal:** Logistic Regression, Decision Trees, Association (Apriori, FP-Growth)
- **Ratio (easy distance metrics):** Neural Nets, Linear Regression, Clustering (K-Means, DBScan)

**Metadata** = data about data (background knowledge). It can **restrict the search space**. Examples: dimensional considerations, **circular orderings (compass degrees)**, **partial orderings (generalization/specialization)**.

**EDA:** the **first** step. It is done **before** building models and looks for patterns, relationships and anomalies.
**5 EDA steps:** Data Cleaning · Statistical Analysis · Data Visualization · **Detect Outliers** · Data Transformation.
- Descriptive stats: mean, median, mode, weighted mean, quartiles, SD, range, variance, IQR. Visual: histograms, boxplots, scatter plots, Q-Q plots.

**Outliers**
- They skew distributions and basic statistics. ⚠️ The ML community is **divided** on them.
- Remove an outlier if it is an **error** (or clamp/correct it). Don't remove it if it is **representative**, because the model then can't account for those cases.
- **Multiple linear regression is very sensitive** to outliers; some algorithms are robust.
- **Boxplot whiskers = ±1.5 × IQR** (IQR = Q3 − Q1). Points beyond them are *candidates* for removal. ⚠️ A **cluster of points right at the whisker** should typically **not** be removed.
- Plotting all features in one boxplot is only useful if they are normalized/standardized or on the same range.
- ⚠️ **Univariate** detection (boxplots) misses unusual **combinations** of variables. **Multivariate** detection uses **DBScan** (low-density = noise/outlier), and ⚠️ **DBScan expects standardized data**.
- Noise example: a mislabelled training example causes nearby queries to be misclassified.

**Scaling** (motivation: in KNN, one feature in [−1000, 1000] dominates Euclidean distance over features in [0, 1])
| | Normalisation (range) | Standardization |
|---|---|---|
| Result | rescaled to **[0, 1]** | **mean 0, SD 1** (standard Gaussian) |
| sklearn | **`MinMaxScaler`** | `StandardScaler` |
| Good for | weighted features (linear regression), **geometric distance (KNN)** | clustering, logistic regression, NNs, SVMs |
| Outliers | more sensitive | ⚠️ **less sensitive to outliers** |
- Standardization statistics are computed **per feature, on the training set**.
- `fit_transform` takes a NumPy array or a DataFrame and **returns a NumPy array**.
- Recommendation: **try both** and see which gives the best accuracy.

**Missing values**
- Causes: malfunctioning equipment, changed experimental design, collating datasets, measurement not possible.
- Fixes: delete the instance, delete the attribute, replace with mean/mode/another statistic.
- ⚠️ A missing value **may be significant in itself** (e.g. a test not done). Most schemes assume it isn't, so "missing" may need coding as an **additional value**.
- **Inaccurate values**: the root reason is that the "**data has not been collected for mining**". Errors can be **deliberate** (wrong zip codes). Watch for typos in nominal values, duplicates and stale data.
- ⚠️ **ALWAYS REPORT/RECORD THE ACTION YOU TOOK.** Look at the data from multiple perspectives.

**Categorical encoding**
- **Ordinal:** there is **no automatic** way to capture the order, because it is domain-specific. Map it **manually** with a dict and `df['Grade'].map(mapping)`, e.g. {'F':0,…,'A':4}.
- **Nominal with `OrdinalEncoder`:** turns values into integers, ⚠️ but this **imposes a false order** (alphabetical: Biology=0 < Chemistry=1 < Computing=2). That is the problem.
- **Fix: one-hot encoding**, one binary dummy column per unique value (`OneHotEncoder`, or **`pd.get_dummies`**, which converts **string columns only and leaves the others unchanged**).
- ⚠️ One-hot's disadvantage: many distinct values → many features → **long training / underperformance (dimensionality)**. Mitigation: **PCA**.

**Feature selection.** Benefits: understanding the data, **interpretability** (⚠️ explainability ≠ interpretability), accuracy, speed.
| Method | What it does | Pros | Cons |
|---|---|---|---|
| **Filter** | **ranks** attributes, **builds no model**; usually univariate (can be multivariate) | **fastest, scalable** | ⚠️ misses **dependencies/redundancies** between variables |
| **Wrapper** | **builds models** to evaluate feature subsets | **best results**, good for interpretability | ⚠️ **very computationally heavy** |
| **PCA** | creates **new** features as **linear combinations** of the originals | very quick, reduces space | ⚠️ new features are **not understandable** |
| **Deep Learning** | selects/creates features automatically | no manual work | **black box**, explanation methods not 100% reliable, **slow** |

**Choosing a method:**
- Need to **understand** the result?
  - **Yes** → a lot of features? **Yes → Filter**, **No → Wrapper**
  - **No** → need a **small model** or have a **small dataset**? **Yes → PCA**, **No → Deep Learning**

How I'd apply that to the slide scenarios (my reading, not given on the slides): gene study (70 patients × 20k fields) → Filter · cancer clinic (250 × 20) → Wrapper · smartwatch activity model (needs to be small) → PCA · 5M adverts for best accuracy / 10B weather samples → Deep Learning.

**PCA:** a dimensionality reduction method. Step 1 is **standardize**, then compute the **covariance matrix**. **Eigenvectors** = directions of **most variance** (the principal components). **Eigenvalues** = the coefficients attached to them, and components are **ranked by variance**. The best projection has the **largest variation**.

**Entropy** (disorder, randomness, uncertainty): H(X) = −Σ p log₂ p
- Boxes: A (4 red) = **0**, B (3 red, 1 blue) = **0.811**, C (2 red, 2 blue) = **1 bit (highest)**.
- Golf data (9 yes, 5 no): H = **0.940**. E(2,3) = **0.971**. E(4,0) = **0**.
- E(T, X) = Σ P(x)·E(x) → E(Golf, Outlook) = 5/14·0.971 + 4/14·0 + 5/14·0.971 = **0.693**.
- **Info gain = entropy before − entropy after split** → IG(Outlook) = **0.247**, Humidity 0.152, Windy 0.048, Temp 0.029. **Outlook becomes the root.**

---

## L4 — Decision Trees

- Used for **classification AND regression** (and multi-output). They are the building block of **Random Forest**.
- **Internal nodes** test an attribute. **Leaves** give a class, a set of classes, or a **probability distribution** (classification), or a **value** (regression).
- Usually an attribute is compared to a **constant**. Alternatives: compare two attributes, or use a function of attributes.
- ⚠️ **Nominal** attribute: one child per value, and it is **not tested more than once** on a path. **Numeric** attribute: < or > a constant, and it **may be tested several times**. Three-way split: integer → less/equal/greater; real → below/within/above.
- Construction: **top-down, recursive divide-and-conquer**. Stop when all instances have the same class. Goal: a **small, accurate** tree with the **purest** nodes.
- ⚠️ **Not all leaves need to be pure**: identical instances can have different classes, and splitting stops when the data can't be split further.
- Golf continuation: under Sunny, Gain(Humidity) = **0.971** (picked), Temp 0.571, Windy 0.020. Final tree: Outlook → Sunny: Humidity (High → No, Normal → Yes); Overcast → Yes; Rainy: Windy (False → Yes, True → No).

**Impurity measure wishlist:** (1) **0 when pure**, (2) **maximal when all classes are equally likely**, (3) **multistage property**.
⚠️ **Entropy is the ONLY function that satisfies all three.** The multistage property is "intellectually pleasing but **not strictly necessary**". Impurity is measured in **bits**.

| | Entropy | Gini |
|---|---|---|
| Used by | **C4.5, J48** (and ID3) | **CART** |
| Behaviour | **slightly more balanced trees**, measures randomness | **slightly faster → good default**, ⚠️ tends to **isolate the most frequent class** in its own branch |
| Formula | −Σ p log₂ p | **1 − Σ pₖ²** |
| Max | 1 bit (2 classes) | **1 − 1/k** (0.5 for 2 classes) |
- Usually they make **little difference** (similar trees).
- ⚠️ SLIDE ERROR / trap: the slide says "a Gini coefficient of 1 expresses maximal inequality". The impurity can **never actually reach 1**. Its max is **1 − 1/k**.
- Iris depth-2 left node: Gini = 1 − (0/54)² − (49/54)² − (5/54)² ≈ **0.168**.
- Gini example (x₁ = 1..8, y = 0,0,0,1,1,1,1,1): parent Gini = **15/32**. Split at 3.5 → ΔGini **15/32** (both children pure, **best**). Split at 4.5 → ΔGini **9/32**.

**Highly-branching attributes (e.g. ID code):** each subset has 1 instance, so entropy is 0 and **IG is maximal (0.940)**. ⚠️ **Information gain is biased toward attributes with many values**, which leads to **overfitting** and **data fragmentation**.

**CART**
- Always **binary splits**: it finds the feature k and threshold tₖ that give the purest subsets **weighted by size**.
- Cost: **J(k, tₖ) = (m_left/m)·G_left + (m_right/m)·G_right**.
- **Greedy**: optimal split at each level; it does **not** look several levels ahead. The result is reasonably good but **not guaranteed optimal**.
- It stops at **max_depth** or when no split **reduces impurity**.
- Detailed version uses a **chi-squared test** to choose the variable and then minimizes Gini, then **prunes**.
- **Regression CART:** minimizes **MSE** (not entropy/Gini). The leaf predicts the **average target value** of the instances in that region.

**Complexity**
- Training: **O(n × m log m)** (n features, m instances). **Presort** speeds up *small* sets (a few thousand instances) but **slows large ones considerably**.
- Prediction: **O(log₂ m)** nodes. ⚠️ **Independent of the number of features**, so very fast.

**Class probabilities:** go to the leaf and return the **ratio of training instances of class k** in it. Petal 5 cm long, 1.5 cm wide → Setosa **0%** (0/54), Versicolor **90.7%** (49/54), Virginica **9.3%** (5/54).

**Parametric vs non-parametric**
- DTs are ⚠️ **non-parametric**: the number of parameters is **not fixed before training**, so the tree can stick closely to the data → risk of **overfitting**.
- Linear models are **parametric**: fixed parameters, limited freedom → less overfitting, **more underfitting**.

**Hyperparameters / regularization**
- A hyperparameter is set **before** learning. Restricting the model's freedom = **regularization**.
- `max_depth`, `min_samples_split` (min samples a node needs **before it can split**), `min_samples_leaf` (min samples **a leaf** must have).
- ⚠️ **Default hyperparameters = no restrictions → overfitting**. Example: min_samples_leaf = 4 generalizes better.
- Regression trees also overfit without regularization.

**Decision boundaries:** ⚠️ **perpendicular / axis-aligned**, so trees are **sensitive to rotation** of the training set. A rotated dataset (45°) still fits perfectly but gives a convoluted boundary that **generalizes badly**.

**Pros:** easy to understand, **handles missing data**, compact (#nodes ≪ #instances), fast testing O(depth), handles non-linear / discontinuous / non-normal data, nominal + numeric, large datasets, non-parametric.
**Cons:** greedy (may not find the best tree), ⚠️ **no extrapolation**, perpendicular boundaries, rotation-sensitive, **needs pruning**, ⚠️ **can't handle complicated relationships between features**, ⚠️ **must regrow the tree when adding instances**.

**Pruning**
- **Prepruning:** stop early when information becomes unreliable. Based on a **statistical significance test**, most popularly **chi-squared**. ID3 (Quinlan) used chi-squared **plus** info gain.
  - ⚠️ Danger is **early stopping**. Classic example: the **XOR/parity problem**. No single attribute is significantly associated with the class, so prepruning **won't even expand the root**. But XOR problems are **rare** in practice, and prepruning is **faster**.
- **Postpruning:** build the full tree, then prune. ⚠️ **Preferred in practice.** The full tree shows all attribute interactions, though some subtrees may be chance effects.
  - **Subtree replacement:** **bottom-up**; a subtree is considered only after all its subtrees.
  - **Subtree raising:** delete a node and redistribute instances. ⚠️ **slower** than replacement.
  - Strategies: error estimation, significance testing, **MDL principle**.
- Prune only if it does **not increase the estimated error**. ⚠️ **Training error is NOT a useful estimator** (it would give almost no pruning).
  - Alternative: a **hold-out set** → "**reduced-error pruning**".
- **C4.5's method:** a confidence interval from the training data, a **Bernoulli-process** heuristic (shaky statistical assumptions).
  - Default **c = 25% → z = 0.69**. f = training error, N = instances at the leaf.
  - Subtree estimate = **weighted sum** of the leaves' estimates. Example: children 0.47, 0.72, 0.47 combined 6:2:6 → **0.51**. Parent e = **0.46** < 0.51 → **prune**.

---

## L5 — Probabilistic classification (1R, Naïve Bayes, SVM)

**"Simplicity first"**: simple algorithms often work very well, and success **depends on the domain**.

**1R**
- Learns a **1-level decision tree**: rules that all test **one** attribute, the one with the **lowest classification error**.
- For each attribute and each value, assign the **majority class**; error = instances not in the majority class. Pick the attribute with the lowest total error.
- Weather data total errors: Outlook **4/14**, Temp 5/14, Humidity **4/14**, Windy 5/14 (* = tie, broken arbitrarily).
- ⚠️ **Missing value = treated as a separate attribute value** (contrast with Naïve Bayes below).
- Numeric attributes: sort the values and place breakpoints where the **class changes** (minimizes error).
  - ⚠️ **Very sensitive to noise**: one mislabelled instance makes its own interval, and a **timestamp attribute gets zero errors** (overfitting).
  - Fix: a **minimum number of instances in the majority class** per interval (**3** in the lecture example; **Holte used 6**).
  - After the fix: Humidity ≤ 82.5 → Yes, 82.5–95.5 → No, > 95.5 → Yes. The **temperature boundary is 77.5**. Humidity's total error becomes **3/14** (the best).
- **Holte (1993):** "Very simple classification rules perform well on most commonly used datasets". **16 datasets**, cross-validation. 1R was **not much worse** than complex trees. ⚠️ It does **worse on more recent, sophisticated** benchmarks.

**Naïve Bayes**
- The "opposite" of 1R: **uses all attributes**. Two assumptions: attributes are **equally important** and **statistically independent given the class**.
- ⚠️ The independence assumption is **almost never correct**, yet NB works well, is **easy to implement and very fast**.
- **Bayes' rule:** P(H|E) = P(E|H)·P(H) / P(E).
  - ⚠️ **SLIDE ERROR:** the slide labels P(H|E) as "a priori" and P(H) as "a posteriori". Correct: **prior = P(H)** (before evidence); **posterior = P(H|E)** (after evidence). The slide's *descriptions* ("before / after evidence is seen") are right; only the formulas are swapped.
- With the naïve assumption: P(H|E) = P(E₁|H)·P(E₂|H)·…·P(Eₙ|H)·P(H) / P(E). **P(E) cancels out** when normalizing.
- **Worked example** (Sunny, Cool, High, True):
  - yes = 2/9·3/9·3/9·3/9·9/14 = **0.0053**
  - no = 3/5·1/5·4/5·3/5·5/14 = **0.0206**
  - → P(yes) = **20.5%**, P(no) = **79.5%** → **No**.
- **Zero-frequency problem:** one zero conditional probability makes the whole posterior **0** regardless of the other values.
  - Fix: the **Laplace estimator**, **add 1** to every attribute-value/class count.
  - Bonus: it **stabilizes estimates from small samples**.
  - Generalization: add a constant μ, e.g. (2 + μ/3)/(9 + μ) or (2 + μp₁)/(9 + μ). ⚠️ The weights pᵢ **don't need to be equal but must sum to 1**.
- **Missing values:**
  - ⚠️ Training: the instance is **not counted** for that attribute-value/class combination.
  - ⚠️ Classification: the attribute is **omitted from the calculation**.
  - Example (?, Cool, High, True): yes 0.0238, no 0.0343 → **41% / 59%**.
- **Numeric attributes:** assume a **normal (Gaussian)** distribution per class, with parameters **mean and SD** (SD uses **N − 1**). Example: f(temp = 66 | yes), μ = 73, σ = 6.2 → **0.0340**. Example day (Sunny, 66, 90, True) → **25% yes / 75% no**.
  - ⚠️ **Densities can be > 1**, so they are not probabilities, but they **integrate to 1**. In likelihoods you can **treat them like probabilities**. P(x − ε/2 ≤ X ≤ x + ε/2) ≈ ε·f(x).
  - Missing values during training are **excluded** from the mean/SD.
- **Why NB works despite violated independence:** classification only needs the **maximum probability assigned to the correct class**, not accurate probabilities.
- ⚠️ **Redundant/identical attributes cause problems.** For non-normal numeric attributes, use **kernel density estimators**.

**SVM**
- **Kernel methods (big idea):** **embed the data into a suitable vector space**, then use **linear methods** there.
  - Kernel methods are **resistant to overfitting and the curse of dimensionality**, **modular**, can incorporate **domain knowledge**, and are used for classification, clustering, novelty detection and ranking.
- **Hyperplane:** w₀ + **w**·**d** = 0 (> 0 above, < 0 below). The best decision boundary is the **hyperplane**; the extreme points that define it are the **support vectors**.
- Negative class outputs **≤ −1**, positive class **≥ +1**. The gap between −1 and +1 is the **margin**.
- **Training = a constrained quadratic optimization problem**, defined by (1) a **set of constraints** and (2) an **optimization criterion**.
  - Constraint: **tᵢ·(w₀ + w·d) ≥ 1** (tᵢ ∈ {−1, +1}).
  - Goal: **maximize the margin 2/‖w‖** subject to the constraint. Distance from the margin edge to the boundary = 1/‖w‖, and **‖w‖ = Euclidean norm**.
- **Model:** M(q) = Σ over the support vectors (tᵢ·α[i]·(dᵢ·q) + w₀). There is one **α per support vector**; the αs are **Lagrange multipliers**.
- **Not linearly separable** → **basis functions φ**. There are usually **more basis functions than features**, which moves the data to a **higher-dimensional space** where a linear separator is *expected* to exist. Example: z = x² + y² (a circle becomes a plane).
- ⚠️ The **kernel trick:** the dot product φ(dᵢ)·φ(q) is **computationally expensive**. A **kernel function** on the *original* features gives the **same result more cheaply**: k(x, z) = ⟨φ(x), φ(z)⟩. You don't need to know φ, **as long as you can prove such a mapping exists**.
- **Kernels:**
  - **Linear:** d·q + c
  - **Polynomial:** (d·q + 1)ᵖ
  - **Gaussian RBF:** exp(−γ‖d − q‖²), where **γ is manually tuned**

---

## L6 — Clustering

**Basics**
- A cluster = objects **similar to each other** within the cluster and **dissimilar** to objects in other clusters.
- Clustering = ⚠️ **unsupervised classification**, with **no predefined classes**.
- Uses: a **stand-alone tool** to get insight into the data distribution, OR a **pre-processing step** for other algorithms.
- Applications: pattern recognition, spatial/GIS (thematic maps), image processing, economics/**market research**, WWW (document classification, weblog access patterns).
- Examples: marketing (customer groups → targeted programs), insurance (motor policy holders with **high average claim cost**), city planning (house type, value, location), earthquakes (**epicentres clustered along continent faults**).
- **Good clustering = HIGH intra-class similarity + LOW inter-class similarity.** ⚠️ Don't mix up intra (within) and inter (between).
- Quality depends on **both the similarity measure and its implementation**, and on the ability to discover hidden patterns.
- Requirements: scalability; different attribute types; **noise and outliers**; **clusters of arbitrary shape**; **minimal domain knowledge needed for input parameters**; **insensitive to the order of input records**; user-specified constraints; interpretability and usability.
- Similarity is expressed as a **distance function d(i, j)**. Its definition differs for interval, Boolean, categorical, ordinal and ratio variables. Variables can be **weighted**. ⚠️ "Similar enough" is **highly subjective**.

**Types of data**
- **Interval-scaled:** standardize using the **mean absolute deviation** s_f = (1/n)Σ|x − m_f|, then the z-score z = (x − m_f)/s_f.
  - ⚠️ The mean absolute deviation is **more robust (to outliers) than the standard deviation**, because deviations aren't squared.
- **Minkowski** distance: d = (Σ|xᵢ − xⱼ|^q)^(1/q). **q = 1 → Manhattan**, **q = 2 → Euclidean**.
  - Metric properties: d ≥ 0; d(i, i) = 0; **symmetric** d(i, j) = d(j, i); **triangle inequality** d(i, j) ≤ d(i, k) + d(k, j).
- **Binary variables** (contingency table: a = 1/1, b = 1/0, c = 0/1, d = 0/0):
  - **Symmetric** binary → **simple matching**: (b + c)/(a + b + c + d)
  - **Asymmetric** binary → **Jaccard**: (b + c)/(a + b + c). ⚠️ It **ignores d** (0/0 negative matches don't count as similarity).
  - Example: **gender (symmetric) is excluded**; Y/P = 1, N = 0. Results: d(jack, mary) = **0.33**, d(jack, jim) = **0.67**, d(jim, mary) = **0.75**. Jack and Mary are the most similar.
- **Nominal** (more than 2 states): (1) simple matching **d = (p − m)/p** (m = number of matches, p = total number of variables), or (2) **one binary variable per state**.
- **Ordinal:** replace each value by its **rank r ∈ {1..M}**, map it to [0, 1] with **z = (r − 1)/(M − 1)**, then treat it as **interval-scaled**.

**Major techniques**
- **Partitioning\***: construct partitions and evaluate them by some criterion.
- **Hierarchical\***: a hierarchical decomposition.
- **Density-based\***: connectivity and density functions.
- **Grid-based**: a **multiple-level granularity structure**.
- **Model-based**: hypothesize a model for each cluster and find the best fit.
- (\* = the ones the module focuses on)

**Partitioning** (database of n objects → k clusters, k given)
- ⚠️ The **global optimum needs exhaustive enumeration** of all partitions, which is infeasible, so **heuristics** are used: k-means and k-medoids.
- **k-means:** a cluster is represented by its **centre (mean)**.
- **k-medoids / PAM:** a cluster is represented by **one of the actual objects** in it (the medoid).

**K-means (4 steps)**
1. Partition into k non-empty subsets.
2. Compute seed points = **centroids** (mean points).
3. Assign each object to the **nearest** seed.
4. Go back to step 2 until **no new assignments**.

- **Strengths:**
  - Relatively efficient: **O(tkn)** (t iterations, k clusters, n objects; k, t ≪ n).
  - ⚠️ "**Often terminates at a local optimum**" is listed on the slide as a *strength* bullet.
  - The **global** optimum can be found with **deterministic annealing and genetic algorithms**.
- **Weaknesses:**
  - Only works where a **mean is defined** (so not categorical data).
  - **k must be specified in advance**.
  - ⚠️ **Can't handle noise and outliers**.
  - ⚠️ **Can't find non-convex clusters**.
- **Variants** differ in the initial means, the dissimilarity calculation, and how means are calculated.
  - **Categorical data → k-modes** (modes instead of means, new dissimilarity measures, **frequency-based** mode updates).
  - **Mixed categorical + numeric → k-prototype.**

**PAM (Partitioning Around Medoids)**
- Pick k medoids **arbitrarily** (randomly).
- For each pair of non-selected object h and selected medoid i, compute the **total swapping cost TC_ih = Σⱼ C_jih**.
- ⚠️ **If TC_ih < 0** (it improves the clustering), **i is replaced by h**.
- Assign each non-selected object to its most similar medoid, and repeat until there's no change.
- ⚠️ **Works well for small data sets, but does NOT scale** to large ones.

**Hierarchical**
- Uses the **distance matrix**. ⚠️ It **does NOT need k** as an input, **but needs a termination condition**.
- **AGNES (Agglomerative Nesting):** **bottom-up**. Uses **single-link** and the dissimilarity matrix, merges the **least dissimilar** nodes, continues in a **non-descending** fashion, and eventually **all nodes are in one cluster**.
- **DIANA (Divisive Analysis):** **top-down**, the inverse of AGNES; eventually **each node is its own cluster**.
- **Dendrogram** = a tree of nested partitions. **Cut it at the desired level** and each **connected component** forms a cluster.
- ⚠️ Weaknesses (agglomerative): **does not scale**, at least **O(n²)**; and it **can never undo** what was done previously.
- Integration with distance-based methods:
  - **BIRCH**: uses a **CF-tree** and incrementally adjusts sub-cluster quality.
  - **CURE**: selects **well-scattered points** from a cluster and **shrinks them towards the centre** by a fraction.
- **Clustering Feature CF = (N, LS, SS)**: N = number of points, LS = **linear sum** Σxᵢ, SS = **square sum** Σxᵢ².
  - Example: points (3,4), (2,6), (4,5), (4,7), (3,8) → **CF = (5, (16,30), (54,190))**.
  - CF tree example parameters: B = 7 (branching), L = 6 (leaf entries).
- **Drawbacks of square-error (distance-based) methods:** only **one point represents a cluster**; they're good only for **convex, similar size and density** clusters, and only if k can be reasonably estimated.

**Density-based (DBSCAN)**
- Features: **arbitrary-shape** clusters, **handles noise**, **one scan**, ⚠️ **needs density parameters as the termination condition**.
- Clusters = **dense regions separated by low-density regions**. A cluster = a **maximal set of density-connected points**.
- **ε-neighbourhood** = objects within radius ε of an object. **High density** = the ε-neighbourhood contains **at least MinPts** objects.
- The two parameters: **ε (Eps)** and **MinPts**.
- Three **mutually exclusive** point types:
  - **Core:** at least MinPts points within Eps; points in the **interior** of a cluster. ⚠️ One slide says "**more than** MinPts", the density-definition slide says "**at least**". "At least" (≥) is the standard definition.
  - **Border:** **fewer than MinPts** within Eps, **but inside the neighbourhood of a core point**.
  - **Noise/outlier:** neither core nor border.
- **Directly density-reachable:** q is directly density-reachable from p **if p is a core point** and q is in p's ε-neighbourhood.
  - ⚠️ It is **NOT symmetric**. If q is a border point, p is **not** directly density-reachable from q.
- **Density-reachable (indirect):** a **chain** of directly density-reachable points (q → p₁ → p₂ → p). p is density-reachable from q, **but q is not density-reachable from p**.
- **Algorithm:** for each unclassified object o: **if o is a core object → collect everything density-reachable from it into a new cluster; else → NOISE**. (Example params: ε = 2 cm, MinPts = 3.)
- **Choosing Eps and MinPts:** the **k-distance plot**. Points in a cluster have their k-th nearest neighbour at roughly the same distance; **noise points have it farther away**. **Plot the sorted k-th nearest-neighbour distances** and pick Eps at the **sharp rise (knee)**.
- ✅ Works well: **resistant to noise**, handles clusters of **different shapes and sizes**.
- ❌ Fails: ⚠️ **varying densities**; ⚠️ **sensitive to parameters** (Eps 9.92 vs 9.75 gives very different results), so the right parameters are hard to find.
- Recall from L2: DBSCAN is also used for **multivariate outlier detection** and expects **standardized** data.

---

## Fast-recall traps (read last thing before the exam)
1. DM = the **analysis step** of KDD. Pre-processing ≈ **60%** of effort.
2. Train/test split happens in **Data Preparation** (CRISP-DM).
3. Clustering success is measured **subjectively**; classification **objectively**.
4. **Year and °F = interval** (no true zero). Interval allows **differences, not ratios/sums**.
5. Nominal → **equality only**; statistics: **count, mode**.
6. OrdinalEncoder on nominal data → **false order**. Use **one-hot** instead.
7. Standardization is **less** outlier-sensitive than normalization. Normalization → **[0, 1]**.
8. Boxplot whiskers **1.5 × IQR**. Univariate methods miss **combinations**. Use DBScan (standardized data).
9. Filter = **no model**, fast, misses redundancy. Wrapper = **builds models**, slow, best results.
10. PCA: **standardize first**. Eigenvectors = **max-variance directions**. Not interpretable.
11. Entropy is the **only** measure meeting all 3 properties. Gini is **faster/default** and isolates the **most frequent class**.
12. Info gain is biased to **many-valued** attributes (ID code: IG = 0.940).
13. CART = **binary splits, Gini, greedy**. Regression tree minimizes **MSE** and predicts the **mean**.
14. DT prediction cost **O(log₂ m)**, **independent of the number of features**.
15. DT = **non-parametric**. Defaults → **overfit**.
16. **Postpruning preferred**. Prepruning is faster but fails on **XOR**. Subtree raising is **slower**.
17. Never use **training error** to decide pruning. C4.5: **c = 25%, z = 0.69**.
18. 1R: missing = **separate value**. NB: missing = **omitted/not counted**.
19. Laplace = **add 1**. Weights don't need to be equal, but **must sum to 1**.
20. Densities can be **> 1**.
21. Prior = **P(H)**, posterior = **P(H|E)** (the slide has them swapped!).
22. SVM margin = **2/‖w‖**. α = **Lagrange multipliers**. Training is a **constrained quadratic optimization**.
23. Kernel trick avoids the **expensive dot product in the higher-dimensional space**.
24. Good clustering = **high intra**, **low inter** similarity.
25. Asymmetric binary → **Jaccard** (drops d). Symmetric binary → **simple matching**.
26. The mean absolute deviation is **more robust** than SD. Ordinal → **(r − 1)/(M − 1)**.
27. k-means: **O(tkn)**, **local optimum**, needs **k**, fails on **outliers/noise and non-convex** shapes. Categorical → **k-modes**; mixed → **k-prototype**.
28. PAM/k-medoid = a **real object**. Swap if **TC < 0**. Doesn't scale.
29. Hierarchical: **no k**, but needs a **termination condition**. **O(n²)**, **can't undo**. AGNES = bottom-up/single-link; DIANA = top-down.
30. CF = (N, **linear sum**, **square sum**). BIRCH → CF-tree. CURE → scattered points shrunk to the centre.
31. DBSCAN: params **Eps and MinPts**. Core/border/noise. Density-reachability is **asymmetric**. Fails with **varying density**. Use the **k-distance plot** to pick Eps.
