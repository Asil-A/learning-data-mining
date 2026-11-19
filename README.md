# DMML MCQ Practice Quiz — Lectures 1–6

109 tricky multiple-choice questions on Data Mining & Machine Learning, Lectures 1–6.

- **Interactive version:** open `index.html` in a browser (download it, or use GitHub Pages if the repo is public). It has shuffling, a per-lecture filter, scoring and a 'retry wrong answers' mode.
- **Here:** answers are hidden. Click ▶ to reveal each one.

## Contents
- [L1 Introduction](#l1-introduction) (11)
- [L2 EDA & Pre-processing](#l2-eda--pre-processing) (17)
- [L3 Pre-processing 2 & Feature Selection](#l3-pre-processing-2--feature-selection) (15)
- [L4 Decision Trees](#l4-decision-trees) (21)
- [L5 Probabilistic Classification & SVM](#l5-probabilistic-classification--svm) (21)
- [L6 Clustering](#l6-clustering) (24)

## L1 Introduction

**1. Which type of learning is explicitly NOT part of the data mining module?**  
A) Supervised learning  
B) Unsupervised learning  
C) Reinforcement learning  
D) Semi-supervised learning  

<details><summary>Answer</summary>

**C) Reinforcement learning** — Reinforcement learning is listed as one of the 4 ML categories but marked '(this will not be part of the DM module)'.

</details>

**2. In CRISP-DM, the conclusions or inferences drawn from the models and from the data mining process are called:**  
A) Findings  
B) Knowledge bases  
C) Patterns  
D) Deliverables  

<details><summary>Answer</summary>

**A) Findings** — Evaluation phase: '...These are known as findings.'

</details>

**3. In the lecture's 'Types of Learning' list, Logistic Regression appears under:**  
A) Clustering  
B) Classification  
C) Association  
D) Regression (numeric prediction)  

<details><summary>Answer</summary>

**B) Classification** — Classification examples: Decision Tree, Neural Networks, Logistic Regression. Don't let the word 'Regression' fool you.

</details>

**4. Which statement matches the lecture's position on ethics?**  
A) Proxy attributes such as area code are ethically neutral  
B) Removing names is enough to anonymize data  
C) Using race as an attribute is unethical in every application  
D) Purely statistical arguments are never sufficient  

<details><summary>Answer</summary>

**D) Purely statistical arguments are never sufficient** — The ethical situation depends on the application (the same information may be fine in a medical setting), and area code can correlate with race. 'Purely statistical arguments are never sufficient!'

</details>

**5. About what proportion of the effort is spent on data cleaning / data preparation?**  
A) 80%  
B) 60%  
C) 40%  
D) 20%  

<details><summary>Answer</summary>

**B) 60%** — Both the KDD and CRISP-DM slides say about 60%.

</details>

**6. In CRISP-DM, splitting the data into training and test sets belongs to which phase?**  
A) Evaluation  
B) Data Preparation  
C) Modelling  
D) Data Understanding  

<details><summary>Answer</summary>

**B) Data Preparation** — The slide lists splitting into training and test sets under Data Preparation, together with merging, sampling, aggregating, deriving attributes, sorting and handling blanks.

</details>

**7. ML models are evaluated in the Evaluation phase of CRISP-DM against:**  
A) The training error  
B) The default model parameters  
C) The business success criteria established at the start of the project  
D) The accuracy on the training set  

<details><summary>Answer</summary>

**C) The business success criteria established at the start of the project** — 'Evaluate the results of your efforts using the business success criteria established at the beginning of the project.'

</details>

**8. Why does association learning need constraints such as minimum coverage and minimum accuracy?**  
A) Because rules can predict any attribute (and several at once), so there are far more association rules than classification rules  
B) Because association learning is supervised  
C) Because association can only handle numeric attributes  
D) Because without class labels accuracy cannot be computed  

<details><summary>Answer</summary>

**A) Because rules can predict any attribute (and several at once), so there are far more association rules than classification rules** — Association rules can predict any attribute's value, and more than one at a time. That gives far more rules, so constraints are necessary.

</details>

**9. According to the ethics slide, about 85% of Americans can be identified from:**  
A) Social security number alone  
B) Name, zip code and phone number  
C) Zip code, birth date and sex  
D) Area code and race  

<details><summary>Answer</summary>

**C) Zip code, birth date and sex** — This shows why anonymizing data is difficult: zip code + birth date + sex identifies about 85%.

</details>

**10. Success in clustering is often measured:**  
A) Objectively, on test data with known labels  
B) Using minimum coverage and minimum accuracy  
C) Using information gain  
D) Subjectively  

<details><summary>Answer</summary>

**D) Subjectively** — Clustering is unsupervised, so 'success is often measured subjectively'. Classification success is measured objectively.

</details>

**11. Data mining is best described as:**  
A) The analysis step of the KDD process  
B) The data-cleaning phase of CRISP-DM  
C) A subset of reinforcement learning  
D) The entire Knowledge Discovery in Databases (KDD) process  

<details><summary>Answer</summary>

**A) The analysis step of the KDD process** — The slides: 'DM is the analysis step of the knowledge discovery in databases process, or KDD.' It is not the whole of KDD.

</details>

## L2 EDA & Pre-processing

**12. According to the lecture, the underlying reason for inaccurate values is:**  
A) Malfunctioning equipment  
B) Data has not been collected for mining  
C) Normalization  
D) Outliers  

<details><summary>Answer</summary>

**B) Data has not been collected for mining** — Inaccurate values: 'Reason: Data has not been collected for mining.'

</details>

**13. Which statistics can be computed for a NOMINAL attribute?**  
A) Mean and median  
B) Count and mode  
C) Range and IQR  
D) Standard deviation  

<details><summary>Answer</summary>

**B) Count and mode** — Nominal values only support equality tests, so you can find the count and the mode.

</details>

**14. A medical test value is missing for some patients. What does the lecture warn?**  
A) The missingness may be significant in itself, so 'missing' may need to be coded as an extra value  
B) Missing values can always be safely replaced by the mean  
C) Instances with missing values must always be deleted  
D) Missing values only arise from typos  

<details><summary>Answer</summary>

**A) The missingness may be significant in itself, so 'missing' may need to be coded as an extra value** — Example: a missing test in a medical examination. Most schemes assume missingness is not significant, but it may be. And always report the action you took.

</details>

**15. The main limitation of univariate (boxplot) outlier detection is that it:**  
A) Removes too many points  
B) Only works for nominal data  
C) Requires standardized data  
D) Cannot detect unusual combinations of multiple variables  

<details><summary>Answer</summary>

**D) Cannot detect unusual combinations of multiple variables** — In the 2-feature example, the boxplot shows no outliers but the scatter plot does. Multivariate methods are needed.

</details>

**16. Which is NOT one of the five EDA steps in the lecture?**  
A) Data cleaning  
B) Data transformation  
C) Model deployment  
D) Detect outliers  

<details><summary>Answer</summary>

**C) Model deployment** — The 5 steps are Data Cleaning, Statistical Analysis, Data Visualization, Detect Outliers and Data Transformation.

</details>

**17. A dense cluster of points sits right at the boundary of a boxplot whisker. According to the lecture they should:**  
A) Be standardized separately  
B) Be replaced by the mean  
C) Typically NOT be removed  
D) Always be removed  

<details><summary>Answer</summary>

**C) Typically NOT be removed** — Outliers are points significantly removed from the main body. A cluster right at the whisker boundary should typically not be removed.

</details>

**18. By default, the whiskers of a Seaborn boxplot extend to:**  
A) ±1 standard deviation  
B) ±1.5 × IQR  
C) The minimum and maximum  
D) ±2 standard deviations  

<details><summary>Answer</summary>

**B) ±1.5 × IQR** — Whiskers are by default ±1.5 IQR. Points beyond them are possible outliers and candidates for removal.

</details>

**19. According to the lecture, which statement about scaling is TRUE?**  
A) fit_transform returns a pandas DataFrame  
B) Standardization is less sensitive to outliers than normalization  
C) Normalization transforms features to mean 0 and SD 1  
D) MinMaxScaler performs standardization  

<details><summary>Answer</summary>

**B) Standardization is less sensitive to outliers than normalization** — Normalization means [0,1] (MinMaxScaler) and standardization means mean 0, SD 1. The slide says standardization is less sensitive to outliers. fit_transform returns a NumPy array.

</details>

**20. Treating degrees on a compass in a special way is an example of which kind of metadata?**  
A) Circular ordering  
B) Interval scaling  
C) Partial ordering  
D) Dimensional consideration  

<details><summary>Answer</summary>

**A) Circular ordering** — Metadata examples: dimensional considerations, circular orderings (compass degrees), partial orderings (generalization/specialization).

</details>

**21. Temperature as 'hot > mild > cool' is ordinal. Which statement is TRUE?**  
A) Addition and subtraction are meaningful  
B) The distance between hot and mild equals that between mild and cool  
C) Order is defined but distance between values is not  
D) It has a true zero point  

<details><summary>Answer</summary>

**C) Order is defined but distance between values is not** — Ordinal imposes an order but no distance between values; addition and subtraction don't make sense.

</details>

**22. Which technique does the lecture name as one of the most popular for MULTIVARIATE outlier detection?**  
A) DBSCAN  
B) K-means  
C) One-hot encoding  
D) PCA  

<details><summary>Answer</summary>

**A) DBSCAN** — DBSCAN treats low-density regions as noise/outliers. Note that it expects standardized data.

</details>

**23. Which scikit-learn class performs range normalisation to [0, 1]?**  
A) OneHotEncoder  
B) OrdinalEncoder  
C) StandardScaler  
D) MinMaxScaler  

<details><summary>Answer</summary>

**D) MinMaxScaler** — preprocessing.MinMaxScaler().

</details>

**24. Temperature measured in degrees Fahrenheit is an example of which level of measurement?**  
A) Ratio  
B) Ordinal  
C) Nominal  
D) Interval  

<details><summary>Answer</summary>

**D) Interval** — It is ordered with fixed equal units, but the zero point is not defined, so it is interval. Year is also interval.

</details>

**25. According to the 'Data Types Suitability' slide, which algorithms are suited to NOMINAL data?**  
A) K-means and DBSCAN  
B) Neural networks and linear regression  
C) Linear regression only  
D) Logistic regression, decision trees and association (Apriori, FP-Growth)  

<details><summary>Answer</summary>

**D) Logistic regression, decision trees and association (Apriori, FP-Growth)** — Nominal: logistic regression, decision trees, association. Ratio (easy distance metrics): neural networks, linear regression, clustering.

</details>

**26. For the attribute 'year', which statement is valid?**  
A) The difference between 2020 and 2010 is meaningful  
B) 2020 is twice as late as 1010  
C) All mathematical operations are allowed  
D) Adding two years gives a meaningful value  

<details><summary>Answer</summary>

**A) The difference between 2020 and 2010 is meaningful** — Interval data: differences make sense; sums, products and ratios don't. All operations are allowed only for ratio data.

</details>

**27. Multi-instance learning: which application is given as an example?**  
A) Weather prediction  
B) Market-basket analysis  
C) Iris classification  
D) Drug activity prediction  

<details><summary>Answer</summary>

**D) Drug activity prediction** — In multi-instance learning each example is a set of instances, and one or more of them may be responsible for the classification. The example given is drug activity prediction.

</details>

**28. A KNN dataset has many features in [0,1] and one feature in [−1000, 1000]. Without scaling:**  
A) All features contribute equally to the distance  
B) Euclidean distance is undefined  
C) The KNN would essentially rely on the single large-range feature  
D) KNN ignores the large-range feature  

<details><summary>Answer</summary>

**C) The KNN would essentially rely on the single large-range feature** — The small-range features become uninformative in the Euclidean distance, which is why scaling is needed.

</details>

## L3 Pre-processing 2 & Feature Selection

**29. In PCA, the eigenvectors of the covariance matrix are:**  
A) The coefficients attached to the components  
B) The original features  
C) The directions of the axes with the most variance  
D) The class labels  

<details><summary>Answer</summary>

**C) The directions of the axes with the most variance** — Eigenvectors are the directions of most variance; eigenvalues are the coefficients attached to them. Components are ranked by variance.

</details>

**30. How should an ORDINAL feature such as Grade (A, B, C...) be encoded according to the lecture?**  
A) Manually with a mapping dictionary and the DataFrame map method  
B) With one-hot encoding  
C) Automatically with OrdinalEncoder, which knows the order  
D) With MinMaxScaler  

<details><summary>Answer</summary>

**A) Manually with a mapping dictionary and the DataFrame map method** — Ordering is domain-specific, so there is no automatic mechanism. Specify the mapping manually, e.g. {'F':0,...,'A':4} with df['Grade'].map(...).

</details>

**31. The main disadvantage of one-hot encoding, and the suggested mitigation, are:**  
A) Loss of order; fix with a mapping dictionary  
B) It cannot handle strings; use OrdinalEncoder  
C) It is sensitive to outliers; use standardization  
D) Many distinct values create many features (long training, poorer models); mitigate with PCA  

<details><summary>Answer</summary>

**D) Many distinct values create many features (long training, poorer models); mitigate with PCA** — The rapid increase in dimensionality can be reduced with dimensionality reduction such as PCA.

</details>

**32. What does pandas get_dummies do?**  
A) Normalizes numeric columns  
B) Converts every column to binary  
C) Converts string columns and leaves the other columns unadjusted  
D) Drops categorical columns  

<details><summary>Answer</summary>

**C) Converts string columns and leaves the other columns unadjusted** — Straight from the slide.

</details>

**33. Three boxes: A = 4 red; B = 3 red + 1 blue; C = 2 red + 2 blue. Which has the HIGHEST entropy?**  
A) A  
B) B  
C) All equal  
D) C  

<details><summary>Answer</summary>

**D) C** — A = 0, B ≈ 0.811, C = 1 bit, the maximum for 2 classes (equally likely).

</details>

**34. Which statement about FILTER feature selection methods is TRUE?**  
A) They rank attributes without building a model, and are less likely to detect dependencies/redundancies  
B) They create new features as linear combinations  
C) They build a model to evaluate each subset  
D) They are slow but give the best results  

<details><summary>Answer</summary>

**A) They rank attributes without building a model, and are less likely to detect dependencies/redundancies** — Filters are quicker and more scalable, normally univariate. Building models describes wrappers; linear combinations describe PCA.

</details>

**35. The entropy of a box with 3 red and 1 blue ball (log base 2) is approximately:**  
A) 0.811  
B) 0.5  
C) 1.0  
D) 0.940  

<details><summary>Answer</summary>

**A) 0.811** — −0.75·log2(0.75) − 0.25·log2(0.25) = 0.311 + 0.5 = 0.811.

</details>

**36. Which feature selection approach is 'very heavy in terms of computation but gets the best results'?**  
A) Deep Learning  
B) Wrapper  
C) Filter  
D) PCA  

<details><summary>Answer</summary>

**B) Wrapper** — Wrappers build models to evaluate subsets of features: very costly, but the best results, and useful for interpretability.

</details>

**37. PCA's main drawback, according to the lecture:**  
A) It requires labelled data  
B) It increases the number of features  
C) The new features are combinations of variables, so they are not possible to understand  
D) It is very slow  

<details><summary>Answer</summary>

**C) The new features are combinations of variables, so they are not possible to understand** — PCA is quick and reduces the space, but the components aren't interpretable.

</details>

**38. What is the FIRST step of PCA?**  
A) Compute eigenvalues  
B) Standardize all values  
C) Compute information gain  
D) One-hot encode  

<details><summary>Answer</summary>

**B) Standardize all values** — First standardize, then calculate the covariance between all variables.

</details>

**39. The lecture distinguishes 'explainability' and 'interpretability'. Feature selection is said to improve:**  
A) Nothing besides accuracy  
B) Only the training time  
C) The interpretability of the methods  
D) Only explainability of deep networks  

<details><summary>Answer</summary>

**C) The interpretability of the methods** — Benefits: understanding the data, improving interpretability (note: explainability ≠ interpretability), accuracy, and speed.

</details>

**40. Using OrdinalEncoder on the nominal feature Department (Biology, Chemistry, Computing) causes what problem?**  
A) It creates too many columns  
B) The algorithm assumes an order: Computing > Chemistry > Biology  
C) It removes the Department column  
D) It raises an error on strings  

<details><summary>Answer</summary>

**B) The algorithm assumes an order: Computing > Chemistry > Biology** — The integers imply an ordering that doesn't exist. The fix is one-hot encoding.

</details>

**41. Using the lecture's feature-selection decision tree: you need to UNDERSTAND the result and the data does NOT have a lot of features. Which method?**  
A) Filter  
B) PCA  
C) Deep Learning  
D) Wrapper  

<details><summary>Answer</summary>

**D) Wrapper** — Understand: yes, then lots of features → Filter, otherwise → Wrapper. Understand: no, then small model or small dataset → PCA, otherwise → Deep Learning.

</details>

**42. Play-golf data: H = 0.940 and E(PlayGolf, Outlook) = 0.693. The information gain of Outlook is:**  
A) 0.940  
B) 0.247  
C) 1.633  
D) 0.693  

<details><summary>Answer</summary>

**B) 0.247** — IG = entropy before − entropy after = 0.940 − 0.693 = 0.247, the highest, so Outlook is the root.

</details>

**43. Gene study: 70 patients, 20,000 fields each, and you need to know WHICH genes matter. Following the decision tree:**  
A) Filter  
B) Wrapper  
C) PCA  
D) Deep Learning  

<details><summary>Answer</summary>

**A) Filter** — You need to understand the result and there are many features, so Filter. A wrapper would be far too expensive with 20k features.

</details>

## L4 Decision Trees

**44. min_samples_leaf is:**  
A) The minimum number of samples a node must have before it can be split  
B) The number of features considered per split  
C) The minimum number of samples a leaf node must have  
D) The maximum depth  

<details><summary>Answer</summary>

**C) The minimum number of samples a leaf node must have** — 'Minimum samples a node must have before it can be split' describes min_samples_split.

</details>

**45. Which statement about Gini vs Entropy is TRUE?**  
A) Gini tends to isolate the most frequent class in its own branch; entropy gives slightly more balanced trees  
B) Entropy is faster to compute, so it is the default  
C) They usually produce very different trees  
D) C4.5 uses Gini and CART uses entropy  

<details><summary>Answer</summary>

**A) Gini tends to isolate the most frequent class in its own branch; entropy gives slightly more balanced trees** — Gini is slightly faster (a good default) and used by CART; entropy is used by C4.5/J48. They usually lead to similar trees.

</details>

**46. Why are decision trees sensitive to rotation of the training data?**  
A) They require normalization  
B) Their decision boundaries are perpendicular to the axes  
C) Because of the Gini index  
D) They use Euclidean distance  

<details><summary>Answer</summary>

**B) Their decision boundaries are perpendicular to the axes** — A linearly separable set rotated 45° gives a convoluted staircase boundary that generalizes poorly.

</details>

**47. In a decision tree, which statement is TRUE?**  
A) Numeric attributes always get a three-way split  
B) A numeric attribute may be tested several times on the same path  
C) A nominal attribute may be tested many times on the same path  
D) Every leaf must be pure  

<details><summary>Answer</summary>

**B) A numeric attribute may be tested several times on the same path** — A nominal attribute won't be tested more than once; a numeric one may be. Leaves need not be pure (identical instances can have different classes).

</details>

**48. Why is error on the training data NOT a useful estimator for pruning?**  
A) It would result in almost no pruning  
B) It would prune everything  
C) It requires a hold-out set  
D) It is too expensive to compute  

<details><summary>Answer</summary>

**A) It would result in almost no pruning** — The alternatives are a hold-out set (reduced-error pruning) or C4.5's confidence-based estimate.

</details>

**49. x1 = 1..8 with y = 0,0,0,1,1,1,1,1 (parent Gini = 15/32). Which split gives the larger Gini decrease?**  
A) x1 < 4.5 (ΔGini = 9/32)  
B) Neither reduces Gini  
C) Both equal  
D) x1 < 3.5 (ΔGini = 15/32)  

<details><summary>Answer</summary>

**D) x1 < 3.5 (ΔGini = 15/32)** — Splitting at 3.5 makes both children pure, so the whole 15/32 is removed.

</details>

**50. Which statement about the CART algorithm is FALSE?**  
A) It splits on a single feature k and a threshold tk  
B) It is greedy  
C) It is guaranteed to find the optimal tree  
D) It minimizes the size-weighted impurity of the two subsets  

<details><summary>Answer</summary>

**C) It is guaranteed to find the optimal tree** — CART is greedy: it does not look ahead several levels, so it produces a reasonably good solution that is not guaranteed optimal.

</details>

**51. Comparing the postpruning operations:**  
A) Subtree replacement is top-down  
B) Subtree raising is faster than subtree replacement  
C) Subtree raising is slower than subtree replacement  
D) They are identical  

<details><summary>Answer</summary>

**C) Subtree raising is slower than subtree replacement** — Subtree replacement works bottom-up. Subtree raising deletes a node and redistributes instances, which makes it slower.

</details>

**52. The prediction complexity of a (roughly balanced) decision tree is about:**  
A) O(n)  
B) O(n × m log m), depending on the number of features  
C) O(m²)  
D) O(log2 m), independent of the number of features  

<details><summary>Answer</summary>

**D) O(log2 m), independent of the number of features** — Prediction traverses root to leaf, checking one feature per node. O(n × m log m) is the TRAINING complexity.

</details>

**53. A node contains 0 Setosa, 49 Versicolor and 5 Virginica. Its Gini impurity is about:**  
A) 0.907  
B) 0  
C) 0.168  
D) 0.5  

<details><summary>Answer</summary>

**C) 0.168** — 1 − (0/54)² − (49/54)² − (5/54)² ≈ 0.168.

</details>

**54. An 'ID code' attribute gets the maximal information gain (0.940 bits). This illustrates that:**  
A) Information gain is biased toward attributes with many values, which can cause overfitting  
B) ID codes are the best predictors  
C) Gini should never be used  
D) Entropy cannot be computed for nominal data  

<details><summary>Answer</summary>

**A) Information gain is biased toward attributes with many values, which can cause overfitting** — Every single-instance subset is pure (entropy 0), so the gain is maximal, but the split is useless for prediction. It also causes data fragmentation.

</details>

**55. Which impurity function is the ONLY one that satisfies all three wishlist properties (zero when pure, max when uniform, multistage)?**  
A) MSE  
B) Misclassification error  
C) Gini  
D) Entropy  

<details><summary>Answer</summary>

**D) Entropy** — 'It can be shown that entropy is the only function that satisfies all three properties.' The multistage property is not strictly necessary in practice.

</details>

**56. A regression decision tree predicts, for each region:**  
A) A linear function of the features  
B) The median feature value  
C) The majority class  
D) The average target value of the training instances in that region  

<details><summary>Answer</summary>

**D) The average target value of the training instances in that region** — CART for regression minimizes MSE (not Gini or entropy), and each leaf predicts the mean target.

</details>

**57. For a 2-class problem, the MAXIMUM possible Gini impurity is:**  
A) 1  
B) 0.5  
C) 0.25  
D) log2(2)  

<details><summary>Answer</summary>

**B) 0.5** — The maximum is 1 − 1/k = 0.5 for k = 2. Trap: the slide says 'a Gini coefficient of 1 expresses maximal inequality', but impurity never reaches 1.

</details>

**58. A flower reaches a leaf containing 0 Setosa, 49 Versicolor and 5 Virginica. The predicted probability of Versicolor is:**  
A) 90.7%  
B) 49%  
C) 9.3%  
D) 100%  

<details><summary>Answer</summary>

**A) 90.7%** — The tree returns the ratio of training instances of class k in that leaf: 49/54 = 90.7%.

</details>

**59. The XOR/parity problem illustrates which danger?**  
A) Postpruning removes too much  
B) Gini fails on binary data  
C) Prepruning may stop growth too early (early stopping)  
D) Information gain is biased  

<details><summary>Answer</summary>

**C) Prepruning may stop growth too early (early stopping)** — No single attribute is significantly associated with the class, so prepruning won't even expand the root. XOR-type problems are rare in practice, though.

</details>

**60. Which of the following is NOT listed as a disadvantage of decision trees?**  
A) Cannot handle missing data  
B) No extrapolation  
C) Must regrow the tree when adding training instances  
D) Sensitive to training set rotation  

<details><summary>Answer</summary>

**A) Cannot handle missing data** — 'Can handle missing data' is listed as a PRO.

</details>

**61. In C4.5's pruning method, the default confidence c = 25% corresponds to z =**  
A) 1.0  
B) 0.69  
C) 0.25  
D) 1.96  

<details><summary>Answer</summary>

**B) 0.69** — If c = 25% then z = 0.69 (from the normal distribution).

</details>

**62. Which pruning strategy is preferred in practice?**  
A) Pruning based on training error  
B) No pruning  
C) Prepruning, because it is faster  
D) Postpruning  

<details><summary>Answer</summary>

**D) Postpruning** — Postpruning is preferred because prepruning can 'stop early'. Prepruning is faster, which is why the 'prepruning, because it is faster' option is a trap.

</details>

**63. Decision trees are called NON-parametric because:**  
A) They only work on nominal data  
B) The number of parameters is not determined prior to training  
C) They have no hyperparameters  
D) They cannot overfit  

<details><summary>Answer</summary>

**B) The number of parameters is not determined prior to training** — The structure is free to stick closely to the data (risk of overfitting). Parametric models such as linear models have a fixed number of parameters (risk of underfitting).

</details>

**64. Quinlan's classic tree learner ID3 used which test in addition to information gain for prepruning?**  
A) F-test  
B) t-test  
C) Kolmogorov–Smirnov  
D) Chi-squared test  

<details><summary>Answer</summary>

**D) Chi-squared test** — Only statistically significant attributes (chi-squared) could be selected by the information gain procedure.

</details>

## L5 Probabilistic Classification & SVM

**65. The 'zero-frequency problem' in Naïve Bayes is fixed by:**  
A) Adding 1 to every attribute value–class count (Laplace estimator)  
B) Standardizing the data  
C) Removing the attribute  
D) Using Gini instead  

<details><summary>Answer</summary>

**A) Adding 1 to every attribute value–class count (Laplace estimator)** — Probabilities are then never zero. A further advantage: it stabilizes estimates computed from small samples.

</details>

**66. Basis functions φ help SVMs with non-linearly separable data because:**  
A) They reduce the dimensionality  
B) They make the data normally distributed  
C) They remove outliers  
D) There are typically more basis functions than features, which moves the data into a higher-dimensional space where a linear separator is expected to exist  

<details><summary>Answer</summary>

**D) There are typically more basis functions than features, which moves the data into a higher-dimensional space where a linear separator is expected to exist** — Example: z = x² + y² turns a circular boundary into a flat plane.

</details>

**67. The support vectors are:**  
A) The extreme points closest to the hyperplane that define the margin  
B) The points farthest from the hyperplane  
C) The misclassified points  
D) All training points  

<details><summary>Answer</summary>

**A) The extreme points closest to the hyperplane that define the margin** — 'SVM chooses the extreme points/vectors that help in creating the hyperplane.'

</details>

**68. How does Naïve Bayes handle a missing attribute value at CLASSIFICATION time?**  
A) Replaces it with the class mean  
B) Omits the attribute from the calculation  
C) Treats it as a separate value  
D) Assigns probability zero  

<details><summary>Answer</summary>

**B) Omits the attribute from the calculation** — During training the instance isn't counted for that combination; during classification the attribute is omitted.

</details>

**69. Training an SVM is framed as:**  
A) Gradient descent on MSE  
B) An unconstrained linear program  
C) A greedy recursive split  
D) A constrained quadratic optimization problem  

<details><summary>Answer</summary>

**D) A constrained quadratic optimization problem** — It is defined by a set of constraints, ti(w0 + w·d) ≥ 1, and an optimization criterion (maximize 2/||w||).

</details>

**70. In Naïve Bayes with Gaussian attributes, a computed density f(x) = 1.4. This is:**  
A) Only possible for discrete attributes  
B) A sign of an error in σ  
C) Possible, since densities can exceed 1 (they must integrate to 1)  
D) Impossible, since probabilities can't exceed 1  

<details><summary>Answer</summary>

**C) Possible, since densities can exceed 1 (they must integrate to 1)** — Densities are not probabilities, but when computing likelihoods you can treat them like probabilities.

</details>

**71. How does 1R handle missing values?**  
A) It treats 'missing' as a separate attribute value  
B) It replaces them with the mode  
C) It deletes the instance  
D) It omits the attribute from the calculation  

<details><summary>Answer</summary>

**A) It treats 'missing' as a separate attribute value** — Contrast with Naïve Bayes, which omits the attribute during classification.

</details>

**72. When estimating mean and SD for a numeric attribute in Naïve Bayes, missing values are:**  
A) Replaced by 0  
B) Not included in the calculation  
C) Replaced by the mean  
D) Counted as outliers  

<details><summary>Answer</summary>

**B) Not included in the calculation** — 'Missing values during training are not included in calculation of mean and standard deviation.'

</details>

**73. Which situation causes problems for Naïve Bayes according to the lecture?**  
A) Adding many redundant/identical attributes  
B) Having small datasets with Laplace smoothing  
C) Having only two classes  
D) Using nominal attributes  

<details><summary>Answer</summary>

**A) Adding many redundant/identical attributes** — Identical attributes effectively count the same evidence several times. For non-normal numeric attributes, kernel density estimators can be used.

</details>

**74. In Bayes' rule P(H|E) = P(E|H)P(H)/P(E), the PRIOR probability of H is:**  
A) P(E|H)  
B) P(H|E)  
C) P(H)  
D) P(E)  

<details><summary>Answer</summary>

**C) P(H)** — Prior = P(H) (before evidence); posterior = P(H|E) (after evidence). Watch out: the slide has the two formulas swapped.

</details>

**75. The α parameters in the SVM model are:**  
A) Lagrange multipliers, one per support vector  
B) Class priors  
C) Learning rates  
D) Kernel widths  

<details><summary>Answer</summary>

**A) Lagrange multipliers, one per support vector** — M(q) = Σ (ti × α[i] × (di·q) + w0), summed over the s support vectors.

</details>

**76. Why does Naïve Bayes work well even when the independence assumption is clearly violated?**  
A) Because attributes are always independent in practice  
B) Because it uses Laplace smoothing  
C) Because it uses kernels  
D) Classification only needs the maximum probability assigned to the correct class, not accurate probabilities  

<details><summary>Answer</summary>

**D) Classification only needs the maximum probability assigned to the correct class, not accurate probabilities** — However, adding too many redundant (e.g. identical) attributes will cause problems.

</details>

**77. In SVM, the size of the margin is:**  
A) w0/||w||  
B) 1/||w||  
C) 2/||w||  
D) ||w||  

<details><summary>Answer</summary>

**C) 2/||w||** — The distance from the margin extent to the boundary is 1/||w||, and the margin is symmetric, so 2/||w||. Training maximizes it.

</details>

**78. Why does a timestamp attribute look perfect for 1R?**  
A) It gets zero errors because every value is unique, which is overfitting  
B) It has the highest entropy  
C) 1R ignores numeric attributes  
D) Timestamps are normally distributed  

<details><summary>Answer</summary>

**A) It gets zero errors because every value is unique, which is overfitting** — Discretization is very sensitive to noise and unique values. The fix is a minimum number of majority-class instances per interval.

</details>

**79. In the generalized Laplace estimate (2 + μp1)/(9 + μ), the weights p1, p2, p3:**  
A) Must be equal  
B) Need not be equal but must sum to 1  
C) Must each equal 1  
D) Must sum to μ  

<details><summary>Answer</summary>

**B) Need not be equal but must sum to 1** — 'Weights don't need to be equal (but they must sum to 1).'

</details>

**80. New day: Sunny, Cool, High, Windy=True. Likelihoods: yes = 0.0053, no = 0.0206. P(no) is about:**  
A) 0.205  
B) 0.0206  
C) 0.5  
D) 0.795  

<details><summary>Answer</summary>

**D) 0.795** — Normalize: 0.0206 / (0.0053 + 0.0206) = 0.795.

</details>

**81. What is the main purpose of the kernel trick?**  
A) To make the margin smaller  
B) To avoid the expensive dot product of basis-function-transformed vectors by computing it from the original features  
C) To normalize the data  
D) To reduce the number of support vectors  

<details><summary>Answer</summary>

**B) To avoid the expensive dot product of basis-function-transformed vectors by computing it from the original features** — k(x,z) = ⟨φ(x), φ(z)⟩. We can use a kernel without knowing φ, as long as we can prove such a mapping exists.

</details>

**82. The two assumptions of Naïve Bayes are that attributes are:**  
A) Normally distributed and numeric  
B) Equally important and statistically independent given the class  
C) Independent of the class and of each other  
D) Ordered and discrete  

<details><summary>Answer</summary>

**B) Equally important and statistically independent given the class** — The independence assumption is almost never correct, but the scheme often works surprisingly well.

</details>

**83. The Gaussian radial basis kernel is:**  
A) (d·q + 1)^p  
B) tanh(d·q)  
C) exp(−γ||d − q||²)  
D) d·q + c  

<details><summary>Answer</summary>

**C) exp(−γ||d − q||²)** — γ is a manually chosen tuning parameter. d·q + c is the linear kernel, (d·q + 1)^p the polynomial kernel, and tanh is not covered.

</details>

**84. In Holte's (1993) 1R study, the minimum number of instances in the majority class per interval was set to:**  
A) 3  
B) 10  
C) 1  
D) 6  

<details><summary>Answer</summary>

**D) 6** — Holte set it to 6 after experimentation. The lecture's own worked example used 3, which is the trap.

</details>

**85. On the weather data, which attributes have the lowest 1R total error (4/14)?**  
A) Outlook only  
B) Temperature and Windy  
C) Outlook and Humidity  
D) Humidity only  

<details><summary>Answer</summary>

**C) Outlook and Humidity** — Outlook 4/14, Temperature 5/14, Humidity 4/14, Windy 5/14. It's a tie.

</details>

## L6 Clustering

**86. Which clustering approach is 'based on a multiple-level granularity structure'?**  
A) Model-based  
B) Partitioning  
C) Grid-based  
D) Hierarchical  

<details><summary>Answer</summary>

**C) Grid-based** — Model-based hypothesizes a model for each cluster and finds the best fit.

</details>

**87. Minkowski distance with q = 1 is:**  
A) Euclidean distance  
B) Chebyshev distance  
C) Jaccard distance  
D) Manhattan distance  

<details><summary>Answer</summary>

**D) Manhattan distance** — q = 1 gives Manhattan, q = 2 gives Euclidean.

</details>

**88. p is a core point and q (a border point) lies in p's ε-neighbourhood. Which is TRUE?**  
A) p is directly density-reachable from q  
B) Neither is reachable from the other  
C) Density-reachability is symmetric  
D) q is directly density-reachable from p, but not vice versa  

<details><summary>Answer</summary>

**D) q is directly density-reachable from p, but not vice versa** — Direct density-reachability requires the SOURCE to be a core point, so it is not symmetric.

</details>

**89. Points (3,4), (2,6), (4,5), (4,7), (3,8). The clustering feature CF = (N, LS, SS) is:**  
A) (5, (16,30), (54,190))  
B) (5, (16,30), (256,900))  
C) (5, (54,190), (16,30))  
D) (5, (3.2,6), (54,190))  

<details><summary>Answer</summary>

**A) (5, (16,30), (54,190))** — LS is the linear sum (16, 30); SS is the square sum (9+4+16+16+9, 16+36+25+49+64) = (54, 190).

</details>

**90. Hierarchical clustering:**  
A) Can undo previous merges  
B) Requires k as input  
C) Does not require k, but needs a termination condition  
D) Requires Eps and MinPts  

<details><summary>Answer</summary>

**C) Does not require k, but needs a termination condition** — It uses the distance matrix. It can never undo what was done previously, and it is at least O(n²).

</details>

**91. In the Jack/Mary/Jim example, why is gender excluded?**  
A) It is a symmetric attribute  
B) It has too many values  
C) It is ordinal  
D) It is missing  

<details><summary>Answer</summary>

**A) It is a symmetric attribute** — Gender is symmetric; the remaining (asymmetric) attributes are used with Jaccard.

</details>

**92. Which is a major feature of density-based clustering?**  
A) It discovers clusters of arbitrary shape and handles noise  
B) It requires k in advance  
C) It needs many scans of the data  
D) It only finds convex clusters  

<details><summary>Answer</summary>

**A) It discovers clusters of arbitrary shape and handles noise** — Also: one scan, and it needs density parameters as the termination condition.

</details>

**93. Which statement about PAM (k-medoids) is TRUE?**  
A) It does not need k  
B) It works effectively for small data sets but does not scale well  
C) It finds arbitrarily shaped clusters  
D) Each cluster is represented by its mean  

<details><summary>Answer</summary>

**B) It works effectively for small data sets but does not scale well** — Medoids are real objects in the cluster.

</details>

**94. A good clustering produces:**  
A) High inter-class similarity and low intra-class similarity  
B) High intra-class similarity and low inter-class similarity  
C) High similarity both within and between clusters  
D) Equal-sized clusters  

<details><summary>Answer</summary>

**B) High intra-class similarity and low inter-class similarity** — Intra means within a cluster (should be similar); inter means between clusters (should be dissimilar).

</details>

**95. How does the lecture suggest choosing Eps for DBSCAN?**  
A) Set Eps = MinPts  
B) Plot the sorted distances of every point to its k-th nearest neighbour  
C) Use the elbow of the SSE curve  
D) Cross-validation on labels  

<details><summary>Answer</summary>

**B) Plot the sorted distances of every point to its k-th nearest neighbour** — Noise points have their k-th nearest neighbour farther away. Pick Eps where the curve rises sharply.

</details>

**96. For ASYMMETRIC binary variables, which dissimilarity is appropriate?**  
A) Simple matching (b+c)/(a+b+c+d)  
B) Manhattan distance  
C) Jaccard (b+c)/(a+b+c)  
D) Euclidean distance  

<details><summary>Answer</summary>

**C) Jaccard (b+c)/(a+b+c)** — Jaccard ignores d (negative matches, 0/0). Simple matching is for symmetric variables.

</details>

**97. An ordinal variable has M = 5 levels. A value with rank r = 3 is mapped to:**  
A) 0.75  
B) 3  
C) 0.5  
D) 0.6  

<details><summary>Answer</summary>

**C) 0.5** — z = (r − 1)/(M − 1) = 2/4 = 0.5. It is then treated as interval-scaled.

</details>

**98. k-means often terminates at a local optimum. According to the lecture, the global optimum may be found using:**  
A) Subtree raising  
B) Laplace estimators  
C) Gini index  
D) Deterministic annealing and genetic algorithms  

<details><summary>Answer</summary>

**D) Deterministic annealing and genetic algorithms** — Exhaustively enumerating all partitions also finds it, but that is infeasible.

</details>

**99. CURE improves hierarchical clustering by:**  
A) Using medoids  
B) Selecting well-scattered points from a cluster and shrinking them towards the centre by a fraction  
C) Building a CF-tree  
D) Using density-reachability  

<details><summary>Answer</summary>

**B) Selecting well-scattered points from a cluster and shrinking them towards the centre by a fraction** — BIRCH is the one that uses a CF-tree.

</details>

**100. When standardizing interval-scaled variables for clustering, the lecture uses the mean absolute deviation because it is:**  
A) Required by k-means  
B) More robust to outliers than the standard deviation  
C) Faster to compute  
D) Always smaller than the variance  

<details><summary>Answer</summary>

**B) More robust to outliers than the standard deviation** — Deviations aren't squared, so outliers have less effect. Then z = (x − m)/s.

</details>

**101. In DBSCAN, a BORDER point is one that:**  
A) Is neither core nor in any core's neighbourhood  
B) Lies on the convex hull  
C) Has fewer than MinPts within Eps but lies in the neighbourhood of a core point  
D) Has at least MinPts neighbours within Eps  

<details><summary>Answer</summary>

**C) Has fewer than MinPts within Eps but lies in the neighbourhood of a core point** — 'Neither core nor in any core's neighbourhood' describes a noise point. Core, border and noise are mutually exclusive.

</details>

**102. Jim vs Mary (asymmetric binary): a = 1, b = 1, c = 2. The Jaccard dissimilarity is:**  
A) 0.33  
B) 0.67  
C) 0.5  
D) 0.75  

<details><summary>Answer</summary>

**D) 0.75** — (b + c)/(a + b + c) = 3/4 = 0.75.

</details>

**103. To cluster purely CATEGORICAL data with a k-means-style approach, use:**  
A) k-prototype  
B) DBSCAN  
C) k-medoids  
D) k-modes  

<details><summary>Answer</summary>

**D) k-modes** — k-modes uses modes instead of means. k-prototype handles mixed categorical + numeric data.

</details>

**104. The time complexity of k-means is:**  
A) O(2^n)  
B) O(tkn)  
C) O(n²)  
D) O(n log n)  

<details><summary>Answer</summary>

**B) O(tkn)** — t iterations, k clusters, n objects, with k and t much smaller than n. It is relatively efficient.

</details>

**105. Which is NOT a weakness of k-means?**  
A) The number of clusters k must be specified in advance  
B) It cannot handle noisy data and outliers well  
C) It is computationally infeasible for large n  
D) It is not suitable for non-convex cluster shapes  

<details><summary>Answer</summary>

**C) It is computationally infeasible for large n** — k-means is relatively efficient, O(tkn). (It is applicable only when a mean is defined.)

</details>

**106. For a nominal variable, simple matching dissimilarity is d = (p − m)/p, where:**  
A) m = number of matches, p = total number of variables  
B) m = mean  
C) p = number of matches, m = total variables  
D) p = number of clusters  

<details><summary>Answer</summary>

**A) m = number of matches, p = total number of variables** — Alternative: create one binary variable per nominal state.

</details>

**107. In PAM, a medoid i is replaced by non-medoid h when the total swapping cost TCih is:**  
A) Less than 0  
B) Equal to 0  
C) Greater than 0  
D) Maximal  

<details><summary>Answer</summary>

**A) Less than 0** — A negative cost means the total distance improves.

</details>

**108. DBSCAN does NOT work well when:**  
A) Clusters have varying densities  
B) Clusters have arbitrary shapes  
C) Clusters have different sizes  
D) There is noise  

<details><summary>Answer</summary>

**A) Clusters have varying densities** — It is also sensitive to its parameters (Eps 9.92 vs 9.75 gave very different results).

</details>

**109. AGNES and DIANA are respectively:**  
A) Both agglomerative  
B) Divisive and agglomerative  
C) Density-based and grid-based  
D) Agglomerative (bottom-up) and divisive (top-down)  

<details><summary>Answer</summary>

**D) Agglomerative (bottom-up) and divisive (top-down)** — AGNES uses single-link, merging the least dissimilar nodes until everything is one cluster. DIANA is the inverse.

</details>
