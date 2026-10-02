# SIT307 HD Task – Heart Disease Prediction

## About This Project

This repository contains the dataset and Jupyter Notebook used for my SIT307 Machine Learning High Distinction task.

For this project, I selected the research paper:

**“An efficient stacking-based ensemble technique for early heart attack prediction”**  
by Monu Bhagat, Aayush Sharma and Piyanshi Agarwal.

The aim of my project was not only to reproduce the machine learning results from the paper, but also to critically investigate the reproduced results and develop my own solution for an issue identified during the experiment.

The project is divided into two main parts.

---

## Part 1 – Reproducing the Published Approach

In Part 1, I reproduced the general machine learning workflow described in the selected paper using the heart disease dataset containing **1,025 records and 14 columns**.

I evaluated the same six types of individual classifiers discussed in the paper:

- Logistic Regression
- Naive Bayes
- K-Nearest Neighbours (KNN)
- Decision Tree
- Random Forest
- XGBoost

I also reproduced the **5-fold stacking ensemble**, which is the main approach proposed in the selected paper.

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC

### Important Finding from Part 1

My reproduction produced unexpectedly high results. Random Forest, XGBoost and the stacking model all achieved **100% accuracy**.

Instead of treating this as an improvement over the published work, I investigated the dataset and train-test split further.

I found that:

- the dataset contained **1,025 total records**;
- only **302 complete records were unique**;
- **723 records were exact duplicates**; and
- **202 of the 205 test feature records also appeared identically in the training data**.

This meant that my Part 1 split had **98.54% identical-feature overlap between the test set and training set**.

This finding became the main motivation for Part 2.

---

## Part 2 – My Proposed Experimental Protocol

For Part 2, I did not simply introduce another classifier or tune the models to try to achieve a higher accuracy.

Instead, I changed the **experimental protocol** to investigate the train-test overlap found in Part 1.

The main difference was the order in which the data was prepared:

### Part 1

```text
1,025 records
      ↓
Random train-test split
      ↓
98.54% identical-feature overlap
      ↓
Model evaluation
```

### Part 2

```text
1,025 records
      ↓
Remove exact duplicates
      ↓
302 unique records
      ↓
Stratified 80:20 split
      ↓
0% identical-feature overlap
      ↓
Model evaluation
      ↓
5-fold stratified cross-validation
```

After duplicate removal, the new dataset contained **302 unique observations**.

The new split contained:

- **241 training observations**
- **61 testing observations**
- **0 out of 61 test feature records appearing identically in training**
- **0% identical-feature overlap**

I deliberately evaluated the same classifier families again so that I could examine the effect of changing the experimental protocol rather than changing both the data treatment and the machine learning algorithms at the same time.

I also added **5-fold stratified cross-validation** because the duplicate-safe dataset was much smaller and I did not want my conclusion to depend only on one train-test split.

---

# Repository Files

```text
SIT307-HD-Heart-Disease-ML/
│
├── heart.csv
├── SIT307_HD_Heart_Disease.ipynb
├── requirements.txt
└── README.md
```

### `heart.csv`

The heart disease dataset used for both Part 1 and Part 2.

It contains **1,025 observations and 14 columns**, consisting of 13 predictor variables and the binary `target` variable.

### `SIT307_HD_Heart_Disease.ipynb`

This is the main notebook for the project. It contains:

- dataset loading and inspection;
- preprocessing and feature scaling;
- Part 1 train-test split;
- training of the six individual classifiers;
- 5-fold stacking ensemble;
- Accuracy, Precision, Recall, F1 and AUC evaluation;
- published vs reproduced result comparison;
- duplicate-record investigation;
- train-test overlap analysis;
- Part 2 duplicate-safe dataset preparation;
- Part 2 model evaluation;
- 5-fold stratified cross-validation; and
- final comparison of the published, Part 1 and Part 2 results.

### `requirements.txt`

Contains the Python packages required to run the notebook.

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/adityakatoch1912/SIT307-HD-Heart-Disease-ML.git
```

Move into the project folder:

```bash
cd SIT307-HD-Heart-Disease-ML
```

The repository can also be downloaded directly as a ZIP file from GitHub.

---

## 2. Create a Virtual Environment

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install the Required Packages

```bash
pip install -r requirements.txt
```

The main packages used in this project are:

- pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost
- Jupyter

---

## 4. Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
SIT307_HD_Heart_Disease.ipynb
```

Keep `heart.csv` in the same directory as the notebook.

Run the notebook cells **from top to bottom in order**.

---

# Reproducing My Results

The notebook is organised so that the complete experiment can be reproduced sequentially.

### Part 1

Run the cells for:

1. dataset loading and inspection;
2. preprocessing and scaling;
3. stratified train-test split;
4. six individual classifiers;
5. 5-fold stacking ensemble;
6. performance evaluation;
7. comparison with the published results; and
8. duplicate and train-test overlap investigation.

### Part 2

Continue with the cells for:

1. exact duplicate removal;
2. creation of the 302-record duplicate-safe dataset;
3. new stratified 80:20 train-test split;
4. verification of 0% identical-feature overlap;
5. retraining the same classifier families;
6. stacking evaluation;
7. 5-fold stratified cross-validation; and
8. final comparison with Part 1 and the published results.

A fixed `random_state=42` is used where specified in the notebook to make the experiment more reproducible.

---

# Main Results

## Part 1 – Reproduction

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.8098 |
| Naive Bayes | 0.8293 |
| KNN | 0.8634 |
| Decision Tree | 0.9854 |
| Random Forest | 1.0000 |
| XGBoost | 1.0000 |
| Stacking | 1.0000 |

The most important finding from Part 1 was not simply the 100% results. The train-test investigation showed:

**202 / 205 test records had an identical feature record in training = 98.54% overlap.**

---

## Part 2 – Proposed Protocol

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.8033 |
| Naive Bayes | 0.7869 |
| KNN | 0.7869 |
| Decision Tree | 0.8033 |
| Random Forest | 0.7541 |
| XGBoost | 0.7213 |
| Stacking | 0.8033 |

The Part 2 stacking model achieved an **AUC of 0.8820**.

The new train-test split had:

**0 / 61 test records appearing identically in training = 0% overlap.**

I also performed 5-fold stratified cross-validation on the duplicate-safe dataset. For example:

- Logistic Regression mean accuracy: **0.8312**
- Random Forest mean accuracy: **0.8214**
- Random Forest mean AUC: **0.9066**

---

# What I Learned from the Comparison

The main purpose of Part 2 was **not to obtain a higher accuracy than Part 1**.

The important difference was how the models were evaluated.

In Part 1, Random Forest, XGBoost and stacking reached 100% accuracy, but **98.54% of the test feature records had an identical representation in the training data**.

After changing the experimental protocol in Part 2, this overlap was reduced to **0%**, and the perfect model scores disappeared.

This does not prove that duplicate overlap was responsible for every correct prediction in Part 1. However, my experiment showed that the measured performance changed substantially when the models were evaluated using the duplicate-safe protocol.

For me, this was the main lesson from the project: **a very high accuracy value should be considered together with the experimental setup that produced it.**

---

# Reproducibility Notes

The selected paper explains the overall machine learning workflow but does not provide every implementation detail required for an exact replication.

Where an exact setting was not clearly specified, I used a reasonable implementation assumption and documented it in the notebook.

For example, Logistic Regression was used as the final meta-classifier in my stacking implementation as a documented reproduction assumption.

Therefore, this project should be considered a **reproduction and critical experimental evaluation of the published methodology**, rather than an exact replication of every undocumented implementation setting.

---

## Author

**Aditya**  
SIT307 – Machine Learning  
Deakin University