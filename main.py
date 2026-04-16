# ----------------------------------------------------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------------------------------------------------

import joblib
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings

os.makedirs("model", exist_ok=True)
os.makedirs("photo", exist_ok=True)

warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------------------------------------------------------------
# Data Preparation
# ----------------------------------------------------------------------------------------------------------------------------------

df_raw = pd.read_csv('online_gaming_behavior_dataset.csv')
df = df_raw.copy()

print("Dataset Overview")
print(df)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# ----------------------------------------------------------------------------------------------------------------------------------

df = df.drop(["PlayerID"], axis=1)
print("\nDataset Overview after PlayerID is excluded:")
print(df)

print("\nMissing values in each column:")
print(df.isnull().sum())

# ----------------------------------------------------------------------------------------------------------------------------------

print("\nSummary statistics for numerical columns:")
print()
print(df['Age'].describe())
print()
print(df['PlayTimeHours'].describe())
print()
print(df['SessionsPerWeek'].describe())
print()
print(df['AvgSessionDurationMinutes'].describe())
print()
print(df['AchievementsUnlocked'].describe())

# Outlier detection using IQR method
print("\nPotential outliers (IQR method):")

numeric_cols = df.select_dtypes(include=np.number).columns
numeric_cols = numeric_cols.drop(["InGamePurchases", "PlayerLevel"])

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"{col}:")
    print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Lower Bound={lower_bound:.2f}, Upper Bound={upper_bound:.2f}")
    print(f"  Potential outliers: {len(outliers)}")
    print()

categorical_cols = list(df.select_dtypes(include=['object']).columns) + ['InGamePurchases', "PlayerLevel"]

print("Unique values in categorical columns:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].unique())
    
# ----------------------------------------------------------------------------------------------------------------------------------
# Data Visualization
# ----------------------------------------------------------------------------------------------------------------------------------

sns.set_theme(style="whitegrid")
total = len(df)

BIN_THRESHOLD = 20
BIN_STEP = 10

def apply_binning(series):
    min_v = (series.min() // BIN_STEP) * BIN_STEP
    max_v = (series.max() // BIN_STEP) * BIN_STEP + BIN_STEP
    bins  = range(int(min_v), int(max_v) + BIN_STEP, BIN_STEP)
    labels = [f"{b}–{b + BIN_STEP - 1}" for b in bins[:-1]]
    binned = pd.cut(series, bins=list(bins), labels=labels, right=False, include_lowest=True)
    return binned.astype(str)

# Data Viualization
cols_to_plot = df.columns.tolist()
n_cols = 2
n_rows = math.ceil(len(cols_to_plot) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 6 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    ax = axes[i]

    if df[col].nunique() > BIN_THRESHOLD:
        plot_series = apply_binning(df[col])
        order = sorted(plot_series.dropna().unique(),
                       key=lambda x: int(x.split('–')[0]))
        is_binned = True
    else:
        plot_series = df[col].astype(str)
        order = sorted(plot_series.dropna().unique(),
                       key=lambda x: int(x) if x.lstrip('-').isdigit() else x)
        is_binned = False

    temp_df = plot_series.rename(col).to_frame()
    sns.countplot(data=temp_df, x=col, ax=ax, palette='viridis', order=order)

    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue
        ax.annotate(
            f'{int(height)}\n({100 * height / total:.1f}%)',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='center',
            xytext=(0, 18),
            textcoords='offset points',
            fontsize=8, fontweight='bold'
        )

    ax.set_title(f'Distribution of {col}' + (' (binned)' if is_binned else ''),
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, plot_series.value_counts().max() * 1.35)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('')

    n_bars = len(order)
    ax.set_xlim(-0.5, n_bars - 0.5)
    fig.subplots_adjust(wspace=0.4)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
_fname = "photo/feature_data_visualisation.png"
plt.savefig(_fname, bbox_inches='tight')
plt.close()
print(f'\nFile saved "{_fname}"')

# # ----------------------------------------------------------------------------------------------------------------------------------

# Correlation Analysis
numeric_cols = df.select_dtypes(include=np.number).columns
numeric_cols = numeric_cols.drop(["InGamePurchases", "PlayerLevel"])
numeric_cols = numeric_cols.tolist()

categorical_cols = list(df.select_dtypes(include=['object']).columns) + ['InGamePurchases', "PlayerLevel"]

print("\nNumeric Correlation (Pearson)")
print(df[numeric_cols].corr().round(2).to_string())

def cramers_v(col1, col2):
    confusion_matrix = pd.crosstab(col1, col2)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

print("\nCategorical Correlation (Cramér's V)")
cramers_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
for col1 in categorical_cols:
    for col2 in categorical_cols:
        cramers_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
print(cramers_matrix.round(2).to_string())

def eta_squared(num_col, cat_col):
    groups = [group.values for _, group in num_col.groupby(cat_col)]
    grand_mean = num_col.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum((x - grand_mean) ** 2 for g in groups for x in g)
    return ss_between / ss_total if ss_total != 0 else 0

print("\nNumeric-Categorical Correlation (Eta Squared)")
eta_matrix = pd.DataFrame(index=numeric_cols, columns=categorical_cols, dtype=float)
for num in numeric_cols:
    for cat in categorical_cols:
        eta_matrix.loc[num, cat] = eta_squared(df[num], df[cat])
print(eta_matrix.round(2).to_string())

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')

# Pearson heatmap
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=axes[0])
axes[0].set_title("Numeric–Numeric\n(Pearson)", fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Cramér's V heatmap
sns.heatmap(cramers_matrix.astype(float), annot=True, fmt='.2f', cmap='YlOrRd',
            vmin=0, vmax=1, linewidths=0.5, ax=axes[1])
axes[1].set_title("Categorical–Categorical\n(Cramér's V)", fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

# Eta squared heatmap
sns.heatmap(eta_matrix.astype(float), annot=True, fmt='.2f', cmap='BuGn',
            vmin=0, vmax=1, linewidths=0.5, ax=axes[2])
axes[2].set_title("Numeric–Categorical\n(Eta Squared)", fontsize=12)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
_fname = "photo/correlation_analysis.png"
plt.savefig(_fname, bbox_inches='tight')
plt.close()
print(f'\nFile saved "{_fname}"')

# ----------------------------------------------------------------------------------------------------------------------------------
# Data Transformation and Encoding
# ----------------------------------------------------------------------------------------------------------------------------------

df = df_raw.copy()
df = df.drop(["PlayerID"], axis=1)

print(f"\nShape before encoding: {df.shape}")

# One-Hot Encoding
df = pd.get_dummies(df, columns=["Location"], drop_first=True)
df = pd.get_dummies(df, columns=["GameGenre"], drop_first=True)

# Ordinal Encoding
df["GameDifficulty"]  = df["GameDifficulty"].map({"Easy": 0, "Medium": 1, "Hard": 2})
df["EngagementLevel"] = df["EngagementLevel"].map({"Low": 0, "Medium": 1, "High": 2})

# Binary Encoding
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

print(f"\nShape after encoding: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"EngagementLevel unique values: {sorted(df['EngagementLevel'].unique())}")
print(f"GameDifficulty unique values: {sorted(df['GameDifficulty'].unique())}")
print(f"Gender unique values: {sorted(df['Gender'].unique())}")

# ----------------------------------------------------------------------------------------------------------------------------------

# Train-Test Split
X = df.drop(columns=['EngagementLevel'])
y = df['EngagementLevel']

print("\nBefore Split:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y dtype: {y.dtype}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nAfter Split:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Save feature columns
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, "model/feature_columns.pkl")

# ----------------------------------------------------------------------------------------------------------------------------------

# Class Imbalance Check
print("\nClass Imbalance Check")
IMBALANCE_THRESHOLD = 0.50

counts = y.value_counts()
majority = counts.max()
ratio = counts.min() / majority

print(f"Class distribution:\n{counts.to_string()}")
print(f"\nPer-class ratio vs majority:")
for cls, cnt in counts.items():
    print(f"  Class {cls}: {cnt} ({100 * cnt / len(y):.1f}%) — ratio: {cnt/majority:.2f}")

print(f"\nMinority/Majority ratio: {ratio:.2f}")
print(f"Imbalance threshold: {IMBALANCE_THRESHOLD}")

if ratio < IMBALANCE_THRESHOLD:
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. New class counts: {dict(zip(*np.unique(y_train, return_counts=True)))}")
else:
    print("No SMOTE needed — classes are sufficiently balanced.")

# ----------------------------------------------------------------------------------------------------------------------------------
# Modelling and Parameter Tuning (Already done, so commented out for faster execution)
# ----------------------------------------------------------------------------------------------------------------------------------

# print("\nModel 1: Decision Tree")

# dt_params = {
#     'max_depth':        [3, 5, 10, 15, 20, None],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf':  [1, 2, 5, 10],
#     'criterion':        ['gini', 'entropy']
# }

# dt_base = DecisionTreeClassifier(random_state=42)

# dt_search = RandomizedSearchCV(
#     dt_base, dt_params,
#     n_iter=20, cv=5,
#     scoring='accuracy',
#     random_state=42, n_jobs=-1, verbose=1
# )

# dt_search.fit(X_train, y_train)

# dt_model = dt_search.best_estimator_
# print(f"Best params: {dt_search.best_params_}")
# print(f"Best CV accuracy: {dt_search.best_score_:.4f}")

# # ----------------------------------------------------------------------------------------------------------------------------------

# print("\nModel 2: Random Forest")

# rf_params = {
#     # number of trees
#     'n_estimators': [100, 200, 300],
#     # max tree depth
#     'max_depth': [None, 10, 20, 30],
#     # min samples to split a node
#     'min_samples_split': [2, 5, 10],
#     # min samples at leaf node
#     'min_samples_leaf': [1, 2, 4],
#     # features considered per split
#     'max_features': ['sqrt', 'log2']
# }

# rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# rf_search = RandomizedSearchCV(
#     rf_base, rf_params,
#     n_iter=20, cv=5,
#     scoring='accuracy',
#     random_state=42, n_jobs=-1, verbose=1
# )

# rf_search.fit(X_train, y_train)   

# rf_model = rf_search.best_estimator_
# print(f"Best params: {rf_search.best_params_}")
# print(f"Best CV accuracy: {rf_search.best_score_:.4f}")

# # ----------------------------------------------------------------------------------------------------------------------------------
# print("\nModel 3: XGBoost")

# xgb_params = {
#     # number of boosting rounds
#     'n_estimators': [100, 200, 300],
#     # tree depth
#     'max_depth': [3, 5, 7, 9],
#     # step size shrinkage
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     # row sampling per tree
#     'subsample': [0.7, 0.8, 1.0],
#     # feature sampling per tree
#     'colsample_bytree': [0.7, 0.8, 1.0]
# }

# xgb_base = XGBClassifier(
#     objective='multi:softmax',
#     num_class=3,
#     eval_metric='mlogloss',
#     random_state=42,
#     n_jobs=-1,
#     verbosity=0
# )

# xgb_search = RandomizedSearchCV(
#     xgb_base, xgb_params,
#     n_iter=20, cv=5,
#     scoring='accuracy',
#     random_state=42, n_jobs=-1, verbose=1
# )

# xgb_search.fit(X_train, y_train)   

# xgb_model = xgb_search.best_estimator_
# print(f"Best params: {xgb_search.best_params_}")
# print(f"Best CV accuracy: {xgb_search.best_score_:.4f}")

# ----------------------------------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------------------------------

# Models after parameter tuning 
models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion='entropy', max_depth=10, min_samples_leaf=10, min_samples_split=20, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, min_samples_split=2, min_samples_leaf=1,
        max_features='log2', max_depth=30, random_state=42
    ),
    "XGBoost": XGBClassifier(
        subsample=1.0, n_estimators=200, max_depth=9,
        learning_rate=0.1, colsample_bytree=1.0,
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    )
}

# ----------------------------------------------------------------------------------------------------------------------------------

label_names = ['Low', 'Medium', 'High']
results = {}

# Fit and Train
for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")
    y_pred = model.predict(X_test)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')
    report    = classification_report(y_test, y_pred, target_names=label_names)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    results[name] = {
        "model"    : model,
        "y_pred"   : y_pred,
        "accuracy" : acc,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1
    }

# ----------------------------------------------------------------------------------------------------------------------------------

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

fig.suptitle('Confusion Matrices', fontsize=15, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm   = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name}\nAcc: {res['accuracy']:.4f}", fontsize=11, fontweight='bold')

plt.tight_layout()
_fname = "photo/confusion_matrix.png"
plt.savefig(_fname, bbox_inches='tight')
plt.close()
print(f'\nFile saved "{_fname}"')

# ----------------------------------------------------------------------------------------------------------------------------------

#Bar Chart for Accuracy, Precision, Recall, F1
metrics    = ['accuracy', 'precision', 'recall', 'f1']
titles     = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors     = ['steelblue', 'seagreen', 'tomato']
names      = list(results.keys())

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold')

for ax, metric, title in zip(axes, metrics, titles):
    values = [results[n][metric] for n in names]
    bars   = ax.bar(names, values, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylim(0.5, 1.05)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(title)
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
_fname = "photo/model_performance_comparison.png"
plt.savefig(_fname, bbox_inches='tight')
plt.close()
print(f'\nFile saved "{_fname}"')

# ----------------------------------------------------------------------------------------------------------------------------------

# Summary
print("\nFinal Summary\n")

# CV Accuracy vs Test Accuracy
print(f"{'Model':<25} {'CV Accuracy':>12} {'Test Accuracy':>14}")
print("=" * 55)
cv_scores = {"Decision Tree": 0.9059, "Random Forest": 0.9086, "XGBoost": 0.9167}
for name in results:
    print(f"{name:<25} {cv_scores[name]:>12.4f} {results[name]['accuracy']:>14.4f}")

print()

# Summarize the big 4
print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
print("=" * 65)
for name in results:
    r = results[name]
    print(f"{name:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")
print("=" * 65)