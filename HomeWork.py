import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings(action="ignore")

def sample_as_img(sample):
    label = sample[0]
    img = np.reshape(sample[1:], (28, 28))
    return img, label

# Загрузка данных
df = pd.read_csv("mnist_train.csv")
df_test = pd.read_csv("mnist_test.csv")

X = df.iloc[:, 1:].values / 255.0
y = df.iloc[:, 0].values

X_test = df_test.iloc[:, 1:].values / 255.0
y_test = df_test.iloc[:, 0].values

idx = 123
label = df.iloc[idx, 0]
sample = df.iloc[idx, 1:]
sample = np.reshape(sample, (28, 28))

plt.title(f"Number: {label}")
plt.imshow(sample, cmap="gray")
plt.show()

fix, axes = plt.subplots(ncols=10, figsize=(15, 5))
for i in range(10):
    sample = df[df["label"] == i].iloc[0]
    img, label = sample_as_img(sample)
    ax = axes[i]
    ax.imshow(img, cmap="gray")
    ax.set_title(label)
    ax.axis("off")
plt.tight_layout()
plt.show()

n_folds = 5
penalties = ['l1', 'l2', 'elasticnet', None]  # Значения параметра penalty
solver_map = {'l1': 'liblinear', 'l2': 'lbfgs', 'elasticnet': 'saga', None: 'lbfgs'}  # Подходящие солверы
results = []

print("Метрика: Accuracy")

for penalty in penalties:
    print(f"\n### Testing with penalty = {penalty} ###")
    solver = solver_map[penalty]

    try:
        for fold_idx, (train_idxs, valid_idxs) in enumerate(KFold(n_splits=n_folds).split(X)):
            x_train, y_train = X[train_idxs], y[train_idxs]
            x_valid, y_valid = X[valid_idxs], y[valid_idxs]

            model = LogisticRegression(penalty=penalty, solver=solver, max_iter=500, l1_ratio=0.5 if penalty == 'elasticnet' else None)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_valid)
            report = classification_report(y_valid, y_pred, output_dict=True)
            print(f"[Fold {fold_idx + 1}/{n_folds}] Accuracy: {report['accuracy']:.4f}")

        model.fit(X, y)
        y_test_pred = model.predict(X_test)
        final_report = classification_report(y_test, y_test_pred, output_dict=True)
        print(f"Test Accuracy: {final_report['accuracy']:.4f}")

        results.append((penalty, final_report['accuracy'], final_report))

    except Exception as e:
        print(f"Error with penalty={penalty}: {e}")

best_model = max(results, key=lambda x: x[1])
print("\nBest Model:")
print(f"Penalty: {best_model[0]}, Test Accuracy: {best_model[1]:.4f}")
print("Full Report:", best_model[2])