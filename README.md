# Python implementation of Friedman's H statistic

Friedman's H calculates, for any ML model, the interaction strength of feature pairs,
check the documentation in the [code](hstats.py).

The implementation is a copy of this PR https://github.com/scikit-learn/scikit-learn/pull/28375 as per June 2025.

Requires scikit-learn >= 1.5

See the [example](example.ipynb) for this code:

## Example

```py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from hstats import h_statistic

X, y = load_diabetes(return_X_y=True, as_frame=True)

est = HistGradientBoostingRegressor(max_iter=50, max_depth=4).fit(X, y)

# Get the top 4 most important features
m = 4
imp = permutation_importance(est, X, y, random_state=0)
top_m = X.columns[np.argsort(imp.importances_mean)[-m:]]

# Calculate H statistic for the top features
H = h_statistic(est, X=X, features=top_m, random_state=4)

H_df = pd.DataFrame(
    {
        "H2": H["h_squared_pairwise"].flatten(),
        "H_unnormalized": np.sqrt(H["numerator_pairwise"]).flatten(),
    },
    index=[str(pair) for pair in H["feature_pairs"]],
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for col, title, ax in zip(H_df.columns, ("$H^2$", "Unnormalized $H$"), axes):
    H_df[col].sort_values().plot.barh(ax=ax, title=f"{title}", xlabel=f"{title}")
plt.tight_layout()
```

![image](example.png)