import numpy as np
import pandas as pd

df = pd.DataFrame({
    "IID": [f"M{i}" for i in range(50)],
    "stdPRS_epilepsy": np.random.normal(loc=0, scale=1, size=50)
})

df.to_csv("../data/prs.csv", index=False)
