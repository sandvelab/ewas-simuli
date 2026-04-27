import numpy as np
import pandas as pd

df = pd.DataFrame({
    "MID": [f"M{i}" for i in range(50)],
    "CID": [f"{i}" for i in range(50)],
    "stdPRS_height": np.random.normal(loc=0, scale=1, size=50)
})

df.to_csv("../data/prs_height.csv", index=False)
