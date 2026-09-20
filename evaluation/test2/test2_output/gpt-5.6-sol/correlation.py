import pandas as pd, numpy as np

ev = pd.read_csv("test2_event_results.csv")
piv = ev.pivot(index="event_date", columns="alpha_id", values="step2_d_bar_event")
c = piv.corr()
iu = np.triu_indices_from(c, k=1)
print(f"median r = {np.median(c.values[iu]):.3f}   "
      f"range {c.values[iu].min():.3f} to {c.values[iu].max():.3f}")
