import numpy as np
s = np.load(r"C:\Users\igorm\Documents\programacao\trabalho-cad-v2\dataset\pca\pca.npy")
print("NaN:", int(np.isnan(s).sum()))          # 0
print("var por PC:", np.round(s.var(0)[:6], 3)) # tem que DECRESCER