import pandas as pd

df = pd.read_csv("cpm_trace_cuda_gpu_trace.csv")
print(df.shape)
print(df["Name"].value_counts().head(20))
