import pandas as pd, json
df = pd.read_csv("output.csv")
df["IPAs"] = df["IPAs"].apply(json.loads)

# Display every single column in the DataFrame
pd.set_option('display.max_columns', None)
# Display every single row in the DataFrame
pd.set_option('display.max_rows', None)

print(df.head(10))