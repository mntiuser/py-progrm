import pandas as pd
data=pd.read_csv(r"C:\\Users\\thela\Downloads\\auto-mpg.csv")
df=pd.DataFrame(data)
print(df)
m1=df['horsepower'].mean()
print("mean:",m1)
