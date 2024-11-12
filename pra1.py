import matplotlib.pyplot as plt
import pandas as pd
##PIVOT TABLE
d={
"First_name":["Aryan","Rohan","Riya","Yash","Siddhant"],
"Last_name":["Singh","Agarwal","Shah","Bhatia","Khanna"],
"Type":["Full-Time","Itern","Full-Time","Part-Time","Full-Time"],
"Dept":["Administration","Technical","Administration","Technical","Management"],
'YoE':[2,3,5,7,6],"Salary":[20000,5000,10000,10000,20000]
}
df=pd.DataFrame(d)
print(df)
av=df.pivot_table(index=['Dept', 'Type'], values='Salary',aggfunc='mean') 
print("Average Salary from ecah dept:\n",av)
sm=df.pivot_table(index='Type',values='Salary', aggfunc=['sum', 'mean','count'])
sm.columns=['Total Salary', 'Mean Salary', 'Number of Employees']
print("\nSum and Mean of:\n",sm)
st=df.pivot_table(values='Salary', index='Type',aggfunc='std')
print("\nStandard Deviation:\n",st)
plt.plot(df["YoE"],df["Salary"],'^-',color='black')
plt.plot(df["Salary"],'o-.r')
plt.xlabel("YoE"),plt.ylabel("salary")
plt.title("Salary of a person")
plt.show()