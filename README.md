## EXNO-3-DS
#### NAME : PREETHI D
#### REGISTER NUMBER : 212224040250
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/Encoding Data (2).csv")
df
```
    ![image](https://github.com/user-attachments/assets/86ed34f4-b1e9-4464-a0af-9a60d0e0ca6c)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm = ['Hot','Warm','Cold']
```
```
e1=OrdinalEncoder(categories=[pm])
```
```
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/d2fcdf3a-d69d-41c8-8a89-2b1c2b166f5a)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/b4467a5c-4162-4011-b50c-1ce0b59878ff)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/9f8c929b-cff8-4b4c-9e23-7a0d6401cc2c)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
```
```
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/a90f9018-2ea6-4a74-9c55-c125bd8a5807)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/4b82dfea-88c4-4023-a16d-4b2bb0867e18)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/22adc802-0113-44a4-adfd-221b076f89ba)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/data.csv")
```
```
df
```
![image](https://github.com/user-attachments/assets/ca3fc96e-5640-4185-a1ef-6d9ff89bec94)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/07196b4c-d5f0-4287-993d-ae9ad3158ed5)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/68fc7c7e-6423-4acd-b81e-a8eb5aa1d3de)
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/207b6d58-9f93-4f66-a296-8616fcc84abe)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/106ac3ff-5378-4a23-850f-644df934d719)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f4163122-4309-4c85-9ecb-828bdbbd5ce2)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2d15b5fc-deea-44c8-a013-d656413b1c56)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/85dc47cb-421a-4146-81da-5ce9afb9bbae)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/15bbfafc-bd8d-4347-a942-d45b7fe13a42)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/be994f06-e1b9-4dae-84e2-c8947791b62b)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/47820774-6a7c-4b06-8f2a-8a8408278f4f)
```
df["Highly Negative Skew_yeohohnson"],parameter=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/8fbce5c3-9553-4f5f-a6b0-b3bb727b5ecf)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Modern Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/51650479-9ddf-4554-8a73-5c9b4c219d88)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
```
![image](https://github.com/user-attachments/assets/f0fb502c-d13e-4694-89db-590604691f3f)
```
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/d5aae494-841f-461b-bc52-0876ee9b57be)
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/fcab3f35-405c-4526-b945-7935c8b44462)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a3c5349d-6503-465a-a54a-49bdf9135c6c)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ce67379d-0946-44c1-b2b6-450b05fd7159)

# RESULT:
      Thus the process of Feature Encoding and Transformation has been successfully performed on the data and the output has been attached.

       
