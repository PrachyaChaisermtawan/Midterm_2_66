from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

File_path = 'D:/65025918/data/'
File_name = 'car_data.csv'

df = pd.read_csv(File_path + File_name)

df.drop(columns=['User ID'], inplace=True)

encoders = []
 
for i in range(0, len(df.columns)-1):
    enc = LabelEncoder()
    df.iloc[:,i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)
    
x = df.iloc[:, 0:5]
y = df['']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x, y) 

x_pred = ['Male',40,20000]

for i in range(0, len(df.columns)-1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
    
x_pred_res = np.array(x_pred).reshape(-1,5)

y_pred = model.predict(x_pred_res)
print('Prediction :', y_pred[0])
 
score = model.score(x, y)
print('Accuracy :', '{:.2f}'.format(score))

feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names = feature,
              class_names = Data_class,
              label = 'all',
              impurity = True,
              fontsize = 16)
plt.show()