import pandas as pd
df = pd.read_csv('PlayTennis1.csv')
def p_xc(X,c=None):
  P = 1
  df_c = df[df['Play Tennis'] == c].drop('Play Tennis', axis=1) if c else df.drop('Play Tennis', axis=1)
  for i in range(len(X)):
    P = P * df_c[df_c.iloc[:,i] == X[i]].shape[0]/df_c.shape[0]
  return P
def p_c(c):
  return df[df['Play Tennis'] == c].shape[0]/df.shape[0]
def naive_bayes(X):
  p_cx = (p_xc(X,'Yes')*p_c('Yes'))/p_xc(X)
  return 'Yes' if p_cx >= 0.5 else 'No'
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.20)
y_pred = []
for i,_y in y_train.iterrows():
  y_pred.append(naive_bayes(_y))
a=accuracy_score(y_test,y_pred)
print("The Accuracy score is", a*100)
