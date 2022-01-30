from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

dataset=load_iris()
X_tr,X_tt,y_tr,y_tt=tts(dataset["data"],dataset["target"],random_state=0)  #tt" test, tr: train

kn=KNeighborsClassifier(n_neighbors=1)
kn.fit(X_tr,y_tr)

for i in range(len(X_tt)):
    x=X_tt[i]
    x_new=np.array([x])
    prediction=kn.predict(x_new)
    print("TARGET=",y_tt[i],dataset["target_names"][y_tt[i]],"PREDICTED=",prediction,dataset["target_names"][prediction])
print(kn.score(X_tt,y_tt))
