import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("seattle-weather.csv")
print(df.head())

X= df[["precipitation",  "temp_max" , "temp_min" , "wind" ]]
y = df["weather"]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.5, random_state=50)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

pickle.dump(classifier, open("model.pkl", "wb"))
