# SalaryPredictionML
Создание классификационных моделей и SVM для предсказания заработку человека по его образованию и возрасту
## Проделанная работа:
- Загрузка и первичный анализ данных
- Визуализация отдельных признаков
- Выбор признаков и кодирование категориальных переменных
- Кодирование целевой переменной
-
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
df = pd.read_csv("adult.csv")
df.isnull().sum()
df.shape
df.head()
df.info()

sns.boxplot(x="income", y="age", data=df)
plt.show()
![Возраст по доходу](boxplot_income_age.png)


sns.barplot(x="income", y="hours-per-week", data=df)
plt.show()

selectedColumns = df[ [ 'age', 'education', 'income']]
X = pd.get_dummies( selectedColumns, columns = [ 'education' ] )
del X['income']
le = LabelEncoder()
le.fit( df['income'] )
le.classes_
y = pd.Series( data = le.transform( df['income'] ) )
y.head()
model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit( X_train, y_train )
predictions = model.predict(X_test)
predictions[:5]
model.predict(X_test)
model.score(X, y)
model.score(X_train, y_train)
model.score(X_test,y_test)
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler
model2 = SVC()
clf = make_pipeline(StandardScaler(), SVC()) 
clf.fit(X_train,y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)
svc = SVC()
svc.fit(X_train, y_train)
svc.score(X_train, y_train)
svc.score(X_test, y_test)
```python
