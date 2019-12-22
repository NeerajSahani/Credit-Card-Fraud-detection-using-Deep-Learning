import matplotlib.pyplot as plt, os, pandas as pd
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


#preprocessing

dataset = pd.read_csv('Dataset/Churn_Modelling.csv')

dataset = dataset[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']]

geography = {dataset.Geography.unique()[i]:i for i in range(len(dataset.Geography.unique()))}

dataset.Geography = dataset.Geography.map(lambda x: geography[x])

gender = {dataset.Gender.unique()[i]:i for i in range(len(dataset.Gender.unique()))}

dataset.Gender = dataset.Gender.map(lambda x: gender[x])

sc = StandardScaler()

X = sc.fit_transform(dataset.iloc[:, :10].values)
y = dataset.iloc[:, -1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



#Neural network
classifier = models.Sequential()
classifier.add(layers.Dense(units=10, kernel_initializer='uniform', activation='relu', input_dim=10))
classifier.add(layers.Dense(units=32, kernel_initializer='uniform', activation='relu'))
classifier.add(layers.Dense(units=64, kernel_initializer='uniform', activation='relu'))
classifier.add(layers.Dense(units=128, kernel_initializer='uniform', activation='relu'))
classifier.add(layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = classifier.fit(X_train, y_train, batch_size=10, epochs=100, validation_data = (X_test, y_test), validation_steps=100)


plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


y_pred = classifier.predict(X_test)
y_pred = (y_pred > .5)
cm = confusion_matrix(y_test, y_pred)
