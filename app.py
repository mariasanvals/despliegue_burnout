from flask import Flask, jsonify, request, render_template, redirect, url_for
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configura el root_path donde estarán guardados los archivos del modelo y el dataset
root_path = '/home/mariasanvals/despliegue_titanic/'
# root_path = os.getcwd()

app = Flask(__name__)
app.config['DEBUG'] = True

# Enruta la landing page (endpoint /)
@app.route('/', methods=['GET'])
def index():  # Ligado al endpoint "/" o sea el home, con el método GET
    return render_template('index.html')

# Enruta la función al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods=['POST'])
def predict():  # Ligado al endpoint '/api/v1/predict', con el método POST

    # Cargar el modelo y los label encoders
    model = joblib.load(os.path.join(root_path, 'titanic_model.pkl'))
    le_sex = joblib.load(os.path.join(root_path, 'le_sex.pkl'))
    le_embarked = joblib.load(os.path.join(root_path, 'le_embarked.pkl'))

    # Obtener los datos del formulario
    sex = request.form['sex']
    embarked = request.form['embarked']
    pclass = request.form['pclass']
    age = request.form['age']

    if sex is None or embarked is None or pclass is None or age is None:
        return "Faltan argumentos, los datos no son suficientes para predecir"
    else:
        # Preprocesar los datos
        sex = le_sex.transform([sex])[0]
        embarked = le_embarked.transform([embarked])[0]
        pclass = int(pclass)
        age = float(age)

        # Crear el DataFrame con las features
        features = pd.DataFrame([[sex, embarked, pclass, age]], columns=['sex', 'embarked', 'pclass', 'age'])

        # Realizar la predicción
        probabilidad_supervivencia = model.predict_proba(features)[:, 1][0]

        # Renderizar la plantilla con el resultado
        return render_template('index.html', probabilidad=round(probabilidad_supervivencia, 2))

# Enruta la función al endpoint /api/v1/retrain
@app.route('/api/v1/retrain', methods=['POST'])
def retrain():  # Rutarlo al endpoint '/api/v1/retrain', metodo POST
    # Verificar si el archivo CSV existe
    if os.path.exists(os.path.join(root_path, "data/titanic.csv")):
        # Cargar el dataset
        data = pd.read_csv(os.path.join(root_path, "data/titanic.csv"))

        # Preprocesamiento de datos
        features = ['sex', 'embarked', 'pclass', 'age']
        target = 'survived'
        
        # Manejo de valores nulos
        data = data.dropna(subset=features)
        
        # Convertir variables categóricas a numéricas
        le_sex = joblib.load(os.path.join(root_path, 'le_sex.pkl'))
        le_embarked = joblib.load(os.path.join(root_path, 'le_embarked.pkl'))

        # data['sex'] = le_sex.transform(data['sex'])
        # data['embarked'] = le_embarked.transform(data['embarked'])

        # Separar variables independientes y dependientes
        X = data[features]
        y = data[target]

        # Dividir el dataset en entrenamiento y prueba. Cambiado random a 24 desde 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

        # Entrenar un nuevo modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Guardar el nuevo modelo entrenado
        joblib.dump(model, os.path.join(root_path, 'titanic_model.pkl'))

        # Devolver un mensaje de éxito a la plantilla
        return render_template('index.html', retrain_message=f"Modelo reentrenado con éxito. Nueva precisión: {accuracy:.2f}")
    else:
        return render_template('index.html', retrain_message="El archivo CSV del Titanic no se encontró. No se realizó el reentrenamiento.")

if __name__ == '__main__':
    app.run()
