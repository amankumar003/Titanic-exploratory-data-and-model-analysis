import numpy as np
from pickle import load
from requests import post

def format_data(input_data: dict, scaler: object) -> np.ndarray:
    # input_data['Sex'] = 1 if input_data['Sex'].lower() == 'male' else 0

    data = [
        input_data['Sex'],
        input_data['SibSp'],
        input_data['Parch'],
        input_data['Fare'],
        any((input_data['Parch'], input_data['SibSp'])),
        input_data['Age'],
        input_data['Pclass']==1,
        input_data['Pclass']==2,
        input_data['Embarked']=='S',
        input_data['Embarked']=='C',
    ]
    data = np.array(data).reshape(1, -1)
    indexes = [1, 2, 3, 5]
    data[:, indexes] = scaler.transform(data[:, indexes])

    return data

def load_model(model_path: str) -> dict:
    with open(model_path, 'rb') as f:
        model_data = load(f)
    return model_data

def predict_survival_chance(input_data: dict) -> float:
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn


    model_data = load_model('models/svm.pkl')
    model, scaler = model_data['model'], model_data['scaler']

    processed_data = format_data(input_data, scaler)
    prediction = model.predict_proba(processed_data)[0][1]

    return prediction


test_data = {
    'Pclass': 1,
    'Sex': 0,
    'Age': 38.0,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 71.2833,
    'Embarked': 'C'
}

def call_api():
    print('Calling API...')
    url = 'http://localhost:8000/predict'
    response = post(url, json=test_data)
    print(response.json())

if __name__ == '__main__':

    print(f' Initial data: {test_data}')
    print(f' Predicted survival chance: {predict_survival_chance(test_data):.2%}')

    call_api()
