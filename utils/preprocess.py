import numpy as np

def preprocess_input(data, scaler):

    data = data.copy()

    data['Amount'] = data['Amount'].fillna(0)
    data['Time'] = data['Time'].fillna(0)

    data['Amount_log'] = np.log1p(data['Amount'])
    data['Hour'] = (data['Time'] % 86400) / 3600

    features = [f'V{i}' for i in range(1, 29)] + ['Amount_log', 'Hour']

    for col in features:
        if col not in data.columns:
            data[col] = 0

    X = data[features]

    return scaler.transform(X)