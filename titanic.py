# fastscore.schema.0: input_schema.avsc
# fastscore.schema.1: output_schema.avsc

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle


# modelop.init
def begin():
    global model
    model = pickle.load(open('model.pkl', 'rb'))


# modelop.training
def train(train_df):

    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']


    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

    categorical_features = X_train.select_dtypes(include=['object']).columns


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])



    model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier())])

    model.fit(X_train, y_train)

    pickle.dump(model, open('model.pkl', 'wb'))

# modelop.metrics
def metrics(df):

    X_test = df.drop('Survived', axis=1)
    y_test = df['Survived']
    yield { "ACCURACY": model.score(X_test, y_test)}

# modelop.score
def predict(X):
    df = pd.DataFrame(X, index=[0])
    y_pred = model.predict(df)
    for p in y_pred:
        yield p



if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    pred_df = pd.read_csv('predict.csv')

    train(train_df)
    begin()

    X = [[519,2,"Bob","male",36.0,1,0,226875,26.0,'C26',"S"]]
    print(predict(X))

    for m in metrics(test_df):
        print(m)







