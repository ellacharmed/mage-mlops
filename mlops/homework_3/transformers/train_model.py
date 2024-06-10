import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import root_mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


mlflow.set_tracking_uri(uri="http://mlflow:5000")
mlflow.set_experiment("homework-03")
# mlflow.sklearn.autolog(log_dataset=False)

version = mlflow.__version__
print(f"MLflow version: {version}")
print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

@transformer
def train(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    # print(df.shape)
    # display(df)
    with mlflow.start_run():

        target = 'duration'
        categorical = ['PULocationID', 'DOLocationID']
        y_train = data[target].values

        mlflow.sklearn.autolog(log_datasets=False)

        dv = DictVectorizer()

        train_dicts = data[categorical].to_dict(orient='records')
        X_train = dv.fit_transform(train_dicts)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)

        rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric("rmse", rmse)
        print(f'intercept: {model.intercept_}')
    
    return (dv, model)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'