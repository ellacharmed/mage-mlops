import mlflow
import mlflow.sklearn
import pickle

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

mlflow.set_tracking_uri(uri="http://mlflow:5000")
mlflow.set_experiment("homework-03")

@custom
def train_lin_reg(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here

    dv, model = data
    # log model
    mlflow.sklearn.log_model(model, artifact_path="models")

    with open("dv.bin", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("dv.bin", artifact_path="dv")

    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
