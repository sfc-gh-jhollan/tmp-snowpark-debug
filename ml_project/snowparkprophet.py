# warning suppresion
import warnings

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
from snowflake.ml.model import custom_model

warnings.simplefilter("ignore")


class SnowParkProphet(custom_model.CustomModel):
    def __init__(
        self, context: custom_model.ModelContext | None = None, session=None
    ) -> None:
        super().__init__(context)

        with open(self.context.path("model_file")) as f:
            self.model: Prophet = model_from_json(f.read())

    @custom_model.inference_api
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        define prophet's predict method here
        """
        X.columns = ["ds"]
        # X["ds"] = pd.to_datetime(X["ds"])
        forecast = self.model.predict(X)
        forecast["ds"] = forecast["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
