import os,sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object, evaluate_model
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2 
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Spliting train and test data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]   
            )
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "CatBoost": CatBoostRegressor(),
                "XgBoost": XGBRegressor(),
            }
            
            
            params = {
                "DecisionTree":{
                   'Criterion': ['squared_error','friedman_mse','absolute_error','poisson'],
            },
            "RandomForest":{
                'n_estimators': [8,16,32]
            },
            "Catboost":{
                'depth':[6,8,10],
                'learning_rate':[0.01,0.05,0.1],
                'iterations': [20,30]     
            },
            "XGBoost":{
                'learning_rate':[.1,0.1,.05],
                'n_estimators':[8,16,32]
            }}
            
            model_report:dict= evaluate_model(X_train, y_train, X_test ,y_test, models, params)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            print("This is the best model")
            print(best_model_name)
            
            
            model_names = list(params.keys())
            
            actual_model=""
            
            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model
            
            best_params = params[actual_model]
            
            mlflow.set_registry_uri("https://dagshub.com/Artistarpit/ml_project.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            #ML-flow:-
            
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                
                # Model registry doesn't work with file store
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
            
                else:
                    mlflow.sklearn.log_model(best_model, "model")
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on train and test dataset")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_squrae = r2_score(y_test, predicted)
            return r2_squrae
        
        except Exception as e:
            raise CustomException(e,sys)
        
                