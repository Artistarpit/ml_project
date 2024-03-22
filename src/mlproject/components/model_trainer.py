import os,sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
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
        
                