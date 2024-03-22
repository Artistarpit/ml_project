import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dotenv import load_dotenv
import pymysql
import pandas as pd
import pickle
import numpy as np

load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
passwd = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    logging.info("Reading sql database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=passwd,
            db=db
        )
        logging.info("Completed connection to sql database",mydb)
        
        df = pd.read_sql_query('select * from student',mydb)
        print(df.head())
        return df
    
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)        
