import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dotenv import load_dotenv
import pymysql
import pandas as pd

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
    
    
