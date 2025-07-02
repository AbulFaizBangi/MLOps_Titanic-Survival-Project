from airflow import DAG # Import the DAG class
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator # Import the operator to download files from GCS
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.operators.python import PythonOperator # Import the Python operator to run custom Python code
from airflow.hooks.base_hook import BaseHook   # Import BaseHook to get connection details
from datetime import datetime # Import datetime for scheduling
import pandas as pd # Import pandas for data manipulation
import sqlalchemy # Import SQLAlchemy for database interaction

## Transfer data from GCS to PostgreSQL
def load_to_sql(file_path):
    conn = BaseHook.get_connection('postgres_default')  
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@titanic-survival_af1bdd-postgres-1:{conn.port}/{conn.schema}")
    df = pd.read_csv(file_path)
    df.to_sql(name="titanic", con=engine, if_exists="replace", index=False)

# Define the DAG
with DAG(
    dag_id="extract_titanic_data",
    schedule_interval=None, 
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Extract data from GCS and load it into PostgreSQL
    list_files = GCSListObjectsOperator(
        task_id="list_files",
        bucket="my-titanic-bucket", 
    )

    download_file = GCSToLocalFilesystemOperator(
        task_id="download_file",
        bucket="my-titanic-bucket",
        object_name="Titanic-Dataset.csv", 
        filename="/tmp/Titanic-Dataset.csv", 
    )

    ## Load the data into PostgreSQL
    load_data = PythonOperator(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        op_kwargs={"file_path": "/tmp/Titanic-Dataset.csv"}
    )

    list_files >> download_file >> load_data