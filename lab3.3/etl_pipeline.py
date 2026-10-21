import json
from contextlib import closing
from datetime import datetime
from math import isfinite
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import pandas as pd
import sqlite3


class DataExtractionError(Exception):
    """Raised when sample data cannot be extracted."""


class DataTransformationError(Exception):
    """Raised when extracted data cannot be transformed."""


class DataValidationError(Exception):
    """Raised when data fails a quality or database validation check."""


class DataLoadError(Exception):
    """Raised when validated data cannot be loaded into SQLite."""

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
}


def extract_data_func():
    """Return JSON-serializable records for Airflow's return-value XCom."""
    try:
        data = {'product_id': [1, 2, 3, 4],
                'name': ['Product A', 'Product B', None, 'Product D'],
                'price': [10, 20, 15, 25]}
        return pd.DataFrame(data).to_dict(orient='records')
    except (TypeError, ValueError) as exc:
        raise DataExtractionError(f"Failed to extract sample data: {exc}") from exc


def transform_data_func(ti):
    """Pull extracted records and return transformed records via XCom."""
    records = ti.xcom_pull(task_ids='extract_data')
    if not isinstance(records, list) or not records:
        raise DataTransformationError("No extracted records found in XCom")
    try:
        df = pd.DataFrame(records)
        df['name'] = df['name'].fillna('Unknown Product')
        df['discounted_price'] = df['price'] * 0.9
        return df.to_dict(orient='records')
    except (KeyError, TypeError, ValueError) as exc:
        raise DataTransformationError(f"Failed to transform records: {exc}") from exc


def validate_data_quality_func(ti):
    """Reject invalid records before making them available to the loader."""
    records = ti.xcom_pull(task_ids='transform_data')
    if not isinstance(records, list) or not records:
        raise DataValidationError("No transformed records found in XCom")
    try:
        df = pd.DataFrame(records)
        required_columns = ['product_id', 'name', 'price', 'discounted_price']
        missing_columns = sorted(set(required_columns) - set(df.columns))
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        if df[required_columns].isna().any().any():
            raise DataValidationError("Required columns contain null values")
        if not pd.api.types.is_integer_dtype(df['product_id']):
            raise DataValidationError("product_id must contain integers")
        if not df['name'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise DataValidationError("name must contain non-empty strings")
        for column in ['price', 'discounted_price']:
            if (not pd.api.types.is_numeric_dtype(df[column])
                    or pd.api.types.is_bool_dtype(df[column])):
                raise DataValidationError(f"{column} must be numeric")
            if not df[column].map(isfinite).all():
                raise DataValidationError(f"{column} must contain finite values")
        if (df['price'] <= 0).any():
            raise DataValidationError("price must be greater than zero")
        if ((df['discounted_price'] < 0)
                | (df['discounted_price'] > df['price'])).any():
            raise DataValidationError("discounted_price must be between zero and price")
        if df.duplicated(subset=['product_id']).any():
            raise DataValidationError("Duplicate product_id values found")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise DataValidationError(f"Unable to validate records: {exc}") from exc
    print(f"Data quality checks passed for {len(records)} products")
    return records


def load_data_func(ti, db_path):
    """Load only records returned by the successful quality-validation task."""
    records = ti.xcom_pull(task_ids='validate_data_quality')
    if not isinstance(records, list) or not records:
        raise DataLoadError("No validated records found in XCom")
    try:
        df = pd.DataFrame(records)
        with closing(sqlite3.connect(db_path)) as conn:
            df.to_sql('products', conn, if_exists='replace', index=False)
    except sqlite3.OperationalError as exc:
        raise DataLoadError(
            f"SQLite operation failed for 'products' in {db_path}: {exc}"
        ) from exc
    except (sqlite3.DatabaseError, pd.errors.DatabaseError, TypeError, ValueError) as exc:
        raise DataLoadError(f"Failed to load 'products' into {db_path}: {exc}") from exc
    print(f"Loaded {len(records)} products into {db_path}")
    return len(records)


def validate_loaded_data_func(ti, db_path):
    """Verify that the same database contains the expected number of rows."""
    expected_count = ti.xcom_pull(task_ids='load_data')
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise DataValidationError("No valid loaded row count found in XCom")
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            actual_count = conn.execute('SELECT COUNT(*) FROM products').fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise DataValidationError(f"Failed to validate database {db_path}: {exc}") from exc
    if actual_count != expected_count:
        raise DataValidationError(
            f"Expected {expected_count} products in {db_path}, found {actual_count}"
        )
    print(f"Database validation passed for {actual_count} products")


def create_etl_dag(config):
    """Create one independent pipeline for each entry in the local JSON file."""
    variable_name = json.dumps(config['db_path_variable'])
    default_path = json.dumps(config['default_db_path'])
    db_path = '{{ var.value.get(' + variable_name + ', ' + default_path + ') }}'
    with DAG(
        dag_id=config['dag_id'],
        default_args=default_args,
        schedule_interval=config.get('schedule_interval', '@once'),
        catchup=False,
        max_active_runs=1,
    ) as dag:
        prepare_task = BashOperator(
            task_id='prepare_data_directory',
            bash_command='mkdir -p -- "$(dirname -- "$DB_FILE_PATH")"',
            env={'DB_FILE_PATH': db_path},
            append_env=True,
            do_xcom_push=False,
        )
        extract_task = PythonOperator(
            task_id='extract_data',
            python_callable=extract_data_func,
        )
        transform_task = PythonOperator(
            task_id='transform_data',
            python_callable=transform_data_func,
        )
        quality_task = PythonOperator(
            task_id='validate_data_quality',
            python_callable=validate_data_quality_func,
        )
        load_task = PythonOperator(
            task_id='load_data',
            python_callable=load_data_func,
            op_kwargs={'db_path': db_path},
        )
        validate_task = PythonOperator(
            task_id='validate_data',
            python_callable=validate_loaded_data_func,
            op_kwargs={'db_path': db_path},
        )
        prepare_task >> extract_task >> transform_task >> quality_task >> load_task >> validate_task
    return dag


config_path = Path(__file__).resolve().parents[2] / 'etl_pipelines.json'
with config_path.open() as config_file:
    pipeline_configs = json.load(config_file)

dag_ids = set()
for pipeline_config in sorted(pipeline_configs, key=lambda config: config['dag_id']):
    dag_id = pipeline_config['dag_id']
    if dag_id in dag_ids:
        raise ValueError(f"Duplicate DAG ID in {config_path}: {dag_id}")
    dag_ids.add(dag_id)
    globals()[dag_id] = create_etl_dag(pipeline_config)