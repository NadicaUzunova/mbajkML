import time
import subprocess
import logging
from datetime import datetime, timedelta


# Configure logging
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_script(script_path):
    """
    Run a Python script using subprocess and log the output.
    """
    try:
        logging.info(f"Starting script: {script_path}")
        result = subprocess.run(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            logging.info(f"Successfully ran {script_path}")
            logging.info(f"Output: {result.stdout}")
        else:
            logging.error(f"Error running {script_path}: {result.stderr}")
    except Exception as e:
        logging.error(f"Failed to run {script_path}. Exception: {e}")


def job():
    """
    Job to run all scripts consecutively.
    """
    logging.info("Starting data processing job...")
    run_script("src/data/fetch_data.py")
    run_script("src/data/preprocess_data.py")
    run_script("src/data/final_processing.py")
    logging.info("Data processing job completed.")


if __name__ == "__main__":
    logging.info("Scheduler started. Waiting for the next scheduled time...")

    # Set the interval to run the job (1 hour and 15 minutes)
    interval = timedelta(hours=1, minutes=15)
    next_run_time = datetime.now()

    while True:
        current_time = datetime.now()
        if current_time >= next_run_time:
            job()
            next_run_time = current_time + interval
            logging.info(f"Next job scheduled for {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Sleep for a short interval to prevent busy waiting
        time.sleep(30)