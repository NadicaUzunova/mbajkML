import subprocess
import logging


# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Log to console for GitHub Actions visibility
)


def run_script(script_path):
    """
    Run a Python script using subprocess and log the output.
    """
    try:
        logging.info(f"Running script: {script_path}")
        result = subprocess.run(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            logging.info(f"Successfully ran {script_path}")
            logging.info(f"Output:\n{result.stdout}")
        else:
            logging.error(f"Error running {script_path}: {result.stderr}")
    except Exception as e:
        logging.error(f"Failed to run {script_path}. Exception: {e}")


if __name__ == "__main__":
    logging.info("Starting data processing job...")
    run_script("src/data/fetch_data.py")
    run_script("src/data/preprocess_data.py")
    run_script("src/data/final_processing.py")
    logging.info("Data processing job completed.")