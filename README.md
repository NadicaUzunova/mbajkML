# mbajkML

**mbajkML** is a machine learning project that utilizes tools such as DVC (Data Version Control), Great Expectations for data validation, and MLflow for experiment tracking. The project is structured to facilitate efficient data management, model training, and evaluation.

## Project Structure

The repository is organized as follows:

- `.dvc/`: Contains DVC configuration files for data version control.
- `.github/workflows/`: Includes GitHub Actions workflows for CI/CD pipelines.
- `data/`: Directory designated for storing datasets.
- `great_expectations/`: Houses configurations and scripts for data validation using Great Expectations.
- `mlruns/`: Directory for MLflow experiment tracking artifacts.
- `reports/`: Contains generated reports and analysis results.
- `src/`: Source code for data processing, model training, and evaluation.

## Installation

To set up the project environment, follow these steps:

1. **Install Poetry**: Ensure that [Poetry](https://python-poetry.org/docs/#installation) is installed on your system.

2. **Install Dependencies**: Run the following command to install the necessary dependencies:

   ```bash
   poetry install
   ```
