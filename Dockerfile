FROM python:3.11.7-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy necessary files
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Install dependencies using Poetry
RUN poetry install --no-root || poetry lock --no-update && poetry install --no-root

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "python", "src/serve/app.py"]