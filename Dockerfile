FROM python:3.11

WORKDIR /app

# Copy the requirements file
COPY ./src /app
COPY ./pyproject.toml /app

# Install the requirements
RUN pip install -e .

# Run the app
ENTRYPOINT ["llm-repl"]
