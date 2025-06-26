# TODO: use an image with Python 3.12 and uv installed
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies
RUN make install

# Set the default command to run the training pipeline
CMD ["make", "train"]
