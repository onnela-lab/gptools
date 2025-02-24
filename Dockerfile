FROM python:3.10.0
WORKDIR /src

# Copy files required for installing dependencies.
COPY util/setup.py util/README.rst util/
COPY stan/setup.py stan/README.rst stan/
COPY requirements.txt .

# Install all the dependencies.
RUN pip install -r requirements.txt --no-cache-dir
RUN install_cmdstan --version=2.31.0 --verbose

# Copy over the remainder of the files.
COPY . .

# Move to the workspace directory and set environment variable for outputs.
ENV WORKSPACE=/workspace
RUN mkdir /workspace
