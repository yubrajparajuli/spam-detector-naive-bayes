#baseimage

FROM python:3.11-slim

# set working directory
WORKDIR /app

# copy requirements
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# download nltk data
RUN python -m nltk.downloader stopwords punkt wordnet

# copy project files
COPY main.py .
COPY src/ src/
COPY data/ data/

# create output dirs
RUN mkdir -p outputs/figures outputs/model

# run pipeline
CMD ["python", "main.py"]