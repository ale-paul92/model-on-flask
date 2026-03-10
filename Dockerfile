FROM python:3.12-slim
LABEL org.opencontainers.image.source=https://github.com/ale-paul92/ml-flask

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "new_app.py"]