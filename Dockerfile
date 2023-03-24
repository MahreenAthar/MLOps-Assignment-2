FROM python:3.9-slim-buster

WORKDIR /templates

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "Model_StockExchange.py" ]