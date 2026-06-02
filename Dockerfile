FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /tmp

EXPOSE 8050

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "1", "--timeout", "120"]