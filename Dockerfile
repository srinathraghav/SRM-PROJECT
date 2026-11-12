FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

EXPOSE 8501 8000

CMD ["./start.sh"]