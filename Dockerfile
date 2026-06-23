FROM python:3.10-slim

# تثبيت ffmpeg وأدوات النظام
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# تثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ الكود
COPY . .

# المنفذ
EXPOSE 8000

# تشغيل التطبيق
CMD ["python", "app.py"]
