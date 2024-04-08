FROM python:3.11-slim
WORKDIR /app
COPY . .
EXPOSE 8000
RUN pip install -r requirements.txt

#docker build -t api .
#docker images
#docker stop api
#docker rm api
#docker run -it -d -p 8000:8000 --name api api python -m uvicorn main:app --host
#docker exec -it -u0 api bash