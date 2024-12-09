FROM python:3.11-slim
WORKDIR /src

# install application dependencies
COPY ./requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt

COPY ./src /src
EXPOSE 8501

# CMD ["streamlit", "run", "frontend/app.py", "--server.port=3000", "--server.enableCORS=false"]
CMD ["streamlit", "run", "frontend/app.py"]
