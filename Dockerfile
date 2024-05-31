FROM python:3.12

WORKDIR /project

COPY /project /project

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3001

CMD streamlit run /project/app.py --server.port=3001