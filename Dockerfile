FROM python:3.7.4-slim

RUN apt-get update && apt-get install -y build-essential

RUN pip install -r requirements.txt

CMD ["bash"]
