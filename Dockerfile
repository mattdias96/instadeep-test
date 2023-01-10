FROM python:3.7.4-slim

RUN apt-get update && apt-get install -y build-essential

RUN pip install matplotlib==3.4.1 numpy==1.18.5 pandas==1.2.3 pytorch-lightning==1.5.3 seaborn==0.11.1 tensorboard==2.2.2 torch==1.8.1 torchmetrics==0.6.0

CMD ["bash"]
