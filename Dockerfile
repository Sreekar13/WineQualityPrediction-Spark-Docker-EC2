FROM ubuntu
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    ln -s /usr/bin/python3 python
ENV TZ=US/Eastern
RUN mkdir -p /dataset
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 
RUN pip3 install pyspark && \
    pip3 install numpy && \
    apt update && \
    apt install -y default-jre
RUN useradd -d /home/ec2-user -ms /bin/bash -g root -G sudo -p ec2-user ec2-user
USER ec2-user
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3
ADD modelLoad.py /
ADD target /
ENTRYPOINT ["python3","/modelLoad.py"]
