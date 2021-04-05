FROM ubuntu:20.04

# installing needed packages for pyhton
RUN apt update
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential

# copy the whole repo
COPY . .

# set as a workdir
WORKDIR .

# installing our python script requirements
RUN pip3 install -r requirements.txt


# executing the script
CMD ["python3", "mains/main.py"]