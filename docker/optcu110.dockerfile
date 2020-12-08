FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive

EXPOSE 8888
USER root
WORKDIR /


RUN apt-get update && apt-get install -y \
	build-essential git python3-pip vim
	
RUN git clone https://github.com/mvkvc/options-research /ddpg_daibing
RUN python3 -m pip install -r /ddpg_daibing/docker/requirements.txt

RUN jupyter notebook --generate-config -y
RUN echo "c.NotebookApp.token = '7u%OV1xWG&7m'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_notebook_config.py
