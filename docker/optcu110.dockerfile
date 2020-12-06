FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

EXPOSE 8888
USER root

WORKDIR /

RUN apt-get update && apt-get install -y \
	python3.6 curl tar tmux build-essential git ninja-build ccache libopenblas-dev \
	libopencv-dev python3-opencv

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-3.6.0-Linux-x86_64.sh && \
	bash Miniconda3-3.6.0-Linux-x86_64.sh -p /miniconda -b && \
	rm Miniconda3-3.6.0-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

RUN conda update -y conda && \
	conda install -y -c conda-forge jupyterlab
RUN mkdir /.local # && chown ${USER_ID} /.local

RUN git clone https://github.com/mvkvc/options-research /ddpg_daibing
RUN python3 -m pip install -r /ddpg_daibing/docker/requirements.txt

RUN jupyter notebook --generate-config -y
RUN echo "c.NotebookApp.token = '7u%OV1xWG&7m'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_notebook_config.py
