FROM nvidia/cuda:10.1-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y vim wget curl git build-essential

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH="/miniconda/bin:$PATH"

RUN git clone https://github.com/estorrs/violet.git 
RUN cd violet && git checkout development
RUN conda env create --file violet/env.yml 

# being a little lazy and just reinstalling the correct pytorch version here
RUN conda install -n violet -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# install violet
RUN /miniconda/envs/violet/bin/pip install -e violet

ENV PATH="/miniconda/envs/violet/bin:$PATH"

CMD /bin/bash