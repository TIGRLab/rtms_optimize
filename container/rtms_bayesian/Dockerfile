FROM ubuntu:16.04

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
	gcc \
	g++ \
	cmake \
	make \
	python2.7 \
	python-dev \
	python-pip \
	python-setuptools \
	git \
	libboost-all-dev \
	libblas-dev \
	liblapack-dev \
	libglu1 \
	libxrender-dev \
	libxcursor1 \
	libxft2 \
	libxinerama1 \
	gfortran && \
	git clone https://github.com/wujian16/Cornell-MOE.git

#Get GMSH and Simnibs modules
COPY ./gmsh-sdk /gmsh-sdk
COPY ./simnibs /simnibs

#Set up environment for installing Cornell-MOE
ENV	MOE_CC_PATH=/usr/bin/gcc \
	MOE_CXX_PATH=/usr/bin/g++ \
	MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/usr/include/python2.7 -D MOE_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so.1.0" \
	PYTHONPATH=$PYTHONPATH:/simnibs/Python_modules/src/:/gmsh-sdk/lib/

COPY	./requirements.txt requirements.txt

#Install MOE
RUN	pip install -r requirements.txt && \
	cd Cornell-MOE && \
	python setup.py install && \
	cd /

#Add to python path
ENV	PYTHONPATH=$PYTHONPATH:/Cornell-MOE
	
ENTRYPOINT /bin/bash
