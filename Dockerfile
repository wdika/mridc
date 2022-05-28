ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.04-py3


# build an image that includes only the mridc dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `mridc-deps`)
FROM ${BASE_IMAGE} as mridc-deps

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    libsndfile1 sox \
    libfreetype6 \
    python-setuptools swig \
    python-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install llvmlite==0.37.0rc2 --ignore-installed

# install mridc dependencies
WORKDIR /tmp/mridc
COPY requirements .
RUN for f in $(ls requirements*.txt); do pip install --disable-pip-version-check --no-cache-dir -r $f; done

# copy mridc source into a scratch image
FROM scratch as mridc-src
COPY . .

# start building the final container
FROM mridc-deps as mridc
ARG MRIDC_VERSION=0.1.0

# Check that MRIDC_VERSION is set. Build will fail without this. Expose MRIDC and base container
# version information as runtime environment variable for introspection purposes
RUN /usr/bin/test -n "MRIDC_VERSION" && \
    /bin/echo "export MRIDC_VERSION=${MRIDC_VERSION}" >> /root/.bashrc && \
    /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc
# TODO: remove sed when PTL has updated their torchtext import check
RUN --mount=from=mridc-src,target=/tmp/mridc cd /tmp/mridc && pip install ".[all]" && \
    sed -i "s/_module_available(\"torchtext\")/False/g" /opt/conda/lib/python3.8/site-packages/pytorch_lightning/utilities/imports.py && \
    python -c "import mridc.collections.reconstruction as mridc_reconstruction"

# TODO: Update to newer numba 0.56.0RC1 for 22.03 container if possible
# install pinned numba version
# RUN conda install -c conda-forge numba==0.54.1

# copy scripts/examples/tests into container for end user
WORKDIR /workspace/mridc
COPY tests /workspace/mridc/tests
# COPY README.rst LICENSE /workspace/mridc/

RUN printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && \
    chmod +x start-jupyter.sh
