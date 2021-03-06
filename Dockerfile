# Multi-Stage build from https://pythonspeed.com/articles/multi-stage-docker-python/
FROM python:3.8-slim AS compile-image
RUN apt-get -y update && apt-get -y upgrade && apt-get clean
RUN apt-get install -y --no-install-recommends build-essential gcc
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements/base.txt .
RUN . /opt/venv/bin/activate && pip install -U -r base.txt


FROM python:3.8-slim AS build-image
RUN useradd --create-home --shell /bin/bash anon_user
RUN apt-get -y update && apt-get -y upgrade && apt-get clean

WORKDIR /home/anon_user
USER anon_user

COPY --from=compile-image /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=anon_user anon ./anon
COPY --chown=anon_user data/datasets/bt_dataset_extended.csv ./examples/bt_dataset_extended.csv
COPY --chown=anon_user data/configurations/bt_dataset.yaml ./examples/bt_dataset_config.yaml
COPY --chown=anon_user data/datasets/paper_example.csv ./examples/paper_blog_example.csv
COPY --chown=anon_user data/configurations/blog_authorship_corpus.yaml ./examples/paper_blog_config.yaml
CMD ["bash"]