FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    bash

RUN useradd -m -s /bin/bash lgorman
USER lgorman
WORKDIR /home/lgorman

CMD ["/bin/bash"]