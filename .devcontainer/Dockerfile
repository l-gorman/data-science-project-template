FROM ubuntu:latest

WORKDIR /home/ubuntu


RUN apt-get update && apt-get install -y \
    bash \
    git \
    python3 \
    curl \
    build-essential 


RUN echo $PATH
RUN echo $HOME



ENV PATH="/root/.local/bin/:$PATH"

    
# Install Pixi for python dependency management
RUN curl -fsSL https://pixi.sh/install.sh | sh -s -- -y
RUN mv /root/.pixi/bin/* /usr/local/bin/

# Add to path
ENV PATH="/root/.pixi/bin:${PATH}"


# Install Rust/Cargo for just setup files
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y 

ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup install stable \
    && rustup default stable

RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
RUN echo 'source $HOME/.cargo/env' >> /home/ubuntu/.bashrc


    
RUN cargo install just

RUN mv /root/.cargo/bin/* /usr/local/bin/

RUN echo 'export PATH="/usr/local/bin:/usr/local/sbin:$PATH"' > /etc/profile.d/global_path.sh

CMD ["/bin/bash"]