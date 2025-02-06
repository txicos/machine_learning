FROM public.ecr.aws/docker/library/python:3.11-bullseye
 
ARG DEPLOY_CMD="streamlit run chat_html.py --server.port 8002 --server.runOnSave true"

ENV DEPLOY_CMD=${DEPLOY_CMD}

WORKDIR /usr/app

COPY ./app /usr/app
COPY ./requirements.txt /usr/app

RUN apt-get update && apt-get install -y \
    gnupg \
    firefox-esr \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Add NodeSource repository and install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN GECKODRIVER_VERSION=$(curl -s https://api.github.com/repos/mozilla/geckodriver/releases/latest | grep 'tag_name' | cut -d '"' -f 4 | sed 's/v//') \
    && wget -q "https://github.com/mozilla/geckodriver/releases/download/v$GECKODRIVER_VERSION/geckodriver-v$GECKODRIVER_VERSION-linux64.tar.gz" -O /tmp/geckodriver.tar.gz \
    && tar -xzf /tmp/geckodriver.tar.gz -C /usr/local/bin \
    && chmod +x /usr/local/bin/geckodriver \
    && rm /tmp/geckodriver.tar.gz

RUN pip install -r /usr/app/requirements.txt


ENTRYPOINT ["bash", "docker-entrypoint.sh"]

CMD ${DEPLOY_CMD}