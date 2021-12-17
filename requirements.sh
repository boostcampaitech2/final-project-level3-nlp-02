conda install python=3.8.5 -y
pip install transformers==4.11.0
pip install datasets==1.15.1
pip install torch==1.10.0
pip install optuna==2.10.0

pip install numpy
pip install pandas
pip3 install sklearn
pip install matplotlib
pip install seaborn
pip install jupyter jupytext notebook ipykernel environment_kernels

pip install python-dotenv
apt-get install tmux -y
pip install wandb

# for rouge
pip install rouge-score
pip install absl-py
pip install nltk
pip install numpy
pip install six>=1.14

# for viz
pip install bertviz
pip install jupyter-dash
pip install plotly
pip install jupyter-dash
pip install streamlit==1.1.0

# ### NLP task
pip install kss
pip install tweepy==3.9.0
pip install konlpy

# ###Soft Pruning task
python -m pip install -U nn_pruning

# install mecab
# apt install g++ -y
# apt update
# apt install default-jre -y
# apt install default-jdk -y
# apt install npm -y
# apt install nodejs
# apt-get install build-essentials

# npm cache clean -f
# npm install -g n
# n stable # lts, lastest, 14.15.4


# wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
# tar xvfz mecab-0.996-ko-0.9.2.tar.gz
# cd mecab-0.996-ko-0.9.2
# ./configure
# make
# make check
# make install
# ldconfig
# cd ~
# wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
# tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
# cd mecab-ko-dic-2.1.1-20180720
# ./configure
# make
# make install
# cd ~
# apt install curl
# bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
# pip install mecab-python