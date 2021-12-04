conda install python=3.8.5 -y
pip install transformers==4.11.0
pip install datasets==1.15.1
conda install pytorch cudatoolkit=11.0 -c pytorch -y
pip install optuna==2.10.0

pip install numpy
pip install pandas
pip3 install sklearn
pip install matplotlib
pip install seaborn
pip install jupyter jupytext notebook ipykernel environment_kernels

apt-get install tmux -y
pip install wandb


# ### NLP task
# pip install tweepy==3.9.0
# pip install konlpy

# install mecab
# apt-get install sudo
# sudo apt install g++ -y
# sudo apt update
# sudo apt install default-jre -y
# sudo apt install default-jdk -y
# sudo apt install npm -y
# sudo apt install nodejs
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
# sudo make install
# sudo ldconfig
# cd ~
# wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
# tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
# cd mecab-ko-dic-2.1.1-20180720
# ./configure
# make
# sudo make install
# cd ~
# apt install curl
# apt install git
# bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
# pip install mecab-python