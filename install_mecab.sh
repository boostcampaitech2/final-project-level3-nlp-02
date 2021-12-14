# ### NLP task
# pip install tweepy==3.9.0
# pip install konlpy

# install mecab
# apt-get install # apt install g++ -y
apt update
apt install default-jre -y
apt install default-jdk -y
apt install npm -y
apt install nodejs
apt-get install build-essentials

npm cache clean -f
npm install -g n
n stable # lts, lastest, 14.15.4


wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
make install
ldconfig

cd ~
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./configure
make
make install
cd ~
apt install curl
apt install git
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
pip install mecab-python