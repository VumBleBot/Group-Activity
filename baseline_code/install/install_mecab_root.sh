### mecab 설치용 쉘 스크립트 파일
#!/bin/bash 

# 파이썬 설치
apt-get -y install libncursesw5-dev libgdbm-dev libc6-dev zlib1g-dev libsqlite3-dev tk-dev
apt-get -y install build-essential python-dev python-setuptools python-pip python-smbus
apt-get -y install libssl-dev openssl libffi-dev python-dev python3-dev
wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz
tar xvfz Python-3.6.9.tgz
cd $HOME/Python-3.6.9 && ./configure
cd $HOME/Python-3.6.9 && make
cd $HOME/Python-3.6.9 && make install
yes | rm -R $HOME/Python-3.6.9
cd $HOME/Python-3.6.9 && ./c.9.tgz

# java 설치
apt -y install openjdk-8-jdk
# pip3 설치 및 konlpy 설치
apt -y install python3-pip
pip3 install konlpy

# mecab_ko 설치
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2 && ./configure
make
make check
make install
yes | 
cd .. && rm -R mecab-0.996-ko-0.9.2.tar.gz
# mecab_ko_dict 설치 
apt -y install automake1.11
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720 && wget https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh
./autogen.sh
./configure
ldconfig
make 
make install
yes | 
cd .. && rm -R mecab-ko-dic-2.1.1-20180720.tar.gz
# mecab pytho 설치 
pip3 install mecab-python3

