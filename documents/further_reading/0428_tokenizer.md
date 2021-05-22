## 📌  요약

- (수행 전) Tokenizing 종류를 조사하고, 한국어 및 현재 Task에 적합한 방식 탐색
- (수행 후) **SentencePiece나 huggingface SentencePiece tokenizer(+를 사용한 모델) 위주로 탐색**

## 📔  상세

### Before

- 최근에 가장 자주 사용되는 Tokenizer
  - WordPiece와 SentencePiece Tokenizing 방식의 차이
- 한국어에 자주 사용되는 Tokenizing 방식
- 우리 데이터셋에 알맞은 Tokenizing 방식(제안)

### After

## 1. Tokeninzer의 종류 조사

### Intro

- Tokenizing은 corpus를 전처리하는 과정 중 하나로, corpus를 의미있는 단위인 토큰으로 나누는 작업을 의미한다.
- 토큰의 기준을 문장으로 할 경우의 토큰화를 Sentence Tokeninzation, 단어으로 할 경우 Word Tokenization이라고 한다.
    - 한국어 기준 Sentence Tokenization : `KSS`
    - 한국어 기준 Word Tokenization은 큰 의미가 없다. 그 이유는 뒤에서 설명한다.

### Subword Tokenization

- 학습하며 기록해둔 vocab에 없는 단어를 마주쳤을 때  OOV(Out-Of-Vocabulary) 문제를 해결하기 위해 Subword 알고리즘이 필요해졌음.
    - `비트코인` - NN // `알트코인` - UNK
    - subword로 분리하면 `비트+코인` - NN+NN // `알트+코인` - UNK+NN
    - 즉, 기존에는 UNK로 분류되던 단어들을 분리하여 아는 단어로 형태소를 분석할 수 있게됨.
- Character와 Word 사이의 애매모호했던 경계선상의 단어들을 포착할 수 있게 됨
    - 이를 subword (unit) 또는 wordpiece 등의 이름으로 부른다.

### Byte Pair Encoding

- [https://arxiv.org/pdf/1508.07909.pdf](https://arxiv.org/pdf/1508.07909.pdf) 최초 언급 논문
- BPE는 대표적인 Subword 분리 알고리즘(즉, segment의 개념)
    - 원래 개념은 압축 알고리즘 - 연속적으로 많이 등장한 문자열을 찾아서 병합하여 치환
    - NLP에서는 bottom-up 방식으로 수행.
        - 일단 단어를 모든 음절로 쪼갠 뒤, 가장 자주 사용되는 연속된 문자열을 unigram으로 치환
- BPE의 좋은 점은, 언어별로 언어학적 분석이 들어갈 필요도 없다는것.
    - 자동적으로 언어에서 하위 의미를 갖는 'subword'를 추출해내는 방법이므로, 굳이 형태소학적 추가 분석을 할 필요가 없음.
- 먼저 훈련 데이터를 단어단위(우리나라는 형태소단위)로 분절하는 pre-tokenize 이후에 수행되어야함.
- `GPT-2`, `Roberta` 등이 사용

### WordPiece

- 최초 논문 - 일본어/한국어 검색문제(2012) [https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37842.pdf](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37842.pdf)
- BPE와 방식은 굉장히 비슷하다(일종의 variation이다). subword tokenization 알고리즘이다.
    - 마찬가지로 언어학적 지식이 필요없다. BPE처럼 데이터에서 빈도수 기반으로 stemming하여 subword를 찾기 때문이다.
    - BPE는 빈도수(frequency)에 기반하여 최빈쌍을 unigram으로 치환하지만, WordPiece는 **likelihood를 기반으로 치환할 쌍을 찾음.**
        - 일단 단어를 character 단위로 모조리 분리하고, vocab에서 만들수 있는 조합 중 병합했을 때 training data의 likehood를 가장 높이는 조합을 그때그때 greedy하게 찾아 병합
- subword임을 나타낼 때 `##`이 붙는다.
- `BERT`, `DistilBERT`, `Electra` 등이 사용

### Unigram

- 최초 논문 [https://arxiv.org/pdf/1804.10959.pdf](https://arxiv.org/pdf/1804.10959.pdf)
- BPE, WordPiece과 반대 개념으로 접근한 subword tokenization 알고리즘이다.
    - 기존(BPE,WordPiece) : 최소단위의 character에서 시작하여 반복적인 병합으로 사전을 늘려가는 것
    - Unigram : Pre-tokenized 토큰, subword를 모두 가진 커다란 vocab에서 시작해서 **점차 vocab을 줄여나가는 방식**
    - 상세
        - 매 step마다 주어진 corpus와 현재 vocab에 대한 loss를 측정하여, 만약 현재 vocab에 있는 subword가 corpus에서 사라졌을 경우 loss가 얼마나 증가하는지 측정
        - loss를 가장 적게 증가시키는(의미를 가장 덜 잃는) 하위 p개의 token을 제거 - 이때 p는 일반적으로  vocab 전체의 10%, 20%로 잡는다.
        - 원하는 크기의 vocab이 될때까지 반복
- 일반적으로 단일 사용되지는 않고, sentencepiece에 내부적으로 사용된다.

### SentencePiece(==WPM)

- 구글이 BERT를 만들때 WPM(SentencePiece)를 사용했다는 논문 [https://arxiv.org/pdf/1609.08144.pdf](https://arxiv.org/pdf/1609.08144.pdf)
- SentencePiece 논문 [https://arxiv.org/pdf/1808.06226.pdf](https://arxiv.org/pdf/1808.06226.pdf)
- **기존 BPE,WordPiece 등과 대치되는 개념이 아니라, 이를 wrapping한 모델이다.**
- 기존 알고리즘들의 문제점 :  일단 단어단위로의 Pre-tokenize를 전제로 하고 있다.
    - 영어는 공백 단위로 단어를 자를 수 있으니 별 문제가 없는데, 한국어같은 언어들은 공백으로 의미를 분리할 수 없다!
    - 따라서, 일단 단어로 자르고나서 subword로 나누는 방식이 아니라 문장 전체를 받아서 의미단위로 나눌 수 있어야 한다.**(Pretokenizing이 없어야한다)**
- 입력 문장 전체를 raw stream으로 받아, 공백을 포함한 모든 단어들을 활용하여 BPE, WordPiece 혹은 **Unigram**을 적용해가며 vocab을 만든다.
    - '공백'까지 받아버리므로, 공백을 표시하기 위한 캐릭터 `_` 를 사용한다.
    - huggingface transformer 라이브러리의 모든 sentencepiece를 이용하는 모델들은 다 unigram 기준으로 토크나이징되었다.
- `ALBERT`, `XLNet` 등이 사용

## 2. 한국어에 적합한 Tokenizing 방식

### 레퍼런스

- [https://velog.io/@metterian/한국어-형태소-분석기POS-분석-3편.-형태소-분석기-비교](https://velog.io/@metterian/%ED%95%9C%EA%B5%AD%EC%96%B4-%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0POS-%EB%B6%84%EC%84%9D-3%ED%8E%B8.-%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0-%EB%B9%84%EA%B5%90)
- [http://www.engear.net/wp/한글-형태소-분석기-비교/](http://www.engear.net/wp/%ED%95%9C%EA%B8%80-%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0-%EB%B9%84%EA%B5%90/)
- [https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0](https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0)

### Intro

- 한국어의 교착어적 특성
    - 각 어절이 독립적 단어로 구성되지 않고, 여러 단어(명사+조사 등)가 붙어있는 형태를 띤다.
    - 영어에서 단어토큰화를 한 것과 비슷한 결과를 얻으려면 한국어에서는 형태소 토큰화를 수행해야한다.

### 요약

- 형태소 분석기를 사용한다면 `mecab`을 사용하는게 시간면으로나, 성능면으로나 좋은편
- **기존의 BERT 모델들 :  wordpiece를 사용**하였는데, 이 경우에는 mecab등을 통해 **subword segmentation한 후 wordpiece를 수행**해주는것이 좋다.
- **ALBERT 등의 모델들 : sentencePiece를 사용**하였는데, ****이 경우에는 **굳이 형태소 분석기를 먼저 돌릴 필요가 없다.** 애초에 공백기준으로 문자를 잘라서 표현하지 않기 때문이다.

### 상세

- 한국어 형태소분석기
    - `khaii`, `한나눔`, `꼬꼬마`, `KOMORAN`, `OKT`, `mecab`, `kiwi` , `twitter`
    - 속도측면에서는 mecab 압승
    - 성능 측면에서는 noise가 적은 데이터의 경우 크게 차이가 없음
        - noise(띄어쓰기, 은어 등)가 있는 데이터의 경우 가장 최근에 나온 twitter가 좋은 성능
- **형태소 분석기를 이용해 tokenization 한 후 subword segmentation이 좋다는데?**
    - 현재 코드
        - Retriever
            - `mecab` 형태소 분석기를 돌린 후 tf-idf Vectorizing할 때 uni/bigram 단위로 vocab을 생성하여 검색 효율을 높임.
        - MRC
            - 모델마다 사용했던 tokenizer(AutoTokenizer) 이용해 학습
            - WPM(SentencePiece)를 사용하는 BERT같은 모델
- 참고
    - [https://monologg.kr/2020/04/27/wordpiece-vocab/](https://monologg.kr/2020/04/27/wordpiece-vocab/)

## 3. 우리 데이터셋에 알맞은 Tokenizer(제안)

### 방식

- **SentencePiece를 사용한 transformers 모델을 사용하는 것**이 시간상 이득
    - ⭕️  [ETRI KoBERT](https://github.com/monologg/KoBERT-Transformers) - pretrained 양이 많고 질이 좋음, **SentencePiece 사용**
        - 완벽히 OOV를 해결하지는 못함(5번 미만 등장 단어는 vocab에 포함 X)
        - 참고
            - [https://github.com/SKTBrain/KoBERT/issues/1](https://github.com/SKTBrain/KoBERT/issues/1)
    - ❌  KoELECTRA (~v2)
        - transformers 라이브러리만으로 바로 모델을 사용가능하게 만들기 위하여, mecab이나 sentencepiece를 사용하지 않고 WordPiece를 직접 구현
        - **→ sentencepiece 모델에 비해 tokenizing이 제대로 이루어지지 않았을 가능성.**
    - ⭕️  **[KoELECTRA (~v3)](https://github.com/monologg/KoELECTRA)**
        - 모두의 말뭉치를 추가로 학습하여 pretrained 데이터가 훨씬 많아짐
        - WordPiece에 Mecab을 추가.
        - **→ mecab과 wordpiece를 같이 사용하였으므로 충분히 좋은 성능을 기대해 볼만함.**

### 사용 해 볼 만한  Tokenizer Library

- [Google SentencePiece](https://github.com/google/sentencepiece)
- Huggingface Tokenizer - **SentencePiece 비교하여 속도도 더욱 더 빠르다.**
    - byte level bpe tokenizer
    - char bpe tokenizer
    - **`SentecePiece BPE Tokenizer`**
    - bert wordpiece tokenizer - (BERT에서 사용된 토크나이저)
- 참고
    - [https://lsjsj92.tistory.com/600](https://lsjsj92.tistory.com/600)
