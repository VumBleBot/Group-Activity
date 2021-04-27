# BERT와 ELMO 요약

#### **BERT와 ELMO**

- **Contextualized Word Embedding**은 단어마다 벡터가 고정되어 있지 않고 문장마다 단어의 Vector가 달라지는 Embedding 방법을 뜻하고, 대표적으로 ELMo, GPT, BERT가 있다. 

- BERT는 GPT와 같이 Transformer를 이용하여 ELMo와 같이 **양방향**으로 학습을 진행한다.
  -  단, ELMo와는 다르다. **ELMo는 단방향으로 좌측 우측 각각 학습**하는 것이고
  - **BERT는 양방향으로 동시에 학습을 진행**하는 것이다.
    - 이때 동시에 양방향적으로 학습하면 **자기참조 문제가 발생할 수 있는데** 이를 해결하기 위해 **MASK를 씌워 학습을 진행**한다.
    - 그러나 MLM을 사용해도 실제 Fine-tuning과정에 학습 데이터에는 Mask를 씌우지 않아서 pretrain ~ finetuning간 간극 발생하는 문제가 있다.
    - 이에 대한 해결책으로 **Mask token의 80%는 mask로 10%는 random word로 10%는 unchanged word로 넣어준다.** ( 일부러 Noise를 넣음으로서 너무 Deep 한 모델에서 발생할 수 있는 Over-fitting문제를 회피하도록 )

# (2강) Extraction-based MRC

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

  - **B**idirectional **E**ncoder **R**epresentations from **T**ransformers

  - **without substantial task-spcific architecture modification**

  - 질문 답변과 언어 추론과 같은 넓은 분야의 작업에서 **state-of-the-art 모델을 만들기 위해 그냥 하나의 출력 레이어만 추가**만 해도 파인튜닝될 수 있다.

  - GLUE 80.5%(7.7% 절대적 향상), MultiNLI 정확도를 86.7%, SQuAD v1.1 질문 답변 테스트 F1을 93.2(1.5 point 절대적 향상) 그리고 SQuAD v2.0 테스트 F1에서 83.1(5.1 point 절대적 향상) 등을 포함해 열 한개의 자연어 처리 작업에서 새로운 state-of-the-art 결과를 얻었었다.

  - openAI의 트랜스포머는 오직 한 방향, 즉 forward language model만 학습시켰는데, **트랜스포머를 베이스로 하면서도 양쪽 방향을 모두 보는 모델을 만들 순 없는 걸까?** 라는 질문에서 출발한게 Bert

    - [Story]

    ```
    - “우리는 transformer를 encoder로 쓸거야”, BERT가 말했습니다.
    - Ernie 는 답장했습니다. “이건 말도 안돼. 우리 모두가 알고 있듯이, bidirectional 하게 한 단어를 본다면, 각 단어는 결국 간접적으로 여러층에서 나타나는 맥락에서 자기 자신을 보게될거야.”
    - “괜찮아. 우린 mask를 쓸거니까”, BERT가 비밀스럽게 말했습니다.
    ```

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning](http://jalammar.github.io/illustrated-bert/))

  ### [BERT]

  ![스크린샷 2021-04-27 오후 1 56 06](https://user-images.githubusercontent.com/46434838/116204440-bdab5080-a777-11eb-954e-fe4fa2425843.png)

  - Step 1 에서는 pre-train된 모델을 다운로드하면 되고, Step 2 에서는 fine-tuning에 대해서만 신경쓰면 된다.
  - 의의
    - 처음부터 학습을 시키는 것 (training from scratch)에 들어가는 **엄청난 시간, 에너지, 지식, 그리고 자원을 아낄 수 있게 되었다.**

  - 사례

    - (문장 분류)
    - Sentiment analysis (감성 분석)    
      - 입력: 영화/제품 리뷰. 출력: 긍정/부정
      - 예시 데이터 셋: [SST](https://nlp.stanford.edu/sentiment/)    
    - Fact-checking (사실 확인)   
      - 입력: 문장. 출력: “주장함” or “주장이 아님”
      - 언젠가 미래에 기대하는 출력:
        - 입력: 주장이 담긴 문장 출력: “사실” or “사실이 아님”

    - ... 

  - 모델 구조

    ![스크린샷 2021-04-27 오후 5 03 13](https://user-images.githubusercontent.com/46434838/116207110-7ecaca00-a77a-11eb-9fee-35727758d9a6.png)

    - 학습된 Transformer Encoder를 쌓아 놓은 것
      - BERT BASE – 이전의 OpenAI Transformer와 성능을 비교하기 위해 설정된 사이즈
      - BERT LARGE – 논문에 명시된 state-of-the-art 결과들을 얻게 해 준 말도 안되게 큰 사이즈의 모델
      - Base 버전에는 12개를 가지며 Large version 에서는 24개를 가진다. feedforward-network의 크기 또한 매우 크고 (768과 1024개의 hidden unit) Transformer의 첫 논문에 나온 구현 설정 (6개의 encoder layers, 512개의 hidden units, 8개의 attention heads) 보다도 많은 attention heads를 가지고 있다.
    - **transformer의 기본 encoder와 동일하게, BERT는 단어의 시퀀스를 입력으로 받아 encoder stack을 계속 타고 올라간다. 각 encoder layer는 self-attention을 적용하고 feed-forward network를 통과시킨 결과를 다음 encoder에게 전달**한다.
    - BERT의 논문에서는 매우 단순한 형태인 single-layer를 classifier로 이용했음에도 불구하고 매우 좋은 결과를 얻는다.
    - 만약 이용하고 싶은 task가 더 다양한 종류의 label을 가진다면 (예를 들어, email을 “spam”,”not spam”, “social”, “promotion” 네가지 label으로 분류하고 싶은 경우),  단순히 classifier network를 조금 변형해주어 더 많은 output neurons를 가지게 하고 softmax를 통과시키면 된다.
    - 이런 방식으로 학습된 **BERT를 fine-tuning할 때는 (Classification task라면)Image task에서의 fine-tuning과 비슷하게 class label 개수만큼의 output을 가지는 Dense Layer를 붙여서 사용**한다.
    - ![스크린샷 2021-04-27 오후 5 10 50](https://user-images.githubusercontent.com/46434838/116208287-b8500500-a77b-11eb-8179-216a10fd966f.png)

    
  ### [ELMO]    
        
  ![스크린샷 2021-04-27 오후 4 46 18](https://user-images.githubusercontent.com/46434838/116204877-25619b80-a778-11eb-9dc9-af1272c433d5.png)

  - 엘모(ELMo, Embeddings from Language Model)는 2018년에 나온 **contextualized word-embeddings** 방법. ELMo은 Word2Vec이나 GloVe가 가지지 못한 장점-주변 문맥에 따라 워드 임베딩을 한다.

  - 각 단어에 **고정된 벡터를 주는 것이 아니라 문맥을 고려하여 임베딩**. ELMo는 임베딩을 할 때 우선 전체 문장을 본다. 그리고 이미 학습된 **양방향 LSTM(Bidirectional LSTM)**을 이용하여 각 단어의 임베딩 벡터를 생성. 

  - 문맥 고려의 의미 - 논문에서는 hidden state들을 concatenate (붙여쓰기)를 한 후 weighted sum 을 구하는 방식을 제안

    

- 참고

  - 모델 flow :  https://vhrehfdl.tistory.com/15
  - bert가 입력받을 수 있는 토크나이저 - https://github.com/google-research/bert/blob/master/tokenization.py

# (3강) Generation-based MRC


- [Introducing BART](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html)
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) ](https://arxiv.org/abs/1910.10683)

- BART는 넓은 분야에 적용할 수 있도록 **seq2seq 구조로 만들어진 denoising auto-encoder**다. Pretrain은 noise function으로 손상된 text를 복구하도록 모델을 학습시키는 방법으로 이뤄진다.

- BERT는 bidirection encoder로 noise된 token을 예측하는데 generation task에서 사용이 어렵다. GPT는 autoregressive하게 다음 token을 예측해 generation에 사용이 가능하지만 bidirectional 정보를 얻지 못한다. **BART는 손상된 text를 입력받아 bidirectional 모델로 인코딩하고, 정답 text에 대한 likelihood를 autoregressive 디코더로 계산**한다.
  이런 설정은 **noising이 자유롭다는 장점이 있다. 이 논문에서는 문장 순서를 바꾸거나 임의 길이의 토큰을 하나의 mask로 바꾸는 등의 여러 noising 기법을 평가**한다.

- BERT는 word prediction을 위해 추가로 feed-forward 레이어를 추가했는데 BART는 그렇지 않다.

- **Pretraining Bert** (Noising 방벙들)

  ![그림2](https://dladustn95.github.io/assets/images/bart_figure2.png)

  - Token Masking: BERT처럼 랜덤 토큰을 masking하고 이를 복구하는 방식이다.
  - Token Deletion: 랜덤 토큰을 삭제하고 이를 복구하는 방식이다. Masking과의 차이점은 어떤 위치의 토큰이 지워졌는지 알 수 없다는 점이다.
  - Text Infilling: 포아송 분포를 따르는 길이의 text span을 생성해서 이를 하나의 mask 토큰으로 masking 한다. 즉 여러 토큰이 하나의 mask 토큰으로 바뀔 수 있고 길이가 0인 경우에는 mask 토큰만 추가될 수도 있다. SpanBERT에서 아이디어를 얻었는데 SpanBERT는 span의 길이를 알려주었으나 여기서는 알려주지 않고 모델이 얼마나 많은 토큰이 빠졌는지 예측하게 한다.
  - Sentence Permutaion: Document를 문장 단위로 나눠서 섞는 방법이다.
  - Document Rotation: 토큰 하나를 정해서 문장이 그 토큰부터 시작하게 한다. 모델이 document의 시작을 구분하게 한다.

* [참고]
* https://dladustn95.github.io/nlp/BART_paper_review/



---

# 토론게시판

### [요약] **dataset 키우고, large model 쓰자**

- train은 3952개로 생각보다 적음. **데이터를 추가해본다면, korquad1.0(37mb) 정도를 고려**해볼 수 있을 것 같다. ( korquad2.0(6.4gb)를 쓰자니 컴퓨팅파워가 떨어진다는 문제점 존재). validation도 240로 많이 적다, traindataset 키우면서 validation dataset도 이번 1주차에 구축해두면 좋지 않을까

- 문서 검색을 포함한 MRC 성능은 많이 떨어짐 -  mrc보다는 **retriever가 성능 향상을 열쇠** (틀린 문서를 찾아주는 경우 mrc는 무조건 틀린 답)

- **글 제목-  대체적으로 20이하**. 해당 문서를 관통하는 키워드이므로 retriever에서 활용할 수 있을 것 같다.

- **문서의 길이는 최소 510자에 최대 2049자**. 생각보다 문서의 길이가 길다. **가능한 문서가 짤리지 않게 large 모델을 사용하면 mrc에서 좋은 성능을 보일 것 같다.**
- title, context, question, answer을 모두 합쳤을 때의 길이 - **대부분의 input을 있는 그대로 받으려면 len()기준으로 2,000자(평균 2라고 가정하면 토큰 기준 약 1,000자) 정도는 돌릴수 있는 모델이어야 한다.**

- [참고]

  - 원글 : http://boostcamp.stages.ai/competitions/31/discussion/post/254
  - colab 참조 - https://colab.research.google.com/drive/1xgla_ghhOlbjDqAWYPieQxNs7nIRnHd1

    

