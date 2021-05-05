# Reading Wikipedia to Answer Open-Domain Questions

Document Reader: Multi-layered LSTM
4 feature encodings
Document Retriever: bigram hashing + TF-IDF matching, inverted index
Link: https://arxiv.org/abs/1704.00051
Property: Apr 26, 2021
Published: 2017
Tags: LSTM, ODQA

![Reading%20Wikipedia%20to%20Answer%20Open-Domain%20Questions%20ee1d0a87f67b4279a6638877d660c768/Untitled.png](Reading%20Wikipedia%20to%20Answer%20Open-Domain%20Questions%20ee1d0a87f67b4279a6638877d660c768/Untitled.png)

### **Document Retriever**

- bigram hashing + TF-IDF matching
    - [TF-IDF](https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/)
- A simple inverted index lookup followed by term vector model scoring performs quite well on this task for many questions types (better than ElasticSearch based Wikipedia Search API)
    - inverted index: word가 key가 되고 그 word가 존재하는 document가 value가 되는 것
- Articles and questions are compared as TF-IDF weighted bag-of-word vectors
- Local word order was considered with n-gram features (bigram showed the best performance
    - Map the bigrams to 2^24 bins with an unsigned murmur3 hash
- It returns 5 Wikipedia articles given any question

### **Document Reader**

- multi-layer recurrent neural network machine comprehension model
- bi-directional LSTM
- Input paragraph encoding: feature vector of token p
    - **Word embeddings**: only fine-tune 1000 most frequent question words (의문문) because the representation of some key words such as what, how, which, many could be crucial for QA systems → 의문문들과 그 유사 문구들을 중심으로 fine-tuning
    - **Exact Match*****: 성능 향상에 가장 도움이 됨
        - 해당 토큰이 질문의 토큰과
            - 완전히 match하는지
            - lower case로 바꾸면 match하는지
            - 원형으로 바꾸면 match하는지
    - **Token feature**: part-of-speech(POS), Named entity recognition(NER), normalized term frequency(TF)
    - **Aligned question embedding*****: attention score capturing the similarity between token i and each question words.
        - dot products between nonlinear mapping of word embeddings
        - this feature add soft alignment between similar but non-identical words (e.g. car and vehicle)
- Question encoding
    - applied RNN on top of the word embeddings of qi and combine the resulting hidden units into one single vector (weights to be learned)

### Prediction

- trained 2 classifiers independently for predicting the two ends of the span
- choose the best span from token i to token i' such that i ≤ i' ≤ i + 15 and P_start(i) and P_end(i') is minimized.
- To make scores compatible across paragraphs in one or several retrieved documents, authors used the unnormalized exponential and take argmax over all considered paragraph spans for the final prediction

### Data

- Wikipedia
    - only the plain text is extracted (structured data sections(lists, figures) are stripped
- SQuAD
    - Document Reader(machine comprehension) 부분은 SQuAD로 학습 후, 테스트 QA 셋에 대해서 SQuAD에서 답을 찾는 대신 위키피디아에서 답을 찾도록 함

- Distantly supervised data
    - Training set

        1) Retrieve top 5 Wikipedia articles with Document Retriever

        2) Discard paragraphs which does not have the known answer

        3) Paragraphs shorter than 25 or longer than 1500 are filtered out

        4) If there is any NER in question, paragraphs do not contain them were discarded

        5) score all positions that match an answer using unigram and bigram overlap between the question and a 20 token window, keeping up to the 5 paragraphs with the highest overlaps

- Document Reader Implementation Detail
    - 3-layer bidirectional LSTMs with h=128 hidden units for both paragraph and question encoding
    - Stanford CoreNLP toolkit for tokenization and also generating lemma, POS, and NER
    - Paragraphs are sorted by length and divided into mini-batches of 32 examples
    - optimizer: Adamax
    - Dropout: 0.3 (word embeddings and all the hidden units of LSTM)

### Results

![Reading%20Wikipedia%20to%20Answer%20Open-Domain%20Questions%20ee1d0a87f67b4279a6638877d660c768/Untitled%201.png](Reading%20Wikipedia%20to%20Answer%20Open-Domain%20Questions%20ee1d0a87f67b4279a6638877d660c768/Untitled%201.png)