# Cluster-Former: Clustering-based Sparse Transformer
for Long-Range Dependency Encoding

Link: https://arxiv.org/pdf/2009.06097v1.pdf
Published: 2020
Tags: ODQA, Transformer

**서론: 극복하고자 한 점**

기존 Transformer가 모든 input에 대한 attention을 계산하기 때문에 input이 길어지면 메모리 사용량과 계산복잡도가 4승으로 증가한다는 데 있음

이 때문에 sliding window를 이용해서 sequence를 짧은 덩어리들로 나눈 후 각 덩어리 간에 connection을 계산해주는 대안이 나옴. 하지만 이 대안들은 전체를 대상으로 self-attention을 계산하는 것에 비해 임의적으로 데이터를 나누었기 때문에, 그리고 필수적 정보가 빠지는 경우들 때문에 낮은 유연성과 정확도를 보임

이 연구에서는 긴 길이의 sequence(수천 단어)에 대하여 hashing-based attention의 가능성을 살펴보려고 함 - sliding window와 hashing 기반 방법의 장점들만 취하고자 함

![Cluster-Former%20Clustering-based%20Sparse%20Transformer%20e4271d1c0ad74d38b4374d31aab82336/Untitled.png](Cluster-Former%20Clustering-based%20Sparse%20Transformer%20e4271d1c0ad74d38b4374d31aab82336/Untitled.png)

QA에서 Long sequence를 다루는 방법

- 가장 많이 사용되는 방법: 가벼운 모델로 쓸모없는 text를 제거한 후 sliding-window 기반의 더 큰 모델을 적용
- Information Retrieval을 위한 노력들
    - bi-gram 적용 (Chen et al, 2017)
    - 특정 paragraph에서 답을 얻을 수 있는지 없는지를 reward로 해서 모델을 train (Wang et al, 2018)
    - Paragraph ranking model (Lin et al, 2018)
    - 여러 문단에 걸쳐 답을 구해야하는 경우 paragraph를 ranking하는 모델을 train (Wang et al, 2019)
    - recurrent retriever 학습시킴 (Asai et al, 2020)
- 하지만 정보가 유실될 수 있음 - 이 논문에서는 바로 큰 모델을 long sequence에 대해 트레이닝하는 방식을 사용

![Cluster-Former%20Clustering-based%20Sparse%20Transformer%20e4271d1c0ad74d38b4374d31aab82336/Untitled%201.png](Cluster-Former%20Clustering-based%20Sparse%20Transformer%20e4271d1c0ad74d38b4374d31aab82336/Untitled%201.png)

2 type의 encoding layer:

1. Sliding-window layer: 국지적 정보에 집중
    1. 각 sliding window 내에 Transformer를 적용
    2. window size(l)와 stride(m)를 이용해서 데이터셋을 쪼갬
        1. QA task에서는 질문 문장과 sliding window로 쪼개진 데이터셋을 concat하여 사용
    3. sliding window로 인해 overlap되는 token에 대해서는 평균값을 사용
2. Cluster-former layer: 전역적 정보를 encoding
    1. clustering을 적용해서 비슷한 input hidden state들이 같은 클러스터에 들어갈 수 있도록 함
    2. cluster되고 sorted된 input들은 쪼개져서 Transformer를 통과함
    3. 계산 비용을 줄이기 위해, clustering centroid는 각 step마다 계산되지 않고 한 epoch, 혹은 몇 epoch마다 업데이트됨
        1. centroid는 nearest neighborhood 기반으로 결정되어 hidden state들이 비슷한 cluster id를 가지고 있으면 서로 유사하다는 것을 의미함
    4. hidden state를 Cluster-Former layer 이전에서 축적시킨 후 클러스터링을 위해 K-Mean 알고리즘을 적용함 (아이디어 적용을 위해 simple한 클러스터링 알고리즘을 적용한 것이고, 더 좋은 클러스터링을 적용하면 개선 가능)
    5.  hidden state와 centroid 간의 내적을 argmax를 통과시켜서 가장 큰 값에 따라 cluster에 할당됨
    6. 클러스터 기반으로 hidden state를 sort → transformer를 통과시킴
    7. transformer 통과시킨 후 clustering 이전 상태로 다시 sort (원상태의 word sequence를 갖도록)

    Implementation Detail

    - RoBERTa-large 사용
        - 24 Transformer layers, 16 head per layers and hidden state dimension of 1024
        - position embedding을 각 chunked sentence에 적용시킴
        - 대부분의 layer에는 sliding-window layer만 적용시키고, cluster-former layer는 15번째와 20번째 layer에 적용시킴
        - sliding window: 256, stride: 224
        - number of clusters: {64, 256, 512}
        - 각 paragraph에 special token 부착 후 전체 paragraph들을 연결시킨 긴 paragraph를 final context sequence로 사용
        - Adam, warm-up update 2,220, maximal updates to 22,200, dropout 0.1, learning rate 5e-5, batch size 160

    **Clustering example**

    ![Cluster-Former%20Clustering-based%20Sparse%20Transformer%20e4271d1c0ad74d38b4374d31aab82336/Untitled%202.png](Cluster-Former%20Clustering-based%20Sparse%20Transformer%20e4271d1c0ad74d38b4374d31aab82336/Untitled%202.png)