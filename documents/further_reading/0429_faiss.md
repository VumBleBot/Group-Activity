# Follow-up (4/29)
[원글](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)

# 요약
1. Faiss : dense 벡터들의 `클러스터링, 유사도 탐색을 위한 라이브러리`
2. CPU, GPU 기반으로 작동하게 할 수 있으며, IVF(InVerted File index)를 활용
    - 강의자료 중 Pruning - `Clustering+Inverted file을 활용한 search`
    - 클러스터 후 가장 가까운 벡터들만 비교 
3. PCA로 original vectors들을 그대로 keep하지 않아 preprocess시 `낮은 RAM 사용`으로도 1 Billions vector를 databases 담을 수 있음 (below 30GB)
4. index(=DB)는 DISK에 저장
    - index : vector가 속한 클러스터 정보, centroid 정보를 모두 가지고 있는 DB. faiss에서 기본으로 사용하는 객체
5. numpy와 호환.  
    - 단, numpy에서는 기본적으로 float64가 default. 근데 faiss는 float32이어야 함.
6. faiss.IndexFlatL2 : 
    - Flat : quantizatioin 없이 vector 그대로 저장한다.
    - L2 : 두 벡터의 거리를 l2 norm으로 찾는다.
    - Index object는 결국 L2 brute force case다.
    - id는 사용자가 임의로 설정할 수 없는 auto increase를 따른다. 
    - index.ntotal : added vectors 개수 반환

7. quantizer : index를 보조하는 index. vector가 들어오면 어떤 cluster에 속해야 할지, 그 클러스터 내에서 nearest vector가 무엇인지 찾는 역할을 한다. 


8. 혼란스러울 수 있는 사례
- index.search : 각 query vector마다 k-nn으로 접근해, **D는 distnace 정보**, **I는 k개의 벡터들의 아이디를 반환**한다.
```python=
# xq is a n2-by-d matrix with query vectors
k = 4                          # we want 4 similar vectors
D, I = index.search(xq, k)     # actual search
print I
```

```
[[  0 393 363  78] # 현재 벡터에서 xq와 가장 유사한 벡터 아이디는 0->393->363->78번이다.
 [  1 555 277 364]
 [  2 304 101  13]]
```

# 토론

- 공백('') 이슈.
- EM이 F1보다 높은 경우는 공백 때문이다. 



[Reference]

- https://checkwhoiam.tistory.com/84
- retrieval 관련 논문 : https://github.com/danqi/acl2020-openqa-tutorial/blob/master/slides/part5-dense-retriever-e2e-training.pdf
- openqa tutorials : https://github.com/danqi/acl2020-openqa-tutorial/tree/master/slides