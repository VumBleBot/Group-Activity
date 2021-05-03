# 1주차 (210426~210502)

* [월요일 회의(210426)](https://github.com/VumBleBot/Group-Activity/tree/main/peer-sessions/Week1/210426.md)
    * 그라운드룰 확정
    * 역할 정의(이주의 관심영역)
    * 안건 논의
        * 베이스라인 코드 읽어보고 오기
        * 찾아볼 논문 정하기
    * 1주차 정보탐색 영역 정하기
        - 모델 탐색 : 종헌
        - survey 논문 : 건모
        - papers with code 데이터셋별 SOTA 논문 훑기 : 수연
        - tokenizing, 전처리 파트 탐색 : 성익
        - follow-up : 지영
    * 범블봇 주제 정하기
        - 쿼리 기반 노래 제목 반환
        - 사용자 쿼리와 IRQA 챗봇 답변을 concat하여 추천(미정)
 
 * [화요일 회의(210427)](https://github.com/VumBleBot/Group-Activity/tree/main/peer-sessions/Week1/210427.md)
    * 정보 탐색 결과 공유
      - (지영) follow-up 정리
      - (건모) 수요일 쯤 survey 논문 정리해서 설명
      - (수연) Reading Wikipedia to Answer Open Domain Question 논문 요약
         - feature vector 어떻게 추가되는건지 코드 보면서 더 알아보기
      - (성익) 수요일까지 베이스라인 코드 tokenizing 부분 + special token을 넣을때 어떤 방식으로 임베딩되는지
         - wordpiece vs sentensepiece 차이
      - (종헌) 기초 모델 논문 + 토크나이저 리뷰 중 - Transformer까지 보고 공유
    * 베이스라인 코드 리뷰
      - Group-Activity repo에 baseline 코드 커밋해서 질문 생기면 issue 활용
    * 베이스라인 기준 모델 비교 결과
      - XML-roberta-large가 가장 높은 EM (2.50)
      - mrm8488/bert-multi-cased-finetuned-xquadv1가 가장 높은 F1 (13.86)

* [수요일 회의(210428)](https://github.com/VumBleBot/Group-Activity/tree/main/peer-sessions/Week1/210428.md)
    * 정보 탐색 결과 공유
        - (지영) BM25 코드, Tokenizer 관련 토론 게시글 리뷰
        - (종헌) Transformer 코드 리뷰
        - (성익) Tokenizer 정리
        - (수연) Cluster-Former(2020) SOTA 논문 리뷰
        - (건모) ODQA(OpenQA) Survey 논문 리뷰
    * 베이스라인 파트 분배
        - retriever: 2명 - 성익,수연
        - train, inference : 2명 - 건모,지영
        - utils_qa : 1명 - 종헌

* [목요일 회의(210429)](https://github.com/VumBleBot/Group-Activity/tree/main/peer-sessions/Week1/210429.md)
    * 정보 탐색 결과 공유
        - (종헌) RoBERTa, XLM
        - (지영) Follow up (faiss)
    * 1주차 결산(issue 정리)
        - retriever: 2명 - 성익,수연
        - train, inference : 2명 - 건모,지영
        - utils_qa : 1명 - 종헌
    * 범블봇 주제 및 아이디어 탐색
        - 어려운 점들 토의하였고 추가 논의 필요
        - 아이디어 있으면 슬랙에 올리기
    * [ETRI KorBERT 신청하기](https://aiopen.etri.re.kr/service_dataset.php)
    * 내일(4/30) 베이스라인 담당 파트 리뷰

* [금요일 회의(210430)](https://github.com/VumBleBot/Group-Activity/tree/main/peer-sessions/Week1/210430.md)
    * 베이스라인 코드 리뷰
        - 각 파트별로 주석 제공
        - 이해 안가는 코드 공유 후 논의