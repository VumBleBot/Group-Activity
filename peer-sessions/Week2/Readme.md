# 2주차 (210503~210509)

* [월요일 회의(210503)](https://github.com/VumBleBot/Group-Activity/tree/main/peer-sessions/Week2/210503.md)
    - EDA, Dense retriever 구현
        - Competition dataset과 KorQUaD dataset 차이, 공통점, 처리 필요한 부분 지적
    - special token 추가가 가지는 의미
        - entity 포착하여 의미가 나누어지지 않도록 하는 용도로 사용은 가능
        - 그러나 일반적이지 않은 방식이고, vocab.txt의 unused token을 원하는 entity로 바꾸는 방식이 일반적.
        - 현재 우리가 수행하고 있는 ODQA는 domain specific하지 않기 때문에 special-token 추가나 unused token 교체가 크게 의미가 없을수도 있다.
            - 그러나 이전 KLUE competition은 general domain이었음에도 불구하고 유의미한 성능 향상을 본 사람들이 있어 실험으로 검증해보면 좋을듯.
        - [추가조사 필요](http://boostcamp.stages.ai/competitions/4/discussion/post/199)
    - new baselinecode 
        - 리팩토링 및 구현 필요한 코드 논의
        - Feature assign
            - **\<refactor\> `Reader 파트 class로 refactoring` -- 종헌, 건모**
                - run_mrc 코드 리팩토링
            - **\<refactor\> `top-k sampling` -- 성익**
                - retriever context 가중치를 무시하고 reader가 top-k에 대해서 모두 정답 추출을 수행
            - **\<refactor, feat\> `retriever refactoring + DPR 클래스 구현` -- 수영, 지영
                - embedding 방식을 내부에서 분기처리하여 하위호출하는 general retriever로 refactoring
                - encoder 학습 및 weight 저장하는 DPR class 구현
                
* [화요일 회의(210504)](https://github.com/VumBleBot/Group-Activity/tree/main/peer-sessions/Week2/210504.md)
    - 성예닮 멘토님 킥오프 미팅
    - 코드리뷰
        - retriever 리팩토링
            - dense embedding으로 구조화 -- 지영(+건모)
            - bm25 구현 -- 수연(+성익)
        - reader 리팩토링 -- 종헌
        - top-k sampling -> reader+retriever 리팩토링 끝날때까지 잠깐 보류

* [목요일 회의(210506)](https://github.com/VumBleBot/Group-Activity/tree/main/peer-sessions/Week2/210506.md)
    - 이슈 정리
    - 베이스라인 코드 논의할 것
    - 이번주 목표치 설정