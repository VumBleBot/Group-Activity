# Rule

## ✨ 이슈 관리

- 이슈 관리는 Group-Activity Repository에서 진행됩니다.
- issue-template을 참고하여 이슈를 남겨주세요.

## 📚 Commit Rule

- [gitmessage template 사용법](#gitmessage-template-사용법)을 참고해 commit 형식 지켜주세요.
- 공용 Repository에서 Commit은 코드 리뷰를 위해 활용될 예정입니다.

</br>

*Organization에 대한 Feedback은 언제나 환영합니다.*


### gitmessage template 사용법
1. home 디렉토리에서 gitmessage.txt를 생성합니다. touch로 파일을 생성해도 됩니다.

```
$ vim ~/.gitmessage.txt
```

2. 사용하는 에디터로 txt 파일을 열고 custom gitmessage template을 만듭니다. (복붙해주세요)

```
# 제목은 최대 50글자까지 작성: ex) <feat>: Add Login

# 본문은 아래에 작성

# 꼬릿말은 아래에 작성: ex) Github issue #1


# --- COMMIT END ---
#   <타입> 리스트
#   feat    : 새로운 기능추가
#   fix     : 버그 혹은 기능 수정
#   refactor: 리팩토링
#   style   : 코드 스타일 변경
#   docs    : 문서 (추가, 수정, 삭제)
#   test    : 테스트 (테스트 코드 추가, 수정, 삭제)
#   chore   : 기타 변경사항 (빌드 스크립트 수정 등)
# ------------------
#   제목
#   1. 첫글자는 대문자로
#   2. 명령문으로 작성
#   3. 끝에 마침표(./,) 금지
#     4. 제목과 본문은 한줄 띄워 분리
#
#   본문
#   1. - 로 시작
#   2. What, Why, How 중 1개 이상 설명
# ------------------

```

3. 작성한 gitmessage를 템플릿으로 지정하겠다고 선언하겠습니다.

```
$ git config --global commit.template ~/.gitmessage.txt
```

4. 이후 `git commit` 시에 템플릿을 확인할 수 있습니다. i 끼워넣기로 템플릿을 참고하여 commit 메시지를 작성합니다.

</br>
