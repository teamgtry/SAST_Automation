# SAST_Automation
sast with llm verification

일단 evidence는 뺴고했음
main.py -> sast_runner -> sarif_normalization -> llm_verifier
 

* TODO
1. 정탐 탐지율 높여야됨 -> 생각보다 탐지가 잘 안되는거 같기도?
   1) 프롬프트 손질하기
   2) 같은 케이스를 여러번 검토하게 하기
   3) 케이스별로 하나씩 검토하는 것 말고 전체적으로 체이닝되는 것도 확인해달라 하기
2. 현재 사용 방법은 대상 프로젝트 폴더 내부에 SAST_Automation을 git clone해서 받아오고 main.py를 실행하는 것임 -> 폴더 내부에 넣지말고 외부에서 argument로 제어할 수 있게 해야 여러개를 동시에 돌릴떄 편할듯
4. runner(semgrep), llm_verification(gemini)추가
5. report 현재 json형태 -> 가독성 좋은걸로 바꿀까?
6. 도커 이미지로 올려버리면 편할듯? 지금 종속성 문제가 조금 있음
   python, node 2025.12 기준 최신버전
   `npm i @openai/codex-sdk` `npm i -D tsx typescript` 설치 필요 -> g옵션 안붙이면 할떄마다 다운해야 하니 g옵션 붙여서 설치하자...`npm i -g @openai/codex-sdk` `npm i -g tsx typescript` 
7. 대시보드 
