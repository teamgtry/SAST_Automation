# SAST_Automation
[Custom Rule](https://github.com/teamgtry/Semgrep_Custom_Ruleset)


# 환경설정
/llm_verifier/.env
.env 구성해야됨
```
GOOGLE_API_KEY={your gemini key}
LLM_MODEL={your llm model}
```

현재 종속성
- python,node,npm
> Python 3.12 -> 최신 버전 사용시 langgraph error<br>
> node v24.12.0<br>
> npm 11.6.2<br>

- semgrep<br>
  ※ pro 사용 시 사전에 pro 엔진 활성화 및 로그인 해야함
pip install -r requirements.txt 로 필요한 python 패키지 한번에 설치
(에러 시 python3 -m pip install --break-system-packages -r requirements.txt로 실행)

# 실행

