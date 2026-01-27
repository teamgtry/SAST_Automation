# SAST_Automation
[Custom Rule]()

/llm_verifier/.env
.env 구성해야됨
```
GOOGLE_API_KEY={your gemini key}
LLM_MODEL={your llm model}
```

현재 종속성(일단 llm 코드것만)
- python,node,npm 
> Python 3.12 -> 최신 버전 사용시 langgraph error<br>
> node v24.12.0<br>
> npm 11.6.2

pip install -r requirements.txt 로 필요한 python 패키지 한번에 설치
