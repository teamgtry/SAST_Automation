[Custom Rule](https://github.com/teamgtry/Semgrep_Custom_Ruleset)
# SAST_Automation

SAST(Semgrep) 실행 결과를 표준 JSON으로 정규화하고, Repo RAG + LLM 검증을 통해 False Positive 후보를 분류하는 자동화 파이프라인입니다.

## 설명
이 프로젝트는 다음 흐름으로 동작합니다.
1) Semgrep 실행 및 SARIF 생성  
2) SARIF 정규화(코드 스니펫/데이터플로우 추출)  
3) Repo RAG 인덱스 생성(코드 청크 + BM25/임베딩)  
4) LLM 검증으로 FP(drop 후보) 분류 및 결과 저장

## 목적
- SAST 결과의 후처리 자동화
- FP 후보를 빠르게 선별해 triage 효율 개선
- 코드 컨텍스트 기반 근거 수집(RAG)으로 판단 정확도 향상

## 환경설정
### 권장 버전
- Python 3.12 (3.12 이후 버전에서는 동작하지 않음)
- node v24.12.0
- npm 11.6.2
- semgrep 1.146.0

### 필수
- Semgrep CLI (PATH에 등록되어 있어야 함)
- Python 패키지 설치
  ```bash
  pip install -r requirements.txt
  ```

### LLM 설정
`llm_verifier/.env` 파일을 생성합니다. (`llm_verifier/example.env` 참고)
```
GOOGLE_API_KEY={your_gemini_api_key}
LLM_MODEL={gemini_model}
```

### (선택) 임베딩 모델 사전 캐시
RAG 임베딩 모델을 미리 내려받아 속도/안정성을 높일 수 있습니다.
```bash
python prepare_models.py --cache-dir ./.hf_cache --model sentence-transformers/all-MiniLM-L6-v2
```
`main.py`는 실행 시 자동으로 `.hf_cache`를 확인하고, 없으면 `prepare_models.py`를 호출합니다.

## 실행 방법
```bash
python main.py
```
실행 중 아래 항목을 순서대로 입력합니다.
- Language (예: python, javascript)
- Semgrep config(s) (공백/쉼표로 여러 개 입력 가능)
- Semgrep target (기본값: 이 프로젝트 상위 디렉터리)
- Semgrep 옵션: jobs/timeout/timeout-threshold/max-target-bytes
- Semgrep Pro 사용 여부

## 출력물
실행 시 현재 디렉터리 하위에 다음 형태의 실행 폴더가 생성됩니다.
```
<oss>-semgrep-<language>-<index>/
  <run_name>.sarif
  <run_name>.sanitized.json
  llm_verifier/
    llm_verified_drop_fp.json
    llm_verified_keep.json -> 주 결과물
    llm_verified_skipped.json
    logs/
    sessions/
```
또한 대상 리포지토리 루트에 `.rag_index/`가 생성됩니다.

## 파일 구조
```
SAST_Automation/
  main.py                      # 전체 파이프라인 실행(인터랙티브)
  prepare_models.py            # 임베딩 모델 사전 캐시
  requirements.txt
  llm_verifier/
    llm_verifier.py            # LLM 검증(LangGraph)
    repo_rag.py                # Repo RAG 검색기(BM25 + 임베딩)
    repo_rag_build.py          # Repo RAG 인덱스 빌더
    example.env                # .env 템플릿
  sarif_normalization/
    generate_json.py           # SARIF -> 정규화 JSON 생성
    normalizator.py            # 코드 스니펫/데이터플로우 추출
    post_drop.py               # 파일명/확장자 기반 후처리 필터
  sast_runner/
    semgrep_runner.py          # Semgrep 실행 래퍼(SARIF 출력 고정)
    pre-drop.ignore            # Semgrep pre-drop 패턴
    exclude_rules/             # auto rules 제외 정책
  .hf_cache/                   # HF 캐시(로컬) -> 최초 실행시 자동 생성
```

## 참고 사항
- Semgrep Pro를 사용하는 경우, 별도 인증이 필요합니다.
- LLM 검증은 기본적으로 Gemini 모델을 사용하며, `.env` 미설정 시 실행 실패할 수 있습니다.
- `.rag_index`는 대상 리포지토리 루트에 생성되며, 재실행 시 재사용됩니다.
