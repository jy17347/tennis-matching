# 🎾 사방팔방 테니스 매칭 시스템

테니스 토너먼트 자동 매칭 시스템 with Streamlit GUI

## 기능

### 📝 참가자 편집
- Excel 파일에서 참가자 데이터 로드
- GUI에서 직접 데이터 편집
- 참가 여부 체크 (O 또는 1 입력)
- 변경사항 저장 및 초기화

### ⚙️ 매칭 생성
- 3코트 × 5타임 = 15경기 자동 배정
- 남복, 여복, 혼복 경기 자동 분배
- 최적화 알고리즘 (1000회 반복)
- 실력 균형 및 제약조건 고려
- Excel 및 PDF 결과 생성
- PDF 미리보기 (이미지 변환)

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run scripts/streamlit_editor.py
```

## 파일 구조

```
matching/
├── scripts/
│   ├── streamlit_editor.py   # Streamlit GUI 앱
│   └── tennis_matching.py     # 매칭 알고리즘
├── dataset/
│   ├── roster.xlsx            # 명단 파일
│   └── participation.xlsx     # 참가자 파일
├── results/                   # 결과 저장 폴더
├── requirements.txt
└── packages.txt              # 시스템 패키지 (poppler)
```

## 배포

Streamlit Cloud에서 배포 가능

## 주요 제약조건

- 남자 최소 4명 이상 필요
- 모든 남자는 혼복 최소 1회 참여
- 실력 균형 유지
- 팀/상대 중복 최소화
