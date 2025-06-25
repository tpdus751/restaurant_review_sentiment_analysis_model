# 음식점 리뷰 감성분석 모델 구축 과정 및 결과 리포트
음식점 리뷰 감성분석 모델 구축, LSTM, beomi/KcELECTRA-base-v2022 비교

## 감성분석 모델 학습 계기


## 데이터 준비

### [리뷰 데이터 크롤링](https://github.com/tpdus751/naver_map_restaurant_review_crawl_process) 

### 리뷰 데이터 라벨링
#### LM Studio API 감정분석 요청 (Gemma 3 12B)
#### 프롬프트 구성
```python
# ✅ 감정 분석 프롬프트 구성
def make_batch_prompt(reviews):
    prompt = (
        "당신은 음식점 리뷰 감성 분석가 입니다. 다음 리뷰 목록을 확인하고, 각 리뷰의 감성을 숫자로 분류하세요.\n"
        "- 부정: 0\n- 중립: 1\n- 긍정: 2\n"
        "형식: 숫자만 쉼표로 나열, 꼭 부정 0, 중립 1, 긍정 2를 지킬 것. 예: 2,1,0,2,1\n\n"
    )
    for i, review in enumerate(reviews, start=1):
        prompt += f"{i}. {str(review).strip()}\n"
    prompt += "\n답:"
    return prompt

# ✅ 응답 파싱
def parse_response(text, expected_length):
    try:
        values = [int(x.strip()) if x.strip().isdigit() else None for x in text.strip().split(',')]
        if len(values) < expected_length:
            values += [None] * (expected_length - len(values))
        elif len(values) > expected_length:
            values = values[:expected_length]
        return values
    except Exception as e:
        print(f"⚠️ 응답 파싱 오류: {e}")
        return [None] * expected_length

# ✅ LM Studio로 감정 분류 요청
def classify_batch_lmstudio(reviews):
    prompt = make_batch_prompt(reviews)
    try:
        response = requests.post(LM_STUDIO_API_URL, json={
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.0,
            "stop": ["\n"]
        })

        # ✅ 응답 상태 코드 확인
        if response.status_code != 200:
            print(f"🚫 응답 실패: HTTP {response.status_code}")
            print("📩 응답 본문:", response.text)
            return [None] * len(reviews)

        result = response.json()

        # ✅ 응답 구조 확인
        if "choices" not in result:
            print("🚫 응답에 'choices' 키가 없습니다.")
            print("📩 전체 응답:", result)
            return [None] * len(reviews)

        response_text = result['choices'][0]['text'].strip()
        print("🧾 응답:", response_text[:100] + "...")

        return parse_response(response_text, len(reviews))

    except Exception as e:
        print(f"🚫 요청 실패: {e}")
        return [None] * len(reviews)
```

## 모델 비교 (LSTM vs beomi/KcELECTRA-base-v2022)

### LSTM
#### 코드
```python
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
```
라이브러리 설치

```python
df = pd.read_csv('___.csv')  # 'text', 'label' 컬럼 포함
```
csv 불러오기

```python
df = df.dropna()
```
결측치 제거

```python
# 전처리
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
    text = re.sub(r'\d+', '', text)      # 숫자 제거
    return text.lower().strip()          # 소문자 + 공백 제거

# 텍스트, 라벨 분리
X = df['content'].astype(str).apply(clean_text)
y = df['label']
```
전처리 및 text, label 분리

```python
print("라벨 분포:\n", y.value_counts())
```
![image](https://github.com/user-attachments/assets/1b8d36a7-482b-403f-a68a-646811afaff8)
라벨 분포 확인

```python
# 라벨 기준으로 데이터프레임 나누기
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1].sample(n=20871, random_state=42)
df_2 = df[df['label'] == 2].sample(n=20871, random_state=42)

# 세 클래스 합치기
df_balanced = pd.concat([df_0, df_1, df_2]).sample(frac=1, random_state=42).reset_index(drop=True)

# 라벨 분포 확인
print("균형 맞춘 라벨 분포:\n", df_balanced['label'].value_counts())
```
![image](https://github.com/user-attachments/assets/cbc565fc-69bb-4076-b14d-d7fb87eb08d3)
라벨 수가 적은 부정을 기준으로 갯수 통일

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
학습/테스트 분리

```python
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
```
토크나이저 로드 및 입력 데이터 정수 인코딩

### KcELECTRA-base-v2022
