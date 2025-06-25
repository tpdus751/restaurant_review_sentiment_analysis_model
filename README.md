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

## 모델 학습 (LSTM, beomi/KcELECTRA-base-v2022)

### LSTM
#### 코드
라이브러리
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

csv 불러오기
```python
df = pd.read_csv('___.csv')  # 'text', 'label' 컬럼 포함
```

결측치 제거
```python
df = df.dropna()
```

전처리 및 text, label 분리
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

라벨 분포 확인
```python
print("라벨 분포:\n", y.value_counts())
```
![image](https://github.com/user-attachments/assets/1b8d36a7-482b-403f-a68a-646811afaff8)

라벨 수가 적은 부정을 기준으로 갯수 통일
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

학습/테스트 분리
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

토크나이저 로드 및 입력 데이터 정수 인코딩
```python
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
```

패딩 (모델 학습 시 입력갯수 통일)
```python
max_len = 80
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
```

라벨 원핫 인코딩
```python
num_classes = len(y.unique())  # 예: 3 (부정/중립/긍정)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)
```

모델 정의
```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])
```

모델 컴파일 (optimizer, loss, metrics)
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

얼리스타핑(val_loss 기준 Epoch5 만큼 개선안되면 멈춤)
```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

모델 학습
```python
history = model.fit(
    X_train_pad, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)
```
![image](https://github.com/user-attachments/assets/0a292528-8aa9-4025-a8ef-bda0d92764b6)

### KcELECTRA-base-v2022
#### 코드
라이브러리
```python
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback, Trainer, DataCollatorWithPadding
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, ElectraConfig, ElectraForSequenceClassification
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from datasets import Value
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
```

csv 불러오기
```python
review_df = pd.read_csv('./___.csv')
```

데이터 전처리
```python
cleaned_review = []

for review in review_df['content']:
  if len(review) < 2:
      review = np.nan
  else:
    if review == '':
      review = np.nan

    review = review.strip()

    if '\n' in review:
      review = review.replace('\n', '')

  cleaned_review.append(review)

review_df['content'] = cleaned_review

# 결측치 확인
print(review_df.isnull().sum())

# 결측치 제거
review_df = review_df.dropna()

# 중복치 확인
print("중복 행 개수:", review_df.duplicated().sum())

# 중복치 제거
review_df = review_df.drop_duplicates()
```

라벨 수가 적은 부정을 기준으로 갯수 통일
```python
sample_size = review_df['label'].value_counts().min()
balanced_df = review_df.groupby('label').sample(n=sample_size, random_state=42)
balanced_df.groupby('label').size().reset_index(name='count')
```
![image](https://github.com/user-attachments/assets/efb68a82-c233-441b-bbd2-902f7bfc2c2c)

train, test 데이터셋 분리
```python
df = balanced_df[['content', 'label']].copy()
df.columns = ['text', 'label']  # HuggingFace 형식에 맞게 컬럼명 변경

# 2. train/test 분리 (stratify)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
```

토크나이저 로드
```python
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

패딩 
```python
def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=35)

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)
```

라벨 타입 정수형으로 지정
```python
train_dataset = train_dataset.cast_column("label", Value("int64"))
test_dataset = test_dataset.cast_column("label", Value("int64"))
```

config 설정 (드랍아웃 기본값 0.1 -> 0.3)
```python
config = ElectraConfig.from_pretrained(
    model_name,
    num_labels=3,
    hidden_dropout_prob=0.3,               # ✅ hidden layer dropout 확률 조정
    attention_probs_dropout_prob=0.3       # ✅ self-attention dropout 확률 조정
)
```

모델 생성
```python
model = ElectraForSequenceClassification.from_pretrained(
    model_name,
    config=config
)
```

평가지표 정의
```python
# 4. 평가지표 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": (preds == labels).mean(),
        "eval_f1_macro": f1_score(labels, preds, average="macro")
    }
```

학습 인자 설정
```python
# 5. Trainer 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",         
    eval_steps=250,                      # 🔄 평가 주기 조정 : 250
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",  # 🔧 'eval_' 붙여야 동작함
    greater_is_better=True,
    
    num_train_epochs=6,                  # ✔️ 짧게. EarlyStopping도 있으니 overfitting 방지
    per_device_train_batch_size=32,      
    per_device_eval_batch_size=64,

    learning_rate=1e-5,                  
    weight_decay=0.01,

    warmup_steps=1000,                    # ✔️ Warmup 적용 (예열 단계)
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,                   # ✔️ 자주 로깅하여 모니터링

    report_to=[],                        # 🔕 tensorboard 끔
    seed=42
)
```
학습 객체 설정
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  # ✔️ patience 줄여 빠른 정지 유도
)
```

모델 학습
```python
trainer.train()
```
![image](https://github.com/user-attachments/assets/94f0ac83-5296-48fe-b9c9-78d371161a02)


## 결과

### Accuracy, Loss 그래프
#### LSTM
![image](https://github.com/user-attachments/assets/001f92cf-3e75-426b-b94e-43295a0e0e87)
Accuracy - LSTM
![image](https://github.com/user-attachments/assets/dcf198de-7612-4c10-bf9b-e56fa578204a)
Loss - LSTM

#### beomi/KcELECTRA-base-v2022
![image](https://github.com/user-attachments/assets/315164c0-ccc0-4a75-bd15-de365c7ffc13)
Accuracy - KcELECTRA
![image](https://github.com/user-attachments/assets/f27aebca-d113-4b79-8764-cf36cea34db0)
Loss - KcELECTRA

### classification_report

#### LSTM
![image](https://github.com/user-attachments/assets/e15b0bad-943b-4851-a6c7-b603c959c7d1)
classification_report - LSTM

#### beomi/KcELECTRA-base-v2022
![image](https://github.com/user-attachments/assets/20ced92a-38ef-435f-bafb-23f47f721665)
classification_report - KcELECTRA

Precision, Recall, F1-score 전체적으로 beomi/KcELECTRA-base-v2022 우수

## beomi/KcELECTRA-base-v2022 하이퍼파라미터 조정
### 하이퍼파라미터 조정으로 성능 개선 목표

config - dropout 0.3 -> 0.4 조절
```python
config = ElectraConfig.from_pretrained(
    model_name,
    num_labels=3,
    hidden_dropout_prob=0.4,               # ✅ hidden layer dropout 확률 조정
    attention_probs_dropout_prob=0.4       # ✅ self-attention dropout 확률 조정
)
```

클래스 별 가중치 조절 (이전 KcELECTRA 모델은 중립에 대한 성능이 전체적으로 낮음 -> 가중치 2배 부여)
```python
from torch.nn import CrossEntropyLoss

# 클래스별 가중치: (예시) [부정:1.0, 중립:2.0, 긍정:1.0]
class_weights = torch.tensor([1.0, 2.0, 1.0]).to(model_0624_5.device)

# Custom Trainer 정의
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # GPU로 class weight 보내기
        weight = class_weights.to(model.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
```

학습 인자 수정
```python
training_args = TrainingArguments(
    output_dir="./best_model",
    evaluation_strategy="steps",
    eval_steps=500,                 # 250 -> 500 steps
    save_strategy="steps",
    save_steps=500,                
    save_total_limit=2,             
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    
    num_train_epochs=6,
    per_device_train_batch_size=64,  # 배치사이즈 32 -> 64 (빠른 학습)
    per_device_eval_batch_size=64,
    
    learning_rate=3e-5,              # 1e-5 -> 3e-5 (모델이 더 빠르게 파라미터를 업데이트)
    weight_decay=0.01,
    warmup_steps=500,                # 1000 -> 500 (예열 구간 감소 : 학습 초기에 학습률(Learning Rate)을 천천히 증가)
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=100,              
    
    fp16=True,                      # GPU 자원을 덜 사용하고 학습 속도를 높이는 기법
    report_to=[],                   
    seed=42
)
```

학습 객체 수정
```python
from transformers import EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,            
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),  
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # 기존 4 -> 3 (더 빠른 멈춤, 과적합 방지)
)
```

학습 실행
```python
trainer.train()
```
![image](https://github.com/user-attachments/assets/d87a3bf5-4db6-439d-ada3-33292cadf1bc)

### beomi/KcELECTRA-base-v2022 Original vs New 성능 비교
#### Accuracy, Loss 그래프
![image](https://github.com/user-attachments/assets/315164c0-ccc0-4a75-bd15-de365c7ffc13)
Accuracy - KcELECTRA
![image](https://github.com/user-attachments/assets/f27aebca-d113-4b79-8764-cf36cea34db0)
Loss - KcELECTRA
