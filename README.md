# 음식점 리뷰 감성분석 모델 구축 과정 및 결과 리포트
음식점 리뷰 감성분석 모델 구축, LSTM, beomi/KcELECTRA-base-v2022 비교

## 감성분석 모델 학습 계기


## 데이터 준비

[### 리뷰 데이터 크롤링](https://github.com/tpdus751/naver_map_restaurant_review_crawl_process) 

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
   
