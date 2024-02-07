from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from gensim.summarization.summarizer import summarize
from transformers import pipeline 

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# '/analyze_news' 엔드포인트 정의
@app.post("/analyze_news")
async def analyze_news(news_data: dict):
    try:
        # 클라이언트 요청 데이터 확인
        if "news_article" not in news_data:
            raise HTTPException(status_code=400, detail="news_article field is required")

        # 기사 내용 가져오기
        news_article = news_data["news_article"]
        
        # 기사 요약
        summarized_article = summarize(news_article, word_count= 50)
        
        # 감정 분석
        classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
        sentiment_results = classifier(summarized_article)
        
        return {"summarized_article": summarized_article, "sentiment_results": sentiment_results}
    except HTTPException as e:
        # HTTP 예외 처리
        raise e
    except Exception as e:
        # 기타 예외 처리
        raise HTTPException(status_code=500, detail=str(e))


# 루트 엔드포인트에 대한 HTML 응답을 반환하는 함수 추가
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>90'Z의 주식분석! 당신도 나락을 갈 수 있다.</title>
    <style>
        /* CSS 초기화 */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            width: 1260px;
            height: 650px;
            display: flex;
        }

        .news-input {
            width: 685px;
            height: 650px;
            background-color: #ffffff;
            padding: 50px 0px;
        }

        .result {
            width: 575px;
            height: 650px;
            color: #ffffff;
            background-color: #ffffff;
            padding: 20px;
            box-sizing: border-box;
        }

        .result-text {
            font-size: 18px;
            margin-bottom: 20px;
            font-family: Arial, sans-serif;
            color : #111111; 
            text-align: center;
        }

        .summary {
            width: 508px;
            height: 25px;
            margin-bottom: 10px;
            
        }

        .summary-text {
            width: 508px;
            height: 185px;
            font-family: Arial, sans-serif;
            color : #111111; 
            font-size : 18px
        }
        .summary, .result-text {
            border: 1px solid #ccc; /* 회색 테두리 추가 */
            padding: 10px; /* 내부 여백 추가 */
            margin-bottom: 20px; /* 아래쪽 여백 추가 */
            width: 535px;
            height: 198px;
}

        .btn {
            width: 250px;
            height: 60px;
            background-color: #007bff;
            color: #ffffff;
            font-size: 18px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: -15px;
        }

        .btn:hover {
            background-color: #0056b3;
        }

        .title {
            background-color: #007bff;
            height: 60px;
        }

        .title h2 {
            text-align: center;
            padding: 15px;
        }
    </style>
</head>
<body>
    <div class="header" style="background-color: #ffffff; height: 60px;">
    <h2 style="text-align: center;"><img src="/static/img/logo.png" alt="90'z 로고" style="max-width: 120px;"></h2>
    </div>
    <div class="container_box" style="display: flex; justify-content: center; margin: 100px;">
    <div class="container">
        <div class="news-input">
            <textarea id="news-article" placeholder="분석하고자 하는 기사 내용을 입력하세요." style="width: 85%; margin: 20px; height: 477px; text-align: center; resize: none;"></textarea>
            <div style="text-align: center;">
            <button class="btn" onclick="analyzeNews()">결과 확인</button>
            </div>
        </div>
        <div class="result">
            <div class="result_box" style="margin-top:45px">
            <div class="title">
            <h2>분석 결과</h2>
          </div>
            <div class="summary">
                <div class="summary-text" id="summary-text"></div>
            </div>
            <div class="title_box" style="margin-top: 35px;">
            <div class="title">
                <h2>결과</h2>
            <div class="result-text" id="result-text" style="height: 125px;"></div>
        </div>
        </div>
        </div>
    </div>
    </div>
    </div>
    <script>
function analyzeNews() {
    var newsArticle = document.getElementById("news-article").value;

    fetch('/analyze_news', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ news_article: newsArticle })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to fetch data from server');
        }
        return response.json();
    })
    .then(data => {
        document.getElementById("summary-text").innerText = data.summarized_article;

        var score = 0.00000;
        var label = "";

        for (var i = 0; i < 3; i++) {
            if (data.sentiment_results[0][i]['label'] != 'neutral') {
                label = data.sentiment_results[0][i]['label'];
                console.log(label)
                
                
                if (data.sentiment_results[0][i]['score'] > score) {
                    score = data.sentiment_results[0][i]['score'];
                    console.log(score)
                    }
            }
            console.log(data.sentiment_results[0])
        }
        document.getElementById("result-text").innerText = label;
           
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

    </script>
</body>
</html>
    """