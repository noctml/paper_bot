import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText

# 1. arXiv 논문 수집 (검색 키워드 최적화)
def fetch_papers():
    print("--- [Step 1] arXiv 논문 수집 중... ---")
    # 학회 이름을 직접 넣기보다 분야별 핵심 키워드로 검색하고 정렬하여 수집합니다.
    queries = [
        'cat:cs.RO AND (SLAM OR "Spatial AI" OR "3D Scene Graph" OR "Visual Odometry")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction" OR "Vision-Language Model")'
    ]
    all_entries = []
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        # 검색 결과 개수를 30개로 늘려 더 많은 후보군 중 고르게 합니다.
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    unique_papers = {p.link: p for p in all_entries}.values()
    print(f"총 {len(unique_papers)}건의 최신 논문 후보 발견")
    return list(unique_papers)

# 2. OpenAI 평가 (사용자 맞춤형 분석 로직)
def evaluate_papers(papers):
    print("--- [Step 2] 논문 큐레이션 및 심층 분석 시작 ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # 분석 프롬프트 고도화
    system_prompt = """
    너는 MIT SPARK Lab의 Luca Carlone과 Meta Reality Labs의 수석 연구원이야.
    논문 리스트 중 사용자가 좋아할 '고전적 엄밀성을 갖춘 SLAM/Robotcs 연구' 2개와 
    최근 이슈가 되는 '최신 Deep Learning/Vision 트렌드' 3개를 엄선해줘.
    
    각 논문은 아래 형식을 엄격히 지켜서 작성해:
    [카테고리: 선호 주제 / 최신 이슈]
    1. 핵심 1줄 요약: 
    2. 제안 방법론 및 기술: (짧고 핵심적인 기술 스택 중심)
    3. 연구 가치 및 사고의 방향: (이 연구가 Luca Carlone이나 Meta의 연구 방향과 어떻게 맞닿아 있는지, 어떤 새로운 시각을 가져야 하는지 분석)
    """

    evaluated_content = ""
    
    # 상위 10개 중 가장 가치 있는 5개를 골라달라고 요청
    candidates = ""
    for i, p in enumerate(papers[:10]):
        candidates += f"ID: {i}\nTitle: {p.title}\nSummary: {p.summary}\n\n"

    prompt = f"다음 논문 후보들 중 최적의 5개를 선정해 분석해줘:\n\n{candidates}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        evaluated_content = response.choices[0].message.content
        
        # 마지막 핵심 질문 추가를 위한 별도 호출 (인사이트 강화)
        insight_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 연구 멘토야. 위 논문들을 관통하는 핵심적인 질문 하나를 던져줘."},
                {"role": "assistant", "content": evaluated_content},
                {"role": "user", "content": "종합적으로 내 연구에 인사이트를 줄만한 하나의 핵심 질문으로 마무리해줘."}
            ]
        )
        final_insight = insight_response.choices[0].message.content
        return evaluated_content + "\n\n" + "="*50 + "\n" + "💡 [Today's Research Insight]\n" + final_insight

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

# 3. 이메일 발송
def send_email(content):
    print("--- [Step 3] 고도화된 리포트 발송 중... ---")
    if not content: return

    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    msg = MIMEText(content)
    msg['Subject'] = "🚀 [Top-tier] 오늘의 맞춤형 연구 브리핑"
    msg['From'] = f"Research Mentor Bot <{sender}>"
    msg['To'] = receiver

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.send_message(msg)
    print("🎉 이메일 발송 성공!")

if __name__ == "__main__":
    papers = fetch_papers()
    report = evaluate_papers(papers)
    send_email(report)