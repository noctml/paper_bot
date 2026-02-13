import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime

# 1. arXiv 논문 수집 (잘 작동했던 안정적인 쿼리 방식)
def fetch_papers():
    print("--- [Step 1] arXiv 논문 수집 중... ---")
    # 날짜 필터를 빼고 키워드로만 검색해야 결과가 잘 나옵니다.
    queries = [
        'cat:cs.RO AND ("SLAM" OR "Spatial AI" OR "Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction")'
    ]
    all_entries = []
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        # 최신순 정렬을 통해 상위 30개를 가져옵니다.
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&max_results=30&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    unique_papers = {p.link: p for p in all_entries}.values()
    print(f"총 {len(unique_papers)}건의 논문 발견")
    return list(unique_papers)

# 2. OpenAI 평가 (날짜 필터링 및 심층 분석)
def evaluate_papers(papers):
    print("--- [Step 2] OpenAI 심층 분석 시작 ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not papers: return None

    # 프롬프트에 '현재 날짜 기준 2년 내 논문 선별' 지침 추가
    system_prompt = f"""
    너는 MIT SPARK Lab의 Luca Carlone과 Meta FAIR의 수석 연구원이야.
    전달받은 논문들 중 **반드시 발행일이 최근 2년(2024년~현재) 이내인** 탑티어 급 논문 5개를 선정해줘.

    [카테고리 구분]
    1. 선호 주제: 수학적 엄밀성을 갖춘 SLAM/Robotics (Luca Carlone 스타일) 2개
    2. 최신 이슈: 최신 Embodied AI/3D Vision 트렌드 (Meta 스타일) 3개

    [보고 형식]
    --------------------------------------------------
    [카테고리: 선호 주제 / 최신 이슈]
    논문 링크: (arXiv URL)
    논문 제목: (한글 번역 병기)
    학회/날짜: (확인 가능한 경우 학회 이름과 날짜 명시)
    1. 핵심 1줄 요약: 
    2. 제안 방법론 및 기술: (기술 스택 중심으로 핵심 요약)
    3. 연구 가치 및 사고의 방향: (Luca Carlone/Meta 관점에서 이 연구가 왜 가치 있고, 어떤 새로운 시각을 가져야 하는지 분석)
    --------------------------------------------------
    """

    candidates = ""
    for i, p in enumerate(papers):
        # 발행 날짜 정보를 OpenAI에게 함께 넘겨줍니다.
        candidates += f"ID: {i}\nTitle: {p.title}\nDate: {p.published}\nSummary: {p.summary}\nLink: {p.link}\n\n"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"오늘 날짜는 {datetime.now().strftime('%Y-%m-%d')}이야. 이 날짜 기준 2년 내의 최적의 논문 5개를 분석해줘:\n\n{candidates}"}
            ],
            temperature=0.7
        )
        evaluated_content = response.choices[0].message.content
        
        # 마지막 인사이트 질문 생성
        insight_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 연구 멘토야. 위 논문들을 관통하는 날카로운 질문 하나를 던져줘."},
                {"role": "assistant", "content": evaluated_content},
                {"role": "user", "content": "종합적으로 내 연구에 인사이트를 줄 핵심 질문 하나로 마무리해줘."}
            ]
        )
        final_insight = insight_response.choices[0].message.content
        return evaluated_content + "\n\n" + "="*50 + "\n" + "💡 [Today's Research Insight]\n" + final_insight

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

# 3. 이메일 발송
def send_email(content):
    print("--- [Step 3] 리포트 발송 중... ---")
    if not content: return

    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    msg = MIMEText(content)
    msg['Subject'] = f"🚀 [Top-tier] {datetime.now().strftime('%Y-%m-%d')} 연구 브리핑"
    msg['From'] = f"Research Mentor <{sender}>"
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("🎉 이메일 발송 성공!")
    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")

if __name__ == "__main__":
    papers = fetch_papers()
    report = evaluate_papers(papers)
    send_email(report)