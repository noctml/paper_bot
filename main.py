import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime

# 1. arXiv 논문 데이터 확보 (안정성이 검증된 최신순 쿼리)
def fetch_papers():
    print("--- [Step 1] arXiv Data Retrieval ---")
    queries = [
        'cat:cs.RO AND ("SLAM" OR "Spatial AI" OR "Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction")'
    ]
    
    all_entries = []
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        # 최신순 정렬(submittedDate)로 상위 25개씩 수집
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    unique_papers = {p.link: p for p in all_entries}.values()
    paper_list = list(unique_papers)
    print(f"Total {len(paper_list)} candidates secured.")
    return paper_list

# 2. 고도화된 분석 로직 (발행 날짜 포함 + 영문 용어 유지)
def evaluate_papers(papers):
    if not papers: return None
    print("--- [Step 2] GPT-powered Deep Analysis ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    MODEL_NAME = "gpt-4o-mini" 

    current_date = datetime.now().strftime('%Y-%m-%d')
    system_prompt = f"""
    당신은 MIT SPARK Lab(Luca Carlone)과 Meta Reality Labs의 Senior Researcher입니다. 
    오늘 날짜는 {current_date}입니다.

    [핵심 미션]
    전달받은 후보 리스트 중 2024-2025년에 발표된 가장 영향력 있는 논문 5개를 엄선하여 리포트를 작성하십시오.

    [작성 규칙]
    1. **영문 용어 유지**: SLAM, VIO, 3D Scene Graph, Backend Optimization, Pose Graph, Factor Graph, Transformer, Embodied AI, Latent Space, Outlier Rejection 등 모든 기술적 전문 용어는 번역하지 말고 반드시 '영문 원어' 그대로 사용하십시오.
    2. **게시 날짜 명시**: 각 논문의 'Published Date'를 리포트에 반드시 포함하십시오.
    3. **Venue**: CVPR, ICRA, IROS 등 학회 정보가 확인되면 명시하고, 없다면 'ArXiv'로 표기하십시오.
    4. **비평**: Luca Carlone의 수학적 엄밀성과 Meta의 실용성 관점에서 해당 연구가 사용자에게 어떤 새로운 Insight를 주는지 분석하십시오.

    [리포트 포맷]
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    [Category: Preferred Research / Recent Trends]
    ■ Title: (영문 제목 및 한글 번역 병기)
    ■ Venue: (학회 이름 혹은 ArXiv)
    ■ Published Date: (논문 게시 날짜)
    ■ Link: (arXiv URL)

    1. 핵심 요약 (1-Line Summary): 
    2. 방법론 (Methodology): (Technical Terms를 영어로 유지하며 핵심 기술 요약)
    3. 비평 (Senior Review): (전문 연구원 관점의 심층 가치 분석)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    candidates = ""
    for i, p in enumerate(papers):
        # 발행 날짜(p.published) 정보를 포함하여 모델에게 전달
        candidates += f"ID: {i}\nTitle: {p.title}\nDate: {p.published}\nSummary: {p.summary}\nLink: {p.link}\n\n"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"오늘의 최신 논문 리스트(2024-2025) 중 5개를 선별해 분석하십시오:\n\n{candidates}"}
            ]
        )
        report_content = response.choices[0].message.content
        
        # 종합 인사이트 질문 생성
        insight_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": f"위 분석 리포트를 바탕으로, 연구자가 자신의 연구 분야에서 Next Step으로 나아가기 위해 고민해야 할 날카로운 Critical Question 하나를 뽑아주세요.\n\n리포트 요약:\n{report_content}"}
            ]
        )
        final_insight = insight_response.choices[0].message.content
        
        header = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n 🧠 Senior Research Briefing ({current_date})\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        footer = f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n💡 [TODAY'S CRITICAL QUESTION]\n\n{final_insight}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        return header + report_content + footer

    except Exception as e:
        print(f"❌ Analysis Failed: {e}")
        return None

# 3. 이메일 발송
def send_email(content):
    if not content: return
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    msg = MIMEText(content)
    msg['Subject'] = f"🚀 [Recent Top-tier] {datetime.now().strftime('%Y-%m-%d')} 연구 리포트"
    msg['From'] = f"Senior Research Bot <{sender}>"
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("🎉 Report successfully sent!")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

if __name__ == "__main__":
    paper_candidates = fetch_papers()
    report = evaluate_papers(paper_candidates)
    send_email(report)