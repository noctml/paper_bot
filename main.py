import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime, timedelta

# 1. arXiv 논문 데이터 확보 (최신 데이터 위주)
def fetch_papers():
    print("--- [Step 1] arXiv 최신 데이터 수집 중... ---")
    queries = [
        'cat:cs.RO AND (SLAM OR "Spatial AI" OR "3D Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction")'
    ]
    
    all_entries = []
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        # 상위 15개씩만 가져와서 오늘/어제 발표된 것에 집중합니다.
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=15&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    unique_papers = {p.link: p for p in all_entries}.values()
    print(f"총 {len(unique_papers)}건의 후보군 확보.")
    return list(unique_papers)

# 2. 고도화된 분석 로직 (중복 제거 및 전문 용어 유지)
def evaluate_papers(papers):
    if not papers: return None
    print("--- [Step 2] GPT-5-mini 기반 선별 및 Deep Analysis ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 현재 모델 설정
    MODEL_NAME = "gpt-4o-mini" # 혹은 gpt-5-mini

    # 프롬프트: 중복 방지 및 학회 표기 수정
    system_prompt = f"""
    당신은 MIT SPARK Lab과 Meta Reality Labs의 Senior Researcher입니다. 
    오늘 날짜({datetime.now().strftime('%Y-%m-%d')}) 기준, 새로 업데이트된 논문들 중 사용자에게 가장 가치 있는 5개를 엄선하세요.

    [핵심 지침]
    1. **Freshness Focus**: 전달된 리스트 중 가급적 오늘 또는 어제 날짜의 논문을 우선적으로 선정하여 중복을 최소화하십시오.
    2. **Technical Terms**: SLAM, VIO, 3D Scene Graph, Backend Optimization, Pose Graph, Latent Space 등 모든 기술 용어는 '영문 원어' 그대로 사용하십시오.
    3. **Venue 표기**: 논문 정보에 학회(CVPR, ICRA 등)가 명시되어 있다면 해당 학회를 적고, 없다면 'ArXiv (Recent Update)'라고만 표기하십시오. 'Expected' 같은 불확실한 추측은 지양합니다.

    [리포트 포맷]
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    [Category: 선호 주제 / 최신 트렌드]
    ■ Title: (영문 제목)
    ■ Venue: (학회 이름 혹은 ArXiv)
    ■ Link: (arXiv URL)

    1. 핵심 요약 (1-Line): 
    2. 방법론 (Methodology): (핵심 기술 스택/알고리즘 위주 영문 혼용)
    3. 비평 (Senior Review): (Luca Carlone/Meta 관점에서 이 연구가 던지는 Insight와 가치 비평)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    candidates = ""
    for i, p in enumerate(papers):
        candidates += f"ID: {i}\nTitle: {p.title}\nDate: {p.published}\nSummary: {p.summary}\nLink: {p.link}\n\n"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"오늘의 신규 논문 분석 리포트를 작성하십시오:\n\n{candidates}"}
            ]
        )
        report_content = response.choices[0].message.content
        
        # 인사이트 질문 생성
        insight_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": f"위 리포트를 바탕으로, 연구자가 자신의 SLAM/VIO 연구 파이프라인에서 당장 고민해봐야 할 아주 날카로운 질문 하나를 던지십시오.\n\n리포트 요약:\n{report_content}"}
            ]
        )
        final_insight = insight_response.choices[0].message.content
        
        header = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n 🧠 Senior Research Briefing ({datetime.now().strftime('%Y-%m-%d')})\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        footer = f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n💡 [TODAY'S CRITICAL QUESTION]\n\n{final_insight}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        return header + report_content + footer

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

# 3. 이메일 발송
def send_email(content):
    if not content: return
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    msg = MIMEText(content)
    msg['Subject'] = f"🚀 [Research Update] {datetime.now().strftime('%Y-%m-%d')} 리포트"
    msg['From'] = f"Research Mentor <{sender}>"
    msg['To'] = receiver

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.send_message(msg)
    print("🎉 리포트 발송 성공!")

if __name__ == "__main__":
    paper_candidates = fetch_papers()
    report = evaluate_papers(paper_candidates)
    send_email(report)