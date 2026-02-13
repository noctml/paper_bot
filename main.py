import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime

# 1. arXiv 논문 수집 (사용자님이 성공했던 쿼리 방식 100% 복구)
def fetch_papers():
    print("--- [Step 1] arXiv 데이터 수집 중 (검증된 안정 쿼리)... ---")
    # 성공이 보장된 쿼리 구조입니다.
    queries = [
        'cat:cs.RO AND ("SLAM" OR "Spatial AI" OR "Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction")'
    ]
    
    all_entries = []
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        # 최신순 정렬하여 상위 25개씩 확보 (총 50개 후보)
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    # 중복 제거 (링크 기준)
    unique_papers = {p.link: p for p in all_entries}.values()
    paper_list = list(unique_papers)
    print(f"총 {len(paper_list)}건의 고품질 후보군 확보.")
    return paper_list

# 2. 고도화된 분석 로직 (전문 용어 영어 유지 + 날짜 필터링 위임)
def evaluate_papers(papers):
    if not papers: return None
    print("--- [Step 2] Senior Researcher 심층 분석 시작 ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 모델 설정 (상황에 따라 gpt-4o 또는 gpt-4o-mini 등 사용)
    MODEL_NAME = "gpt-4o-mini" 

    # 프롬프트: 날짜 필터링 지침 및 기술 용어 영어 유지
    current_date = datetime.now().strftime('%Y-%m-%d')
    system_prompt = f"""
    당신은 MIT SPARK Lab의 Luca Carlone과 Meta Reality Labs의 Senior Researcher입니다. 
    오늘 날짜는 {current_date}입니다.

    [핵심 미션]
    1. **최신성 검증**: 전달받은 리스트 중 발행일(Date)이 **최근 2년(2024년~현재) 이내인** 논문만 고려하십시오.
    2. **용어 원어 유지**: SLAM, VIO, 3D Scene Graph, Factor Graph, Optimization, Transformer, Latent Space 등 모든 전문 용어는 번역하지 말고 반드시 '영문 원어' 그대로 표기하십시오. 설명 문구만 한글로 작성합니다.
    3. **Venue 표기**: 논문에 학회(CVPR, ICRA 등) 정보가 명시되어 있다면 표기하고, 없다면 'ArXiv (Recent Update)'라고 적으십시오. 'Expected' 같은 표현은 쓰지 마십시오.

    [리포트 형식]
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    [Category: 선호 주제 / 최신 트렌드]
    ■ Title: (영문 제목 및 한글 번역 병기)
    ■ Venue: (학회 혹은 ArXiv)
    ■ Link: (arXiv URL)

    1. 핵심 요약 (1-Line): 
    2. 방법론 (Methodology): (핵심 기술 스택/알고리즘 위주로 기술용어 영어 유지하며 정리)
    3. 비평 (Senior Review): (Luca Carlone/Meta 관점에서 이 연구가 던지는 Insight와 가치 비평)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    candidates = ""
    for i, p in enumerate(papers):
        # 발행 날짜 정보를 OpenAI에게 넘겨주어 직접 필터링하게 합니다.
        candidates += f"ID: {i}\nTitle: {p.title}\nDate: {p.published}\nSummary: {p.summary}\nLink: {p.link}\n\n"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 논문 중 2024-2025년 최적의 5개를 선별해 분석 리포트를 작성하십시오:\n\n{candidates}"}
            ]
        )
        report_content = response.choices[0].message.content
        
        # 종합 인사이트 질문 생성
        insight_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "당신은 연구 멘토입니다. 기술 용어는 영어를 유지하며 질문하십시오."},
                {"role": "assistant", "content": report_content},
                {"role": "user", "content": "이 논문들을 관통하는 내 연구의 패러다임 시프트를 위한 핵심 질문 하나를 던져주십시오."}
            ]
        )
        final_insight = insight_response.choices[0].message.content
        
        header = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n 🧠 Senior Research Briefing ({current_date})\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
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
    msg['From'] = f"Research Mentor Bot <{sender}>"
    msg['To'] = receiver

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.send_message(msg)
    print("🎉 리포트 발송 성공!")

if __name__ == "__main__":
    paper_candidates = fetch_papers()
    report = evaluate_papers(paper_candidates)
    send_email(report)