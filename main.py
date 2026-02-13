import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime

# 1. arXiv 논문 데이터 확보 (가장 안정적인 쿼리 방식)
def fetch_papers():
    print("--- [Step 1] arXiv 논문 데이터 확보 중... ---")
    queries = [
        'cat:cs.RO AND (SLAM OR "Spatial AI" OR "Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction")'
    ]
    
    all_entries = []
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    unique_papers = {p.link: p for p in all_entries}.values()
    paper_list = list(unique_papers)
    print(f"총 {len(paper_list)}건의 고품질 후보군 확보.")
    return paper_list

# 2. 고도화된 분석 로직 (전문 용어 영어 유지 + GPT-5-mini 최적화)
def evaluate_papers(papers):
    if not papers: return None
    print("--- [Step 2] 시니어 연구원 페르소나 기반 심층 분석 시작 ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # GPT-5-mini 모델 적용 (사용하시는 환경에 따라 모델명 확인 필요)
    MODEL_NAME = "gpt-4o-mini" 

    # 프롬프트: 기술 용어 영어 유지 가이드 추가
    system_prompt = f"""
    너는 MIT SPARK Lab의 Luca Carlone과 Meta FAIR의 시니어 연구원이야. 
    오늘 날짜({datetime.now().strftime('%Y-%m-%d')}) 기준, 최근 2년 내 발표된 탑티어(CVPR, ICRA, IROS 등)급 논문 5개를 엄선해.

    [핵심 작성 규칙]
    1. 용어 표기: SLAM, 3D Scene Graph, VIO, Factor Graph, Optimization, Transformer 등 모든 전문 용어와 기술적 키워드는 번역하지 말고 반드시 '영문 원어' 그대로 표기해. 설명 문구만 한글로 작성해.
    2. 학회 정보: 초록을 분석하여 예상 학회(예: CVPR 2024)를 명시해. 불확실하면 'ArXiv'로 표기.
    3. 똑똑한 비평: Luca Carlone의 '수학적 엄밀성'과 Meta의 '시스템적 효율성' 관점에서 이 연구가 사용자에게 어떤 새로운 Perspective를 주는지 분석해.
    4. 가독성: 이모지와 굵은 구분선을 사용하여 시각적으로 구조화해.
    """

    candidates = ""
    for i, p in enumerate(papers[:40]):
        candidates += f"ID: {i}\nTitle: {p.title}\nDate: {p.published}\nSummary: {p.summary}\nLink: {p.link}\n\n"

    try:
        # GPT-5-mini 등 최신 모델과의 호환성을 위해 temperature를 제거하거나 1로 설정
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 논문 후보 중 최적의 5개를 선별해 분석 리포트를 작성해:\n\n{candidates}"}
            ]
        )
        report_content = response.choices[0].message.content
        
        # 종합 인사이트 질문 생성
        insight_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "너는 연구 멘토야. 기술 용어는 영어로 쓰되, 연구자의 뇌를 자극할 날카로운 질문 하나를 뽑아줘."},
                {"role": "assistant", "content": report_content},
                {"role": "user", "content": "이 논문들을 관통하는 하나의 거대한 '핵심 질문'으로 마무리해줘."}
            ]
        )
        final_insight = insight_response.choices[0].message.content
        
        header = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n 🧠 Senior Researcher Briefing ({datetime.now().strftime('%Y-%m-%d')})\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        footer = f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n💡 [MASTER QUESTION FOR TODAY]\n\n{final_insight}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        return header + report_content + footer

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

# 3. 이메일 발송
def send_email(content):
    if not content: return
    print("--- [Step 3] 가공된 리포트 이메일 발송 중... ---")

    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    msg = MIMEText(content)
    msg['Subject'] = f"🚀 [Intelligence Report] {datetime.now().strftime('%Y-%m-%d')} 연구 브리핑"
    msg['From'] = f"Senior Research Bot <{sender}>"
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("🎉 리포트 발송 성공!")
    except Exception as e:
        print(f"❌ 발송 에러: {e}")

if __name__ == "__main__":
    paper_candidates = fetch_papers()
    report = evaluate_papers(paper_candidates)
    send_email(report)