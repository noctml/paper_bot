import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime

# 1. arXiv 논문 데이터 확보
def fetch_papers():
    print("--- [Step 1] arXiv 최신 논문 데이터 확보 중... ---")
    queries = [
        'cat:cs.RO AND (SLAM OR "Spatial AI" OR "3D Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction")'
    ]
    all_entries = []
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    unique_papers = {p.link: p for p in all_entries}.values()
    print(f"총 {len(unique_papers)}건의 고품질 후보를 확보했습니다.")
    return list(unique_papers)

# 2. GPT-5-mini 전용 심층 분석 로직
def evaluate_papers(papers):
    if not papers: return None
    print("--- [Step 2] GPT-5-mini 기반 고도의 추론 분석 시작 ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 최신 모델명 적용
    MODEL_NAME = "gpt-5-mini" 

    # GPT-5-mini는 지시사항을 매우 잘 따르므로, 제약 조건을 유저 메시지에 명확히 통합합니다.
    full_prompt = f"""
    당신은 MIT SPARK Lab의 Luca Carlone과 Meta FAIR의 수석 연구원입니다.
    사용자는 3D Scene Graph, VIO, SLAM 분야의 권위자입니다.
    
    [미션]
    전달받은 리스트 중 2024년 이후 발표된 탑티어(CVPR, ICRA 등)급 논문 5개를 엄선하여 분석 리포트를 작성하십시오.
    
    [출력 가이드라인]
    1. 학회 추론: 기술적 성숙도를 분석하여 예상 학회를 명시하십시오.
    2. 비평적 분석: 단순 요약 대신, Luca Carlone 스타일의 수학적 엄밀성과 Meta의 실용적 혁신 관점을 섞어 비평하십시오.
    3. 가독성: 굵은 선과 이모지를 사용하여 시각적으로 구조화하십시오.

    [후보 논문 리스트]
    """
    
    candidates = ""
    for i, p in enumerate(papers[:35]):
        candidates += f"ID: {i}\nTitle: {p.title}\nDate: {p.published}\nSummary: {p.summary}\nLink: {p.link}\n\n"

    try:
        # GPT-5-mini는 temperature=1(기본값)에서 가장 안정적이므로 파라미터를 생략하거나 1로 설정합니다.
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": full_prompt + candidates}
            ]
        )
        report_content = response.choices[0].message.content
        
        # 종합 인사이트 질문 추출
        insight_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": f"위 분석 내용을 바탕으로, 연구자가 자신의 연구 주제에서 패러다임 시프트를 일으키기 위해 스스로에게 던져야 할 가장 파괴적인 질문 하나를 도출하십시오.\n\n분석내용:\n{report_content}"}
            ]
        )
        final_insight = insight_response.choices[0].message.content
        
        header = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n 🚀 GPT-5-mini Intelligence Report ({datetime.now().strftime('%Y-%m-%d')})\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
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
    msg['Subject'] = f"🔥 [GPT-5-mini] 오늘의 전략적 연구 리포트"
    msg['From'] = f"Senior AI Research Bot <{sender}>"
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("🎉 리포트 발송 성공!")
    except Exception as e:
        print(f"❌ 발송 에러: {e}")

if __name__ == "__main__":
    candidates = fetch_papers()
    report = evaluate_papers(candidates)
    send_email(report)