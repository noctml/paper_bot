import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime

# 1. arXiv 논문 수집 (안정성 최우선 쿼리)
def fetch_papers():
    print("--- [Step 1] arXiv 논문 데이터 확보 중... ---")
    # 키워드를 분리하여 검색 확률을 높입니다.
    queries = [
        'cat:cs.RO AND (SLAM OR "Spatial AI" OR "Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction")'
    ]
    
    all_entries = []
    for q in queries:
        encoded_q = urllib.parse.quote(q)
        # 최신순 정렬하여 상위 25개씩 확보
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    unique_papers = {p.link: p for p in all_entries}.values()
    paper_list = list(unique_papers)
    print(f"총 {len(paper_list)}건의 고품질 후보군을 확보했습니다.")
    return paper_list

# 2. 고도화된 분석 로직 (학회 추론 및 가독성 최적화)
def evaluate_papers(papers):
    if not papers: return None
    print("--- [Step 2] 시니어 연구원 페르소나 기반 심층 분석 시작 ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 프롬프트: 학회 정보 추출 및 시니어 연구원급 비평 요구
    system_prompt = f"""
    너는 MIT SPARK Lab의 Luca Carlone과 Meta FAIR의 시니어 연구원이야. 
    사용자는 현재 3D Scene Graph, VIO, SLAM 분야의 연구자야. 
    전달받은 논문 중 2024년~현재(오늘: {datetime.now().strftime('%Y-%m-%d')}) 사이에 발표된 탑티어 학회(CVPR, ICRA, IROS, ECCV, NeurIPS 등)급 논문 5개를 엄선해줘.

    [작성 가이드라인]
    1. 학회 정보: 초록 내용이나 저자 정보를 토대로 발표된 학회(예: ICRA 2024)를 반드시 추론해 명시해. 불확실하면 'ArXiv (Top-tier candidate)'라고 적어.
    2. 가독성: 각 섹션을 이모지와 구분선으로 명확히 나눠.
    3. 똑똑한 비평: 단순 요약이 아니라, 이 연구가 사용자 연구의 '엄밀성'이나 '실용성' 측면에서 어떤 사고의 전환을 요구하는지Luca Carlone 스타일로 비평해.

    [카테고리]
    - 선호 주제: 수학적 엄밀성 기반 SLAM/Robotics (2개)
    - 최신 이슈: Embodied AI 및 최신 Vision 트렌드 (3개)
    """

    candidates = ""
    for i, p in enumerate(papers[:40]):
        candidates += f"ID: {i}\nTitle: {p.title}\nDate: {p.published}\nSummary: {p.summary}\nLink: {p.link}\n\n"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 논문 리스트에서 최고의 5개를 선별해 분석 리포트를 작성해줘:\n\n{candidates}"}
            ],
            temperature=0.6
        )
        report_content = response.choices[0].message.content
        
        # 종합 인사이트 질문 생성 (가장 중요한 마무리)
        insight_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 연구 멘토야. 사용자가 오늘 논문을 쓰거나 연구를 할 때 스스로에게 던져야 할 단 하나의 본질적인 질문을 뽑아줘."},
                {"role": "assistant", "content": report_content},
                {"role": "user", "content": "이 논문들을 종합해 볼 때, 내가 내 연구주제에서 '다음 단계'로 넘어가기 위해 답해야 할 핵심 질문은 무엇일까?"}
            ]
        )
        final_insight = insight_response.choices[0].message.content
        
        header = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n 📚 오늘의 시니어 연구원 브리핑 ({datetime.now().strftime('%Y-%m-%d')})\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
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
    msg['Subject'] = f"🚀 [Top-tier Update] 오늘의 고도화된 연구 브리핑"
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