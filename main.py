import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime, timedelta

# 1. arXiv 논문 수집 (최신 1~2년 논문 타겟팅)
def fetch_papers():
    print("--- [Step 1] arXiv 최신 논문 수집 중 (2024-2025 타겟)... ---")
    
    # 현재 날짜로부터 2년 전 날짜 계산 (예: 2024년 이후 논문만 검색)
    two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d%H%M%S')
    
    # 쿼리 설명: 핵심 키워드 + 최근 2년 내 제출된 논문
    queries = [
        'cat:cs.RO AND (SLAM OR "Spatial AI" OR "3D Scene Graph") AND lastUpdatedDate:[202401010000 TO 202612312359]',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction" OR "Multimodal") AND lastUpdatedDate:[202401010000 TO 202612312359]'
    ]
    
    all_entries = []
    for q in queries:
        encoded_q = urllib.parse.quote(q, safe=':[]')
        # 최신순 정렬 및 상위 40개 수집
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=40&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    unique_papers = {p.link: p for p in all_entries}.values()
    print(f"총 {len(unique_papers)}건의 최신 논문(2024-2025) 후보 발견")
    return list(unique_papers)

# 2. OpenAI 평가 (Luca Carlone & Meta 스타일 + 최신성 검증)
def evaluate_papers(papers):
    print("--- [Step 2] 탑티어 학회 논문 큐레이션 및 심층 분석 ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not papers:
        return "수집된 최신 논문이 없습니다."

    system_prompt = """
    너는 MIT SPARK Lab의 Luca Carlone과 Meta Reality Labs의 수석 연구원이야.
    전달받은 후보들 중 **반드시 2024년~2025년 사이에 발표된** CVPR, ICRA, IROS 등 탑티어 수준의 논문 5개를 선정해줘.

    선정 기준:
    1. 고전적 엄밀성을 갖춘 SLAM/Robotics 연구 (Luca Carlone 스타일) 2개
    2. 최신 Embodied AI/3D Vision 트렌드 (Meta 스타일) 3개

    각 논문 보고 형식:
    --------------------------------------------------
    [카테고리: 선호 주제 / 최신 이슈]
    논문 제목: (한글 번역 병기)
    학회 정보: (예: CVPR 2024, ICRA 2025 등 확인 가능한 경우 명시)
    1. 핵심 1줄 요약: 
    2. 제안 방법론 및 기술: (기술 스택 중심으로 간결하게)
    3. 연구 가치 및 사고의 방향: (이 연구가 왜 중요한지, 어떤 새로운 시각을 가져야 하는지 Luca Carlone/Meta 관점에서 심도 있게 분석)
    --------------------------------------------------
    """

    candidates = ""
    for i, p in enumerate(papers[:30]): 
        candidates += f"ID: {i}\nTitle: {p.title}\nSummary: {p.summary}\nDate: {p.published}\n\n"

    prompt = f"다음 최신 논문 후보들 중 최적의 5개를 선정해 분석해줘:\n\n{candidates}"

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
        
        # 인사이트 질문 생성
        insight_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 연구 멘토야. 위 논문들을 기반으로 사용자가 오늘 깊게 고민해볼 만한 질문을 하나 던져줘."},
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
    print("--- [Step 3] 최신 논문 리포트 발송 ---")
    if not content: return

    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    msg = MIMEText(content)
    msg['Subject'] = f"🚀 [Recent Top-tier] 연구 브리핑 ({datetime.now().strftime('%Y-%m-%d')})"
    msg['From'] = f"Research Mentor <{sender}>"
    msg['To'] = receiver

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.send_message(msg)
    print("🎉 이메일 발송 성공!")

if __name__ == "__main__":
    papers = fetch_papers()
    report = evaluate_papers(papers)
    send_email(report)