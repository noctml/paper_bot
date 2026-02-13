import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime, timedelta

# 1. arXiv 논문 수집 (현재 날짜 기준 최근 2년 자동 계산)
def fetch_papers():
    # 현재 시점 기준 2년 전 날짜 계산
    two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d%H%M%S')
    current_date = datetime.now().strftime('%Y%m%d%H%M%S')
    
    print(f"--- [Step 1] arXiv 수집 시작 (범위: {two_years_ago[:4]}년 ~ 현재) ---")
    
    # 쿼리에 동적으로 날짜 범위를 주입합니다.
    queries = [
        f'cat:cs.RO AND (SLAM OR "Spatial AI" OR "3D Scene Graph") AND lastUpdatedDate:[{two_years_ago} TO {current_date}]',
        f'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction" OR "Multimodal") AND lastUpdatedDate:[{two_years_ago} TO {current_date}]'
    ]
    
    all_entries = []
    for q in queries:
        # arXiv API는 대괄호([])와 콜론(:)을 특수문자로 처리하므로 인코딩 시 예외처리
        encoded_q = urllib.parse.quote(q, safe=':[]')
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&start=0&max_results=40&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    
    # 중복 제거
    unique_papers = {p.link: p for p in all_entries}.values()
    print(f"총 {len(unique_papers)}건의 최신 연구 후보 발견")
    return list(unique_papers)

# 2. OpenAI 평가 (Luca Carlone & Meta 스타일 분석)
def evaluate_papers(papers):
    if not papers:
        print("⚠️ 검색된 논문이 없습니다. 쿼리 범위를 넓힙니다.")
        return None

    print("--- [Step 2] 탑티어 큐레이션 및 심층 인사이트 분석 ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_prompt = """
    너는 MIT SPARK Lab의 Luca Carlone과 Meta FAIR의 수석 연구원이야.
    전달받은 후보들 중 **최근 1~2년 내 발표된** CVPR, ICRA, IROS 등 탑티어 수준의 논문 5개를 선정해줘.

    선정 및 분석 기준:
    1. 선호 주제: 수학적 엄밀성을 갖춘 SLAM/Robotics (Luca Carlone 스타일) 2개
    2. 최신 이슈: 최신 Embodied AI/3D Vision 트렌드 (Meta 스타일) 3개

    각 논문 리포트 형식:
    --------------------------------------------------
    [카테고리: 선호 주제 / 최신 이슈]
    논문 링크: (arXiv URL)
    논문 제목: (한글 번역 병기)
    학회/날짜: (학회 이름과 정확한 발표 날짜 명시)
    1. 핵심 1줄 요약: 
    2. 제안 방법론 및 기술: (기술 스택 중심으로 핵심 요약)
    3. 연구 가치 및 사고의 방향: (이 연구가 왜 중요한지, 어떤 시각을 가져야 하는지 Luca Carlone/Meta 관점에서 분석)
    --------------------------------------------------
    """

    candidates = ""
    for i, p in enumerate(papers[:30]): 
        candidates += f"ID: {i}\nTitle: {p.title}\nDate: {p.published}\nSummary: {p.summary}\nLink: {p.link}\n\n"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 최신 논문 리스트 중 사용자에게 가장 가치 있는 5개를 분석해줘:\n\n{candidates}"}
            ],
            temperature=0.7
        )
        evaluated_content = response.choices[0].message.content
        
        # 마지막 핵심 질문 생성
        insight_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 연구 멘토야. 위 논문들의 흐름을 관통하는 본질적인 질문 하나를 던져줘."},
                {"role": "assistant", "content": evaluated_content},
                {"role": "user", "content": "내 연구와 논문 작성에 거대한 영감을 줄 핵심 질문 하나로 마무리해줘."}
            ]
        )
        final_insight = insight_response.choices[0].message.content
        return evaluated_content + "\n\n" + "="*50 + "\n" + "💡 [Today's Research Insight]\n" + final_insight

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

# 3. 이메일 발송
def send_email(content):
    if not content: return
    print("--- [Step 3] 고도화된 리포트 발송 중... ---")

    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    msg = MIMEText(content)
    msg['Subject'] = f"🚀 [Recent Top-tier] {datetime.now().strftime('%Y-%m-%d')} 연구 브리핑"
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