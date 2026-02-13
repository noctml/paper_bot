import feedparser
import requests
from openai import OpenAI
import os
import smtplib
import json
from email.mime.text import MIMEText

# 1. arXiv에서 논문 수집
def fetch_papers():
    print("--- [Step 1] arXiv 논문 수집 중... ---")
    queries = [
        'cat:cs.RO AND ("SLAM" OR "Spatial AI" OR "Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction" OR "Multimodal")'
    ]
    
    all_entries = []
    for q in queries:
        url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=5&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    print(f"수집 완료: {len(all_entries)}건")
    return all_entries

# 2. GPT로 논문 평가 및 요약
def evaluate_papers(papers):
    print("--- [Step 2] GPT 평가 시작... ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    evaluated_list = []

    for p in papers[:5]: # 비용 절감을 위해 상위 5개만 정밀 분석
        prompt = f"""
        너는 MIT SPARK Lab과 Meta FAIR의 시니어 연구원이야. 
        다음 논문 초록을 읽고, 'Luca Carlone 스타일의 수치적 엄밀성'과 
        'Meta 스타일의 실용적 Embodied AI' 관점에서 중요도를 0~10점으로 평가해.
        반드시 다음 JSON 형식으로만 응답해: {{"score": 9.5, "reason": "...", "summary": "..."}}

        Title: {p.title}
        Summary: {p.summary}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful research assistant."},
                          {"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            result = json.loads(response.choices[0].message.content)
            result['title'] = p.title
            result['link'] = p.link
            evaluated_list.append(result)
            print(f"평가 완료: {p.title[:30]}... ({result['score']}점)")
        except Exception as e:
            print(f"평가 실패: {e}")
            
    # 점수 높은 순으로 정렬
    evaluated_list.sort(key=lambda x: x['score'], reverse=True)
    return evaluated_list

# 3. 이메일 발송
def send_email(evaluated_papers):
    print("--- [Step 3] 이메일 발송 중... ---")
    if not evaluated_papers:
        print("발송할 내용이 없습니다.")
        return

    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    content = "📚 오늘의 맞춤형 논문 리포트\n\n"
    for p in evaluated_papers:
        content += f"[{p['score']}점] {p['title']}\n"
        content += f"🔗 링크: {p['link']}\n"
        content += f"📝 요약: {p['summary']}\n"
        content += f"💡 추천 이유: {p['reason']}\n"
        content += "-"*30 + "\n"

    msg = MIMEText(content)
    msg['Subject'] = "🚀 Robotics & CV 최신 논문 리포트"
    msg['From'] = f"Research Bot <{sender}>"
    msg['To'] = receiver

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.send_message(msg)
    print("이메일 발송 성공!")

# ==========================================
# 실제 실행 부분 (이게 있어야 작동합니다!)
# ==========================================
if __name__ == "__main__":
    try:
        papers = fetch_papers()
        evaluated = evaluate_papers(papers)
        send_email(evaluated)
    except Exception as e:
        print(f"❌ 최종 실행 에러: {e}")