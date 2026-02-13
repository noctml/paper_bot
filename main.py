import os
import feedparser
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText

# 1. arXiv 논문 수집
def fetch_papers():
    print("--- [Step 1] arXiv 논문 수집 중... ---")
    queries = [
        'cat:cs.RO AND ("SLAM" OR "Spatial AI" OR "Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction")'
    ]
    all_entries = []
    for q in queries:
        url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=5&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    return all_entries

# 2. Gemini로 논문 평가 (무료 버전)
def evaluate_papers(papers):
    print("--- [Step 2] Gemini 평가 시작 (무료 모드) ---")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    evaluated_list = []
    for p in papers[:5]:
        prompt = f"""
        너는 MIT SPARK Lab과 Meta FAIR의 시니어 연구원이야. 
        다음 논문 초록을 읽고 중요도를 0~10점으로 평가하고 한줄요약해줘.
        응답은 반드시 아래 형식을 지켜줘:
        점수: [점수]
        이유: [추천이유]
        요약: [한줄요약]

        Title: {p.title}
        Summary: {p.summary}
        """
        try:
            response = model.generate_content(prompt)
            evaluated_list.append({"title": p.title, "link": p.link, "analysis": response.text})
            print(f"평가 완료: {p.title[:30]}...")
        except Exception as e:
            print(f"평가 실패: {e}")
    return evaluated_list

# 3. 이메일 발송
def send_email(evaluated_papers):
    print("--- [Step 3] 이메일 발송 중... ---")
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    content = "📚 오늘의 Gemini 맞춤형 논문 리포트\n\n"
    for p in evaluated_papers:
        content += f"📌 {p['title']}\n🔗 {p['link']}\n{p['analysis']}\n"
        content += "-"*30 + "\n"

    msg = MIMEText(content)
    msg['Subject'] = "🚀 Robotics & CV 최신 논문 (Gemini Bot)"
    msg['From'] = f"Gemini Bot <{sender}>"
    msg['To'] = receiver

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.send_message(msg)
    print("이메일 발송 성공!")

if __name__ == "__main__":
    papers = fetch_papers()
    evaluated = evaluate_papers(papers)
    send_email(evaluated)