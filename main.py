import os
import feedparser
import smtplib
import urllib.parse
from openai import OpenAI
from email.mime.text import MIMEText

# 1. arXiv 논문 수집 (URL 인코딩 완벽 적용)
def fetch_papers():
    print("--- [Step 1] arXiv 논문 수집 중... ---")
    queries = [
        'cat:cs.RO AND ("SLAM" OR "Spatial AI" OR "Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction")'
    ]
    all_entries = []
    for q in queries:
        # 공백과 특수문자를 웹 주소용으로 변환 (핵심 해결책)
        encoded_q = urllib.parse.quote(q)
        url = f"http://export.arxiv.org/api/query?search_query={encoded_q}&max_results=5&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    print(f"총 {len(all_entries)}건의 논문 발견")
    return all_entries

# 2. OpenAI 평가
def evaluate_papers(papers):
    print("--- [Step 2] OpenAI 평가 시작 ---")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return []
        
    client = OpenAI(api_key=api_key)
    evaluated_list = []

    for p in papers[:5]:
        prompt = f"""
        너는 MIT SPARK Lab과 Meta FAIR의 시니어 연구원이야. 
        다음 논문 초록을 읽고 중요도를 0~10점으로 평가하고 한줄요약해줘.
        형식 - 점수: [점수], 이유: [추천이유], 요약: [한줄요약]

        Title: {p.title}
        Summary: {p.summary}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            analysis = response.choices[0].message.content
            evaluated_list.append({"title": p.title, "link": p.link, "analysis": analysis})
            print(f"✅ 평가 완료: {p.title[:20]}...")
        except Exception as e:
            print(f"❌ 평가 실패: {e}")
    return evaluated_list

# 3. 이메일 발송
def send_email(evaluated_papers):
    print("--- [Step 3] 이메일 발송 중... ---")
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("RECEIVER_EMAIL")

    if not evaluated_papers:
        print("⚠️ 발송할 데이터가 없습니다.")
        return

    content = "📚 오늘의 Robotics & CV 논문 리포트 (OpenAI)\n\n"
    for p in evaluated_papers:
        content += f"📌 {p['title']}\n🔗 {p['link']}\n{p['analysis']}\n"
        content += "-"*30 + "\n"

    msg = MIMEText(content)
    msg['Subject'] = "🚀 Robotics & CV 최신 논문 리포트"
    msg['From'] = f"Research Bot <{sender}>"
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("🎉 이메일 발송 성공!")
    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")

if __name__ == "__main__":
    try:
        papers = fetch_papers()
        evaluated = evaluate_papers(papers)
        send_email(evaluated)
    except Exception as e:
        print(f"❌ 최종 실행 에러: {e}")