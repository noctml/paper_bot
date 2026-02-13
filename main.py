import feedparser
import requests
from openai import OpenAI
import os
import smtplib
from email.mime.text import MIMEText

# 1. arXiv에서 취향 저격 논문 수집
def fetch_papers():
    # Meta & MIT Spark Lab 스타일의 키워드 조합
    queries = [
        'cat:cs.RO AND ("SLAM" OR "Spatial AI" OR "Scene Graph")',
        'cat:cs.CV AND ("Embodied AI" OR "3D Reconstruction" OR "Multimodal")'
    ]
    
    all_entries = []
    for q in queries:
        url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=15&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        all_entries.extend(feed.entries)
    return all_entries

# 2. 전문적인 페르소나를 가진 GPT 평가
def evaluate_with_gpt(papers):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # 사용자 맞춤형 프롬프트
    system_prompt = """
    너는 MIT SPARK Lab과 Meta FAIR의 시니어 연구원이야. 
    다음 논문 초록을 읽고, 'Luca Carlone 스타일의 수치적 엄밀성'과 
    'Meta 스타일의 실용적 Embodied AI' 관점에서 중요도를 0~10점으로 평가해.
    반드시 JSON 형식으로 응답해: {"score": 9.5, "reason": "...", "summary": "..."}
    """
    
    selected_papers = []
    for p in papers[:10]: # 상위 10개만 검토 (비용 절감)
        user_content = f"Title: {p.title}\nSummary: {p.summary}"
        # ... (API 호출 로직) ...
        # 점수가 8점 이상인 것만 필터링하는 로직 추가 가능
