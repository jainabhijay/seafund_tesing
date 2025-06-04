import os
import re
import json
import requests
import PyPDF2
import spacy
from bs4 import BeautifulSoup
from datetime import datetime
import streamlit as st
from openai import OpenAI  # Groq-compatible OpenAI SDK

# Initialize Groq API client with hardcoded key (for testing only)
client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

PDF_PATH = "/Users/abhijayjain/Desktop/Seafund/Streamlit demo02/uploads/Executive Summary - iHub Robotics.pdf"

st.set_page_config(layout="wide")
st.title("📄 Pitch Deck + Investor Memo Analyzer")

MODEL = "en_core_web_sm"

try:
    nlp = spacy.load(MODEL)
except OSError:
    print(f"{MODEL} not found. Downloading now...")
    subprocess.run(["python", "-m", "spacy", "download", MODEL], check=True)
    nlp = spacy.load(MODEL)

def extract_pdf_content(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return analyze_text_content(text)

def analyze_text_content(text):
    doc = nlp(text)
    entities = {
        "companies": list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"])),
        "people": list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"])),
        "locations": list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "GPE"])),
        "dates": list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "DATE"])),
        "money": list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "MONEY"])),
        "raw_text": text
    }
    return entities

def search_google(query, max_results=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://www.google.com/search?q={query}"
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for g in soup.find_all('div', class_='tF2Cxc')[:max_results]:
            title = g.find('h3').text if g.find('h3') else 'No Title'
            link = g.find('a')['href']
            results.append({"title": title, "link": link})
        return results
    except Exception:
        return []

def fetch_google_news(query):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://news.google.com/search?q={query}"
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('a', class_='DY5T1d')
        return [{"title": a.text.strip(), "link": f"https://news.google.com{a['href'][1:]}"} for a in articles[:5]]
    except Exception:
        return []

def extract_website_from_text(text):
    urls = re.findall(r'(https?://[\w\.-]+)', text)
    for url in urls:
        if "linkedin.com" not in url:
            return url
    return None

def scrape_site_metadata(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return {
            "title": soup.title.string if soup.title else "",
            "meta_description": next((tag.get("content") for tag in soup.find_all("meta") if tag.get("name") == "description"), ""),
            "meta_keywords": next((tag.get("content") for tag in soup.find_all("meta") if tag.get("name") == "keywords"), ""),
        }
    except Exception:
        return {}

def web_enrichment(company_name, founders, raw_text):
    results = {"company_name": company_name}
    results["company_news"] = fetch_google_news(company_name)
    results["company_web"] = search_google(company_name)
    results["founders"] = {}
    for founder in founders:
        results["founders"][founder] = {
            "news": fetch_google_news(founder),
            "web": search_google(founder)
        }
    website = extract_website_from_text(raw_text)
    if website:
        results["website"] = website
        domain = website.split("//")[-1].split("/")[0]
        results["metadata"] = scrape_site_metadata(website)
    else:
        results["website"] = "Not Found"
        results["metadata"] = {}
    return results

def generate_structured_output(pdf_data, web_data):
    prompt = f"""
Act as a VC investment analyst.
Based on the pitch deck content and enriched web research below, write an exhaustive, structured investor memo with subpoints.

--- PITCH DECK ---
{pdf_data['raw_text']}

--- WEB RESEARCH (company, founders, product news, site metadata) ---
{json.dumps(web_data, indent=2)}

Your output should include:
1. Executive Summary
2. Company Overview
   - Name, founding year, location, website, domains of operation
3. Team & Founders
   - Backgrounds, strengths, possible gaps
4. Product & IP
   - Product line, technology stack, innovation, defensibility
5. Market
   - TAM/SAM, ICP, timing
6. Traction
   - Deployments, revenues, partnerships
7. Financial Summary
   - Previous rounds, ask, use of funds
8. Unit Economics
   - Pricing model, margin, CAC, LTV
9. Competitive Landscape
   - Global + Indian peers with funding and stage comparisons
10. Website & Public Presence
   - Meta info, brand perception, credibility
11. Strategic Concerns & Risks
12. Roadmap & Execution Readiness
13. Exit Potential (IPO, acquisition)
14. Final Recommendation with analyst verdict and confidence rating

Also highlight:
- Any interesting headlines or PR news from your search
- Any investor backers or partner signals
- Gaps or mismatches between vision and execution capability
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Groq AI failed: {str(e)}]"

# MAIN UI
with st.sidebar:
    st.header("Upload Pitch Deck PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    file_path = f"/tmp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    pdf_data = extract_pdf_content(file_path)
    company = pdf_data["companies"][0] if pdf_data["companies"] else "Unknown_Company"
    founders = pdf_data["people"][:3]
    web_data = web_enrichment(company, founders, pdf_data["raw_text"])
    report = generate_structured_output(pdf_data, web_data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📑 Extracted Pitch Deck Text")
        st.code(pdf_data['raw_text'][:5000])

    with col2:
        st.subheader("🧠 Investor Memo Output")
        st.write(report)

    st.success("✅ Memo generated.")
else:
    st.info("Please upload a pitch deck PDF to begin.")
