import os
import json
import re
import PyPDF2
import streamlit as st
import spacy
from openai import OpenAI
import base64
import pandas as pd

# Setup Groq-compatible OpenAI client
client = OpenAI(
    api_key="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",  # Replace with your actual Groq API key
    base_url="https://api.groq.com/openai/v1"
)

nlp = spacy.load("en_core_web_sm")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memo_generated" not in st.session_state:
    st.session_state.memo_generated = False
if "final_memo" not in st.session_state:
    st.session_state.final_memo = ""
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = ""
if "user_company_name" not in st.session_state:
    st.session_state.user_company_name = None

st.set_page_config(layout="wide")
st.title("📄 AI Investor Memo Generator (Groq + Web Simulation)")

if not st.session_state.user_company_name:
    with st.form("company_form"):
        st.markdown("### 🚀 Before we begin, tell us the startup's name:")
        entered_name = st.text_input("Company Name", placeholder="e.g., Zepto, Bluelearn, etc.", max_chars=100)
        submitted = st.form_submit_button("Start")
        if submitted and entered_name.strip():
            st.session_state.user_company_name = entered_name.strip()
            st.rerun()
    st.stop()

with st.sidebar:
    st.markdown("### 🏷️ Company Details")
    st.markdown(f"**Current Company:** `{st.session_state.user_company_name}`")
    if st.button("🔁 Change Company Name"):
        st.session_state.user_company_name = None
        st.session_state.memo_generated = False
        st.rerun()
    uploaded_file = st.file_uploader("Upload a Pitch Deck (PDF)", type="pdf")

def chat_with_groq(user_input):
    messages = st.session_state.chat_history + [{"role": "user", "content": user_input}]
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.5
        )
        reply = response.choices[0].message.content.strip()
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"[Chatbot Error: {str(e)}]"

def extract_text_by_page(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return [page.extract_text() for page in reader.pages if page.extract_text()]

def summarize_page_content(page_text, page_number):
    prompt = f"Summarize this pitch deck page {page_number} for investment analysis:\n\n{page_text}"
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq failed on page {page_number}: {str(e)}]"

def summarize_entire_deck(summary_text, company, founders):
    prompt = f"""
You are helping compress a pitch deck for a VC analyst. Retain important context from the top including:
- Company name: {company}
- Founders: {', '.join(founders)}
- Any funding, product, or traction details if mentioned.

Now, summarize the pitch content concisely (within 1500 words) preserving critical information for investor analysis.

--- Full Pitch Text ---
{summary_text}
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq summarization failed: {str(e)}]"

def analyze_entities(text):
    doc = nlp(text)
    return {
        "companies": list(set(ent.text.strip() for ent in doc.ents if ent.label_ == "ORG")),
        "people": list(set(ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON")),
    }

def groq_simulate_web_research(company, founders):
    people_str = ", ".join(founders)
    prompt = f"""
You are an advanced AI with access to the internet and VC databases.

Do an exhaustive online research for a startup called **{company}**, founded by {people_str}.
Collect and summarize publicly available information such as:
Map out direct and indirect competitors at various stages (Seed to Series E and beyond).
Compare product, funding, traction, and GTM strategies.
Assess market size (TAM/SAM), funding climate, and investor behavior.
Highlight differentiation or gaps in the startup’s positioning.
🔍 Research Scope:
🥊 1. Competitor Landscape
Identify 8–12 startups or companies in the same or adjacent space.
For each competitor, collect:
Name, HQ, founding year
Core product(s)/technology
Funding stage and total raised
Key investors
Notable clients or deployments
Recent strategic moves (launches, pivots, M&A)
📊 2. Comparison Matrix
Create a table comparing the target startup with key competitors across:
Product depth
what is the tech differentiation
Revenue traction
Funding raised
Stage and GTM approach
IP or defensibility
📈 3. Funding Trends
Summarize recent fundraising in this space (last 2–3 years).
Flag active VCs, corporate investors, or government grants.
Note valuation shifts or SPAC/IPO exits if relevant.
🌍 4. Market Size & Demand Signals
Estimate TAM/SAM using credible market reports, government data, or proxy benchmarks.
Include emerging trends, customer behaviors, or regulatory shifts.
Use tools like Crunchbase, Tracxn, Gartner, Reddit, job boards, and LinkedIn signals.
🔑 5. Strategic Insights
What are the whitespace opportunities?
What moat (if any) does the startup have?
How fast can incumbents replicate the startup’s playbook?
Where are competitors ahead or behind?

Provide the research output in JSON format under keys like:
- name, website, domain, location, founded_year, product_overview, news, social_links, funding_info, investor_names, media_mentions, awards, team_highlights

If anything is missing, say \"unknown\".
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq search simulation failed: {str(e)}]"

def generate_final_memo(condensed_summary, simulated_web_data):
    prompt = f"""
Act as a VC investment analyst.
Based on the pitch deck content and enriched web research below, write an exhaustive, structured investor memo with subpoints.

--- PITCH DECK (condensed summary) ---
{condensed_summary}

--- WEB RESEARCH (company, founders, product news, site metadata) ---
{simulated_web_data}

Your output should include:
1. Executive Summary
2. Company Overview
   - Name, founding year, location, website, domains of operation
3. Team & Founders
   - Backgrounds, strengths, possible gaps
4. Technology Breakdown (NEW)
   - What kind of technology is being used? (e.g. AI, ML, Blockchain)
   - Is this technology necessary and beneficial for the solution?
   - Could this work without tech? Why or why not?
   - Explain in layman's language using jugaad-style Indian analogies
   - Evaluate tech defensibility: is it hard to copy or replicate?
5. Product & IP
   - Product line, tech stack, innovation, and protectability
6. Market
   - TAM/SAM, ICP, timing
7. Traction
   - Deployments, revenues, partnerships
8. Financial Summary
   - Previous rounds, ask, use of funds
9. Unit Economics
   - Pricing model, margin, CAC, LTV
10. Competitive Landscape
   - Global + Indian peers with funding/stage comparisons
   - Table-style competitor comparison
11. Website & Public Presence
   - Meta info, credibility, brand impression
12. Strategic Concerns & Risks
13. Roadmap & Execution Readiness
14. Exit Potential (IPO, acquisition)
15. Final Recommendation
   - Analyst verdict, confidence score, VC thesis match
16.Executive Summary
17.Competitor Profiles (1-paragraph each + data table)
18.Competitive Matrix
19.Funding and Market Landscape
20.Key Takeaways / Strategic Notes
30.Recommendation (Proceed / Hold / Pass)

Also include:
- PR highlights, awards, endorsements
- Vision vs execution gaps
- Differentiators for investors
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq final memo failed: {str(e)}]"

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def build_summary_table():
    prompt = f"""
Act as a VC analyst.
Extract a detailed, in-depth structured summary from the following investor memo.
Provide extensive insights under each section for granular analysis.

**Important:** Your response must contain only the JSON array wrapped in ```json and ``` markers, with no additional text, explanations, or comments.

Output as a JSON array of objects with keys:
- Section
- Subsections (optional): Bullet-point highlights under the section
- Details: Clear, specific takeaways or descriptions
- Links (if any): A list of relevant external URLs mentioned in that section

Ensure that the JSON is properly formatted, with all strings correctly escaped, and no trailing commas or other syntax errors.

Example format:
```json
[
  {{
    "Section": "Market Overview",
    "Subsections": ["Target Segments", "Customer Pain Points"],
    "Details": "• TAM is estimated at $8B...\\n• Primarily focused on Gen Z urban consumers...",
    "Links": ["https://example.com/report"]
  }},
  ...
]
```

Investor Memo:
{st.session_state.final_memo}
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        content = response.choices[0].message.content.strip()
        # Extract the JSON block
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in response.")
        json_str = match.group(1)
        # Parse the JSON
        json_data = json.loads(json_str)
        return pd.DataFrame(json_data)
    except Exception as e:
        return pd.DataFrame([{
            "Section": "Error",
            "Details": str(e),
            "Raw Response": content
        }])

st.markdown("""
    <style>
    .css-1xarl3l, .css-1r6slb0, .css-1fcbfmj, .stDataFrame td {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    </style>
""", unsafe_allow_html=True)

if uploaded_file and not st.session_state.memo_generated:
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.uploaded_file_path = temp_path

    status_box = st.empty()

    status_box.info("📄 Extracting and summarizing PDF pages...")
    page_texts = extract_text_by_page(temp_path)
    combined_page_summaries = []

    for i, page in enumerate(page_texts):
        status_box.info(f"🧠 Summarizing page {i+1}...")
        summary = summarize_page_content(page[:3000], i + 1)
        combined_page_summaries.append(f"[Page {i+1}]\n{summary}")

    full_summary_text = "\n\n".join(combined_page_summaries)

    status_box.info("🔍 Extracting entities...")
    all_text = "\n".join(page_texts)
    entities = analyze_entities(all_text)
    company = st.session_state.user_company_name
    founders = entities["people"][:3]

    status_box.info("🧠 Condensing pitch content...")
    condensed_summary = summarize_entire_deck(full_summary_text, company, founders)

    status_box.info("🌐 Simulating web research...")
    simulated_web_data = groq_simulate_web_research(company, founders)

    status_box.info("📊 Generating investor memo...")
    final_memo = generate_final_memo(condensed_summary, simulated_web_data)

    st.session_state.memo_generated = True
    st.session_state.final_memo = final_memo
    status_box.empty()

if st.session_state.memo_generated:
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Memo", "📄 PDF Preview", "💬 Chat", "📋 Summary Table"])

    with tab1:
        st.subheader("📘 Final Investor Memo")
        st.markdown(st.session_state.final_memo)
        st.download_button("📥 Download Memo", st.session_state.final_memo, file_name="Investor_Memo.txt")

    with tab2:
        st.subheader("📄 Original Pitch Deck Preview")
        show_pdf(st.session_state.uploaded_file_path)

    with tab3:
        st.subheader("💬 VC Chat Assistant")
        user_input = st.text_input("Ask a question about the startup, market, team...", key="chat_input")
        if user_input and user_input.strip():
            response = chat_with_groq(user_input)
            st.markdown(f"**🧑‍💼 You:** {user_input}")
            st.markdown(f"**🤖 AI Analyst:** {response}")

        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💬 Chat History")
            for msg in st.session_state.chat_history:
                role = "🧑‍💼 You" if msg["role"] == "user" else "🤖 AI Analyst"
                st.markdown(f"**{role}:** {msg['content']}")

    with tab4:
        st.subheader("📋 In-Depth AI-Filled Executive Summary Table (with Links)")
        df = build_summary_table()
        if df.iloc[0]["Section"] == "Error":
            st.error("There was an error generating the summary table.")
            st.write("**Error Details:**", df.iloc[0]["Details"])
            st.write("**Raw API Response:**")
            st.code(df.iloc[0]["Raw Response"], language="json")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
