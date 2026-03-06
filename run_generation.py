#!/usr/bin/env python3
"""
Standalone document generation runner.
Executes the same logic as notebook 02 but without nbconvert cell timeouts.
Skips applicants whose documents already exist.

Usage:
    python run_generation.py              # uses .env settings
    SAMPLE_SIZE=50 python run_generation.py  # override via env
"""

import os, sys, json, time, random, hashlib, base64, re, io, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv('.env')

SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "10"))
SAMPLE_START = int(os.environ.get("SAMPLE_START", "0"))

ALLOWED_MODELS = {"gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview"}
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-image-preview")
assert MODEL_NAME in ALLOWED_MODELS, f"GEMINI_MODEL must be one of {ALLOWED_MODELS}"

from google import genai
LLM_CLIENT = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

print(f"Model: {MODEL_NAME}")
print(f"SAMPLE_START={SAMPLE_START}, SAMPLE_SIZE={SAMPLE_SIZE}")

# ── Retry helper ──────────────────────────────────────────────────────────────

def retry_with_backoff(func, max_retries=5, initial_delay=1.0, multiplier=2.0):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"    Attempt {attempt+1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= multiplier

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_PATH = 'unstructured_data/generation_log.jsonl'

def log_generation(sk_id, doc_type, pt, rt, gen_time, success):
    entry = {
        'SK_ID_CURR': int(sk_id), 'doc_type': doc_type, 'model': MODEL_NAME,
        'prompt_tokens': pt, 'response_tokens': rt,
        'generation_time_s': round(gen_time, 2), 'success': success
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')

# ── LLM helpers ───────────────────────────────────────────────────────────────

def generate_text_via_llm(prompt):
    def _call():
        response = LLM_CLIENT.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = response.text
        pt = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        rt = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        return text, pt, rt
    return retry_with_backoff(_call)

def generate_image_via_llm(prompt):
    def _call():
        response = LLM_CLIENT.models.generate_content(
            model=MODEL_NAME, contents=prompt,
            config={"response_modalities": ["IMAGE", "TEXT"]},
        )
        if not response.candidates or not response.candidates[0].content:
            raise ValueError("Empty response from API")
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                pt = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                rt = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
                return part.inline_data.data, pt, rt
        raise ValueError("No image in response")
    return retry_with_backoff(_call)

# ── Logo / seal helpers ──────────────────────────────────────────────────────

def _make_logo_b64(text, bg_color, text_color='white', width=420, height=70):
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) / 2, (height - th) / 2), text, fill=text_color, font=font)
    buf = io.BytesIO(); img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def _make_seal_b64(text, ring_color='#2F4F4F', width=120, height=120):
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    r, g, b = int(ring_color[1:3], 16), int(ring_color[3:5], 16), int(ring_color[5:7], 16)
    draw.ellipse([4, 4, width-4, height-4], outline=(r, g, b), width=4)
    draw.ellipse([12, 12, width-12, height-12], outline=(r, g, b), width=2)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = ImageFont.load_default()
    lines = text.split('\n')
    y_start = (height - len(lines) * 14) // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((width - tw) / 2, y_start + i * 14), line, fill=(r, g, b), font=font)
    buf = io.BytesIO(); img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

BANK_LOGO_B64 = {
    'JPMorgan Chase': _make_logo_b64('JPMORGAN CHASE & CO.', '#003087'),
    'Bank of America': _make_logo_b64('BANK OF AMERICA', '#012169'),
    'Wells Fargo': _make_logo_b64('WELLS FARGO', '#D71E28'),
    'Citibank': _make_logo_b64('CITIBANK', '#003B70'),
    'U.S. Bank': _make_logo_b64('U.S. BANK', '#0C2340'),
}
BUREAU_LOGO_B64 = {
    'Experian': _make_logo_b64('EXPERIAN', '#1D4F91'),
    'TransUnion': _make_logo_b64('TransUnion', '#00857C'),
    'Equifax': _make_logo_b64('EQUIFAX', '#C41230'),
}
COUNTY_SEAL_B64 = _make_seal_b64('COUNTY\nASSESSOR\nOFFICIAL\nSEAL')

# ── HTML / PDF helpers ────────────────────────────────────────────────────────

def _inject_logo_into_html(html, logo_b64, alt_text='Logo'):
    logo_tag = f'<div style="text-align:center;margin-bottom:10px;"><img src="data:image/png;base64,{logo_b64}" alt="{alt_text}" style="max-width:400px;height:auto;" /></div>'
    if '<body' in html.lower():
        idx = html.lower().find('<body'); close = html.find('>', idx) + 1
        html = html[:close] + logo_tag + html[close:]
    else:
        html = logo_tag + html
    return html

def _inject_seal_into_html(html, seal_b64, alt_text='Official Seal'):
    seal_tag = f'<div style="text-align:center;margin-bottom:10px;"><img src="data:image/png;base64,{seal_b64}" alt="{alt_text}" style="width:100px;height:100px;" /></div>'
    if '<body' in html.lower():
        idx = html.lower().find('<body'); close = html.find('>', idx) + 1
        html = html[:close] + seal_tag + html[close:]
    else:
        html = seal_tag + html
    return html

def _sanitize_html_for_xhtml2pdf(html):
    html = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1-\2', html)
    html = re.sub(r'var\(--[^)]*\)', '#333333', html)
    html = re.sub(r'calc\([^)]*\)', '50%', html)
    html = re.sub(r'@media[^{]*\{[^}]*\}', '', html)
    for prop in ['box-shadow','text-shadow','backdrop-filter','transition','transform']:
        html = re.sub(rf'{prop}\s*:[^;]+;', '', html)
    html = re.sub(r'animation[^:]*:[^;]+;', '', html)
    html = re.sub(r'display\s*:\s*flex[^;]*;', '', html)
    html = re.sub(r'display\s*:\s*grid[^;]*;', '', html)
    for prop in ['gap','justify-content','align-items']:
        html = re.sub(rf'{prop}\s*:[^;]+;', '', html)
    html = re.sub(r'flex[^:]*:[^;]+;', '', html)
    html = re.sub(r'grid[^:]*:[^;]+;', '', html)
    html = re.sub(r'letter-spacing\s*:[^;]+;', '', html)
    html = re.sub(r'overflow\s*:[^;]+;', '', html)
    html = re.sub(r'white-space\s*:\s*nowrap[^;]*;', '', html)
    html = re.sub(r'position\s*:\s*(?:sticky|fixed)[^;]*;', '', html)
    return html

def html_to_pdf(html_content, output_path, logo_b64=None, seal_b64=None, logo_alt='Logo'):
    if logo_b64:
        html_content = _inject_logo_into_html(html_content, logo_b64, logo_alt)
    if seal_b64:
        html_content = _inject_seal_into_html(html_content, seal_b64)
    html_content = _sanitize_html_for_xhtml2pdf(html_content)
    buf = io.BytesIO()
    from xhtml2pdf import pisa
    status = pisa.CreatePDF(html_content, dest=buf)
    if status.err:
        raise RuntimeError(f"xhtml2pdf conversion failed with {status.err} errors")
    with open(output_path, 'wb') as f:
        f.write(buf.getvalue())

def strip_markdown_fences(text):
    text = text.strip()
    if text.startswith('```'): text = text.split('\n', 1)[1]
    if text.endswith('```'): text = text.rsplit('```', 1)[0]
    return text.strip()

# ── Persona generator ────────────────────────────────────────────────────────

MALE_FIRST = ['James','Robert','Michael','William','David','Richard','Joseph',
              'Thomas','Charles','Christopher','Daniel','Matthew','Anthony',
              'Mark','Steven','Paul','Andrew','Joshua','Kenneth','Kevin']
FEMALE_FIRST = ['Mary','Patricia','Jennifer','Linda','Barbara','Elizabeth',
                'Susan','Jessica','Sarah','Karen','Lisa','Nancy','Betty',
                'Margaret','Sandra','Ashley','Dorothy','Kimberly','Emily','Donna']
LAST_NAMES = ['Smith','Johnson','Williams','Brown','Jones','Garcia','Miller',
              'Davis','Rodriguez','Martinez','Hernandez','Lopez','Wilson',
              'Anderson','Thomas','Taylor','Moore','Jackson','Martin','Lee',
              'Thompson','White','Harris','Clark','Lewis','Robinson','Walker',
              'Young','Allen','King']
STREETS = ['Oak St','Maple Ave','Cedar Ln','Pine Dr','Elm Blvd','Walnut Rd',
           'Birch Way','Willow Ct','Spruce Pl','Ash Ter','Main St','Park Ave',
           'Lake Dr','River Rd','Hill St','Meadow Ln','Forest Ave','Valley Rd']
CITIES_STATES = [
    ('Springfield','IL','62701'),('Columbus','OH','43215'),('Portland','OR','97201'),
    ('Charlotte','NC','28202'),('Austin','TX','78701'),('Denver','CO','80202'),
    ('Nashville','TN','37201'),('Raleigh','NC','27601'),('Tampa','FL','33602'),
    ('Phoenix','AZ','85001'),('San Diego','CA','92101'),('Minneapolis','MN','55401'),
    ('Atlanta','GA','30301'),('Kansas City','MO','64101'),('Pittsburgh','PA','15201'),
]
EMPLOYER_MAP = {
    'Business Entity Type 3': ['Vertex Solutions Inc.','CoreAxis LLC','Summit Partners Group'],
    'Business Entity Type 2': ['Pinnacle Corp','NexGen Industries','Atlas Holdings LLC'],
    'Business Entity Type 1': ['Omega Enterprises Inc.','Prime Holdings Group','Delta Staffing Services'],
    'Self-employed': ['Independent Contractor'],
    'Government': ['City of Springfield','County Administrative Office','State Department of Labor'],
    'School': ['Central Valley Unified School District','State University of New York','National Academy of Sciences'],
    'Medicine': ['Mercy General Hospital','Kaiser Permanente','HealthFirst Medical Group'],
    'Construction': ['Turner Construction Co.','Bechtel Group Inc.'],
    'Trade: type 7': ['Global Trade Partners LLC','East Coast Trading Co.'],
}
BANK_NAMES = ['JPMorgan Chase','Bank of America','Wells Fargo','Citibank','U.S. Bank']

def generate_persona(row):
    sk_id = int(row['SK_ID_CURR'])
    rng = random.Random(sk_id)
    gender = row.get('CODE_GENDER', 'M')
    if pd.isna(gender): gender = 'M'
    first = rng.choice(MALE_FIRST if gender == 'M' else FEMALE_FIRST)
    last = rng.choice(LAST_NAMES)
    street_num = rng.randint(100, 9999)
    street = rng.choice(STREETS)
    city, state, zipcode = rng.choice(CITIES_STATES)
    days_birth = abs(row.get('DAYS_BIRTH', -10000))
    if pd.isna(days_birth) or days_birth > 36500: days_birth = 10000
    dob = pd.Timestamp.now() - pd.Timedelta(days=days_birth)
    days_emp = abs(row.get('DAYS_EMPLOYED', -365))
    if pd.isna(days_emp) or days_emp > 36500: days_emp = 365
    hire_date = pd.Timestamp.now() - pd.Timedelta(days=days_emp)
    days_id = abs(row.get('DAYS_ID_PUBLISH', -1000))
    if pd.isna(days_id) or days_id > 36500: days_id = 1000
    id_issue_date = pd.Timestamp.now() - pd.Timedelta(days=days_id)
    org_type = str(row.get('ORGANIZATION_TYPE', 'General'))
    employer = rng.choice(EMPLOYER_MAP.get(org_type, [f'{org_type} Corp.']))
    ssn_hash = int(hashlib.md5(str(sk_id).encode()).hexdigest()[:8], 16)
    ssn_last4 = f"{ssn_hash % 10000:04d}"
    acct_num = f"{rng.randint(1000,9999)}-{rng.randint(100000,999999)}"
    routing = f"0{rng.randint(10000000,99999999)}"
    phone = f"({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"
    email = f"{first.lower()}.{last.lower()}{rng.randint(1,99)}@email.com"
    bank_name = rng.choice(BANK_NAMES)
    bureau_name = rng.choice(['Experian','TransUnion','Equifax'])
    return {
        'full_name': f"{first} {last}", 'first_name': first, 'last_name': last,
        'gender': gender, 'gender_word': 'male' if gender == 'M' else 'female',
        'dob': dob.strftime('%Y-%m-%d'), 'age': int(days_birth / 365),
        'address_line': f"{street_num} {street}", 'city': city, 'state': state, 'zipcode': zipcode,
        'full_address': f"{street_num} {street}, {city}, {state} {zipcode}",
        'phone': phone, 'email': email, 'ssn_last4': ssn_last4,
        'employer': employer, 'org_type': org_type,
        'hire_date': hire_date.strftime('%Y-%m-%d'), 'years_employed': round(days_emp / 365, 1),
        'id_issue_date': id_issue_date.strftime('%Y-%m-%d'),
        'acct_num': acct_num, 'routing_num': routing,
        'bank_name': bank_name, 'bureau_name': bureau_name, 'sk_id': sk_id,
    }

# ── Prompt templates (imported from notebook) ────────────────────────────────

PROMPT_TEMPLATES = {

'paystub': """Generate a realistic paystub image for an employee. The paystub should look like a printed/scanned payroll document. Render it as a single complete PNG image.

{style_instruction}

Employee: {full_name}
SSN (last 4): ***-**-{ssn_last4}
Address: {full_address}
Employer: {employer} | Industry: {org_type} | Pay frequency: Biweekly
Pay date: 2024-01-15 | Pay period: 2024-01-01 to 2024-01-14
Gross pay: ${gross_pay} | Federal tax: ${federal_tax} | State tax: ${state_tax}
Health insurance: ${insurance} | 401k/Retirement: ${retirement}
Net pay: ${net_pay} | Hire date: {hire_date}
YTD Gross: ${ytd_gross} | YTD Net: ${ytd_net}

Stability level: {stability:.2f} (0=low, 1=high). Apply document quality accordingly.

CRITICAL IMAGE REQUIREMENTS:
- The image MUST be fully self-contained with no content cut off at any edge.
- Leave adequate padding/margins on all four sides.
- Render the complete document in a single frame.
- DO NOT crop any text, UI elements, or borders.

Return ONLY the image. No commentary.""",

'linkedin': """Generate a realistic LinkedIn profile card image showing a complete, uncropped profile.

{style_instruction}

Name: {full_name} | Gender: {gender_word} | Age: {age}
Headline: {occupation} at {employer}
Location: {city}, {state}
Industry: {org_type} | Income type: {income_type}
Years employed: {years_employed} | Education: {education}
Connections: {connections}

Stability score: {stability:.2f} (0=low, 1=high)

Include these sections in the card:
- A realistic, photographic profile picture of a {gender_word} person aged approximately {age} years old — this MUST be a realistic human face photo, NOT a silhouette, avatar, icon, or placeholder. Generate an actual face.
- Name, headline, location
- About summary (2-3 sentences)
- Experience (list {num_jobs} positions with dates)
- Education
- Skills ({skill_count} skills listed)

CRITICAL IMAGE REQUIREMENTS:
- The profile photo MUST be a realistic human face — NOT a gray silhouette or placeholder avatar.
- The image MUST be fully self-contained with no content cut off at any edge.
- Leave generous padding/margins on all four sides (at least 20px).
- Render the COMPLETE profile card in a single frame.
- DO NOT crop any text, sections, or UI elements.

Return ONLY the image. No commentary.""",

'id_document': """Generate a realistic scanned image of an American driver's license or state ID card.

{style_instruction}

Holder: {full_name}
Date of birth: {dob}
Gender: {gender_word}
Address: {full_address}
Issue date: {id_issue_date}
Marital status: {family_status} | Children: {children} | Family members: {fam_members}

Include typical US ID features: state seal, ID number, class, expiration date (4 years from issue), height/weight, eye color.

PHOTO REQUIREMENT: The ID card MUST include a realistic, photographic headshot of a {gender_word} person aged approximately {age} years old. This MUST be a realistic human face photograph — NOT a silhouette, avatar, icon, or gray placeholder. Generate an actual face photo in the ID photo area.

CRITICAL IMAGE REQUIREMENTS:
- The ID photo area MUST contain a realistic human face, not a placeholder.
- The image MUST be fully self-contained with no content cut off at any edge.
- Leave adequate padding/margins on all four sides.
- Render the complete ID card in a single frame.
- DO NOT crop any text, borders, or card edges.

Return ONLY the image. No commentary.""",

'bank_statement': """Generate complete, well-structured HTML markup for a realistic 3-month bank statement from {bank_name}.

{style_instruction}

Use {bank_name} brand colors (as direct hex values like #003087, NOT CSS variables).

IMPORTANT CSS RULES: Use ONLY simple inline CSS or a <style> block with direct hex color values. Do NOT use CSS variables (var(--...)), flexbox, grid, calc(), box-shadow, or any modern CSS features. Use only tables for layout.

Account holder: {full_name}
Address: {full_address}
Account number: {acct_num} | Routing: {routing_num}
Statement period: November 1, 2024 — January 31, 2025

Monthly transactions:
- Regular salary/direct-deposit credits of ${monthly_income:.2f} per month from {employer}
- Regular loan repayment debits of ${annuity:.2f} per month
- {overdue_text}
- Typical household debits (utilities, groceries, subscriptions)

Summary:
- Total credit from other sources: ${credit_sum:.2f}
- Outstanding debt: ${credit_debt:.2f}

Include: {bank_name} header with address ({city}, {state}), daily balance summary table, FDIC insured footer, page number.

Return ONLY the HTML. Do not include markdown fences.""",

'property_doc': """Generate complete, well-structured HTML for a realistic US property tax assessment notice or utility bill.

{style_instruction}

IMPORTANT CSS RULES: Use ONLY simple inline CSS or a <style> block with direct hex color values. Do NOT use CSS variables, flexbox, grid, calc(), box-shadow, or modern CSS. Use only tables for layout.

Addressee: {full_name}
Property address: {full_address}
Mailing address: {full_address}

Property details:
- Housing type: {housing_type}
- Ownership status: {own_realty_text}
- Vehicle registered: {own_car_text} (age {car_age} years)
- Building floors: {floors}
- Living area: {living_area:.1f} sq ft | Total area: {total_area:.1f} sq ft
- Property region rating: {region}/3
- Registration/living city mismatch: {reg_live_mismatch} | Living/work city mismatch: {live_work_mismatch}

Include: issuing authority header (county assessor's office or utility company in {city}, {state}), assessment date, account number, billing period, amount due, due date, payment instructions, footer with office hours and contact info.

Return ONLY the HTML. Do not include markdown fences.""",

'credit_report': """Generate complete, well-structured HTML for a professional US consumer credit bureau report.

{style_instruction}

IMPORTANT CSS RULES: Use ONLY simple inline CSS or a <style> block with direct hex color values. Do NOT use CSS variables, flexbox, grid, calc(), box-shadow, or modern CSS. Use only tables for layout.

Consumer: {full_name}
SSN (last 4): ***-**-{ssn_last4}
Address: {full_address}
Report date: January 31, 2025
Report number: {report_number}

Credit scores:
- Score 1: {score_1} | Score 2: {score_2} | Score 3: {score_3} (out of 850)
- Risk tier: {risk_tier}

Inquiry history:
- Last hour: {inq_hour} | Last day: {inq_day} | Last week: {inq_week}
- Last month: {inq_month} | Last quarter: {inq_quarter} | Last year: {inq_year}

Include: bureau logo header, personal info, score summary, tradeline summary (2-3 accounts), public records section, inquiry details, disclaimer footer.

Return ONLY the HTML. Do not include markdown fences.""",
}

STYLE_VARIANTS = {
'paystub': [
    "Clean modern design with a blue-and-white color scheme, sans-serif fonts, and a company logo in the top-left corner.",
    "Traditional payroll format with a gray header bar, tabular deductions layout, and monospace font for numbers.",
    "Minimalist design with a green accent, employee info on the left, earnings/deductions on the right.",
    "Corporate style with a dark navy header, gold accents, centered company name, and boxed sections.",
],
'linkedin': [
    "Standard LinkedIn white-card layout with blue accents, a circular profile photo, and clean section dividers.",
    "LinkedIn mobile-app style card with rounded corners, compact sections, and a banner image placeholder at the top.",
    "LinkedIn premium-style card with a gold 'Premium' badge, dark background header section.",
    "LinkedIn classic desktop view with a wide banner, left-aligned photo, and detailed experience timeline.",
],
'id_document': [
    "US driver's license style with a blue gradient header, state seal watermark, and horizontal layout.",
    "State ID card style with a green accent bar, vertical layout, and prominent photo on the left.",
    "Modern Real ID compliant design with a gold star indicator and machine-readable zone.",
    "Classic DMV-issued card with a plain white background and standard field layout.",
],
'bank_statement': [
    "Clean modern design with a blue accent bar header, sans-serif typography, and alternating row colors.",
    "Traditional columnar layout with serif fonts, double-ruled lines between sections, and formal letterhead.",
    "Minimalist design with a sidebar account summary and large statement period header.",
    "Premium banking style with a dark header, gold accents, and wealth management branding.",
],
'property_doc': [
    "Official county assessor notice with a government seal header, formal table, and legal disclaimer footer.",
    "Utility bill format with a colorful company header and usage bar chart in CSS.",
    "Property tax statement with a clean municipal letterhead and assessed vs market value comparison table.",
    "Modern digital property report with a sidebar property summary and neighborhood rating indicators.",
],
'credit_report': [
    "Experian-style layout with a blue gradient header, circular score gauge, and color-coded tradeline status.",
    "TransUnion-style report with teal accent, horizontal score bar, and tabular account summaries.",
    "Equifax-style format with red-and-gray color scheme, score range visualization, and payment history grid.",
    "Generic bureau report with clean black-and-white design, large score number, and summary statistics boxes.",
],
}

# ── Fallback templates ────────────────────────────────────────────────────────

FALLBACK_BANK_STMT = """<html><head><style>
body {{ font-family: Arial, sans-serif; margin: 30px; color: #333; }}
h1 {{ color: {header_color}; font-size: 22px; border-bottom: 3px solid {header_color}; padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
th {{ background-color: {header_color}; color: white; padding: 8px; text-align: left; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #ddd; }}
.footer {{ margin-top: 30px; font-size: 10px; color: #666; border-top: 1px solid #ccc; padding-top: 10px; }}
</style></head><body>
<h1>{bank_name}</h1>
<p><strong>Account Holder:</strong> {full_name}<br>
<strong>Address:</strong> {full_address}<br>
<strong>Account:</strong> {acct_num} &nbsp; <strong>Routing:</strong> {routing_num}</p>
<p><strong>Statement Period:</strong> November 1, 2024 — January 31, 2025</p>
<table><tr><th>Date</th><th>Description</th><th>Credit</th><th>Debit</th><th>Balance</th></tr>
<tr><td>11/01</td><td>Direct Deposit - {employer}</td><td>${monthly_income:.2f}</td><td></td><td>${monthly_income:.2f}</td></tr>
<tr><td>11/05</td><td>Loan Payment</td><td></td><td>${annuity:.2f}</td><td>${balance_1:.2f}</td></tr>
<tr><td>12/01</td><td>Direct Deposit - {employer}</td><td>${monthly_income:.2f}</td><td></td><td>${balance_3:.2f}</td></tr>
<tr><td>12/05</td><td>Loan Payment</td><td></td><td>${annuity:.2f}</td><td>${balance_4:.2f}</td></tr>
<tr><td>01/02</td><td>Direct Deposit - {employer}</td><td>${monthly_income:.2f}</td><td></td><td>${balance_5:.2f}</td></tr>
<tr><td>01/05</td><td>Loan Payment</td><td></td><td>${annuity:.2f}</td><td>${balance_6:.2f}</td></tr>
</table>
<div class="footer">Member FDIC. Equal Housing Lender. {bank_name}, {city}, {state}.</div>
</body></html>"""

FALLBACK_PROPERTY = """<html><head><style>
body {{ font-family: Arial, sans-serif; margin: 30px; color: #333; }}
h1 {{ color: #2F4F4F; font-size: 20px; text-align: center; }}
h2 {{ color: #2F4F4F; font-size: 16px; border-bottom: 2px solid #2F4F4F; padding-bottom: 5px; }}
table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #ddd; }}
.footer {{ margin-top: 30px; font-size: 10px; color: #666; text-align: center; }}
</style></head><body>
<h1>{city} County Assessor's Office</h1>
<p style="text-align:center;">Property Tax Assessment Notice</p>
<h2>Property Information</h2>
<table>
<tr><td><strong>Owner:</strong></td><td>{full_name}</td></tr>
<tr><td><strong>Property Address:</strong></td><td>{full_address}</td></tr>
<tr><td><strong>Housing Type:</strong></td><td>{housing_type}</td></tr>
<tr><td><strong>Living Area:</strong></td><td>{living_area:.1f} sq ft</td></tr>
<tr><td><strong>Total Area:</strong></td><td>{total_area:.1f} sq ft</td></tr>
</table>
<h2>Assessment</h2>
<p>Assessment Date: January 15, 2025<br>Amount Due: $1,250.00<br>Due Date: March 31, 2025</p>
<div class="footer">Office Hours: Mon-Fri 8AM-5PM | {city}, {state}</div>
</body></html>"""

FALLBACK_CREDIT = """<html><head><style>
body {{ font-family: Arial, sans-serif; margin: 30px; color: #333; }}
h1 {{ color: {header_color}; font-size: 22px; }}
h2 {{ color: {header_color}; font-size: 16px; border-bottom: 2px solid {header_color}; padding-bottom: 5px; }}
table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
th {{ background-color: {header_color}; color: white; padding: 8px; text-align: left; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #ddd; }}
.footer {{ margin-top: 30px; font-size: 9px; color: #999; border-top: 1px solid #ccc; padding-top: 10px; }}
</style></head><body>
<h1>{bureau_name} Credit Report</h1>
<p><strong>Consumer:</strong> {full_name}<br>
<strong>SSN:</strong> ***-**-{ssn_last4}<br>
<strong>Address:</strong> {full_address}<br>
<strong>Report Date:</strong> January 31, 2025 &nbsp; <strong>Report #:</strong> {report_number}</p>
<h2>Credit Scores</h2>
<p>Score 1: {score_1} | Score 2: {score_2} | Score 3: {score_3}</p>
<p><strong>Risk Tier:</strong> {risk_tier}</p>
<h2>Inquiry Summary</h2>
<table><tr><th>Period</th><th>Count</th></tr>
<tr><td>Last Hour</td><td>{inq_hour}</td></tr><tr><td>Last Day</td><td>{inq_day}</td></tr>
<tr><td>Last Week</td><td>{inq_week}</td></tr><tr><td>Last Month</td><td>{inq_month}</td></tr>
<tr><td>Last Quarter</td><td>{inq_quarter}</td></tr><tr><td>Last Year</td><td>{inq_year}</td></tr>
</table>
<div class="footer">This report is furnished per the Fair Credit Reporting Act. {bureau_name}.</div>
</body></html>"""

BANK_HEADER_COLORS = {
    'JPMorgan Chase': '#003087', 'Bank of America': '#012169',
    'Wells Fargo': '#D71E28', 'Citibank': '#003B70', 'U.S. Bank': '#0C2340',
}
BUREAU_HEADER_COLORS = {
    'Experian': '#1D4F91', 'TransUnion': '#00857C', 'Equifax': '#C41230',
}

# ── Document generators ──────────────────────────────────────────────────────

def _safe(val, default):
    return default if pd.isna(val) else val

def normalize_series(s):
    s_min, s_max = s.min(), s.max()
    if s_max == s_min: return pd.Series(0.5, index=s.index)
    return (s - s_min) / (s_max - s_min)

def compute_stability_scores(df):
    ext2 = normalize_series(df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median()))
    days_emp = normalize_series(-df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median()))
    income = normalize_series(df['AMT_INCOME_TOTAL'].fillna(df['AMT_INCOME_TOTAL'].median()))
    realty = df['FLAG_OWN_REALTY'].map({'Y':1,'N':0}).fillna(0) if df['FLAG_OWN_REALTY'].dtype == 'object' else df['FLAG_OWN_REALTY'].fillna(0)
    return (ext2 * 0.4 + days_emp * 0.3 + income * 0.2 + realty * 0.1).clip(0, 1)


def generate_paystub(row, persona):
    sk_id = persona['sk_id']
    rng = random.Random(sk_id + 1)
    stability = row['stability_score']
    income = _safe(row.get('AMT_INCOME_TOTAL', 50000), 50000)
    gross_pay = round((income / 24) * (1 + rng.uniform(-0.05, 0.05)), 2)
    if income < 100000: net_ratio = rng.uniform(0.80, 0.85)
    elif income < 250000: net_ratio = rng.uniform(0.70, 0.80)
    else: net_ratio = rng.uniform(0.60, 0.75)
    net_pay = round(gross_pay * net_ratio, 2)
    tax_gap = gross_pay - net_pay
    federal_tax = round(tax_gap * 0.6, 2)
    state_tax = round(tax_gap * 0.25, 2)
    fam = _safe(row.get('CNT_FAM_MEMBERS', 1), 1)
    insurance = round(50 + fam * 30, 2)
    retirement = round(gross_pay * 0.05, 2) if stability > 0.5 else 0
    style_instruction = rng.choice(STYLE_VARIANTS['paystub'])
    prompt = PROMPT_TEMPLATES['paystub'].format(
        **persona, style_instruction=style_instruction,
        gross_pay=gross_pay, net_pay=net_pay,
        federal_tax=federal_tax, state_tax=state_tax,
        insurance=insurance, retirement=retirement,
        ytd_gross=round(gross_pay*2,2), ytd_net=round(net_pay*2,2), stability=stability,
    )
    start = time.time()
    try:
        img_bytes, pt, rt = generate_image_via_llm(prompt)
        with open(f'unstructured_data/paystubs/{sk_id}_paystub.png', 'wb') as f: f.write(img_bytes)
        log_generation(sk_id, 'paystub', pt, rt, time.time()-start, True); return True
    except Exception as e:
        log_generation(sk_id, 'paystub', 0, 0, time.time()-start, False)
        print(f"    ERROR paystub {sk_id}: {e}"); return False


def generate_linkedin(row, persona):
    sk_id = persona['sk_id']
    rng = random.Random(sk_id + 2)
    stability = row['stability_score']
    occ = _safe(row.get('OCCUPATION_TYPE','Professional'),'Professional')
    income_type = _safe(row.get('NAME_INCOME_TYPE','Working'),'Working')
    edu = _safe(row.get('NAME_EDUCATION_TYPE','Higher education'),'Higher education')
    if stability > 0.7: num_jobs, skill_count, connections = rng.randint(1,2), rng.randint(8,15), rng.randint(400,500)
    elif stability > 0.4: num_jobs, skill_count, connections = rng.randint(2,4), rng.randint(5,10), rng.randint(200,400)
    else: num_jobs, skill_count, connections = rng.randint(4,6), rng.randint(2,5), rng.randint(50,150)
    style_instruction = rng.choice(STYLE_VARIANTS['linkedin'])
    prompt = PROMPT_TEMPLATES['linkedin'].format(
        **persona, style_instruction=style_instruction,
        occupation=occ, income_type=income_type, education=edu,
        stability=stability, connections=connections, num_jobs=num_jobs, skill_count=skill_count,
    )
    start = time.time()
    try:
        img_bytes, pt, rt = generate_image_via_llm(prompt)
        with open(f'unstructured_data/linkedin/{sk_id}_linkedin.png', 'wb') as f: f.write(img_bytes)
        log_generation(sk_id, 'linkedin', pt, rt, time.time()-start, True); return True
    except Exception as e:
        log_generation(sk_id, 'linkedin', 0, 0, time.time()-start, False)
        print(f"    ERROR linkedin {sk_id}: {e}"); return False


def generate_id_document(row, persona):
    sk_id = persona['sk_id']
    rng = random.Random(sk_id + 3)
    family_status = _safe(row.get('NAME_FAMILY_STATUS','Single / not married'),'Single / not married')
    children = int(_safe(row.get('CNT_CHILDREN',0),0))
    fam_members = int(_safe(row.get('CNT_FAM_MEMBERS',1),1))
    style_instruction = rng.choice(STYLE_VARIANTS['id_document'])
    prompt = PROMPT_TEMPLATES['id_document'].format(
        **persona, style_instruction=style_instruction,
        family_status=family_status, children=children, fam_members=fam_members,
    )
    start = time.time()
    try:
        img_bytes, pt, rt = generate_image_via_llm(prompt)
        with open(f'unstructured_data/id_documents/{sk_id}_id.png', 'wb') as f: f.write(img_bytes)
        log_generation(sk_id, 'id_document', pt, rt, time.time()-start, True); return True
    except Exception as e:
        log_generation(sk_id, 'id_document', 0, 0, time.time()-start, False)
        print(f"    ERROR id_document {sk_id}: {e}"); return False


def generate_bank_statement(row, persona):
    sk_id = persona['sk_id']
    rng = random.Random(sk_id + 4)
    income = _safe(row.get('AMT_INCOME_TOTAL',50000),50000)
    monthly_income = income / 12
    annuity = _safe(row.get('AMT_ANNUITY',0),0)
    overdue = _safe(row.get('BUREAU_CREDIT_DAY_OVERDUE',0),0)
    credit_sum = _safe(row.get('BUREAU_AMT_CREDIT_SUM',0),0)
    credit_debt = _safe(row.get('BUREAU_AMT_CREDIT_SUM_DEBT',0),0)
    overdue_text = "Include a late fee row ($35.00) and an overdue notice banner" if overdue > 0 else "No overdue items"
    style_instruction = rng.choice(STYLE_VARIANTS['bank_statement'])
    prompt = PROMPT_TEMPLATES['bank_statement'].format(
        **persona, style_instruction=style_instruction,
        monthly_income=monthly_income, annuity=annuity,
        credit_sum=credit_sum, credit_debt=credit_debt, overdue_text=overdue_text,
    )
    out = f'unstructured_data/bank_statements/{sk_id}_bank_statement.pdf'
    bank_logo = BANK_LOGO_B64.get(persona['bank_name'])
    start = time.time()
    try:
        html_content, pt, rt = generate_text_via_llm(prompt)
        html_content = strip_markdown_fences(html_content)
        html_to_pdf(html_content, out, logo_b64=bank_logo, logo_alt=persona['bank_name'])
        log_generation(sk_id, 'bank_statement', pt, rt, time.time()-start, True); return True
    except Exception as e:
        print(f"    WARN bank_statement LLM/PDF failed for {sk_id}: {e}. Using fallback.")
        try:
            bal = monthly_income
            fb = FALLBACK_BANK_STMT.format(
                **persona, monthly_income=monthly_income, annuity=annuity,
                header_color=BANK_HEADER_COLORS.get(persona['bank_name'],'#003087'),
                balance_1=bal-annuity, balance_3=2*bal-2*annuity,
                balance_4=2*bal-3*annuity, balance_5=3*bal-3*annuity, balance_6=3*bal-4*annuity,
            )
            html_to_pdf(fb, out, logo_b64=bank_logo, logo_alt=persona['bank_name'])
            log_generation(sk_id, 'bank_statement', 0, 0, time.time()-start, True); return True
        except Exception as e2:
            log_generation(sk_id, 'bank_statement', 0, 0, time.time()-start, False)
            print(f"    ERROR bank_statement fallback failed for {sk_id}: {e2}"); return False


def generate_property_doc(row, persona):
    sk_id = persona['sk_id']
    rng = random.Random(sk_id + 5)
    housing_type = _safe(row.get('NAME_HOUSING_TYPE','House / apartment'),'House / apartment')
    own_realty = _safe(row.get('FLAG_OWN_REALTY','N'),'N')
    own_car = _safe(row.get('FLAG_OWN_CAR','N'),'N')
    car_age = int(_safe(row.get('OWN_CAR_AGE',0),0))
    region = int(_safe(row.get('REGION_RATING_CLIENT',2),2))
    floors = int(_safe(row.get('FLOORSMAX_AVG',10),10))
    living_area = _safe(row.get('LIVINGAREA_AVG',50),50)
    total_area = _safe(row.get('TOTALAREA_MODE',70),70)
    reg_live = int(_safe(row.get('REG_CITY_NOT_LIVE_CITY',0),0))
    live_work = int(_safe(row.get('LIVE_CITY_NOT_WORK_CITY',0),0))
    style_instruction = rng.choice(STYLE_VARIANTS['property_doc'])
    prompt = PROMPT_TEMPLATES['property_doc'].format(
        **persona, style_instruction=style_instruction, housing_type=housing_type,
        own_realty_text='Owner' if own_realty == 'Y' else 'Renter/Tenant',
        own_car_text='Yes' if own_car == 'Y' else 'No',
        car_age=car_age, floors=floors, living_area=living_area, total_area=total_area,
        region=region, reg_live_mismatch='Yes' if reg_live else 'No',
        live_work_mismatch='Yes' if live_work else 'No',
    )
    out = f'unstructured_data/property_docs/{sk_id}_property.pdf'
    start = time.time()
    try:
        html_content, pt, rt = generate_text_via_llm(prompt)
        html_content = strip_markdown_fences(html_content)
        html_to_pdf(html_content, out, seal_b64=COUNTY_SEAL_B64)
        log_generation(sk_id, 'property_doc', pt, rt, time.time()-start, True); return True
    except Exception as e:
        print(f"    WARN property_doc LLM/PDF failed for {sk_id}: {e}. Using fallback.")
        try:
            fb = FALLBACK_PROPERTY.format(
                **persona, housing_type=housing_type,
                own_realty_text='Owner' if own_realty == 'Y' else 'Renter/Tenant',
                own_car_text='Yes' if own_car == 'Y' else 'No',
                car_age=car_age, floors=floors, living_area=living_area, total_area=total_area, region=region,
            )
            html_to_pdf(fb, out, seal_b64=COUNTY_SEAL_B64)
            log_generation(sk_id, 'property_doc', 0, 0, time.time()-start, True); return True
        except Exception as e2:
            log_generation(sk_id, 'property_doc', 0, 0, time.time()-start, False)
            print(f"    ERROR property_doc fallback failed for {sk_id}: {e2}"); return False


def generate_credit_report(row, persona):
    sk_id = persona['sk_id']
    rng = random.Random(sk_id + 6)
    ext1 = _safe(row.get('EXT_SOURCE_1',0.5),0.5)
    ext2 = _safe(row.get('EXT_SOURCE_2',0.5),0.5)
    ext3 = _safe(row.get('EXT_SOURCE_3',0.5),0.5)
    score_1 = int(300 + ext1 * 550); score_2 = int(300 + ext2 * 550); score_3 = int(300 + ext3 * 550)
    avg = (score_1+score_2+score_3)/3
    if avg >= 740: risk_tier = 'Excellent'
    elif avg >= 670: risk_tier = 'Good'
    elif avg >= 580: risk_tier = 'Fair'
    else: risk_tier = 'Poor'
    def gv(name, default=0):
        v = row.get(name, default); return default if pd.isna(v) else int(v)
    report_number = f"CR-{sk_id}-{rng.randint(100000,999999)}"
    style_instruction = rng.choice(STYLE_VARIANTS['credit_report'])
    prompt = PROMPT_TEMPLATES['credit_report'].format(
        **persona, style_instruction=style_instruction, report_number=report_number,
        score_1=score_1, score_2=score_2, score_3=score_3, risk_tier=risk_tier,
        inq_hour=gv('AMT_REQ_CREDIT_BUREAU_HOUR'), inq_day=gv('AMT_REQ_CREDIT_BUREAU_DAY'),
        inq_week=gv('AMT_REQ_CREDIT_BUREAU_WEEK'), inq_month=gv('AMT_REQ_CREDIT_BUREAU_MON'),
        inq_quarter=gv('AMT_REQ_CREDIT_BUREAU_QRT'), inq_year=gv('AMT_REQ_CREDIT_BUREAU_YEAR'),
    )
    out = f'unstructured_data/credit_reports/{sk_id}_credit_report.pdf'
    bureau_logo = BUREAU_LOGO_B64.get(persona['bureau_name'])
    start = time.time()
    try:
        html_content, pt, rt = generate_text_via_llm(prompt)
        html_content = strip_markdown_fences(html_content)
        html_to_pdf(html_content, out, logo_b64=bureau_logo, logo_alt=persona['bureau_name'])
        log_generation(sk_id, 'credit_report', pt, rt, time.time()-start, True); return True
    except Exception as e:
        print(f"    WARN credit_report LLM/PDF failed for {sk_id}: {e}. Using fallback.")
        try:
            fb = FALLBACK_CREDIT.format(
                **persona, report_number=report_number,
                header_color=BUREAU_HEADER_COLORS.get(persona.get('bureau_name','Experian'),'#1D4F91'),
                score_1=score_1, score_2=score_2, score_3=score_3, risk_tier=risk_tier,
                inq_hour=gv('AMT_REQ_CREDIT_BUREAU_HOUR'), inq_day=gv('AMT_REQ_CREDIT_BUREAU_DAY'),
                inq_week=gv('AMT_REQ_CREDIT_BUREAU_WEEK'), inq_month=gv('AMT_REQ_CREDIT_BUREAU_MON'),
                inq_quarter=gv('AMT_REQ_CREDIT_BUREAU_QRT'), inq_year=gv('AMT_REQ_CREDIT_BUREAU_YEAR'),
            )
            html_to_pdf(fb, out, logo_b64=bureau_logo, logo_alt=persona.get('bureau_name','Bureau'))
            log_generation(sk_id, 'credit_report', 0, 0, time.time()-start, True); return True
        except Exception as e2:
            log_generation(sk_id, 'credit_report', 0, 0, time.time()-start, False)
            print(f"    ERROR credit_report fallback failed for {sk_id}: {e2}"); return False


# ── Main ──────────────────────────────────────────────────────────────────────

DOC_GENERATORS = {
    'paystub': generate_paystub,
    'linkedin': generate_linkedin,
    'bank_statement': generate_bank_statement,
    'id_document': generate_id_document,
    'property_doc': generate_property_doc,
    'credit_report': generate_credit_report,
}

DOC_FILES = {
    'paystub': ('paystubs', '{sk_id}_paystub.png'),
    'linkedin': ('linkedin', '{sk_id}_linkedin.png'),
    'bank_statement': ('bank_statements', '{sk_id}_bank_statement.pdf'),
    'id_document': ('id_documents', '{sk_id}_id.png'),
    'property_doc': ('property_docs', '{sk_id}_property.pdf'),
    'credit_report': ('credit_reports', '{sk_id}_credit_report.pdf'),
}

if __name__ == '__main__':
    # Load data
    bureau = pd.read_csv('datasets/bureau.csv')
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_SUM': 'sum', 'AMT_CREDIT_SUM_DEBT': 'sum',
        'AMT_CREDIT_MAX_OVERDUE': 'max', 'CREDIT_DAY_OVERDUE': 'max',
        'CNT_CREDIT_PROLONG': 'sum', 'CREDIT_ACTIVE': lambda x: (x == 'Active').sum()
    }).reset_index()
    bureau_agg.columns = ['SK_ID_CURR', 'BUREAU_AMT_CREDIT_SUM', 'BUREAU_AMT_CREDIT_SUM_DEBT',
                           'BUREAU_AMT_CREDIT_MAX_OVERDUE', 'BUREAU_CREDIT_DAY_OVERDUE',
                           'BUREAU_CNT_CREDIT_PROLONG', 'BUREAU_CREDIT_ACTIVE_COUNT']

    app_train_full = pd.read_csv('datasets/application_train.csv')
    app_train_full = app_train_full.merge(bureau_agg, on='SK_ID_CURR', how='left')

    sample = app_train_full.iloc[SAMPLE_START : SAMPLE_START + SAMPLE_SIZE].copy()
    print(f"Selected {len(sample)} applicants (iloc[{SAMPLE_START}:{SAMPLE_START + SAMPLE_SIZE}])")
    print(f"SK_ID range: {sample['SK_ID_CURR'].iloc[0]} → {sample['SK_ID_CURR'].iloc[-1]}")

    sample['stability_score'] = compute_stability_scores(sample)

    for d in ['paystubs','linkedin','bank_statements','id_documents','property_docs','credit_reports']:
        os.makedirs(f'unstructured_data/{d}', exist_ok=True)

    results = {dt: {'success': 0, 'fail': 0, 'skip': 0} for dt in DOC_GENERATORS}

    for idx, (_, row) in enumerate(sample.iterrows()):
        sk_id = int(row['SK_ID_CURR'])
        persona = generate_persona(row)

        print(f"\n{'='*60}")
        print(f"Applicant {idx+1}/{len(sample)}: SK_ID={sk_id}  [{persona['full_name']}]")
        print(f"  Address: {persona['full_address']}  |  Bank: {persona['bank_name']}")

        for doc_type, generator in DOC_GENERATORS.items():
            subdir, fname_tpl = DOC_FILES[doc_type]
            fname = fname_tpl.format(sk_id=sk_id)
            fpath = f'unstructured_data/{subdir}/{fname}'

            if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                results[doc_type]['skip'] += 1
                print(f"  {doc_type}: SKIP (exists)")
                continue

            print(f"  {doc_type}...", end=" ", flush=True)
            success = generator(row, persona)
            if success:
                results[doc_type]['success'] += 1
                print("OK")
            else:
                results[doc_type]['fail'] += 1
                print("FAILED")

    print(f"\n{'='*60}")
    print("Generation Summary:")
    total_s = total_f = total_k = 0
    for dt, c in results.items():
        s, f, k = c['success'], c['fail'], c['skip']
        total_s += s; total_f += f; total_k += k
        total = s + f + k
        print(f"  {dt}: {s} ok, {f} fail, {k} skip  ({total} total)")
    print(f"\nGenerated: {total_s}  |  Failed: {total_f}  |  Skipped: {total_k}")
    overall = total_s + total_f
    if overall > 0:
        print(f"Success rate (excluding skips): {total_s}/{overall} ({total_s/overall*100:.1f}%)")
