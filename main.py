import yfinance as yf
import math
import os
import smtplib
import unicodedata
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime, timedelta
from scipy.stats import norm
from email.mime.text import MIMEText
from email.header import Header

# === 全局配置 ===
DEFAULT_SPAXX_YIELD = 0.034
TAX_ST = 0.37       # 短期税率
TAX_LT = 0.238      # 长期税率

# 邮件通知触发门槛
DEFAULT_THRESHOLD_SCHD = 11.0
DEFAULT_THRESHOLD_AMZN = 2.0
DEFAULT_THRESHOLD_MSFT = 2.0

# 流动性风控配置
# 最大允许价差比例。例如 0.6 表示如果 (Ask-Bid)/Ask > 60%，则认为流动性太差，丢弃。
MAX_SPREAD_RATIO = 0.6 

# 数据保存文件名
HISTORY_FILE = "option_history.csv"

# === 辅助函数：强力清洗字符串 ===
def clean_str(text):
    if not text: return ""
    return str(text).replace(u'\xa0', ' ').strip()

# === 辅助函数：保存数据到 CSV (包含 Bid/Ask) ===
def save_history_to_csv(schd_items, amzn_items, msft_items):
    all_records = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if schd_items:
        for item in schd_items:
            record = item.copy()
            record['ticker'] = 'SCHD'
            record['timestamp'] = timestamp
            record['type'] = 'Put'
            all_records.append(record)
            
    if amzn_items:
        for item in amzn_items:
            record = item.copy()
            record['ticker'] = 'AMZN'
            record['timestamp'] = timestamp
            record['type'] = 'Call'
            all_records.append(record)

    if msft_items:
        for item in msft_items:
            record = item.copy()
            record['ticker'] = 'MSFT'
            record['timestamp'] = timestamp
            record['type'] = 'Call'
            all_records.append(record)

    if not all_records:
        return

    df_new = pd.DataFrame(all_records)
    
    # 智能四舍五入
    numeric_cols = ['strike', 'price', 'bid', 'ask', 'ltcg', 'prob', 'raw_yield', 'gross', 'real_profit', 'otm', 'mid_raw']
    for col in numeric_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(float).round(2)

    # 整理列顺序
    columns_order = [
        'timestamp', 'ticker', 'type', 'date', 'strike', 'price', 
        'bid', 'ask', 
        'ltcg', 'prob', 'raw_yield', 'gross', 'real_profit', 'otm', 'mid_raw'
    ]
    final_cols = [c for c in columns_order if c in df_new.columns]
    df_new = df_new[final_cols]

    file_exists = os.path.isfile(HISTORY_FILE)
    try:
        df_new.to_csv(HISTORY_FILE, mode='a', header=not file_exists, index=False)
        print(f"💾 已保存 {len(df_new)} 条记录到 {HISTORY_FILE}")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")

# === 辅助函数：调用 Gemini 进行分析 (🔥 Prompt 深度修正) ===
def get_gemini_analysis(report_text):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ 未配置 GEMINI_API_KEY，跳过智能分析。"
    
    try:
        genai.configure(api_key=api_key)
        # 使用 latest 别名
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # 🔥🔥🔥 Prompt 核心修改：重新定义概率的含义 🔥🔥🔥
        prompt = f"""
        你是一位专业的期权卖方交易员（Seller Strategies）。请阅读以下 SCHD (Put), AMZN 和 MSFT (Call) 的扫描数据。
        
        【重要定义 - 务必遵守】：
        1. **概率 (Prob)**：这里指“被行权概率”(Probability of Assignment/ITM)。
        2. **核心逻辑**：作为期权卖方，我们希望**「概率」越低越好**（意味着更安全，股票不会被卖飞或被迫接盘），同时**「收益率」越高越好**。
        3. **任务**：请寻找“低风险（低概率）”下的“高性价比”机会。不要推荐那些收益虽高但概率极高（例如 >20%）的危险选项！

        请完成以下任务（总字数 200 字以内）：
        
        1. 【风控核查】：重点核查 SCHD 的「除息日」风险。
        2. 【策略建议】：
           - 语气专业客观。
           - 分别针对 SCHD, AMZN 和 MSFT 推荐一个最佳行权价。
           - **理由必须基于：在较低的行权概率（安全）下，获得了不错的收益。**
        3. 【观望建议】：如果所有选项的收益率都很低，或者行权概率都太高（不安全），请直说“建议观望”。

        数据如下：
        {report_text}
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8000,
                temperature=0.2
            )
        )
        return response.text.strip()

    except Exception as e:
        return f"❌ Gemini 分析失败: {str(e)}"

# === 辅助函数：发送邮件 ===
def send_notification(subject, body):
    raw_pass = os.environ.get('EMAIL_PASS', '')
    password = raw_pass.replace(u'\xa0', '').replace(' ', '').strip()
    
    sender = clean_str(os.environ.get('EMAIL_USER'))
    receiver = clean_str(os.environ.get('TO_EMAIL'))

    if not sender or not password or not receiver:
        print("\n⚠️ 未配置邮件 Secrets，跳过发送通知。")
        return

    clean_body = clean_str(body)
    clean_subject = clean_str(subject)

    try:
        msg = MIMEText(clean_body, 'plain', 'utf-8')
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = Header(clean_subject, 'utf-8')

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30)
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())
        server.quit()
        print(f"✅ 通知已发送给 {receiver}")
    except Exception as e:
        print(f"❌ 发送通知失败: {e}")

# === 辅助函数：计算行权概率 (Delta) ===
def calculate_probability(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(-d1)

# === 模块 1: SCHD Put 扫描 ===
def scan_schd():
    print(f"\n🔎 [SCHD Put] 扫描开始...")
    TICKER = "SCHD"
    stock = yf.Ticker(TICKER)
    
    try:
        hist = stock.history(period='1d')
        current_price = hist['Close'].iloc[-1]
    except: return None, [], ""

    spaxx_yield = DEFAULT_SPAXX_YIELD
    try:
        fetched = yf.Ticker("SPAXX").info.get('sevenDayAverageReturn')
        if fetched and fetched > 0: spaxx_yield = fetched
    except: pass

    # 获取除息信息
    ex_div_date_obj = None
    ex_div_date_str = "N/A"
    dividend_amount = 0.0

    try:
        if len(stock.dividends) > 0:
            dividend_amount = stock.dividends.iloc[-1]
        
        cal = stock.calendar
        if cal and isinstance(cal, dict) and 'Ex-Dividend Date' in cal:
             dates = cal['Ex-Dividend Date']
             future_dates = [d for d in dates if d > datetime.now().date()]
             if future_dates:
