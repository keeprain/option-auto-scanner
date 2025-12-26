import yfinance as yf
import math
import os
import smtplib
import unicodedata
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from email.mime.text import MIMEText
from email.header import Header

# === 全局配置 ===
DEFAULT_SPAXX_YIELD = 0.034
TAX_ST = 0.37       # 短期税率
TAX_LT = 0.238      # 长期税率

# 邮件通知触发门槛
DEFAULT_THRESHOLD_SCHD = 10.0
DEFAULT_THRESHOLD_AMZN = 3.0

# === 辅助函数：强力清洗字符串 ===
def clean_str(text):
    if not text: return ""
    return str(text).replace(u'\xa0', ' ').strip()

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

# === 模块 1: SCHD Put 扫描 (Top 5 + 真实收益) ===
def scan_schd():
    print(f"\n🔎 [SCHD Put] 扫描开始...")
    TICKER = "SCHD"
    stock = yf.Ticker(TICKER)
    
    try:
        hist = stock.history(period='1d')
        current_price = hist['Close'].iloc[-1]
        divs = stock.dividends
        if len(divs) >= 4: last_4_divs = divs.iloc[-4:].sum()
        else: last_4_divs = divs.sum()
    except: return None, [], ""

    spaxx_yield = DEFAULT_SPAXX_YIELD
    try:
        fetched = yf.Ticker("SPAXX").info.get('sevenDayAverageReturn')
        if fetched and fetched > 0: spaxx_yield = fetched
    except: pass

    try:
        dates = stock.options
    except: return None, [], ""

    opportunities = []
    
    for date in dates:
        dt = datetime.strptime(date, "%Y-%m-%d")
        dte = (dt - datetime.now()).days
        if not (25 <= dte <= 50): continue
        T = dte / 365.0

        try:
            chain = stock.option_chain(date).puts
            
            # 范围: 95% - 102%
            min_strike = current_price * 0.95
            max_strike = current_price * 1.02
            
            chain = chain[(chain['strike'] >= min_strike) & (chain['strike'] <= max_strike)]
            
            for _, row in chain.iterrows():
                mid = (row['bid'] + row['ask']) / 2
                if mid == 0: continue
                price = math.floor(mid / 0.05) * 0.05
                if price <= 0.01: continue
                
                iv = row.get('impliedVolatility', 0) or 0.12
                prob = calculate_probability(current_price, row['strike'], T, spaxx_yield, iv, 'put')

                # 计算真实外在价值 (剔除 ITM 水分)
                intrinsic_value = max(0.0, row['strike'] - current_price)
                extrinsic_value = price - intrinsic_value
                if extrinsic_value < 0: extrinsic_value = 0
                
                # 基于真实价值计算收益率
                opt_roi = (extrinsic_value / row['strike']) * (365 / dte)
                total_gross = opt_roi + spaxx_yield
                net_yield = total_gross * (1 - TAX_ST)
                ltcg_equiv = net_yield / (1 - TAX_LT)
                
                opportunities.append({
                    "date": date,
                    "strike": row['strike'],
                    "mid_raw": mid,
                    "price": price,              
                    "real_profit": extrinsic_value, 
                    "opt_roi": opt_roi * 100,    
                    "gross": total_gross * 100,
                    "ltcg": ltcg_equiv * 100,
                    "prob": prob * 100
                })
        except: continue
    
    # 🔥 修改点：Top 3 -> Top 5
    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"🔵 [SCHD Put Top 5] (现价 ${current_price:.2f})\n"
        header = "到期日        行权价      原价      挂单价    真实年化%   真实LTCG%   概率      \n"
        report_str += header
        report_str += "-" * 105 + "\n"
        
        for op in top_ops:
            prob_str = f"{op['prob']:.1f}%"
            report_str += (
                f"{op['date']:<14} "
                f"{op['strike']:<12.2f} "
                f"{op['mid_raw']:<10.2f} "
                f"{op['price']:<10.2f} "
                f"{op['opt_roi']:<12.2f} "
                f"{op['ltcg']:<12.2f} "
                f"{prob_str:<8}\n"
            )
        report_str += "-" * 105 + "\n"
        report_str += "💡 注: '真实'收益已剔除行权价高于现价带来的虚高水分 (只算时间价值)。\n\n"
        
    return current_price, top_ops, report_str

# === 模块 2: AMZN Covered Call 扫描 ===
def scan_amzn():
    print(f"\n🔎 [AMZN Call] 扫描开始...")
    TICKER = "AMZN"
    stock = yf.Ticker(TICKER)
    
    try:
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        print(f"📦 AMZN 当前价格: ${current_price:.2f}")
    except: return None, [], ""

    # 获取下次财报日期
    earnings_limit_date = None
    try:
        cal = stock.calendar
        if cal and isinstance(cal, dict) and 'Earnings Date' in cal:
             earnings_dates = cal['Earnings Date']
             future_dates = [d for d in earnings_dates if d > datetime.now().date()]
             if future_dates:
                 earnings_limit_date = min(future_dates)
                 print(f"📅 下次财报日: {earnings_limit_date}")
    except: pass
    
    if not earnings_limit_date:
        print("⚠️ 无法确认财报日，将扫描未来 45 天内的期权")
        earnings_limit_date = datetime.now().date() + timedelta(days=45)

    try:
        dates = stock.options
    except: return None, [], ""

    opportunities = []

    for date in dates:
        dt = datetime.strptime(date, "%Y-%m-%d")
        if earnings_limit_date and dt.date() >= earnings_limit_date:
            continue
            
        dte = (dt - datetime.now()).days
        if dte < 5: continue
        
        T = dte / 365.0

        try:
            chain = stock.option_chain(date).calls
            min_strike = current_price * 1.08
            max_strike = current_price * 1.20
            chain = chain[(chain['strike'] >= min_strike) & (chain['strike'] <= max_strike)]
            
            for _, row in chain.iterrows():
                mid = (row['bid'] + row['ask']) / 2
                if mid == 0: continue
                price = math.floor(mid / 0.05) * 0.05
                if price <= 0.01: continue
                
                iv = row
