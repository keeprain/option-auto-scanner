import yfinance as yf
import math
import os
import smtplib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from email.mime.text import MIMEText
from email.header import Header

# === 全局配置 ===
DEFAULT_SPAXX_YIELD = 0.045
TAX_ST = 0.37      # 短期税率
TAX_LT = 0.238     # 长期税率

# 邮件通知触发门槛
NOTIFY_THRESHOLD_SCHD = 11.0 
NOTIFY_THRESHOLD_AMZN = 12.0 

# === 辅助函数：发送邮件 ===
def send_notification(subject, body):
    sender = os.environ.get('EMAIL_USER')
    password = os.environ.get('EMAIL_PASS')
    receiver = os.environ.get('TO_EMAIL')

    if not sender or not password or not receiver:
        print("\n⚠️ 未配置邮件 Secrets，跳过发送通知。(请检查 GitHub Settings -> Secrets)")
        return

    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = Header(subject, 'utf-8')

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
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

# === 模块 1: SCHD Put 扫描 (无条件 Top 3) ===
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
            chain = chain[(chain['strike'] < current_price) & (chain['strike'] > current_price * 0.95)]
            
            for _, row in chain.iterrows():
                mid = (row['bid'] + row['ask']) / 2
                if mid == 0: continue
                price = math.floor(mid / 0.05) * 0.05
                if price <= 0.01: continue
                
                iv = row.get('impliedVolatility', 0) or 0.12
                prob = calculate_probability(current_price, row['strike'], T, spaxx_yield, iv, 'put')

                opt_roi = (price / row['strike']) * (365 / dte)
                total_gross = opt_roi + spaxx_yield
                net_yield = total_gross * (1 - TAX_ST)
                ltcg_equiv = net_yield / (1 - TAX_LT)
                
                opportunities.append({
                    "date": date,
                    "strike": row['strike'],
                    "mid_raw": mid,
                    "price": price,
                    "opt_roi": opt_roi * 100,
                    "gross": total_gross * 100,
                    "ltcg": ltcg_equiv * 100,
                    "prob": prob * 100
                })
        except: continue
    
    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:3]
    
    report_str = ""
    if top_ops:
        report_str += f"🔵 [SCHD Put Top 3] (现价 ${current_price:.2f})\n"
        # 格式修复：先定义表头
        header = f"{'到期日':<12} {'行权价':<8} {'原价':<6} {'挂单价':<8} {'期权年化%':<10} {'双吃税前%':<10} {'LTCG等效%':<10} {'概率':<6}\n"
        report_str += header
        report_str += "-" * len(header) + "\n"
        
        for op in top_ops:
            # 格式修复核心：先变成字符串带%，再对齐
            prob_str = f"{op['prob']:.1f}%"
            
            report_str += (
                f"{op['date']:<12} "
                f"{op['strike']:<8.1f} "
                f"{op['mid_raw']:<6.2f} "
                f"{op['price']:<8.2f} "
                f"{op['opt_roi']:<10.2f} "
                f"{op['gross']:<10.2f} "
                f"{op['ltcg']:<10.2f} "
                f"{prob_str:<6}\n" # 这里就没有空格了
            )
        report_str += "-" * len(header) + "\n\n"
        
    return current_price, top_ops, report_str

# === 模块 2: AMZN Covered Call 扫描 (财报日前 + 格式修复) ===
def scan_amzn():
    print(f"\n🔎 [AMZN Call] 扫描开始...")
    TICKER = "AMZN"
    stock = yf.Ticker(TICKER)
    
    try:
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        print(f"📦 AMZN 当前价格: ${current_price:.2f}")
    except: return None, [], ""

    # === 获取下次财报日期 ===
    earnings_limit_date = None
    try:
        # yfinance 的 calendar 经常变，尝试抓取下一次财报日
        cal = stock.calendar
        if cal and isinstance(cal, dict) and 'Earnings Date' in cal:
             # 获取列表中的第一个日期
             earnings_dates = cal['Earnings Date']
             # 找到第一个未来的日期
             future_dates = [d for d in earnings_dates if d > datetime.now().date()]
             if future_dates:
                 earnings_limit_date = min(future_dates)
                 print(f"📅 下次财报日: {earnings_limit_date} (扫描将截止于此日期前)")
    except: 
        pass
    
    # 如果没抓到，给一个默认的 30 天安全期，或者你可以注释掉这行不设限
    if not earnings_limit_date:
        print("⚠️ 无法确认财报日，将扫描未来 45 天内的期权")
        earnings_limit_date = datetime.now().date() + timedelta(days=45)

    try:
        dates = stock.options
    except: return None, [], ""

    opportunities = []

    for date in dates:
        dt = datetime.strptime(date, "%Y-%m-%d")
        
        # 🔥 核心过滤: 必须在财报日之前到期 (或当天)
        if earnings_limit_date and dt.date() >= earnings_limit_date:
            continue
            
        dte = (dt - datetime.now()).days
        if dte < 5: continue # 剔除太短的
        
        T = dte / 365.0

        try:
            chain = stock.option_chain(date).calls
            
            # 行权价范围: 现价+8% ~ 现价+20%
            min_strike = current_price * 1.08
            max_strike = current_price * 1.20
            
            chain = chain[(chain['strike'] >= min_strike) & (chain['strike'] <= max_strike)]
            
            for _, row in chain.iterrows():
                mid = (row['bid'] + row['ask']) / 2
                if mid == 0: continue
                price = math.floor(mid / 0.05) * 0.05
                if price <= 0.01: continue
                
                iv = row.get('impliedVolatility', 0) or 0.25
                prob_assign = calculate_probability(current_price, row['strike'], T, DEFAULT_SPAXX_YIELD, iv, 'call')
                
                if prob_assign >= 0.20: continue 
                
                otm_pct = (row['strike'] - current_price) / current_price * 100
                raw_yield = (price / current_price) * (365 / dte)
                net_yield = raw_yield * (1 - TAX_ST)
                ltcg_equiv = net_yield / (1 - TAX_LT)
                
                opportunities.append({
                    "date": date,
                    "strike": row['strike'],
                    "otm": otm_pct,
                    "premium": price,
                    "prob": prob_assign * 100,
                    "raw": raw_yield * 100,
                    "ltcg": ltcg_equiv * 100
                })
        except: continue

    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"📦 [AMZN Call Top 5] (财报日前 | 10%-20% OTM)\n"
        # 更新表头
        header = f"{'到期日':<12} {'行权价':<8} {'价差%':<8} {'概率':<8} {'挂单价':<8} {'税前%':<8} {'LTCG%':<8}\n"
        report_str += header
        report_str += "-" * len(header) + "\n"
        
        for op in top_ops:
            # 🔥 格式修复：紧凑型百分比
            otm_str = f"{op['otm']:.1f}%"
            prob_str = f"{op['prob']:.1f}%"
            
            report_str += (
                f"{op['date']:<12} "
                f"{op['strike']:<8.0f} "
                f"{otm_str:<8} "  # 修复后
                f"{prob_str:<8} " # 修复后
                f"{op['premium']:<8.2f} "
                f"{op['raw']:<8.1f} "
                f"{op['ltcg']:<8.1f}\n"
            )
        report_str += "-" * len(header) + "\n"
    else:
        print(f"⚠️ AMZN: 在财报日 ({earnings_limit_date}) 前未找到符合条件的期权")
    
    return current_price, top_ops, report_str

# === 主程序 ===
def job():
    print(f"🚀 任务启动: {datetime.now()} UTC")
    
    schd_price, schd_list, schd_text = scan_schd()
    amzn_price, amzn_list, amzn_text = scan_amzn()
    
    if schd_text: print(schd_text)
    if amzn_text: print(amzn_text)
    
    should_notify = False
    title_parts = []

    if schd_list and schd_list[0]['ltcg'] > NOTIFY_THRESHOLD_SCHD:
        should_notify = True
        title_parts.append(f"SCHD {schd_list[0]['ltcg']:.1f}%")
        
    if amzn_list and amzn_list[0]['ltcg'] > NOTIFY_THRESHOLD_AMZN:
        should_notify = True
        title_parts.append(f"AMZN {amzn_list[0]['ltcg']:.1f}%")

    if should_notify:
        full_report = schd_text + "\n" + amzn_text
        subject = "🚨 捡钱机会: " + " | ".join(title_parts)
        send_notification(subject, full_report)
    else:
        print(f"😴 结果未达通知门槛 (SCHD > {NOTIFY_THRESHOLD_SCHD}%, AMZN > {NOTIFY_THRESHOLD_AMZN}%)")

if __name__ == "__main__":
    job()
