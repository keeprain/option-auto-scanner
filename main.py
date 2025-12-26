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
DEFAULT_THRESHOLD_SCHD = 10.0
DEFAULT_THRESHOLD_AMZN = 3.0

# 数据保存文件名
HISTORY_FILE = "option_history.csv"

# === 辅助函数：强力清洗字符串 ===
def clean_str(text):
    if not text: return ""
    return str(text).replace(u'\xa0', ' ').strip()

# === 辅助函数：保存数据到 CSV (列名已统一) ===
def save_history_to_csv(schd_items, amzn_items):
    """
    将当天的 Top 机会追加保存到 CSV 文件中
    """
    all_records = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 处理 SCHD 数据
    if schd_items:
        for item in schd_items:
            record = item.copy()
            record['ticker'] = 'SCHD'
            record['timestamp'] = timestamp
            record['type'] = 'Put'
            all_records.append(record)
            
    # 处理 AMZN 数据
    if amzn_items:
        for item in amzn_items:
            record = item.copy()
            record['ticker'] = 'AMZN'
            record['timestamp'] = timestamp
            record['type'] = 'Call'
            all_records.append(record)

    if not all_records:
        return

    # 转换为 DataFrame
    df_new = pd.DataFrame(all_records)
    
    # 整理列顺序 (统一使用 price 和 raw_yield)
    columns_order = [
        'timestamp', 'ticker', 'type', 'date', 'strike', 'price', 
        'ltcg', 'prob', 'raw_yield', 'gross', 'real_profit', 'otm', 'mid_raw'
    ]
    # 只保留存在的列，防止报错
    final_cols = [c for c in columns_order if c in df_new.columns]
    df_new = df_new[final_cols]

    # 检查文件是否存在
    file_exists = os.path.isfile(HISTORY_FILE)
    
    try:
        # 追加模式 'a'，如果文件不存在则写入表头
        df_new.to_csv(HISTORY_FILE, mode='a', header=not file_exists, index=False)
        print(f"💾 已保存 {len(df_new)} 条记录到 {HISTORY_FILE}")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")

# === 辅助函数：调用 Gemini 进行分析 (回归专业详尽版) ===
def get_gemini_analysis(report_text):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ 未配置 GEMINI_API_KEY，跳过智能分析。"
    
    try:
        genai.configure(api_key=api_key)
        
        # 使用你 Log 里确认可用的 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 🔥🔥🔥 回归旧版本的 Prompt 🔥🔥🔥
        prompt = f"""
        你是一位专业的期权交易员。请阅读以下 SCHD (Cash-Secured Put) 和 AMZN (Covered Call) 的期权扫描数据。
        请给出一段非常简练的分析和操作建议（总字数控制在 200 字以内）。
        
        要求：
        1. 语气专业、客观。
        2. 分别针对 SCHD 和 AMZN 推荐一个性价比最高的行权价，并一句话解释原因（基于真实收益率和安全性）。
        3. 如果所有机会都很差，请直说“建议观望”。

        数据如下：
        {report_text}
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8000,  # 🔥 给足空间，防止截断
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

# === 模块 1: SCHD Put 扫描 (含 Debug Log) ===
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
            
            # [DEBUG]
            print(f"   [DEBUG] {date}: 原始 Put 数量 {len(chain)}")

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

                intrinsic_value = max(0.0, row['strike'] - current_price)
                extrinsic_value = price - intrinsic_value
                if extrinsic_value < 0: extrinsic_value = 0
                
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
                    "raw_yield": opt_roi * 100,  # 统一列名
                    "gross": total_gross * 100,
                    "ltcg": ltcg_equiv * 100,
                    "prob": prob * 100
                })
        except Exception as e:
            print(f"   [DEBUG] 处理 {date} 时出错: {e}")
            continue
    
    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"🔵 [SCHD Put Top 5] (现价 ${current_price:.2f})\n"
        header = "到期日        行权价      原价      挂单价    真实年化%   双吃税前%   真实LTCG%   概率      \n"
        report_str += header
        report_str += "-" * 115 + "\n"
        
        for op in top_ops:
            prob_str = f"{op['prob']:.1f}%"
            report_str += (
                f"{op['date']:<14} "
                f"{op['strike']:<12.2f} "
                f"{op['mid_raw']:<10.2f} "
                f"{op['price']:<10.2f} "
                f"{op['raw_yield']:<12.2f} " # 统一列名
                f"{op['gross']:<12.2f} "
                f"{op['ltcg']:<12.2f} "
                f"{prob_str:<8}\n"
            )
        report_str += "-" * 115 + "\n"
        report_str += "💡 注: '真实'收益已剔除行权价高于现价带来的虚高水分 (只算时间价值)。\n\n"
        
    return current_price, top_ops, report_str

# === 模块 2: AMZN Covered Call 扫描 (含 Debug Log) ===
def scan_amzn():
    print(f"\n🔎 [AMZN Call] 扫描开始...")
    TICKER = "AMZN"
    stock = yf.Ticker(TICKER)
    
    try:
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        print(f"📦 AMZN 当前价格: ${current_price:.2f}")
    except: return None, [], ""

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
            
            # [DEBUG]
            print(f"   [DEBUG] {date}: 原始 Call 数量 {len(chain)}")

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
                    "price": price,              # 统一列名
                    "prob": prob_assign * 100,
                    "raw_yield": raw_yield * 100,# 统一列名
                    "ltcg": ltcg_equiv * 100
                })
        except Exception as e:
            print(f"   [DEBUG] 处理 {date} 时出错: {e}")
            continue

    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"📦 [AMZN Call Top 5] (现价 ${current_price:.2f} | 财报日前 | 10%-20% OTM)\n"
        header = "到期日        行权价    价差%     挂单价    税前%     LTCG%     概率      \n"
        report_str += header
        report_str += "-" * 105 + "\n"
        
        for op in top_ops:
            otm_str = f"{op['otm']:.1f}%"
            prob_str = f"{op['prob']:.1f}%"
            
            report_str += (
                f"{op['date']:<14} "
                f"{op['strike']:<10.0f} "
                f"{otm_str:<10} "
                f"{op['price']:<10.2f} "      # 统一列名
                f"{op['raw_yield']:<10.1f} "  # 统一列名
                f"{op['ltcg']:<10.1f} "
                f"{prob_str:<10}\n"
            )
        report_str += "-" * 105 + "\n"
    else:
        print(f"⚠️ AMZN: 在财报日 ({earnings_limit_date}) 前未找到符合条件的期权")
    
    return current_price, top_ops, report_str

# === 主程序 ===
def job():
    print(f"🚀 任务启动: {datetime.now()} UTC")
    
    run_mode = os.environ.get('RUN_MODE', 'MONITOR')
    
    if run_mode == 'SUMMARY':
        threshold_schd = -100.0
        threshold_amzn = -100.0
        subject_prefix = "📅 [每日汇总]"
        print("📊 运行模式: 每日汇总")
    else:
        threshold_schd = DEFAULT_THRESHOLD_SCHD
        threshold_amzn = DEFAULT_THRESHOLD_AMZN
        subject_prefix = "🚨 [捡钱机会]"
        print(f"👀 运行模式: 实时监控 (阈值 >{threshold_schd}, >{threshold_amzn})")

    schd_price, schd_list, schd_text = scan_schd()
    amzn_price, amzn_list, amzn_text = scan_amzn()
    
    if schd_text: print(schd_text)
    if amzn_text: print(amzn_text)
    
    # 🔥 保存数据到 CSV (无论是否发邮件)
    save_history_to_csv(schd_list, amzn_list)
    
    should_notify = False
    title_parts = []

    if schd_list and schd_list[0]['ltcg'] > threshold_schd:
        should_notify = True
        title_parts.append(f"SCHD {schd_list[0]['ltcg']:.1f}%")
        
    if amzn_list and amzn_list[0]['ltcg'] > threshold_amzn:
        should_notify = True
        title_parts.append(f"AMZN {amzn_list[0]['ltcg']:.1f}%")

    if should_notify:
        full_report = schd_text + "\n" + amzn_text
        
        print("🤖 正在请求 Gemini 进行分析...")
        gemini_analysis = get_gemini_analysis(full_report)
        print("🤖 分析完成")
        
        final_body = full_report + "\n" + "="*40 + "\n🤖 [Gemini 智能分析建议]\n" + "="*40 + "\n" + gemini_analysis
        final_body += f"\n\n(自动生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC)"
        
        subject = f"{subject_prefix} " + " | ".join(title_parts)
        send_notification(subject, final_body)
    else:
        print("😴 结果未达阈值")

if __name__ == "__main__":
    job()
