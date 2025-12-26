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
DEFAULT_THRESHOLD_MSFT = 3.0

# 🔥 [新增] 流动性风控配置
# 最大允许价差比例。例如 0.6 表示如果 (Ask-Bid)/Ask > 60%，则认为流动性太差，丢弃。
MAX_SPREAD_RATIO = 0.6 

# 数据保存文件名
HISTORY_FILE = "option_history.csv"

# === 辅助函数：强力清洗字符串 ===
def clean_str(text):
    if not text: return ""
    return str(text).replace(u'\xa0', ' ').strip()

# === 辅助函数：保存数据到 CSV ===
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
    numeric_cols = ['strike', 'price', 'ltcg', 'prob', 'raw_yield', 'gross', 'real_profit', 'otm', 'mid_raw', 'bid', 'ask']
    for col in numeric_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(float).round(2)

    # 整理列顺序
    columns_order = [
        'timestamp', 'ticker', 'type', 'date', 'strike', 'price', 
        'bid', 'ask', 'ltcg', 'prob', 'raw_yield', 'gross', 'real_profit', 'otm', 'mid_raw'
    ]
    final_cols = [c for c in columns_order if c in df_new.columns]
    df_new = df_new[final_cols]

    file_exists = os.path.isfile(HISTORY_FILE)
    try:
        df_new.to_csv(HISTORY_FILE, mode='a', header=not file_exists, index=False)
        print(f"💾 已保存 {len(df_new)} 条记录到 {HISTORY_FILE}")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")

# === 辅助函数：调用 Gemini 进行分析 ===
def get_gemini_analysis(report_text):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ 未配置 GEMINI_API_KEY，跳过智能分析。"
    
    try:
        genai.configure(api_key=api_key)
        # 使用 latest 别名
        model = genai.GenerativeModel('gemini-flash-latest')
        
        prompt = f"""
        你是一位专业的期权交易员。请阅读以下 SCHD (Put), AMZN 和 MSFT (Call) 的期权扫描数据。
        
        请完成以下任务（总字数控制在 200 字以内）：
        
        1. 【风控核查】：请重点核查 SCHD 的「除息日」风险。如果报告中指出期权跨越了除息日，请在建议中明确指出是否值得为了这点收益去承担除息后的股价下跌风险。
        2. 【策略建议】：语气专业客观。分别针对 SCHD, AMZN 和 MSFT 推荐一个最佳行权价。
        3. 【观望建议】：如果所有机会都很差，请直说“建议观望”。

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
                 ex_div_date_obj = min(future_dates)
                 ex_div_date_str = ex_div_date_obj.strftime("%Y-%m-%d")
        
        if not ex_div_date_obj:
             info = stock.info
             if 'exDividendDate' in info and info['exDividendDate']:
                 ex_div_date_obj = datetime.fromtimestamp(info['exDividendDate']).date()
                 ex_div_date_str = ex_div_date_obj.strftime("%Y-%m-%d")
        
        if ex_div_date_obj:
            print(f"📅 SCHD 下次除息日: {ex_div_date_str}, 预估分红: ${dividend_amount:.2f}")
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
            min_strike = current_price * 0.95
            max_strike = current_price * 1.02
            chain = chain[(chain['strike'] >= min_strike) & (chain['strike'] <= max_strike)]
            
            for _, row in chain.iterrows():
                # 🔥 [关键修改] 流动性过滤
                bid = row['bid']
                ask = row['ask']
                
                # 1. 必须有人买
                if bid <= 0 or ask <= 0: continue
                
                # 2. 价差不能太离谱 (比如 Bid 0.1, Ask 1.0)
                if (ask - bid) / ask > MAX_SPREAD_RATIO: continue
                
                # 3. 计算 Mid Price
                mid = (bid + ask) / 2
                price = math.floor(mid / 0.05) * 0.05
                if price <= 0.01: continue
                
                iv = row.get('impliedVolatility', 0) or 0.12
                
                # 股息调整逻辑
                adj_current_price = current_price
                is_impacted = False
                if ex_div_date_obj and dt.date() >= ex_div_date_obj:
                    adj_current_price = current_price - dividend_amount
                    is_impacted = True
                
                prob = calculate_probability(current_price, row['strike'], T, spaxx_yield, iv, 'put')

                intrinsic_value = max(0.0, row['strike'] - adj_current_price)
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
                    "bid": bid,   # 保存 Bid 方便查阅
                    "ask": ask,   # 保存 Ask 方便查阅
                    "price": price,              
                    "real_profit": extrinsic_value, 
                    "raw_yield": opt_roi * 100,
                    "gross": total_gross * 100,
                    "ltcg": ltcg_equiv * 100,
                    "prob": prob * 100,
                    "div_impact": is_impacted
                })
        except: continue
    
    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"🔵 [SCHD Put Top 5] (现价 ${current_price:.2f})\n"
        if ex_div_date_str != "N/A":
            report_str += f"📅 下次除息日: {ex_div_date_str} (已扣减预估股息 ${dividend_amount:.2f})\n"
            
        header = "到期日        行权价      Bid/Ask     挂单价    真实年化%   双吃税前%   真实LTCG%   概率      \n"
        report_str += header
        report_str += "-" * 115 + "\n"
        
        for op in top_ops:
            prob_str = f"{op['prob']:.1f}%"
            bid_ask_str = f"{op['bid']:.2f}/{op['ask']:.2f}"
            
            date_display = op['date']
            if op.get('div_impact'):
                date_display += "*"

            report_str += (
                f"{date_display:<14} "
                f"{op['strike']:<12.2f} "
                f"{bid_ask_str:<12} " # 显示 Bid/Ask
                f"{op['price']:<10.2f} "
                f"{op['raw_yield']:<12.2f} "
                f"{op['gross']:<12.2f} "
                f"{op['ltcg']:<12.2f} "
                f"{prob_str:<8}\n"
            )
        report_str += "-" * 115 + "\n"
        report_str += "💡 注: '真实'收益已剔除除息日股价下跌影响及实值水分。\n"
        report_str += "💡 过滤: 已自动剔除无买单(Bid=0)或价差过大(Spread>60%)的废单。\n\n"
        
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
                # 🔥 [关键修改] 流动性过滤
                bid = row['bid']
                ask = row['ask']
                
                if bid <= 0 or ask <= 0: continue
                if (ask - bid) / ask > MAX_SPREAD_RATIO: continue

                mid = (bid + ask) / 2
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
                    "bid": bid,
                    "ask": ask,
                    "price": price,              
                    "prob": prob_assign * 100,
                    "raw_yield": raw_yield * 100,
                    "ltcg": ltcg_equiv * 100
                })
        except: continue

    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"📦 [AMZN Call Top 5] (现价 ${current_price:.2f} | 财报日前 | 10%-20% OTM)\n"
        if earnings_limit_date:
            report_str += f"📅 下次财报日: {earnings_limit_date}\n"

        header = "到期日        行权价    Bid/Ask   挂单价    税前%     LTCG%     概率      \n"
        report_str += header
        report_str += "-" * 105 + "\n"
        
        for op in top_ops:
            otm_str = f"{op['otm']:.1f}%"
            prob_str = f"{op['prob']:.1f}%"
            bid_ask_str = f"{op['bid']:.2f}/{op['ask']:.2f}"

            report_str += (
                f"{op['date']:<14} "
                f"{op['strike']:<10.0f} "
                f"{bid_ask_str:<10} " # 显示 Bid/Ask
                f"{op['price']:<10.2f} "      
                f"{op['raw_yield']:<10.1f} "  
                f"{op['ltcg']:<10.1f} "
                f"{prob_str:<10}\n"
            )
        report_str += "-" * 105 + "\n"
    else:
        print(f"⚠️ AMZN: 在财报日 ({earnings_limit_date}) 前未找到符合条件的期权")
    
    return current_price, top_ops, report_str

# === 模块 3: MSFT Covered Call 扫描 ===
def scan_msft():
    print(f"\n🔎 [MSFT Call] 扫描开始...")
    TICKER = "MSFT"
    stock = yf.Ticker(TICKER)
    
    try:
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        print(f"📦 MSFT 当前价格: ${current_price:.2f}")
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
            
            # MSFT 和 AMZN 逻辑一样，筛选 8%-20% OTM
            min_strike = current_price * 1.08
            max_strike = current_price * 1.20
            chain = chain[(chain['strike'] >= min_strike) & (chain['strike'] <= max_strike)]
            
            for _, row in chain.iterrows():
                # 🔥 [关键修改] 流动性过滤
                bid = row['bid']
                ask = row['ask']
                
                if bid <= 0 or ask <= 0: continue
                if (ask - bid) / ask > MAX_SPREAD_RATIO: continue

                mid = (bid + ask) / 2
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
                    "bid": bid,
                    "ask": ask,
                    "price": price,              
                    "prob": prob_assign * 100,
                    "raw_yield": raw_yield * 100,
                    "ltcg": ltcg_equiv * 100
                })
        except: continue

    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"📦 [MSFT Call Top 5] (现价 ${current_price:.2f} | 财报日前 | 10%-20% OTM)\n"
        if earnings_limit_date:
            report_str += f"📅 下次财报日: {earnings_limit_date}\n"

        header = "到期日        行权价    Bid/Ask   挂单价    税前%     LTCG%     概率      \n"
        report_str += header
        report_str += "-" * 105 + "\n"
        
        for op in top_ops:
            otm_str = f"{op['otm']:.1f}%"
            prob_str = f"{op['prob']:.1f}%"
            bid_ask_str = f"{op['bid']:.2f}/{op['ask']:.2f}"
            
            report_str += (
                f"{op['date']:<14} "
                f"{op['strike']:<10.0f} "
                f"{bid_ask_str:<10} "
                f"{op['price']:<10.2f} "      
                f"{op['raw_yield']:<10.1f} "  
                f"{op['ltcg']:<10.1f} "
                f"{prob_str:<10}\n"
            )
        report_str += "-" * 105 + "\n"
    else:
        print(f"⚠️ MSFT: 在财报日 ({earnings_limit_date}) 前未找到符合条件的期权")
    
    return current_price, top_ops, report_str

# === 主程序 ===
def job():
    print(f"🚀 任务启动: {datetime.now()} UTC")
    
    run_mode = os.environ.get('RUN_MODE', 'MONITOR')
    
    if run_mode == 'SUMMARY':
        threshold_schd = -100.0
        threshold_amzn = -100.0
        threshold_msft = -100.0
        subject_prefix = "📅 [每日汇总]"
        print("📊 运行模式: 每日汇总")
    else:
        threshold_schd = DEFAULT_THRESHOLD_SCHD
        threshold_amzn = DEFAULT_THRESHOLD_AMZN
        threshold_msft = DEFAULT_THRESHOLD_MSFT
        subject_prefix = "🚨 [捡钱机会]"
        print(f"👀 运行模式: 实时监控 (阈值 SCHD>{threshold_schd}, AMZN>{threshold_amzn}, MSFT>{threshold_msft})")

    # 执行三个扫描
    schd_price, schd_list, schd_text = scan_schd()
    amzn_price, amzn_list, amzn_text = scan_amzn()
    msft_price, msft_list, msft_text = scan_msft()
    
    if schd_text: print(schd_text)
    if amzn_text: print(amzn_text)
    if msft_text: print(msft_text)
    
    # 保存数据到 CSV
    save_history_to_csv(schd_list, amzn_list, msft_list)
    
    should_notify = False
    title_parts = []

    # 检查阈值
    if schd_list and schd_list[0]['ltcg'] > threshold_schd:
        should_notify = True
        title_parts.append(f"SCHD {schd_list[0]['ltcg']:.1f}%")
        
    if amzn_list and amzn_list[0]['ltcg'] > threshold_amzn:
        should_notify = True
        title_parts.append(f"AMZN {amzn_list[0]['ltcg']:.1f}%")
        
    if msft_list and msft_list[0]['ltcg'] > threshold_msft:
        should_notify = True
        title_parts.append(f"MSFT {msft_list[0]['ltcg']:.1f}%")

    if should_notify:
        full_report = schd_text + "\n" + amzn_text + "\n" + msft_text
        
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
