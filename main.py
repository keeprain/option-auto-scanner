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

# 邮件通知触发门槛 (年化收益率 %)
DEFAULT_THRESHOLD_SCHD = 11.0
DEFAULT_THRESHOLD_AMZN = 2.0
DEFAULT_THRESHOLD_MSFT = 2.0

# 流动性风控配置
MAX_SPREAD_RATIO = 0.6  # Bid/Ask 价差超过 60% 丢弃
MIN_PREMIUM = 0.15      # 权利金少于 $15 不做

# 策略风控配置 (Delta 动态防御)
TARGET_DELTA_MIN = 0.01 # Delta 下限 (1% 概率)
TARGET_DELTA_MAX = 0.09 # Delta 上限 (9% 概率 - 极致安全)
RSI_PERIOD = 14         # RSI 计算周期

# 数据保存文件名
HISTORY_FILE = "option_history.csv"

# === 辅助函数：强力清洗字符串 ===
def clean_str(text):
    if not text: return ""
    return str(text).replace(u'\xa0', ' ').strip()

# === 辅助函数：计算 RSI ===
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# === 辅助函数：保存数据到 CSV ===
def save_history_to_csv(schd_items, amzn_items, msft_items):
    all_records = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def add_records(items, ticker, type_):
        if items:
            for item in items:
                record = item.copy()
                record['ticker'] = ticker
                record['timestamp'] = timestamp
                record['type'] = type_
                all_records.append(record)

    add_records(schd_items, 'SCHD', 'Put')
    add_records(amzn_items, 'AMZN', 'Call')
    add_records(msft_items, 'MSFT', 'Call')

    if not all_records:
        return

    df_new = pd.DataFrame(all_records)
    
    numeric_cols = ['strike', 'price', 'bid', 'ask', 'ltcg', 'prob', 'raw_yield', 'gross', 'real_profit', 'otm', 'mid_raw', 'rsi']
    for col in numeric_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(float).round(2)

    columns_order = [
        'timestamp', 'ticker', 'type', 'date', 'strike', 'price', 
        'bid', 'ask', 
        'ltcg', 'prob', 'raw_yield', 'gross', 'real_profit', 'otm', 'rsi', 'mid_raw'
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
        model = genai.GenerativeModel('gemini-flash-latest')
        
        prompt = f"""
        你是一位精通量化策略的期权交易员。请分析以下 SCHD, AMZN 和 MSFT 的期权数据。
        
        【策略核心】：
        1. **极致安全 (Delta 1%-9%)**：寻找行权概率极低的机会，保证股票安全。
        2. **RSI风控**：
           - 卖 Call：RSI < 30 (超卖) 是危险信号。
           - 卖 Put：RSI > 70 (超买) 是回调风险。
        
        【任务】：
        1. **风控核查**：检查 RSI 状态和财报/除息日风险。
        2. **最佳推荐**：推荐一个“性价比最高”的期权。
        3. **决策建议**：如果收益太低或风险过高，直接建议“空仓观望”。

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

    try:
        msg = MIMEText(clean_str(body), 'plain', 'utf-8')
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = Header(clean_str(subject), 'utf-8')

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
        hist = stock.history(period='3mo')
        if hist.empty: return None, [], ""
        current_price = hist['Close'].iloc[-1]
        rsi_series = calculate_rsi(hist['Close'])
        current_rsi = rsi_series.iloc[-1]
        print(f"📊 SCHD 当前 RSI(14): {current_rsi:.2f}")
    except: return None, [], ""

    spaxx_yield = DEFAULT_SPAXX_YIELD
    try:
        fetched = yf.Ticker("SPAXX").info.get('sevenDayAverageReturn')
        if fetched and fetched > 0: spaxx_yield = fetched
    except: pass

    ex_div_date_obj = None
    ex_div_date_str = "N/A"
    dividend_amount = 0.0
    try:
        if len(stock.dividends) > 0: dividend_amount = stock.dividends.iloc[-1]
        cal = stock.calendar
        if cal and isinstance(cal, dict) and 'Ex-Dividend Date' in cal:
            dates = cal['Ex-Dividend Date']
            future = [d for d in dates if d > datetime.now().date()]
            if future:
                ex_div_date_obj = min(future)
                ex_div_date_str = ex_div_date_obj.strftime("%Y-%m-%d")
        if not ex_div_date_obj:
            info = stock.info
            if 'exDividendDate' in info and info['exDividendDate']:
                ex_div_date_obj = datetime.fromtimestamp(info['exDividendDate']).date()
                ex_div_date_str = ex_div_date_obj.strftime("%Y-%m-%d")
        if ex_div_date_obj:
            print(f"📅 SCHD 下次除息日: {ex_div_date_str}")
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
            min_strike = current_price * 0.90
            max_strike = current_price * 1.05
            chain = chain[(chain['strike'] >= min_strike) & (chain['strike'] <= max_strike)]
            
            for _, row in chain.iterrows():
                bid, ask = row['bid'], row['ask']
                if bid <= 0 or ask <= 0: continue
                if (ask - bid) / ask > MAX_SPREAD_RATIO: continue
                
                mid = (bid + ask) / 2
                price = math.floor(mid / 0.05) * 0.05
                if price < MIN_PREMIUM: continue
                
                iv = row.get('impliedVolatility', 0) or 0.12
                adj_price = current_price
                is_impacted = False
                if ex_div_date_obj and dt.date() >= ex_div_date_obj:
                    adj_price = current_price - dividend_amount
                    is_impacted = True
                
                prob = calculate_probability(current_price, row['strike'], T, spaxx_yield, iv, 'put')
                if prob > 0.45: continue

                intrinsic = max(0.0, row['strike'] - adj_price)
                extrinsic = price - intrinsic
                if extrinsic < 0: extrinsic = 0
                
                opt_roi = (extrinsic / row['strike']) * (365 / dte)
                total_gross = opt_roi + spaxx_yield
                net_yield = total_gross * (1 - TAX_ST)
                ltcg = net_yield / (1 - TAX_LT)
                
                opportunities.append({
                    "date": date, "strike": row['strike'], "mid_raw": mid,
                    "bid": bid, "ask": ask, "price": price,
                    "real_profit": extrinsic, "raw_yield": opt_roi * 100,
                    "gross": total_gross * 100, "ltcg": ltcg * 100,
                    "prob": prob * 100, "div_impact": is_impacted,
                    "rsi": current_rsi
                })
        except: continue
    
    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"🔵 [SCHD Put Top 5] (现价 ${current_price:.2f})\n"
        if current_rsi > 70: report_str += "⚠️ RSI 超买警报 (>70)：股价可能回调，卖 Put 需谨慎！\n"
        if ex_div_date_str != "N/A": report_str += f"📅 下次除息日: {ex_div_date_str}\n"
            
        # 🔥 UI 强力对齐优化 (宽间距)
        # 手动对齐中文表头，不依赖自动 padding
        header = (
            "到期日          行权价      Bid/Ask         挂单价      真实年化%     双吃税前%     真实LTCG%     概率      \n"
        )
        report_str += header + "-" * 115 + "\n"
        
        for op in top_ops:
            date_disp = op['date'] + ("*" if op.get('div_impact') else "")
            bid_ask_str = f"{op['bid']:.2f}/{op['ask']:.2f}"
            
            # 使用宽间距格式化
            report_str += (
                f"{date_disp:<15} "
                f"{op['strike']:<11.2f} "
                f"{bid_ask_str:<15} "
                f"{op['price']:<11.2f} "
                f"{op['raw_yield']:<13.2f} "
                f"{op['gross']:<13.2f} "
                f"{op['ltcg']:<13.2f} "
                f"{op['prob']:.1f}%\n"
            )
        report_str += "-" * 115 + "\n"
        
    return current_price, top_ops, report_str

# === 模块 2: AMZN Covered Call 扫描 ===
def scan_amzn():
    print(f"\n🔎 [AMZN Call] 扫描开始...")
    TICKER = "AMZN"
    stock = yf.Ticker(TICKER)
    
    try:
        hist = stock.history(period='3mo')
        if hist.empty: return None, [], ""
        current_price = hist['Close'].iloc[-1]
        rsi_series = calculate_rsi(hist['Close'])
        current_rsi = rsi_series.iloc[-1]
        print(f"📊 AMZN 当前 RSI(14): {current_rsi:.2f}")
        print(f"📦 AMZN 当前价格: ${current_price:.2f}")
    except: return None, [], ""

    earnings_limit_date = None
    try:
        cal = stock.calendar
        if cal and isinstance(cal, dict) and 'Earnings Date' in cal:
            dates = cal['Earnings Date']
            future = [d for d in dates if d > datetime.now().date()]
            if future:
                earnings_limit_date = min(future)
                print(f"📅 下次财报日: {earnings_limit_date}")
    except: pass
    
    if not earnings_limit_date:
        earnings_limit_date = datetime.now().date() + timedelta(days=45)

    try:
        dates = stock.options
    except: return None, [], ""

    opportunities = []

    for date in dates:
        dt = datetime.strptime(date, "%Y-%m-%d")
        if earnings_limit_date and dt.date() >= earnings_limit_date: continue
        
        dte = (dt - datetime.now()).days
        if not (25 <= dte <= 50): continue
        T = dte / 365.0

        try:
            chain = stock.option_chain(date).calls
            min_strike = current_price * 1.05
            max_strike = current_price * 1.35 
            chain = chain[(chain['strike'] >= min_strike) & (chain['strike'] <= max_strike)]
            
            for _, row in chain.iterrows():
                bid, ask = row['bid'], row['ask']
                if bid <= 0 or ask <= 0: continue
                if (ask - bid) / ask > MAX_SPREAD_RATIO: continue

                mid = (bid + ask) / 2
                price = math.floor(mid / 0.05) * 0.05
                if price < MIN_PREMIUM: continue
                
                iv = row.get('impliedVolatility', 0) or 0.25
                prob = calculate_probability(current_price, row['strike'], T, DEFAULT_SPAXX_YIELD, iv, 'call')
                
                if not (TARGET_DELTA_MIN <= prob <= TARGET_DELTA_MAX): continue
                
                otm_pct = (row['strike'] - current_price) / current_price * 100
                raw_yield = (price / current_price) * (365 / dte)
                net_yield = raw_yield * (1 - TAX_ST)
                ltcg = net_yield / (1 - TAX_LT)
                
                opportunities.append({
                    "date": date, "strike": row['strike'], "otm": otm_pct,
                    "bid": bid, "ask": ask, "price": price,
                    "prob": prob * 100, "raw_yield": raw_yield * 100,
                    "ltcg": ltcg * 100, "rsi": current_rsi
                })
        except: continue

    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"📦 [AMZN Call Top 5] (现价 ${current_price:.2f} | 财报日前 | 5-15% OTM)\n"
        if current_rsi < 30: report_str += "🛑 RSI 超卖警报 (<30)：股价随时反弹，建议空仓观望！\n"
        if earnings_limit_date: report_str += f"📅 下次财报日: {earnings_limit_date}\n"

        # 🔥 UI 强力对齐优化 (宽间距)
        header = (
            "到期日          行权价      价差%       Bid/Ask         挂单价      税前%       LTCG%       概率      \n"
        )
        report_str += header + "-" * 115 + "\n"
        
        for op in top_ops:
            otm_str = f"{op['otm']:.2f}%"
            bid_ask_str = f"{op['bid']:.2f}/{op['ask']:.2f}"

            report_str += (
                f"{op['date']:<15} "
                f"{op['strike']:<11.0f} "
                f"{otm_str:<11} "
                f"{bid_ask_str:<15} "
                f"{op['price']:<11.2f} "
                f"{op['raw_yield']:<11.1f} "
                f"{op['ltcg']:<11.1f} "
                f"{op['prob']:.1f}%\n"
            )
        report_str += "-" * 115 + "\n"
    else:
        print(f"⚠️ AMZN: 未找到符合 Delta ({TARGET_DELTA_MIN*100:.0f}%-{TARGET_DELTA_MAX*100:.0f}%) 且避开财报的期权")
    
    return current_price, top_ops, report_str

# === 模块 3: MSFT Covered Call 扫描 ===
def scan_msft():
    print(f"\n🔎 [MSFT Call] 扫描开始...")
    TICKER = "MSFT"
    stock = yf.Ticker(TICKER)
    
    try:
        hist = stock.history(period='3mo')
        if hist.empty: return None, [], ""
        current_price = hist['Close'].iloc[-1]
        rsi_series = calculate_rsi(hist['Close'])
        current_rsi = rsi_series.iloc[-1]
        print(f"📊 MSFT 当前 RSI(14): {current_rsi:.2f}")
        print(f"📦 MSFT 当前价格: ${current_price:.2f}")
    except: return None, [], ""

    earnings_limit_date = None
    try:
        cal = stock.calendar
        if cal and isinstance(cal, dict) and 'Earnings Date' in cal:
            dates = cal['Earnings Date']
            future = [d for d in dates if d > datetime.now().date()]
            if future:
                earnings_limit_date = min(future)
                print(f"📅 下次财报日: {earnings_limit_date}")
    except: pass
    
    if not earnings_limit_date:
        earnings_limit_date = datetime.now().date() + timedelta(days=45)

    try:
        dates = stock.options
    except: return None, [], ""

    opportunities = []

    for date in dates:
        dt = datetime.strptime(date, "%Y-%m-%d")
        if earnings_limit_date and dt.date() >= earnings_limit_date: continue
        
        dte = (dt - datetime.now()).days
        if not (25 <= dte <= 50): continue
        T = dte / 365.0

        try:
            chain = stock.option_chain(date).calls
            min_strike = current_price * 1.05
            max_strike = current_price * 1.25
            chain = chain[(chain['strike'] >= min_strike) & (chain['strike'] <= max_strike)]
            
            for _, row in chain.iterrows():
                bid, ask = row['bid'], row['ask']
                if bid <= 0 or ask <= 0: continue
                if (ask - bid) / ask > MAX_SPREAD_RATIO: continue

                mid = (bid + ask) / 2
                price = math.floor(mid / 0.05) * 0.05
                if price < MIN_PREMIUM: continue
                
                iv = row.get('impliedVolatility', 0) or 0.25
                prob = calculate_probability(current_price, row['strike'], T, DEFAULT_SPAXX_YIELD, iv, 'call')
                
                if not (TARGET_DELTA_MIN <= prob <= TARGET_DELTA_MAX): continue
                
                otm_pct = (row['strike'] - current_price) / current_price * 100
                raw_yield = (price / current_price) * (365 / dte)
                net_yield = raw_yield * (1 - TAX_ST)
                ltcg = net_yield / (1 - TAX_LT)
                
                opportunities.append({
                    "date": date, "strike": row['strike'], "otm": otm_pct,
                    "bid": bid, "ask": ask, "price": price,
                    "prob": prob * 100, "raw_yield": raw_yield * 100,
                    "ltcg": ltcg * 100, "rsi": current_rsi
                })
        except: continue

    top_ops = sorted(opportunities, key=lambda x: x['ltcg'], reverse=True)[:5]
    
    report_str = ""
    if top_ops:
        report_str += f"📦 [MSFT Call Top 5] (现价 ${current_price:.2f} | 财报日前 | 5-15% OTM)\n"
        if current_rsi < 30: report_str += "🛑 RSI 超卖警报 (<30)：股价随时反弹，建议空仓观望！\n"
        if earnings_limit_date: report_str += f"📅 下次财报日: {earnings_limit_date}\n"

        # 🔥 UI 强力对齐优化 (宽间距)
        header = (
            "到期日          行权价      价差%       Bid/Ask         挂单价      税前%       LTCG%       概率      \n"
        )
        report_str += header + "-" * 115 + "\n"
        
        for op in top_ops:
            otm_str = f"{op['otm']:.2f}%"
            bid_ask_str = f"{op['bid']:.2f}/{op['ask']:.2f}"

            report_str += (
                f"{op['date']:<15} "
                f"{op['strike']:<11.0f} "
                f"{otm_str:<11} "
                f"{bid_ask_str:<15} "
                f"{op['price']:<11.2f} "
                f"{op['raw_yield']:<11.1f} "
                f"{op['ltcg']:<11.1f} "
                f"{op['prob']:.1f}%\n"
            )
        report_str += "-" * 115 + "\n"
    else:
        print(f"⚠️ MSFT: 未找到符合 Delta ({TARGET_DELTA_MIN*100:.0f}%-{TARGET_DELTA_MAX*100:.0f}%) 且避开财报的期权")
    
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
        print(f"👀 运行模式: 实时监控 (阈值 SCHD>{threshold_schd}%, AMZN>{threshold_amzn}%, MSFT>{threshold_msft}%)")

    schd_price, schd_list, schd_text = scan_schd()
    amzn_price, amzn_list, amzn_text = scan_amzn()
    msft_price, msft_list, msft_text = scan_msft()
    
    if schd_text: print(schd_text)
    if amzn_text: print(amzn_text)
    if msft_text: print(msft_text)
    
    save_history_to_csv(schd_list, amzn_list, msft_list)
    
    should_notify = False
    title_parts = []

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
        
        # 🔥 RSI 策略速查表
        rsi_cheat_sheet = (
            "\n" + "="*40 + "\n"
            "📊 [RSI 策略速查]\n"
            "RSI < 30 (超卖)：别卖 Call，敢卖 Put。\n"
            "RSI > 70 (超买)：敢卖 Call，别卖 Put。\n"
            "RSI 中间 (30-70)：随便卖，收租金。"
        )
        
        final_body = full_report + "\n" + "="*40 + "\n🤖 [Gemini 智能分析建议]\n" + "="*40 + "\n" + gemini_analysis
        final_body += rsi_cheat_sheet
        final_body += f"\n\n(自动生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC)"
        
        subject = f"{subject_prefix} " + " | ".join(title_parts)
        send_notification(subject, final_body)
    else:
        print("😴 结果未达阈值")

if __name__ == "__main__":
    job()
