import os
import smtplib
import unicodedata
from email.mime.text import MIMEText
from email.header import Header

def clean_string(text):
    """
    强力清洗函数：
    1. NFKD 标准化：把各种怪异的字符转为标准字符
    2. encode/decode: 过滤掉非 UTF-8 字符
    3. replace: 再次确保 \xa0 (不换行空格) 变成了普通空格
    """
    if not text: return ""
    # 1. 标准化 (把 \xa0 变成空格)
    normalized = unicodedata.normalize('NFKD', str(text))
    # 2. 再次强制替换
    cleaned = normalized.replace(u'\xa0', u' ')
    return cleaned

def send_test_email():
    print("🚀 [Debug模式] 开始测试邮件发送功能...")

    # 1. 读取 Secrets
    sender = os.environ.get('EMAIL_USER')
    password = os.environ.get('EMAIL_PASS')
    receiver = os.environ.get('TO_EMAIL')

    # 2. 检查 Secrets 是否读取成功
    if not sender:
        print("❌ 错误: EMAIL_USER 未找到")
        return
    if not password:
        print("❌ 错误: EMAIL_PASS 未找到")
        return
    if not receiver:
        print("❌ 错误: TO_EMAIL 未找到")
        return

    print(f"📧 发件人: {sender}")
    print(f"📧 收件人: {receiver}")
    print("🔑 密码状态: 已读取 (长度: {})".format(len(password) if password else 0))

    # 3. 构造测试内容 (模拟之前的报错场景)
    # 我们故意放入一些中文、Emoji 和竖线，看看是否能正常发送
    raw_subject = "🚨 测试: GitHub Action Test | 检查点"
    raw_body = """
    你好！
    
    这是一封测试邮件。
    如果收到这封信，说明：
    1. Secrets 配置正确。
    2. 字符编码问题已解决。
    3. 你的程序可以正常发信了。
    
    Test Time: 2025-12-25
    """

    # 4. 清洗字符串 (关键步骤)
    safe_subject = clean_string(raw_subject)
    safe_body = clean_string(raw_body)

    try:
        # 5. 构造邮件对象
        msg = MIMEText(safe_body, 'plain', 'utf-8')
        msg['From'] = sender
        msg['To'] = receiver
        # 显式指定 UTF-8 编码
        msg['Subject'] = Header(safe_subject, 'utf-8')

        print("🔄 正在连接 Gmail 服务器...")
        
        # 6. 发送 (设置 30秒 超时)
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30)
        
        print("🔄 正在登录...")
        server.login(sender, password)
        
        print("🔄 正在发送数据...")
        server.sendmail(sender, [receiver], msg.as_string())
        
        server.quit()
        print("-" * 30)
        print(f"✅✅✅ 成功！测试邮件已发送给 {receiver}")
        print("-" * 30)

    except Exception as e:
        print("-" * 30)
        print(f"❌❌❌ 发送严重失败: {e}")
        print("-" * 30)
        # 打印更多调试信息
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    send_test_email()
