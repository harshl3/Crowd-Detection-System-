import os
import wave
import math
import struct
import base64
import time
import streamlit as st

def generate_alert_sound(file_path="assets/alert.wav"):
    """
    Synthesizes a short warning beep programmatically to avoid external asset dependency.
    """
    if os.path.exists(file_path):
        return
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    sample_rate = 44100.0
    duration = 0.5  # seconds
    frequency = 1200.0  # 1200 Hz alert pitch
    num_samples = int(duration * sample_rate)
    
    try:
        wave_file = wave.open(file_path, 'w')
        # channels, sampwidth (2 bytes = 16 bit), framerate, nframes, comptype, compname
        wave_file.setparams((1, 2, int(sample_rate), num_samples, 'NONE', 'not compressed'))
        
        for i in range(num_samples):
            # Generate sine wave
            value = int(32767.0 * math.sin(2.0 * math.pi * frequency * (i / sample_rate)))
            data = struct.pack('<h', value)
            wave_file.writeframesraw(data)
            
        wave_file.close()
    except Exception as e:
        st.error(f"Failed to generate alert audio file: {e}")

def play_audio_alert(file_path="assets/alert.wav", throttle_seconds=5):
    """
    Plays an alert sound directly in the user's web browser using HTML5 audio.
    Throttles the playbacks to avoid overwhelming the user during frequent streamlit reruns.
    """
    current_time = time.time()
    
    # Check if we need to initialize session state for audio alert tracking
    if "last_audio_alert_time" not in st.session_state:
        st.session_state.last_audio_alert_time = 0.0
        
    if current_time - st.session_state.last_audio_alert_time < throttle_seconds:
        return # Skip playing to prevent audio overlaps
        
    if not os.path.exists(file_path):
        generate_alert_sound(file_path)
        
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            audio_html = f"""
                <audio autoplay="true" style="display:none;">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                </audio>
                """
            st.markdown(audio_html, unsafe_allow_html=True)
            st.session_state.last_audio_alert_time = current_time
    except Exception as e:
        pass # Silently fail audio if there is an issue with file read or HTML rendering

def trigger_external_alerts(message, settings):
    """
    Triggers Telegram and Email alerts depending on the configuration settings.
    Returns status messages of the actions taken.
    """
    alert_logs = []
    
    # 1. Telegram alert
    if settings.get("telegram_enabled", False):
        bot_token = settings.get("telegram_token", "").strip()
        chat_id = settings.get("telegram_chat_id", "").strip()
        
        if bot_token and chat_id:
            success, msg = send_telegram_alert(message, bot_token, chat_id)
            alert_logs.append((f"Telegram: {'✅' if success else '❌'}", msg))
        else:
            alert_logs.append(("Telegram: ⚠️ Mock", "Credentials missing. Msg: " + message))
            
    # 2. Email alert
    if settings.get("email_enabled", False):
        sender = settings.get("email_sender", "").strip()
        pwd = settings.get("email_password", "").strip()
        receiver = settings.get("email_receiver", "").strip()
        
        if sender and pwd and receiver:
            success, msg = send_email_alert(message, sender, pwd, receiver)
            alert_logs.append((f"Email: {'✅' if success else '❌'}", msg))
        else:
            alert_logs.append(("Email: ⚠️ Mock", "Credentials missing. Msg: " + message))
            
    return alert_logs

def send_telegram_alert(message, bot_token, chat_id):
    import urllib.request
    import urllib.parse
    import json
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=5) as response:
            res = response.read().decode("utf-8")
            res_json = json.loads(res)
            if res_json.get("ok"):
                return True, "Alert message dispatched to Telegram channel!"
            else:
                return False, f"Telegram Error: {res_json.get('description')}"
    except Exception as e:
        return False, f"HTTP Error: {str(e)}"

def send_email_alert(message, sender_email, sender_pwd, receiver_email):
    import smtplib
    from email.mime.text import MIMEText
    try:
        msg = MIMEText(message)
        msg['Subject'] = '🚨 OVERCROWD ALERT: AI Crowd Monitor System'
        msg['From'] = sender_email
        msg['To'] = receiver_email

        # Connect to Gmail SMTP (SSL)
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, sender_pwd)
        server.sendmail(sender_email, [receiver_email], msg.as_string())
        server.close()
        return True, "Alert email sent to " + receiver_email
    except Exception as e:
        return False, f"SMTP Error: {str(e)}"
