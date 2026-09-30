import html
import re

def highlight_email_content(raw_text: str, is_threat: bool = True, threat_tokens: list = None) -> tuple:
    if not raw_text:
        return "", 0, 0, []

    # Prepare custom threat tokens from ML if available
    extra_red = []
    if threat_tokens:
        for t in threat_tokens:
            token = t.get("token", "") if isinstance(t, dict) else str(t)
            if len(token) >= 3 and token.lower() not in {"the", "and", "for", "with", "this", "from", "that", "your"}:
                extra_red.append(re.escape(token))

    red_words = [
        r"passwords?", r"passcode", r"log\s*in", r"login", r"sign\s*in", r"credentials?",
        r"verify\s+your\s+account", r"confirm\s+your\s+identity", r"verify\s+your\s+identity",
        r"reset\s+password", r"security\s+alert", r"unauthorized", r"account\s+suspension",
        r"reactivate", r"otp", r"2fa", r"pin", r"ssn", r"social\s+security",
        r"bank\s+account", r"banking", r"credit\s+card", r"update-verification",
        r"secure-login-portal", r"account-alert", r"rnicrosoft", r"paypa1", r"amaz0n",
        r"invoice_scan\.pdf\.exe", r"payload", r"exploit"
    ]
    if extra_red:
        red_words.extend(extra_red[:5])

    red_patterns = [
        r"\b(?:" + "|".join(red_words) + r")\b",
        r"\.(?:exe|scr|bat|vbs|iso|zip|js|cmd|pif|hta)\b",
        r"https?://[^\s<>\"']*(?:rnicrosoft|paypa1|amaz0n|update|verify|login|secure|auth|\.xyz|\.top)[^\s<>\"']*",
    ]

    yellow_patterns = [
        r"\b(?:immediately?|urgently?|within\s+24\s+hours|within\s+2\s+hours|24\s+hours|final\s+notice|final\s+warning|action\s+required|terminated|suspended|suspend|locked|expires?|immediate\s+action|restricted|restriction|limited\s+time|act\s+now|critical\s+alert|deadline|quota\s+exceeded|blocked|freeze)\b",
        r"\b(?:winners?|winning|congratulations|inheritance|prizes?|wire\s+transfer|western\s+union|crypto(?:currency)?|bitcoins?|lottery|funds?\s+transfer|million\s+dollars|beneficiary|compensation|claim\s+now|risk\s+free|grant|donation|unclaimed|reward|100%\s+free)\b",
        r"\b(?:buy\s+now|special\s+offer|exclusive\s+deal|viagra|guaranteed|unsubscribe|click\s+here|click\s+below|visit\s+link|free\s+gift|earn\s+extra|work\s+from\s+home)\b",
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    ]

    green_patterns = [
        r"\b(?:meeting|agenda|deliverables?|roadmap|quarterly|project\s+sync|engineering|standup|calendar\s+invite|attached\s+report|colleagues?|team|discussion|schedule|updates?|notes|budget|approved|quarter|presentation|milestones?)\b",
    ]

    matches = []
    detected_tags = set()

    for pat in red_patterns:
        for m in re.finditer(pat, raw_text, re.IGNORECASE):
            matches.append((m.start(), m.end(), "red", m.group(0)))
            detected_tags.add(m.group(0).lower())

    for pat in yellow_patterns:
        for m in re.finditer(pat, raw_text, re.IGNORECASE):
            matches.append((m.start(), m.end(), "yellow", m.group(0)))
            detected_tags.add(m.group(0).lower())

    if not is_threat:
        for pat in green_patterns:
            for m in re.finditer(pat, raw_text, re.IGNORECASE):
                matches.append((m.start(), m.end(), "green", m.group(0)))

    # Sort matches by start position, prioritizing longer matches if tie
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Filter overlaps
    filtered_matches = []
    last_end = 0
    for start, end, cat, text in matches:
        if start >= last_end:
            filtered_matches.append((start, end, cat, text))
            last_end = end

    # Build highlighted output safely
    out = []
    last_idx = 0
    red_count = 0
    yellow_count = 0

    for start, end, cat, text in filtered_matches:
        out.append(html.escape(raw_text[last_idx:start]))
        if cat == "red":
            red_count += 1
            out.append(f'<span class="hl-red">{html.escape(text)}</span>')
        elif cat == "yellow":
            yellow_count += 1
            out.append(f'<span class="hl-yellow">{html.escape(text)}</span>')
        else:
            out.append(f'<span class="hl-green">{html.escape(text)}</span>')
        last_idx = end

    out.append(html.escape(raw_text[last_idx:]))
    formatted = "".join(out).replace("\n", "<br>")
    return formatted, red_count, yellow_count, list(detected_tags)

if __name__ == "__main__":
    sample = "Urgent: Your account is suspended. Please login to verify your account within 24 hours: https://rnicrosoft.com/login. Also wire transfer to claim prize or download invoice.pdf.exe"
    res, rc, yc, tags = highlight_email_content(sample, is_threat=True)
    print(f"Red: {rc}, Yellow: {yc}")
    print("Tags:", tags)
    print("HTML:", res)
