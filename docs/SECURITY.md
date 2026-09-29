# PhishGuard AI — Security Considerations

## Application Security

PhishGuard AI is itself a cybersecurity application. The following security measures are implemented:

### Input Handling
- ✅ Email content is processed as plain text — no execution of embedded code
- ✅ HTML tags are stripped before processing (BeautifulSoup)
- ✅ Input length is limited to 50,000 characters
- ✅ No user-uploaded files are executed
- ✅ URLs found in email text are analyzed statically — never visited

### Data Privacy
- ✅ No email content is sent to external APIs
- ✅ All processing is performed locally
- ✅ Analysis history is stored in session state only (not persisted to disk)
- ✅ No sensitive email content is logged
- ✅ Raw datasets are not committed to git

### Configuration Security
- ✅ Configuration via environment variables (.env)
- ✅ .env files excluded from git (.gitignore)
- ✅ .env.example provided with placeholder values
- ✅ No API keys or secrets in source code
- ✅ No hardcoded paths (all from config module)

### Static Analysis Only
- ✅ URLs are parsed but NEVER visited, resolved, or fetched
- ✅ Attachment mentions are detected from text but files are NEVER opened or executed
- ✅ No DNS lookups, no network requests to analyzed URLs
- ✅ No sandboxing or dynamic malware analysis

### Error Handling
- ✅ Graceful error messages for invalid input
- ✅ No raw Python stack traces shown to users
- ✅ Model unavailability handled with user-friendly message
- ✅ Empty/missing input returns appropriate guidance

## Threat Model

### What this system protects against
- Automated analysis of suspicious email text content
- Pattern-based detection of phishing language
- Identification of malicious indicators (urgency, credential requests, suspicious URLs)

### What this system does NOT protect against
- Real-time email interception
- Binary malware analysis
- Zero-day attacks
- Image-based phishing
- Attacks on the system itself (the app has no authentication)

## Ethical Considerations
- The system does not make security decisions autonomously
- Human review is always recommended
- The system is transparent about its limitations
- No false sense of security is created
- Clearly labeled as a research prototype
