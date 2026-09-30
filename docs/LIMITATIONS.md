# Limitations

This document describes known limitations of Mail-Lens AI. These are genuine constraints, not conservative disclaimers.

## Dataset limitations

**MALICIOUS class is severely underrepresented.** The training corpus contains 177 malicious samples out of 17,960 total (0.99%). While the test recall is 94.3% (33/35), this is measured on a small holdout set of 35 emails. Real-world performance against diverse malware delivery campaigns is unknown and should not be assumed from this figure.

**English-only training.** All training data is in English. Performance on emails in other languages is untested and expected to degrade significantly.

**Snapshot datasets.** Phishing and malware language evolves constantly. The training data represents a snapshot of known attack patterns. Novel techniques that differ substantially from the training distribution may not be caught.

**Limited social engineering diversity.** The Legitimate/Phishing split in the training data is primarily drawn from a single HuggingFace dataset. This may not represent the full range of real-world phishing styles.

## Model limitations

**False positives on aggressive legitimate emails.** Legitimate marketing emails with urgency language (e.g., "limited time offer", "act now") may trigger threat indicators and receive elevated risk scores. The model attempts to distinguish context, but miscalibration is possible.

**False negatives on sophisticated phishing.** A carefully crafted phishing email that avoids all keyword patterns and uses novel phrasing may be classified as legitimate. The model learned from historical patterns — it cannot detect zero-day social engineering approaches.

**MALICIOUS vs PHISHING boundary.** The distinction between malicious (payload delivery) and phishing (credential harvesting) can be ambiguous in real emails. Some emails that carry both social engineering and payload components may be classified as one or the other, not both.

**No header protocol analysis.** SPF, DKIM, and DMARC are the most reliable signals for sender authenticity in production email security. This system analyzes only the text content of emails, not their headers. An attacker who spoofs a legitimate domain (and passes header checks) while sending obviously malicious content may still be caught, but a sophisticated domain spoofing attack paired with subtle content might not be.

## Feature limitations

**Static URL inspection only.** URLs are detected and flagged for suspicious TLDs, IP addresses, and URL shorteners. The application does not resolve or visit URLs, so it cannot inspect the landing page content or check against real-time blocklists (e.g., VirusTotal, Google Safe Browsing).

**Attachment content is not inspected.** References to attachment filenames (e.g., `.exe`) are detected, but actual attachment content is not accessible or analyzed.

**No email metadata.** Sender reputation, email routing headers, bounce patterns, and volume information are not available in the static text analysis.

## Application limitations

**Prototype architecture.** The application runs as a single Streamlit process. It has no rate limiting, authentication, or multi-user isolation beyond Streamlit's session state mechanism.

**Inference explanation is probabilistic, not causal.** The token attribution (top threat tokens / top safe tokens) shows which n-grams statistically correlate with threat classes based on LinearSVC coefficients. It does not definitively explain why the model made its decision in a causal sense — it is a post-hoc approximation.

**Risk score is heuristic.** The 0.0–1.0 risk score combines ML probabilities and keyword-based security heuristics. The weighting formula (`_compute_risk_score` in `engine.py`) was designed to be directionally correct but is not calibrated against a validated security ground truth.

## What this system cannot replace

- Production email security gateways (Proofpoint, Mimecast, Microsoft Defender for Office 365)
- Sandboxed URL and attachment detonation (Any.run, Cuckoo)
- Real-time threat intelligence feeds
- Human security analyst judgment for ambiguous cases
- DNS/SMTP-level sender authentication (SPF/DKIM/DMARC enforcement)
