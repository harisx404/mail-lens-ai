# Project Overview

## What is Mail-Lens AI?

Mail-Lens AI is an NLP-based email security analysis tool. It classifies emails as **legitimate**, **phishing**, or **malicious**, then explains the signals driving that classification in plain language.

The core problem it addresses: phishing and socially-engineered emails consistently bypass simple keyword filters because they adapt their language to look plausible. A trained ML model learns statistical patterns across thousands of examples and catches attacks that rule-based systems miss.

## Problem statement

Email remains the primary attack vector in most security incidents. Phishing attacks specifically rely on psychological manipulation — urgency, authority impersonation, fear — rather than technical exploits. These patterns are learnable from data.

The challenge is that:
- Real phishing emails use evolving language and avoid obvious keywords
- Legitimate emails sometimes use language that superficially resembles threats (e.g., password reset notifications)
- Malicious emails (delivering actual payloads) have different linguistic fingerprints than social engineering phishing

A three-class model that distinguishes all three categories is more useful than a binary spam detector.

## Intended users

- **Security-aware individuals** who want to verify suspicious emails before acting on them
- **Students and researchers** learning practical NLP/ML applications in cybersecurity
- **Developers** looking at a reference implementation of a text classification pipeline with explainability

## Main capabilities

| Capability | Description |
|---|---|
| **Email classification** | Three-class: LEGITIMATE / PHISHING / MALICIOUS |
| **Risk scoring** | 0.0–1.0 composite score combining ML probability and heuristic signals |
| **Security indicators** | Detects urgency language, credential requests, suspicious domains, lookalike brands, executable references |
| **Keyword highlighting** | Visually marks threat-relevant terms in the input text |
| **Plain-English explanations** | Describes why the system flagged an email, in non-technical language |
| **Token attribution** | Shows which n-grams most influenced the ML model's decision |
| **Preset scenarios** | Four built-in test emails demonstrating all three threat classes |

## Scope and boundaries

**In scope:**
- Static analysis of email text content (body, subject, sender address)
- Classification and risk assessment
- Explainability of model decisions

**Out of scope:**
- Real-time email interception or mailbox integration
- Email header protocol analysis (SPF, DKIM, DMARC)
- Dynamic URL scanning or sandbox execution
- Attachment content analysis
- Non-English emails (model trained on English corpus only)
