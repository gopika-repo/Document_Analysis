
### **DEMO_SCRIPT.md**
```markdown
# Vision-Fusion Demo Script (3 Minutes)

## [0:00-0:30] Introduction

**Presenter:** "Welcome to Vision-Fusion, a cutting-edge multi-modal document intelligence system. Traditional AI systems suffer from 'blindness' - they can read text but can't see. This causes critical errors when documents contain charts, tables, or signatures that contradict the text."

**[Visual: Show example document with contradictory chart and text]**

**Presenter:** "Today, I'll show you how Vision-Fusion solves this by combining computer vision with language understanding."

---

## [0:30-1:30] Vision Agent Demo

**Presenter:** "Let's start with our Vision Agent analyzing a scanned invoice PDF."

**[Action: Upload sample_invoice.pdf via /upload endpoint]**

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@sample_data/synthetic_test_document.pdf"