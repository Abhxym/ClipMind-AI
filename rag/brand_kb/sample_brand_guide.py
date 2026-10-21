"""
Sample Brand Style Guide — for RAG demo indexing

This document is used to populate the FAISS/ChromaDB vector store.
Upload this PDF (or text version) via the dashboard to enable brand-aware
style decisions in the Brand RAG Agent.
"""

BRAND_GUIDE_TEXT = """
AI Video Director — Sample Brand Style Guide
=============================================

1. BRAND IDENTITY
-----------------
Brand Name: HackFest 2024
Tagline: "Build. Innovate. Disrupt."
Primary Color: #6C63FF (Electric Purple)
Secondary Color: #FF6584 (Coral Pink)
Accent Color: #43D9AD (Mint Green)

2. TYPOGRAPHY
-------------
Heading Font: Inter Bold, weight 800
Body Font: Inter Regular, weight 400
Monospace: JetBrains Mono for code/data
Caption Font: Inter SemiBold, weight 600
Caption Style: All caps, letter-spacing 0.1em

3. CAPTION GUIDELINES
---------------------
Tone: High energy, motivational, tech-forward
Style: Short punchy sentences. Max 8 words per caption.
Examples:
  - "Code hard. Ship fast."
  - "48 hours. Infinite possibilities."
  - "The future is being built right now."
  - "Every great product starts here."
  - "Hack the impossible."

4. TRANSITION STYLE
-------------------
Preferred: Fast cut transitions (0.1–0.2s)
Secondary: Quick zoom-in on hero moments
Between sections: Smooth fade (0.3s)
Avoid: Long slow crossfades
Overall pacing: Energetic, rhythm-matched to beat

5. MUSIC PREFERENCES
--------------------
Primary: Upbeat electronic / EDM
BPM Range: 120–145 BPM
Style: Hype, energetic, motivational
References: Festival drops, tech product reveals
Avoid: Slow tempo, acoustic, sad/melancholy

6. VISUAL STYLE
---------------
Aesthetic: Dark background, neon accents
Overlay: Subtle gradient (#6C63FF to transparent, 20% opacity)
Logo Position: Bottom-right corner, 10% opacity watermark
Aspect Ratio: 9:16 (Instagram/TikTok vertical)
Color Grade: High contrast, slightly desaturated midtones, vibrant highlights

7. CONTENT PRIORITIES
---------------------
Priority 1: Show people collaborating, working together
Priority 2: Close-up shots of screens, keyboards, code
Priority 3: Celebration moments — high-fives, cheering
Priority 4: Speaker presentations and demos
Priority 5: Energy drinks, late-night coding vibes
Avoid: Empty rooms, blurry shots, awkward pauses

8. HASHTAG STRATEGY
-------------------
Always include: #hackathon #coding #tech #startup
Platform-specific:
  Instagram: #developer #innovation #buildinpublic #SiliconValley
  TikTok: #coderlife #hackathonlife #techtok #devlife
  YouTube: #hackathon2024 #buildwithus #techcompetition
Trending (update regularly): #AI #machinelearning #GenAI #LLM

9. MUSIC TIMING
---------------
Intro (0–3s): Beat drop or sound ramp
Main content: Full beat, on-rhythm cuts
Climax (best moment): Loudest, most energetic section
Outro (last 3s): Fade out or logo sting

10. EXPORT SETTINGS
-------------------
Platform: Instagram Reels
Resolution: 1080x1920 (9:16)
Frame Rate: 30 FPS
Bitrate: 4 Mbps
Max Duration: 90 seconds recommended
Caption Safe Zone: Bottom 200px reserved for Instagram UI
"""

if __name__ == "__main__":
    # Generate a simple text file for RAG indexing
    from pathlib import Path
    out = Path("rag/brand_kb/sample_brand_guide.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(BRAND_GUIDE_TEXT)
    print(f"Brand guide written to {out}")
    print("Index it with: db.index_document('rag/brand_kb/sample_brand_guide.txt')")
