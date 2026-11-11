# Evaluation corpus (A1)

**Source:** FLORES-200 devtest (NLLB Team, "No Language Left Behind", 2022), downloaded from
https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz. **License:** CC-BY-SA 4.0.

**Content:** 1,012 English sentences sampled from Wikinews, Wikijunior and Wikivoyage
(news, children's educational text, travel), professionally translated into every language.
Line N is the same sentence in every file. That makes it parallel, which is the whole point:
the meaning is held constant.

**Languages (8):** eng_Latn, hin_Deva, kan_Knda, tam_Taml, tel_Telu, mal_Mlym (required:
English, Hindi, 2+ Dravidian) + ben_Beng, mar_Deva (Part C languages; Marathi shares Hindi's
Devanagari script, so it's a same-script control).

**Preprocessing:** strip leading/trailing whitespace; NFC-normalize. No lowercasing, no filtering,
no deduplication. Script: `prepare_corpus.py`.
NFC changes 93 Hindi / 609 Bengali lines: it decomposes precomposed nukta letters (e.g. য় U+09DF,
ड़ U+095C), adding +1.0% code points and +0.5–0.9% tokens in Bengali (`nfc_check.py`, `extras.py`).

**Size:** see corpus_stats.txt (eng 21,901 words / 132,096 bytes; Indic 14,930–25,643 words).

**What this corpus cannot tell us**
FLORES is formal, edited, encyclopedic text, and every non-English sentence is a *translation
from English*. Translated text tends to follow English structure and vocabulary more closely than
native text, which could make the premiums look closer to English than real native writing would.
It has no conversational chat, no code-mixed Hinglish/Kanglish, no romanized Indic text (e.g.
"kya haal hai"), and no domain jargon. Our real traffic contains all of these, and they tokenize
differently. 1,012 sentences make the averages tight (bootstrap CIs ±1%), but there are too few
sentences to say anything reliable about rare words, names or long-tail inputs. These numbers are
a well-controlled lower-variance baseline for the *formal* register, not a forecast of
production token costs. Production monitoring (see memo) is what checks that gap.