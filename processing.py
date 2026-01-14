import re
import html

BRAND_PATTERNS = [
    r"(brand|brands)",
]

NOISE_PHRASES = [
    "мои соц сети", "мои соц. сети", "подпишись", "подписывайтесь",
    "ставь лайк", "ставьте лайк", "не забудьте подписаться",
    "поддержать канал", "поддержать меня", "делаю обзоры магазинов",
    "смотрите полное видео", "полная версия на канале",
    "мой instagram", "мой инстаграм", "telegram-канал", "телеграм-канал",
    "dzen.ru", "дзен -", "youtube.com", "vk.com", "tiktok.com",
    "tiktok-", "insta:", "instagram:"
]

RU_STOP = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так",
    "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг",
    "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
    "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где",
    "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто",
    "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы",
    "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец",
    "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про",
    "них", "какая", "много", "разве", "сказал", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более",
    "всегда", "конечно", "всю", "между",
}

TOPIC_KEYWORDS = {
    "Topic1": [r"key_word11", r"key-word12"],
    "Topic2": [r"key_word21", r"key-word22"]
}


class CommentPreprocessor:
    def __init__(self, BRAND_PATTERNS, NOISE_PHRASES, RU_STOP, TOPIC_KEYWORDS, max_len: int = 800):
        self.max_len = max_len

        # --- brands ---
        self.BRAND_PATTERNS = BRAND_PATTERNS
        self.BRAND_RE = re.compile(
            "|".join(self.BRAND_PATTERNS), re.IGNORECASE)

        # --- blogger/social noise ---
        self.NOISE_PHRASES = NOISE_PHRASES
        self.NOISE_RE = re.compile("|".join(self.NOISE_PHRASES), re.IGNORECASE)

        # --- stopwords (basic core) ---
        self.RU_STOP = RU_STOP

        # --- punctuation to replace with spaces ---
        self.PUNCT_TABLE = str.maketrans({c: " " for c in "«»“”\"'`´,()=><"})

        # --- emoji / latin-only detector ---
        self.EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]+")
        self.LATIN_ONLY_RE = re.compile(r"^[A-Za-z0-9\s\W]+$")

        # --- topic keywords (can be passed from outside) ---
        self.TOPIC_KEYWORDS = TOPIC_KEYWORDS
        self.TOPIC_RE = {
            t: re.compile("|".join(kw), re.IGNORECASE)
            for t, kw in self.TOPIC_KEYWORDS.items()
        }

    # ===== Basic building blocks =====

    def basic_clean_v2(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = text.replace("\r", "\n")
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"\[([^\]|]+)\|([^\]]+)\]",
                          r"\2", line)  # [club..|Name]
            line = re.sub(r"https?://\S+", " ", line)               # URLs
            line = re.sub(r"#(\w+)", r"\1", line)                   # hashtags
            line = re.sub(r"@\w+", " ", line)                       # @handle
            line = re.sub(r"<[^>]+>", " ", line)                    # HTML tags
            # quotes, etc.
            line = line.translate(self.PUNCT_TABLE)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                clean_lines.append(line)
        return "\n".join(clean_lines)

    def strip_blogger_noise(self, text: str) -> str:
        if not text:
            return ""
        lines = text.split("\n")
        keep = []
        for line in lines:
            low = line.lower()
            if self.BRAND_RE.search(line):   # keep brand mentions intact
                keep.append(line)
                continue
            if self.NOISE_RE.search(low):    # drop blogger/promotional junk
                continue
            keep.append(line)
        return "\n".join(keep).strip()

    def squash_repeats(self, text: str, max_repeat: int = 3) -> str:
        return re.sub(r"(.)\1{" + str(max_repeat) + r",}", r"\1" * max_repeat, text)

    def remove_emoji_only_lines(self, text: str) -> str:
        if not text:
            return ""
        lines = text.split("\n")
        keep = []
        for line in lines:
            no_emoji = self.EMOJI_RE.sub("", line)
            if sum(ch.isalpha() for ch in no_emoji) == 0:
                continue
            keep.append(line)
        return "\n".join(keep).strip()

    def drop_pure_english(self, text: str):
        # drop lines with no Cyrillic at all
        if not re.search(r"[А-Яа-яЁё]", text):
            return None
        return text

    def tokenize(self, text: str):
        return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower())

    def _iter_tokens_with_positions(self, text: str):
        """
        Split text into tokens and store, for each token:
        (token, start_char_pos, end_char_pos) in the original string.
        """
        return [
            (m.group(0), m.start(), m.end())
            for m in re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+", text)
        ]

    def _char_pos_to_token_idx(self, tokens_with_pos, char_pos: int) -> int:
        """
        Find token index by a character position in the original string.
        Take the token that contains char_pos, or the nearest one.
        """
        for i, (_, start, end) in enumerate(tokens_with_pos):
            if start <= char_pos < end:
                return i
            if char_pos < start:
                return max(0, i - 1)
        return len(tokens_with_pos) - 1

    def _extract_by_sentences(self, text: str, max_len: int) -> str:
        """
        Old sentence-based fallback, in case no brands/keywords are found.
        Here max_len is treated as NUMBER OF WORDS.
        """
        sents = self.split_sentences(text)
        if not sents:
            # if sentences didn't split well — cut by words
            toks = self.tokenize(text)
            return " ".join(toks[:max_len])

        brand_idxs = [i for i, s in enumerate(
            sents) if self.BRAND_RE.search(s)]
        chosen = []

        if brand_idxs:
            # as before: sentence with a brand ± one sentence around it
            i = brand_idxs[0]
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(sents):
                    chosen.append(sents[j])
        else:
            # "content" scoring excluding stopwords
            def score(sent):
                toks = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", sent.lower())
                content = [t for t in toks if t not in self.RU_STOP]
                return len(content)

            scored = sorted(((score(s), s) for s in sents), reverse=True)
            for _, s in scored[:5]:
                chosen.append(s)

        out = " ".join(chosen).strip()
        toks = self.tokenize(out)
        if len(toks) > max_len:
            # important: keep the TAIL
            toks = toks[-max_len:]
        out = " ".join(toks)
        out = out.replace("!", "").replace("?", "")
        return out

    def is_meaningful_v2(self, text: str) -> bool:
        if not text:
            return False
        letters = sum(ch.isalpha() for ch in text)
        if letters < 3 and not self.BRAND_RE.search(text):
            return False

        toks = self.tokenize(text)
        if not toks:
            return False

        if self.BRAND_RE.search(text):
            return True

        content = [t for t in toks if t not in self.RU_STOP]
        if len(content) == 0:
            return False

        low = text.lower()
        if self.NOISE_RE.search(low) and len(content) <= 3:
            return False

        return True

    def _iter_tokens_with_positions(self, text: str):
        """
        Split text into tokens + store each token's positions
        in the original string. Needed to cut a window ± N words
        around a brand / keyword.
        """
        return [
            (m.group(0).lower(), m.start(), m.end())
            for m in re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+", text)
        ]

    def _char_pos_to_token_idx(self, tokens_with_pos, char_pos: int) -> int:
        """
        Find token index by a character position in the original string.
        Take the token that contains char_pos, or the nearest one.
        """
        for i, (_, start, end) in enumerate(tokens_with_pos):
            if start <= char_pos < end:
                return i
            if char_pos < start:
                return max(0, i - 1)
        return len(tokens_with_pos) - 1

    def _window_around_center(self, center_idx: int, n_tokens: int, max_len: int):
        """
        A window of length max_len around center_idx.
        If there aren't enough tokens on the left — extend to the right.
        If there aren't enough tokens on the right — extend to the left.
        max_len here is treated as NUMBER OF WORDS.
        """
        if n_tokens <= max_len:
            return 0, n_tokens

        half = max_len // 2  # e.g. for 800 → ~400 left, ~400 right

        start = center_idx - half
        end = start + max_len  # [start, end) — total max_len tokens

        # not enough on the left → shift window to the beginning
        if start < 0:
            start = 0
            end = max_len

        # not enough on the right → shift window to the end
        if end > n_tokens:
            end = n_tokens
            start = n_tokens - max_len

        return start, end

    def split_sentences(self, text: str):
        parts = re.split(r"(?<=[\.\!\?\…])\s+|\n+", text)
        return [p.strip() for p in parts if p.strip()]

    def extract_relevant_span(self, text: str, max_len: int = None) -> str:
        """
        Trimming without topic_hint:
        - if the comment is shorter than max_len words — return it as-is;
        - otherwise: find ALL brand occurrences, and for each one
          cut a 100-words-left / 100-words-right window, then merge windows.
        """
        if max_len is None:
            max_len = self.max_len

        # word count
        all_tokens = self.tokenize(text)
        if len(all_tokens) <= max_len:
            # do not trim short comments
            return text.strip()

        # all brand occurrences
        centers = [m.start() for m in self.BRAND_RE.finditer(text)]
        if not centers:
            # no brands — sentence-based fallback
            return self._extract_by_sentences(text, max_len=max_len)

        tokens_with_pos = self._iter_tokens_with_positions(text)
        if not tokens_with_pos:
            return self._extract_by_sentences(text, max_len=max_len)

        n_tokens = len(tokens_with_pos)
        half_window = 100  # 100 words to the left and 100 to the right

        # build windows around EACH brand occurrence
        windows = []
        for char_pos in centers:
            center_idx = self._char_pos_to_token_idx(tokens_with_pos, char_pos)
            start = max(0, center_idx - half_window)
            # +1 to include the center token
            end = min(n_tokens, center_idx + half_window + 1)
            windows.append((start, end))

        # merge overlapping windows
        windows.sort()
        merged = []
        for s, e in windows:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)

        # convert token windows into text spans
        segments = []
        for s, e in merged:
            start_char = tokens_with_pos[s][1]
            end_char = tokens_with_pos[e - 1][2]
            segments.append(text[start_char:end_char])

        out = " ".join(seg.strip() for seg in segments).strip()

        # final cap by number of words
        toks = self.tokenize(out)
        if len(toks) > max_len:
            # keep the last max_len words
            toks = toks[-max_len:]
        out = " ".join(toks)
        out = out.replace("!", "").replace("?", "")
        return out

    def extract_relevant_span_with_keywords(self, text: str, topic_hint: str = None, max_len: int = None):
        """
        Trimming with a topic:
        - if the text is shorter than max_len words — return it as-is;
        - find ALL brand occurrences;
        - plus ALL keyword occurrences for topic_hint (if it exists in TOPIC_RE);
        - for each occurrence cut a 100-words-left / 100-words-right window and merge windows.
        """
        if max_len is None:
            max_len = self.max_len

        all_tokens = self.tokenize(text)
        if len(all_tokens) <= max_len:
            return text.strip()

        centers = []

        # brands
        centers.extend(m.start() for m in self.BRAND_RE.finditer(text))

        # topic keywords
        if topic_hint in self.TOPIC_RE:
            re_topic = self.TOPIC_RE[topic_hint]
            centers.extend(m.start() for m in re_topic.finditer(text))

        centers = sorted(set(centers))

        if not centers:
            # no brands and no keywords — fallback
            return self._extract_by_sentences(text, max_len=max_len)

        tokens_with_pos = self._iter_tokens_with_positions(text)
        if not tokens_with_pos:
            return self._extract_by_sentences(text, max_len=max_len)

        n_tokens = len(tokens_with_pos)
        half_window = 100

        windows = []
        for char_pos in centers:
            center_idx = self._char_pos_to_token_idx(tokens_with_pos, char_pos)
            start = max(0, center_idx - half_window)
            end = min(n_tokens, center_idx + half_window + 1)
            windows.append((start, end))

        # merge overlaps
        windows.sort()
        merged = []
        for s, e in windows:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)

        segments = []
        for s, e in merged:
            start_char = tokens_with_pos[s][1]
            end_char = tokens_with_pos[e - 1][2]
            segments.append(text[start_char:end_char])

        out = " ".join(seg.strip() for seg in segments).strip()

        toks = self.tokenize(out)
        if len(toks) > max_len:
            # again, keep the tail
            toks = toks[-max_len:]
        out = " ".join(toks)
        out = out.replace("!", "").replace("?", "")
        return out

    # ===== External interface =====

    def preprocess_base(self, text: str):
        x = self.basic_clean_v2(text)
        x = self.strip_blogger_noise(x)
        x = self.remove_emoji_only_lines(x)
        x = self.squash_repeats(x)
        x = self.drop_pure_english(x)
        if x is None:
            return None
        if not self.is_meaningful_v2(x):
            return None
        return x.lower()

    def preprocess(self, text: str, topic_hint: str = None, max_len: int = None):
        """
        Main method:
        - if topic_hint is provided → use keywords + brands,
        - if not → just cut a relevant fragment without a topic hint.
        """
        base = self.preprocess_base(text)
        if base is None:
            return None

        if max_len is None:
            max_len = self.max_len

        if topic_hint is None:
            return self.extract_relevant_span(base, max_len=max_len)
        else:
            return self.extract_relevant_span_with_keywords(base, topic_hint=topic_hint, max_len=max_len)
