"""Upload-scoped hybrid retrieval. No MMR or NumPy matrix operations."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def normalize(text):
    return unicodedata.normalize("NFKC", text).casefold()


def tokens(text):
    text = normalize(text)
    result = re.findall(r"[a-z][a-z0-9]*|\d+(?:\.\d+)?", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        result.extend(run[i:i + 2] for i in range(len(run) - 1))
    return result


# Query expansion is vocabulary only; no company facts are injected.
FACETS = {
    "介绍": "overview company about founded product employees mission",
    "分析": "business product funding valuation risks limitations",
    "融资": "funding financing raised investors series",
    "估值": "valuation post money market cap premium",
    "风险": "risk assumptions limitations transferability debt",
    "股权": "capital structure ownership shares liquidation preference",
    "投资人": "investors shareholders",
    "产品": "product humanoid robot technology",
    "收入": "revenue ARR sales",
    "成立": "founded founding company",
    "竞争": "competitors competitive comparison",
}
TITLE_STOP = set("ai report pdf inc the and for of about funding valuation analysis overview company".split())


@dataclass
class FileReport:
    name: str
    pages: int = 0
    text_pages: int = 0
    chars: int = 0
    chunks: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def extract_page_text(page):
    """Layout mode can silently return empty text for Chinese PDF fonts."""
    plain = page.extract_text() or ''
    try:
        layout = page.extract_text(extraction_mode='layout', layout_mode_space_vertically=False) or ''
        plain_count = len(re.sub(r'\s', '', plain))
        layout_count = len(re.sub(r'\s', '', layout))
        if layout_count and layout_count >= .85 * plain_count:
            return re.sub(r'[ \t]{3,}', '    ', layout), False
    except Exception:
        pass
    return plain, bool(plain.strip())


def read_pdf(path):
    path = Path(path)
    report = FileReport(path.name)
    documents = []
    layout_fallback_pages = []
    try:
        reader = PdfReader(path)
        report.pages = len(reader.pages)
        labels = reader.page_labels
        for index, page in enumerate(reader.pages):
            try:
                text, used_fallback = extract_page_text(page)
                if used_fallback:
                    layout_fallback_pages.append(index + 1)
                text = text.replace("\x00", "").strip()
                # Join English words split by PDF line wrapping.
                text = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", text)
            except Exception as exc:
                report.warnings.append(f"第 {index + 1} 页提取失败（{type(exc).__name__}）")
                continue
            if not text:
                report.warnings.append(f"第 {index + 1} 页无可提取文字，可能需要 OCR")
                continue
            report.chars += len(text)
            report.text_pages += 1
            documents.append(Document(page_content=text, metadata={
                    "source": path.name, "page": index + 1,
                "page_label": str(labels[index]) if index < len(labels) else str(index + 1),
            }))
        if layout_fallback_pages:
            report.warnings.append(f"{len(layout_fallback_pages)} 页使用普通文本提取以保留正文，表格列需对照原 PDF")
        if not documents:
            report.error = "未提取到正文；扫描件请先 OCR 后重新上传"
    except Exception as exc:
        report.error = f"无法读取 PDF（{type(exc).__name__}）"
    return documents, report


class LexicalIndex:
    """Small BM25 index using Python arithmetic (avoids native BLAS)."""
    def __init__(self, documents):
        self.counts = [Counter(tokens(d.metadata['source'] + " " + d.page_content)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter(t for counts in self.counts for t in counts)

    def rank(self, query, limit=40):
        scores = []
        for index, counts in enumerate(self.counts):
            score = 0.0
            for term in set(tokens(query)):
                freq = counts.get(term, 0)
                if freq:
                    idf = math.log(1 + (len(self.counts) - self.df[term] + .5) / (self.df[term] + .5))
                    norm = 1.2 * (.25 + .75 * self.lengths[index] / max(1, self.average))
                    score += idf * freq * 2.2 / (freq + norm)
            if score > 0:
                scores.append((index, score))
        return sorted(scores, key=lambda row: (-row[1], row[0]))[:limit]


@dataclass
class KnowledgeBase:
    pages: list[Document]
    chunks: list[Document]
    reports: list[FileReport]
    vectors: object = None
    warning: str = ""
    lexical: LexicalIndex = field(init=False)

    def __post_init__(self):
        self.lexical = LexicalIndex(self.chunks)

    def __deepcopy__(self, memo):
        # The corpus is immutable after creation; never deepcopy native Chroma clients.
        return self

    def close(self):
        if self.vectors is not None:
            try:
                self.vectors.delete_collection()
            except Exception:
                pass
            self.vectors = None


def ingest(pdf_files, embedding_factory=None):
    """Only explicit uploaded paths, never a directory glob or old data folder."""
    pages, reports, hashes, names = [], [], set(), set()
    for uploaded in pdf_files:
        path = Path(uploaded if isinstance(uploaded, (str, Path)) else uploaded.name)
        if path.suffix.lower() != ".pdf" or not path.is_file():
            reports.append(FileReport(path.name, error="不是可读取的 PDF 文件"))
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest in hashes:
            reports.append(FileReport(path.name, error="与本批次其他文件内容重复，未重复索引"))
            continue
        hashes.add(digest)
        if path.name in names:
            reports.append(FileReport(path.name, error="存在同名但内容不同的文件，请重命名后上传"))
            continue
        names.add(path.name)
        documents, report = read_pdf(path)
        pages.extend(documents)
        reports.append(report)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=180,
        separators=["\n\n", "\n", "。", ". ", "；", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for index, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = str(index)
    counts = Counter(d.metadata["source"] for d in chunks)
    for report in reports:
        if not report.error:
            report.chunks = counts[report.name]
    corpus = KnowledgeBase(pages, chunks, reports)
    if chunks and embedding_factory:
        # Independent in-memory collection per upload/session. Existing on-disk
        # collections are untouched, and no previous upload can enter this corpus.
        from langchain_chroma import Chroma
        store = None
        try:
            store = Chroma(collection_name="upload_" + uuid.uuid4().hex,
                           embedding_function=embedding_factory())
            for offset in range(0, len(chunks), 128):
                store.add_documents(chunks[offset:offset + 128])
            corpus.vectors = store
        except Exception as exc:
            if store:
                store.delete_collection()
            corpus.warning = f"语义索引暂不可用（{type(exc).__name__}），当前使用文件名与关键词检索。"
    return corpus


def ingestion_status(corpus):
    success = sum(bool(r.chunks) for r in corpus.reports)
    lines = [f"本次选择 {len(corpus.reports)} 份，成功索引 {success} 份；共 {len(corpus.pages)} 个文本页、{len(corpus.chunks)} 个文本块。",
             "回答仅使用这次成功索引的文件。"]
    for r in corpus.reports:
        if r.error:
            lines.append(f"未索引：{r.name} — {r.error}")
        else:
            lines.append(f"✓ {r.name}：{r.text_pages}/{r.pages} 页，{r.chars:,} 字符，{r.chunks} 块")
        lines.extend("  注意：" + warning for warning in r.warnings)
    if corpus.warning:
        lines.append(corpus.warning)
    return "\n".join(lines)


def matched_sources(query, corpus):
    query_norm = normalize(query)
    qtokens = set(tokens(query))
    result = []
    for name in dict.fromkeys(d.metadata["source"] for d in corpus.pages):
        distinctive = {t for t in tokens(Path(name).stem)
                       if t not in TITLE_STOP and not t.isdigit() and len(t) > 2 and t.isascii()}
        # Strong literal filename/entity match complements weak cross-language vectors.
        if distinctive & qtokens or normalize(Path(name).stem) in query_norm:
            result.append(name)
        elif any(term in query_norm and term in normalize(name)
                 for term in ("宇树", "智元", "优必选", "小鹏", "特斯拉", "傅利叶")):
            result.append(name)
    return result


@dataclass
class Retrieval:
    documents: list[Document]
    query: str
    matched_files: list[str]
    warnings: list[str]


def retrieve(corpus, question, rewritten="", deep=False):
    query = question + ("\n" + rewritten if rewritten else "")
    expanded = query + " " + " ".join(v for k, v in FACETS.items() if k in query)
    lexical = corpus.lexical.rank(expanded)
    scores = defaultdict(float)
    for rank, (index, _) in enumerate(lexical):
        scores[index] += 1.4 / (30 + rank)
    warnings = []
    if corpus.vectors is not None:
        try:
            # Keep similarity: MMR np.dot previously killed this Windows process.
            dense = corpus.vectors.similarity_search(query, k=min(32, len(corpus.chunks)))
            for rank, doc in enumerate(dense):
                index = int(doc.metadata["chunk_id"])
                if 0 <= index < len(corpus.chunks):
                    scores[index] += 1 / (30 + rank)
        except Exception as exc:
            warnings.append(f"语义检索不可用（{type(exc).__name__}），已使用关键词检索。")
    matched = matched_sources(question, corpus) or matched_sources(rewritten, corpus)
    # Respect explicit requests to base the answer on a named report only.
    source_only = bool(matched and re.search(r'根据|依据|仅用|只用', question)
                       and re.search(r'report|报告|研报', question, re.I)
                       and not re.search(r'比较|对比|结合|综合', question))
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    page_map = {(d.metadata['source'], d.metadata['page']): d for d in corpus.pages}
    selected, seen = [], set()
    budget = 36000 if deep else 26000
    used = 0

    def include(doc):
        nonlocal used
        key = (doc.metadata['source'], doc.metadata['page'])
        if key in seen:
            return
        text = doc.page_content
        if len(text) > 6500:
            # Use relevant chunks when an unusually long PDF page exceeds the budget.
            hits = [corpus.chunks[i].page_content for i in ranked
                    if (corpus.chunks[i].metadata['source'], corpus.chunks[i].metadata['page']) == key]
            text = "\n[…节选…]\n".join(hits[:4]) if hits else text[:6000]
        if used + len(text) > budget:
            return
        used += len(text)
        seen.add(key)
        selected.append(Document(page_content=text, metadata=dict(doc.metadata)))

    # Short, explicitly named reports can be read as a whole, preserving financing
    # tables, page 1 company description, and appendix footnotes together.
    for name in matched[:3]:
        source_pages = [d for d in corpus.pages if d.metadata['source'] == name]
        if len(source_pages) <= 14 and sum(len(d.page_content) for d in source_pages) <= 18000:
            for doc in source_pages:
                include(doc)
        else:
            source_indices = [i for i in ranked if corpus.chunks[i].metadata['source'] == name]
            for i in source_indices[:8]:
                d = corpus.chunks[i]
                include(page_map[(name, d.metadata['page'])])

    per_source = Counter(d.metadata['source'] for d in selected)
    for i in ranked:
        if len(selected) >= (22 if deep else 16):
            break
        d = corpus.chunks[i]
        name = d.metadata['source']
        if source_only and name not in matched:
            continue
        if per_source[name] >= 4 and name not in matched:
            continue
        before = len(selected)
        include(page_map[(name, d.metadata['page'])])
        per_source[name] += len(selected) - before
    for index, doc in enumerate(selected, 1):
        doc.metadata["evidence_id"] = index
    return Retrieval(selected, query, matched, warnings)


def evidence_payload(retrieval):
    return [{"id": d.metadata['evidence_id'], "file": d.metadata['source'],
             "page": d.metadata['page'], "text": d.page_content} for d in retrieval.documents]


def render_answer(raw, retrieval):
    """Only cited, valid evidence IDs are listed as answer references."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
            raise ValueError("invalid answer schema")
        answer = parsed["answer"].strip()
    except (ValueError, TypeError):
        return "回答格式未通过校验，请重试。已找到的原文仍可在下方“本次检索证据”查看。"
    valid = {str(d.metadata['evidence_id']): d for d in retrieval.documents}
    invalid = set(re.findall(r"\[(\d+)\]", answer)) - valid.keys()
    answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if m[1] in valid else "[引用无效]", answer)
    cited = list(dict.fromkeys(re.findall(r"\[(\d+)\]", answer)))
    if cited:
        answer += "\n\n---\n引用原文（编号与页码已核对，论据是否支持结论仍需复核）：\n\n"
        for key in cited:
            d = valid[key]
            answer += f"- [{key}] {d.metadata['source']}（PDF 第 {d.metadata['page']} 页）\n"
    elif parsed.get("evidence_status") != "insufficient":
        answer += "\n\n⚠️ 模型未提供有效的逐项引用；以上分析需要对照下方原文核查。"
    if invalid:
        answer += "\n\n⚠️ 模型使用了不存在的证据编号，已标出，请勿直接采信相关结论。"
    return answer
