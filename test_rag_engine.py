"""Regression checks for upload scope, multilingual retrieval and evidence labels."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from pypdf import PdfWriter
from rag_engine import (FileReport, KnowledgeBase, Retrieval, ingest, ingestion_status,
                        retrieve, render_answer, read_pdf, matched_sources, extract_page_text)


def doc(text, name='Figure AI Report.pdf', page=1, index=0):
    return Document(page_content=text, metadata={'source': name, 'page': page,
                    'page_label': str(page), 'chunk_id': str(index)})


class RetrievalTests(unittest.TestCase):
    def corpus(self):
        ds = [doc('About Figure AI. An autonomous humanoid robotics company founded in 2022.'),
              doc('Funding. Series B post-money valuation and authorized shares.', page=2, index=1),
              doc('Appendix. Financing estimates assume sale of all authorized shares.', page=3, index=2),
              doc('Figure 2 illustrates matrix control in a materials experiment.', 'materials.pdf', index=3)]
        return KnowledgeBase(ds, ds, [])

    def test_english_report_is_selected_for_chinese_query_with_name(self):
        result = retrieve(self.corpus(), '为我介绍一下Figure AI')
        self.assertEqual(result.documents[0].metadata['source'], 'Figure AI Report.pdf')
        self.assertEqual({d.metadata['page'] for d in result.documents
                          if d.metadata['source'] == 'Figure AI Report.pdf'}, {1, 2, 3})

    def test_dense_distractors_cannot_push_out_named_report(self):
        corpus = self.corpus()
        class DistractingStore:
            def similarity_search(self, query, k):
                return [corpus.chunks[-1]]
        corpus.vectors = DistractingStore()
        result = retrieve(corpus, '介绍 Figure AI')
        self.assertEqual(result.documents[0].metadata['source'], 'Figure AI Report.pdf')

    def test_keyword_financing_expansion_without_english_name(self):
        result = retrieve(self.corpus(), '融资估值')
        self.assertEqual(result.documents[0].metadata['page'], 2)

    def test_new_entity_overrides_old_rewritten_entity(self):
        corpus = self.corpus()
        corpus.pages.append(doc('Unitree company', 'Unitree Report.pdf'))
        self.assertEqual(matched_sources('介绍 Unitree', corpus), ['Unitree Report.pdf'])
        result = retrieve(corpus, '介绍 Unitree', rewritten='Figure AI')
        self.assertEqual(result.matched_files, ['Unitree Report.pdf'])

    def test_only_cited_valid_sources_are_listed(self):
        result = retrieve(self.corpus(), 'Figure AI')
        answer = render_answer(json.dumps({'answer': '公司简介。[1] 虚构数据。[99]',
                                          'evidence_status': 'partial'}), result)
        self.assertIn('Figure AI Report.pdf', answer)
        self.assertIn('[引用无效]', answer)
        self.assertNotIn('materials.pdf', answer)
        self.assertNotIn('可信度', answer)

    def test_explicit_named_report_limits_evidence(self):
        result = retrieve(self.corpus(), '根据 Figure AI Report 分析公司')
        self.assertEqual({d.metadata['source'] for d in result.documents}, {'Figure AI Report.pdf'})

    def test_refusal_does_not_attach_candidates_as_references(self):
        result = retrieve(self.corpus(), 'Figure AI')
        answer = render_answer(json.dumps({'answer': '资料未披露收入。',
                                          'evidence_status': 'insufficient'}), result)
        self.assertNotIn('引用原文', answer)

    def test_malformed_answer_is_not_passed_through(self):
        self.assertIn('格式未通过校验', render_answer('not json', Retrieval([], '', [], [])))


class IngestionTests(unittest.TestCase):
    def test_layout_silent_empty_does_not_drop_chinese_text(self):
        class ChinesePage:
            def extract_text(self, **kwargs):
                return '' if kwargs else '这是中文报告中的融资与业务正文。'
        text, fallback = extract_page_text(ChinesePage())
        self.assertIn('融资与业务正文', text)
        self.assertTrue(fallback)

    def test_layout_partial_text_does_not_silently_drop_main_body(self):
        class PartialPage:
            def extract_text(self, **kwargs):
                return '标题' if kwargs else '标题\n' + '业务数据融资金额公司产品。' * 10
        text, fallback = extract_page_text(PartialPage())
        self.assertIn('融资金额', text)
        self.assertTrue(fallback)

    def test_only_selected_file_is_loaded_no_directory_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected, stale = Path(tmp) / 'Selected.pdf', Path(tmp) / 'Old.pdf'
            for p in (selected, stale):
                writer = PdfWriter()
                writer.add_blank_page(width=100, height=100)
                writer.write(p)
            def loader(path):
                return [doc('selected company', path.name)], FileReport(path.name, 1, 1, 16)
            with patch('rag_engine.read_pdf', side_effect=loader) as mock:
                corpus = ingest([selected])
            mock.assert_called_once_with(selected)
            self.assertEqual({d.metadata['source'] for d in corpus.chunks}, {'Selected.pdf'})

    def test_duplicate_content_and_blank_scan_are_not_reported_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'Scan.pdf'
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.write(p)
            corpus = ingest([p, p])
            self.assertEqual(len(corpus.chunks), 0)
            status = ingestion_status(corpus)
            self.assertIn('成功索引 0 份', status)
            self.assertIn('OCR', status)
            self.assertIn('重复', status)

    def test_partial_page_failure_is_visible(self):
        r = FileReport('partial.pdf', 3, 1, 100, 1, ['第 2 页无可提取文字，可能需要 OCR'])
        corpus = KnowledgeBase([doc('text')], [doc('text')], [r])
        self.assertIn('1/3 页', ingestion_status(corpus))

    def test_actual_figure_report_when_available(self):
        path = Path(__file__).parent / 'data' / 'Figure AI Report 2024.12.30_09.00.34 (1).pdf'
        if not path.exists():
            self.skipTest('User PDF is not distributed with the tests')
        pages, report = read_pdf(path)
        self.assertEqual(report.text_pages, 9)
        self.assertIn('Founded', pages[0].page_content)
        self.assertIn('2022', pages[0].page_content)
        corpus = ingest([path])
        result = retrieve(corpus, '根据 Figure AI Report 分析融资、估值和风险', deep=True)
        self.assertEqual({d.metadata['page'] for d in result.documents}, set(range(1, 10)))


if __name__ == '__main__':
    unittest.main()
