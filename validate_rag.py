"""Run on local PDFs. --live sends one grounded question to configured DeepSeek."""
import argparse
import json
from pathlib import Path

import main
from rag_engine import ingest, ingestion_status, retrieve


def run(live=False):
    folder = Path(__file__).parent / 'data'
    names = [
        'Figure AI Report 2024.12.30_09.00.34 (1).pdf',
        'H3_AP202602011819006121_1 (1).pdf',
        '中国机电一体化技术应用协会-中国具身智能行业产业发展报告2026-260708.pdf',
        '华鑫证券-汽车行业深度报告：从宇树科技看国内整机及供应链投资机会（一），全球通用机器人龙头，争做硬件生态第一流-260710.pdf',
        '国泰海通证券-机器人行业2026H1人形机器人产业盘点：2026H1人形机器人盘点，量产前的加速与分化-260704.pdf',
    ]
    paths = [folder / name for name in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path.name)
    corpus = ingest(paths, main.get_embeddings)
    try:
        print(ingestion_status(corpus), flush=True)
        assert corpus.vectors is not None, corpus.warning
        assert {d.metadata['source'] for d in corpus.chunks} == set(names)
        for q in ('为我介绍一下Figure AI', 'Figure AI 的融资、估值和股权风险是什么？',
                  '比较 Figure AI 和宇树的商业化证据与风险'):
            result = retrieve(corpus, q, deep=True)
            sources = list(dict.fromkeys(d.metadata['source'] for d in result.documents))
            assert names[0] in sources
            assert all(s in names for s in sources)
            print(json.dumps({'query': q, 'first_source': result.documents[0].metadata['source'],
                              'pages': len(result.documents), 'sources': sources}, ensure_ascii=False), flush=True)
        if live:
            answer, evidence = main.answer_question(
                '根据 Figure AI Report，介绍 Figure AI，并分析融资、估值口径、风险和数据矛盾。明确区分事实与推断，注明报告日期。',
                corpus, [], deep=True)
            print('LIVE_ANSWER_BEGIN', flush=True)
            print(answer, flush=True)
            print('LIVE_ANSWER_END', flush=True)
            assert 'Figure AI Report' in answer and '引用原文' in answer, 'No grounded live answer returned'
        print('VALIDATION_OK', flush=True)
    finally:
        corpus.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    run(parser.parse_args().live)
