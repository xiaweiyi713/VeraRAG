#!/usr/bin/env python3
"""Make VeraBench evidence spans exactly or segment-wise traceable to corpus text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SPAN_UPDATES = {
    ("V010", "E1"): (
        "星辰科技（StarTech Inc.）2022财年全年营收为458亿元人民币...全年净利润为68亿元，净利率14.8%"
    ),
    ("V011", "E2"): (
        "Willow，拥有105个量子比特...首次在 Increasing code distance 下实现了指数级降低错误率"
    ),
    ("V014", "E1"): (
        'Transformer架构由Vaswani等人在2017年的论文"Attention Is All You Need"中提出'
        "...2022年ChatGPT（基于GPT-3.5）引发全球AI热潮"
    ),
    ("V016", "E1"): (
        "DPO（直接偏好优化）通过直接利用偏好数据优化策略模型，绕过了奖励模型训练，"
        "训练更稳定、更高效...目前主流大模型厂商大多已转向DPO或其变体进行对齐训练"
    ),
    ("V020", "E3"): ("另有报道称该公司目前员工已超过6万人...该报道称星辰科技于2010年成立"),
    ("V032", "E1"): (
        "截至2024年初，全球可再生能源装机容量达到3870GW"
        "...报告预计，按当前增速，全球可在2028年前实现可再生能源装机容量翻倍"
    ),
    ("V029", "E1"): (
        "日本丰田拥有最多的固态电池专利（超过1300项），"
        "计划于2027-2028年开始量产"
        "...卫蓝新能源已于2024年开始小批量装车供货"
        "...宁德时代预计2027年实现全固态电池量产"
    ),
    ("V042", "E3"): ("另有报道称该公司目前员工已超过6万人...该报道称星辰科技于2010年成立"),
    ("V060", "E1"): (
        "2024年全球新能源汽车（含纯电动和插电混动）销量达到约2100万辆"
        "...中国继续领跑，销量约1200万辆"
        "...欧洲销量约340万辆"
        "...比亚迪以约430万辆的年销量成为全球最大新能源汽车制造商"
    ),
    ("V061", "E2"): (
        "主流RAG系统的整体幻觉率在15-30%之间...CoT（思维链）推理可以将幻觉率降低约5个百分点"
    ),
    ("V063", "E1"): (
        "Willow，拥有105个量子比特...首次在 Increasing code distance 下实现了指数级降低错误率"
    ),
    ("V071", "E1"): (
        "检索质量对幻觉率影响最大——检索到相关文档时，幻觉率可降至8%以下"
        "...增加检索文档数量（top-k从3增至10）并不总是降低幻觉率，"
        "因为不相关文档会干扰生成"
    ),
    ("V079", "E1"): (
        "2024年全球新能源汽车（含纯电动和插电混动）销量达到约2100万辆"
        "...中国继续领跑，销量约1200万辆，占全球57%"
        "...比亚迪以约430万辆的年销量成为全球最大新能源汽车制造商"
    ),
    ("V101", "E1"): (
        "美国企业占据全球半导体市场份额约46%，韩国约22%，欧洲约9%，"
        "日本约9%，中国大陆约7%"
        "...台积电在先进制程（7nm及以下）代工市场占比超过90%。"
        "三星和英特尔正在追赶3nm及以下制程"
    ),
    ("V110", "E1"): (
        "2022年成立的Rapidus公司目标是量产2nm先进制程芯片...Rapidus计划2027年开始量产2nm芯片"
    ),
    ("V114", "E3"): (
        "美国NASA的毅力号（Perseverance）火星车自2021年2月着陆以来，"
        "已在耶泽罗陨石坑行驶超过28公里，采集了24个岩石样本管"
        "...毅力号携带的机智号（Ingenuity）直升机完成了72次火星飞行"
        "...祝融号火星车在火星表面行驶约1924米"
    ),
    ("V117", "E1"): (
        "日本在1980年代曾是全球半导体霸主，市占率超过50%，但到2020年已降至约9%"
        "...Rapidus计划2027年开始量产2nm芯片"
        "...台积电在日本熊本建设的晶圆厂（JASM）已于2024年2月投产"
        "...信越化学和SUMCO合计占全球硅晶圆市场约60%份额"
    ),
    ("V127", "E2"): (
        "美国NASA的毅力号（Perseverance）火星车自2021年2月着陆以来，"
        "已在耶泽罗陨石坑行驶超过28公里，采集了24个岩石样本管"
    ),
    ("V128", "E2"): "于2022年底全面建成",
    ("V131", "E2"): "COP28提出到2030年三倍于2022年水平",
    ("V140", "E1"): (
        "2022年成立的Rapidus公司目标是量产2nm先进制程芯片，"
        "获得了日本政府约2万亿日元的资金支持，并与IBM合作获得"
        "GAA（全环绕栅极）晶体管技术授权"
    ),
    ("V146", "E1"): (
        "因延误和超支（总预算已超220亿欧元），预计推迟至2030年前后"
        "...达到3.15兆焦耳输出对比2.05兆焦耳激光输入"
    ),
    ("V148", "E1"): (
        "AlphaFold3（2024年发布）进一步扩展至预测蛋白质与DNA、RNA、"
        "小分子配体的复合物结构...然而AlphaFold并非万能——对蛋白质动力学、"
        "内在无序区域和蛋白质-蛋白质相互作用的预测仍有局限"
    ),
    ("V149", "E1"): (
        "全球已宣布的绿氢项目总产能超过3800万吨/年，"
        "但实际已投产的绿氢产能仅约100万吨/年"
        "...绿氢的生产成本目前约为4-6美元/公斤，"
        "远高于灰氢（天然气制氢，约1-2.5美元/公斤）"
        "...IRENA预计到2030年绿氢成本可降至2-3美元/公斤"
    ),
    ("V150", "E1"): (
        "美国NASA的毅力号（Perseverance）火星车自2021年2月着陆以来，"
        "已在耶泽罗陨石坑行驶超过28公里，采集了24个岩石样本管"
        "...祝融号火星车在火星表面行驶约1924米，获取了大量科学数据"
    ),
}


def migrate_question(question: dict[str, Any]) -> dict[str, Any]:
    """Return a question with traceable evidence spans."""
    question = dict(question)
    evidence_rows = []
    for evidence in question.get("evidence", []):
        evidence = dict(evidence)
        replacement = SPAN_UPDATES.get((question["id"], evidence["evidence_id"]))
        if replacement is not None:
            evidence["text_span"] = replacement
        evidence_rows.append(evidence)
    question["evidence"] = evidence_rows
    return question


def migrate_file(path: Path, *, check: bool = False) -> int:
    """Rewrite one JSONL file deterministically and return changed row count."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    migrated = [migrate_question(row) for row in rows]
    changed = sum(before != after for before, after in zip(rows, migrated, strict=True))
    if not check:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in migrated
            ),
            encoding="utf-8",
        )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("data/verabench/questions.jsonl"),
            Path("src/benchmark/data/verabench/questions.jsonl"),
        ],
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any file still requires migration.",
    )
    args = parser.parse_args()
    pending = 0
    for path in args.paths:
        changed = migrate_file(path, check=args.check)
        pending += changed
        verb = "pending" if args.check else "changed"
        print(f"{path}: {changed} {verb} question rows")
    if args.check and pending:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
