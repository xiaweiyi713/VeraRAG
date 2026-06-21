import pytest

from experiments.filter_verabench_report import filter_report


def _report():
    return {
        "metadata": {"provider": "deepseek"},
        "question_results": [
            {
                "question_id": "V001",
                "question_type": "single_evidence",
                "question": "欧盟AI法案将违规罚款上限设定为多少？",
                "ground_truth": "3500万欧元或全球年营业额7%。",
                "predicted": "3500万欧元或全球年营业额7%。",
                "expected_behavior": "answer_with_citation",
                "actual_behavior": "answer_with_citation",
                "correct": True,
                "answer_f1": 0.1,
                "confidence": 0.8,
                "difficulty": "easy",
            },
            {
                "question_id": "V002",
                "question_type": "single_evidence",
                "question": "中国生成式AI管理暂行办法是什么时候施行的？",
                "ground_truth": "2023年8月15日。",
                "predicted": "该办法于2023年8月15日正式施行。",
                "expected_behavior": "answer_with_citation",
                "actual_behavior": "answer_with_citation",
                "correct": False,
                "answer_f1": 0.1,
                "confidence": 0.7,
                "difficulty": "easy",
            },
        ],
    }


def test_filter_report_keeps_requested_order_and_rescores():
    filtered = filter_report(_report(), ["V002", "V001"], allow_unverified=True)

    assert filtered["completed"] == 2
    assert [row["question_id"] for row in filtered["question_results"]] == [
        "V002",
        "V001",
    ]
    assert filtered["metadata"]["filtered_offline"] is True
    assert filtered["metadata"]["question_ids"] == ["V002", "V001"]
    assert filtered["overall_answer_f1"] > 0.8


def test_filter_report_rejects_missing_ids():
    with pytest.raises(ValueError, match="missing requested question IDs"):
        filter_report(_report(), ["V404"], allow_unverified=True)
