import json
import os
from openai import AzureOpenAI
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import argparse
import re

# Initialize Azure OpenAI client
API_BASE = ""
API_KEY = ""
DEPLOYMENT_NAME = ""
API_VERSION = ""

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    base_url=f"{API_BASE}/openai/deployments/{DEPLOYMENT_NAME}"
)



def extract_model_answer(sample: Dict[str, Any]) -> str:
    """
    Extract the model's answer from the sample.
    1. First check if final_answer exists and is non-empty, return it if true
    2. If no final_answer, for free-form questions return entire model_answer
    3. For multiple-choice questions:
       - First try to extract answer after "Answer: " pattern
       - If not found, use entire model_answer as fallback
    """

    final_answer = sample.get("final_answer", "").strip()
    if final_answer:
        return final_answer

    model_answer = sample.get("model_answer", "").strip()
    if not model_answer:
        return ""

    question_type = sample.get("question_type", "free_form")

    if question_type == "free_form":
        return model_answer
    else:
        match = re.search(r"Answer:\s*([A-Da-d])", model_answer)
        if match:
            return match.group(1).upper()
        return model_answer


def get_evaluation_prompt(question: str, model_answer: str, ground_truth: str, question_type: str,
                          choices: str = "") -> str:
    """
    Build the evaluation prompt based on question type.
    """
    if question_type == "multi_choice":
        prompt = f"""
You are an impartial evaluator assessing the correctness of a model's answer to a multiple-choice question.

Question: {question}
Choices: {choices}
Model's Answer: {model_answer}
Correct Answer: {ground_truth}

Please evaluate whether the model's answer is correct by considering:
1. Whether the model's answer matches the correct answer exactly (e.g., same option letter).
2. If the model's answer is a value, whether it matches the value of the correct option.
3. Whether the model's reasoning (if provided) supports its answer.

Your response should be a JSON object with the following structure:
{{
    "is_correct": <true or false>,
    "reason": "<brief explanation of your evaluation>"
}}
"""
    else:  # free_form
        prompt = f"""
You are an impartial evaluator assessing the correctness of a model's answer to a free-form question requiring a numerical answer.

Question: {question}
Model's Answer: {model_answer}
Correct Answer: {ground_truth}

Please evaluate whether the model's answer is correct by considering:
1. Whether the model's answer matches the correct answer exactly.
2. If not exact, whether the model's answer is numerically equivalent to the correct answer.
3. Whether the model's reasoning (if provided) supports its answer.

Your response should be a JSON object with the following structure:
{{
    "is_correct": <true or false>,
    "reason": "<brief explanation of your evaluation>"
}}
"""
    return prompt.strip()


def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single sample based on its question type.
    """
    try:
        model_answer = extract_model_answer(sample)
        if not model_answer:
            return {
                "id": sample["id"],
                "is_correct": False,
                "evaluation_reason": "No valid answer found in final_answer or model_answer",
                "model_answer": model_answer,
                "ground_truth": sample["answer"]
            }

        question_type = sample.get("question_type", "free_form")
        choices = sample["question"].split("Choices:")[1].strip() if "Choices:" in sample[
            "question"] and question_type == "multi_choice" else ""

        prompt = get_evaluation_prompt(
            question=sample["question"],
            model_answer=model_answer,
            ground_truth=sample["answer"],
            question_type=question_type,
            choices=choices
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an impartial evaluator. Evaluate answers strictly based on the provided criteria."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        evaluation = json.loads(response.choices[0].message.content)
        return {
            "id": sample["id"],
            "is_correct": evaluation["is_correct"],
            "evaluation_reason": evaluation["reason"],
            "model_answer": model_answer,
            "ground_truth": sample["answer"]
        }
    except Exception as e:
        return {
            "id": sample["id"],
            "is_correct": False,
            "evaluation_reason": f"Evaluation failed: {str(e)}",
            "model_answer": model_answer,
            "ground_truth": sample["answer"]
        }


def process_json_file(input_path: str, output_path: str) -> tuple:
    """
    Process a single JSON file, return accuracy and sample statistics.
    """
    results = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            samples = data if isinstance(data, list) else [data]

        for sample in tqdm(samples, desc=f"Processing {os.path.basename(input_path)}"):
            evaluation = evaluate_sample(sample)
            results.append(evaluation)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {input_path}: {str(e)}")
        return input_path, 0, 0, 0
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return input_path, 0, 0, 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"Evaluation completed for {input_path}. Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    return input_path, accuracy, correct_count, total_count


def batch_process_json_files(input_dir: str, output_dir: str, summary_file: str):
    """
    Batch process all JSON files in the input directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)


    if not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} does not exist or is not a directory")
        return


    if not output_dir.exists():
        output_dir.mkdir(parents=True)


    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"Warning: No .json files found in directory {input_dir}")
        return


    summary_results = []


    for json_file in tqdm(json_files, desc="Processing JSON files"):
        output_file = output_dir / f"{json_file.stem}_evaluated.jsonl"
        input_path = str(json_file)
        output_path = str(output_file)


        file_path, accuracy, correct_count, total_count = process_json_file(input_path, output_path)
        summary_results.append({
            "file": file_path,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count
        })


    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("JSON File Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        for result in summary_results:
            f.write(f"File: {result['file']}\n")
            f.write(f"Accuracy: {result['accuracy']:.2%} ({result['correct_count']}/{result['total_count']})\n")
            output_file_path = output_dir / f"{Path(result['file']).stem}_evaluated.jsonl"
            f.write(f"Output File: {output_file_path}\n")
            f.write("-" * 50 + "\n")
        f.write("\nBatch processing completed!\n")

    print(f"Batch processing completed, summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process JSON files for MathVista answer evaluation")
    parser.add_argument("--input_dir", type=str,
                        default="/data/steering_reason/",
                        help="Input directory containing JSON files")
    parser.add_argument("--output_dir", type=str,
                        default="/data/steering_reason/score/",
                        help="Output directory for evaluation results")
    parser.add_argument("--summary_file", type=str,
                        default="/data/steering_reason/evaluation_summary.txt",
                        help="TXT file for summary statistics")
    args = parser.parse_args()

    batch_process_json_files(args.input_dir, args.output_dir, args.summary_file)