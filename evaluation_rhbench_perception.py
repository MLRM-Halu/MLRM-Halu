
import json
import os
from openai import AzureOpenAI
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import argparse
import re
import uuid

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


def get_hallucination_prompt(question: str, model_answer: str, ground_truth: str) -> str:
    """
    Build the hallucination evaluation prompt for free-form questions using the provided template.
    """
    image_contents = "Not provided"

    template = '''
Please act as an impartial and objective judge to evaluate the presence and severity of hallucination in the response provided by a Large Multimodal Model (LMM) to the user question. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail not grounded in the image.

Your task is to determine whether hallucination exists and, if present, to categorize its severity based on the extent and impact of the false information. Use the provided image contents, question, standard human-generated answer, and LMM response to make your judgment.

Evaluation Guidelines:
Compare the LMM response with the standard human-generated answer.
Identify any claims in the LMM response that are not supported by the image or question.
Assess the severity of hallucination based on the nature and extent of the false information.
The standard human-generated answer may not describe all objects or details comprehensively, so focus on whether the LMM introduces false claims rather than omitting details.
Detailed reasoning or analysis by the LMM that is consistent with the image contents is acceptable and should not be considered hallucination.

Task:
Evaluate the LMM response for hallucination based on the provided image contents, question, and standard human-generated answer. Provide a brief explanation of your analysis, identifying any false claims and their severity. Then, assign a rating from the following six levels:

Rating: 5: No hallucination (response is fully consistent with image contents).
Rating: 4: Minimal hallucination (minor, negligible false detail that does not significantly alter the response's accuracy, e.g., a slight misdescription of color or background).
Rating: 3: Mild hallucination (small false claim that adds unsupported but non-critical detail, e.g., mentioning a minor object or attribute not present).
Rating: 2: Moderate hallucination (clear false claim that affects the response's accuracy, e.g., incorrect object count or unsupported environmental detail).
Rating: 1: Severe hallucination (significant false claim that misrepresents key elements, e.g., entirely wrong objects or actions).
Rating: 0: Extreme hallucination (response is largely or entirely detached from the image contents, with multiple or critical false claims).


Question
{}

Standard Human-Generated Answer
{}

LMM Response to Evaluate
{}

Your response should be a JSON object with the following structure:
{{
    "hallucination_score": <integer from 0 to 5>,
    "reason": "<brief explanation of your evaluation>"
}}
'''
    return template.format(question, ground_truth, model_answer).strip()


def get_multi_choice_evaluation_prompt(question: str, options: List[str], model_answer: str, ground_truth: str) -> str:
    """
    Build the evaluation prompt for multiple-choice questions.
    """
    options_text = "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])

    template = '''Please evaluate whether the model's answer to the multiple-choice question is correct by considering:
1. Whether the model's answer matches the correct answer exactly (same option letter).
2. If the model's answer is a value, whether it matches the value of the correct option.
3. Whether the model's reasoning (if provided) supports its answer.

Question:
{}

Options:
{}

Correct Answer: {}

Model's Answer:
{}

Your evaluation should be a JSON object with the following structure:
{{
    "is_correct": <boolean>,
    "reason": "<explanation of your evaluation>",
    "model_answer_extracted": "<the extracted answer from the model's response>"
}}
'''
    return template.format(question, options_text, ground_truth, model_answer)


def evaluate_free_form_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a free-form sample for hallucination level.
    """
    try:
        model_answer = extract_model_answer(sample)
        if not model_answer:
            return {
                "id": sample["id"],
                "is_positive": False,
                "hallucination_score": None,
                "evaluation_reason": "No valid answer found in final_answer or model_answer",
                "model_answer": model_answer,
                "ground_truth": sample["answer"]
            }

        prompt = get_hallucination_prompt(
            question=sample["question"],
            model_answer=model_answer,
            ground_truth=sample["answer"]
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an impartial evaluator. Evaluate hallucination levels strictly based on the provided criteria."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        evaluation = json.loads(response.choices[0].message.content)
        hallucination_score = evaluation["hallucination_score"]
        is_positive = hallucination_score > 2 if hallucination_score is not None else False

        return {
            "id": sample["id"],
            "is_positive": is_positive,
            "hallucination_score": hallucination_score,
            "evaluation_reason": evaluation["reason"],
            "model_answer": model_answer,
            "ground_truth": sample["answer"]
        }
    except Exception as e:
        return {
            "id": sample["id"],
            "is_positive": False,
            "hallucination_score": None,
            "evaluation_reason": f"Evaluation failed: {str(e)}",
            "model_answer": model_answer,
            "ground_truth": sample["answer"]
        }


def evaluate_multi_choice_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a multiple-choice sample using model-based evaluation.
    """
    try:
        model_answer = extract_model_answer(sample)
        if not model_answer:
            return {
                "id": sample["id"],
                "is_positive": False,
                "hallucination_score": None,
                "evaluation_reason": "No valid answer found in final_answer or model_answer",
                "model_answer": model_answer,
                "ground_truth": sample["answer"]
            }

        options = sample.get("options", [])

        prompt = get_multi_choice_evaluation_prompt(
            question=sample["question"],
            options=options,
            model_answer=model_answer,
            ground_truth=sample["answer"]
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an impartial evaluator. Evaluate multiple-choice answers carefully considering both the selected option and any provided reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        evaluation = json.loads(response.choices[0].message.content)
        is_positive = evaluation["is_correct"]

        return {
            "id": sample["id"],
            "is_positive": is_positive,
            "hallucination_score": None,
            "evaluation_reason": evaluation["reason"],
            "model_answer": model_answer,
            "ground_truth": sample["answer"],
            "extracted_answer": evaluation.get("model_answer_extracted", model_answer)
        }
    except Exception as e:
        return {
            "id": sample["id"],
            "is_positive": False,
            "hallucination_score": None,
            "evaluation_reason": f"Evaluation failed: {str(e)}",
            "model_answer": model_answer,
            "ground_truth": sample["answer"]
        }


def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single sample based on its question type.
    """
    question_type = sample.get("question_type", "free_form")
    if question_type == "free_form":
        return evaluate_free_form_sample(sample)
    else:  
        return evaluate_multi_choice_sample(sample)


def process_json_file(input_path: str, output_path: str) -> tuple:
    """
    Process a single JSON file, return positive/negative sample statistics.
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
        return input_path, 0, 0, 0, 0
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return input_path, 0, 0, 0, 0


    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


    positive_count = sum(1 for r in results if r["is_positive"])
    total_count = len(results)
    negative_count = total_count - positive_count
    positive_ratio = positive_count / total_count if total_count > 0 else 0

    print(
        f"Evaluation completed for {input_path}. Positive samples: {positive_count}/{total_count} ({positive_ratio:.2%})")
    return input_path, positive_ratio, positive_count, negative_count, total_count


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

    # Process each .json file
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        output_file = output_dir / f"{json_file.stem}_evaluated.jsonl"
        input_path = str(json_file)
        output_path = str(output_file)

        # Process single file
        file_path, positive_ratio, positive_count, negative_count, total_count = process_json_file(input_path,
                                                                                                   output_path)
        summary_results.append({
            "file": file_path,
            "positive_ratio": positive_ratio,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_count": total_count
        })

    # Save summary to file
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("JSON File Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        for result in summary_results:
            f.write(f"File: {result['file']}\n")
            f.write(
                f"Positive Samples: {result['positive_count']}/{result['total_count']} ({result['positive_ratio']:.2%})\n")
            f.write(f"Negative Samples: {result['negative_count']}/{result['total_count']}\n")
            output_file_path = output_dir / f"{Path(result['file']).stem}_evaluated.jsonl"
            f.write(f"Output File: {output_file_path}\n")
            f.write("-" * 50 + "\n")
        f.write("\nBatch processing completed!\n")

    print(f"Batch processing completed, summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process JSON files for answer evaluation with hallucination scoring")
    parser.add_argument("--input_dir", type=str,
                        default="/data/steering_reason/",
                        help="Input directory containing JSON files")
    parser.add_argument("--output_dir", type=str,
                        default="/data/steering_reason/score",
                        help="Output directory for evaluation results")
    parser.add_argument("--summary_file", type=str,
                        default="/data/steering_reason/evaluation_summary.txt",
                        help="TXT file for summary statistics")
    args = parser.parse_args()

    batch_process_json_files(args.input_dir, args.output_dir, args.summary_file)