import os
import json
import asyncio
from openai import AsyncOpenAI
from slime.utils.types import Sample


EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

JUDGE_PROMPT_TEMPLATE = """You are an expert mathematics grader. Your task is to evaluate a student's final answer to a math problem based on a reference solution.

**Problem:**
{problem}

**Reference Answer:**
{reference_answer}

**Student's Answer to Evaluate:**
{student_answer}

---
**Instructions:**
1. Compare the student's final numerical answer with the reference answer.
2. The student's answer is considered correct if it is mathematically equivalent to the reference answer.
3. Provide your evaluation in a JSON format with two keys: "score" and "reasoning".
   - "score": A float value, 1.0 for a correct answer, and 0.0 for an incorrect answer.
   - "reasoning": A brief explanation for your score.

**Your JSON Evaluation:**
"""

client = None

async def get_llm_as_judge_reward(args, sample: Sample, **kwargs) -> float:
    global client
    if client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        client = AsyncOpenAI()

    problem = sample.prompt[0]['content'] if isinstance(sample.prompt, list) else sample.prompt
    reference_answer = sample.label
    student_answer = sample.response

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        problem=problem,
        reference_answer=reference_answer,
        student_answer=student_answer
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result_str = response.choices[0].message.content
        result_json = json.loads(result_str)
        
        score = float(result_json.get("score", 0.0))
        reasoning = result_json.get("reasoning", "No reasoning provided.")

        sample.metadata['judge_reasoning'] = reasoning
        sample.metadata['judge_score'] = score
        
        return score

    except Exception as e:
        print(f"Error calling LLM-as-a-Judge for instance {sample.index}: {e}")
        return 0.0
