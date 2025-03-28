import requests
import json
import time
import csv
import os
from datetime import datetime

# --- Configuration ---
API_KEY = ""

# Optional, leave blank if not needed
API_BASE_URL = "https://openrouter.ai/api/v1"
# Timeout for the API request in seconds (e.g., 5 minutes)
REQUEST_TIMEOUT = 300
DELAY_BETWEEN_REQUESTS = 10  # Seconds to wait between API calls to avoid rate limits

# --- Helper Functions ---


def get_api_headers():
    """Constructs the API headers."""
    if not API_KEY:
        raise ValueError("Missing environment variable: OPENROUTER_API_KEY")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    # Add optional headers if provided
    return headers


def measure_llm_performance(model_name: str, prompt: str) -> dict:
    """
    Calls the OpenRouter API for a given model and prompt, measures performance.

    Args:
        model_name: The name of the model to use (e.g., "openai/gpt-4o").
        prompt: The user prompt to send to the model.

    Returns:
        A dictionary containing performance metrics and results.
    """
    headers = get_api_headers()
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,  # Crucial for measuring TTFT
    }

    start_time = None
    first_token_time = None
    last_token_time = None
    completion_text = ""
    total_prompt_tokens = None
    total_completion_tokens = None
    error_message = None
    finish_reason = None

    try:
        start_time = time.perf_counter()
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=headers,
            data=json.dumps(data),
            stream=True,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    json_data_str = decoded_line[len("data: "):].strip()

                    if json_data_str == "[DONE]":
                        last_token_time = (
                            time.perf_counter()
                        )  # Mark end time on DONE signal
                        break  # Exit loop once DONE is received

                    try:
                        chunk = json.loads(json_data_str)
                        # Check for usage stats often sent at the end (though might be in the last content chunk too)
                        if usage := chunk.get("usage"):
                            total_prompt_tokens = usage.get("prompt_tokens")
                            total_completion_tokens = usage.get(
                                "completion_tokens")

                        # Check for content delta
                        if choices := chunk.get("choices"):
                            if len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                content_chunk = delta.get("content")
                                finish_reason = (
                                    choices[0].get(
                                        "finish_reason") or finish_reason
                                )  # Store last known finish reason

                                if content_chunk:
                                    if first_token_time is None:
                                        first_token_time = time.perf_counter()
                                    completion_text += content_chunk
                                    # Update last_token_time with every chunk containing content
                                    last_token_time = time.perf_counter()

                    except json.JSONDecodeError:
                        print(
                            f"Warning: Could not decode JSON line: {json_data_str}")
                        # Continue processing other lines if possible
                    except Exception as e:
                        print(
                            f"Warning: Error processing chunk: {e} - Line: {json_data_str}"
                        )
                        # Continue if possible, or potentially set error message

        # Sometimes usage is sent in the very last chunk before [DONE]
        # Re-check last chunk if totals are still missing (less common now with explicit usage object)
        if (
            (total_completion_tokens is None or total_prompt_tokens is None)
            and "chunk" in locals()
            and (usage := chunk.get("usage"))
        ):
            total_prompt_tokens = usage.get(
                "prompt_tokens", total_prompt_tokens)
            total_completion_tokens = usage.get(
                "completion_tokens", total_completion_tokens
            )

    except requests.exceptions.RequestException as e:
        error_message = f"Request failed: {e}"
        if start_time:  # Record time even if request fails after starting
            last_token_time = time.perf_counter()
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        if start_time:  # Record time even if error occurs after starting
            last_token_time = time.perf_counter()

    # Calculate metrics
    ttft = (first_token_time -
            start_time) if first_token_time and start_time else None
    ttlt = (last_token_time - start_time) if last_token_time and start_time else None
    # Simple token calculation based on response if not provided (less accurate)
    # if total_completion_tokens is None and completion_text:
    #     total_completion_tokens = len(completion_text.split()) # Very rough estimate

    return {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "prompt": prompt,
        "ttft_seconds": ttft,
        "ttlt_seconds": ttlt,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "completion_text": completion_text,
        "finish_reason": finish_reason,
        "error": error_message,
    }


def run_experiments(models: list[str], prompts: list[str]) -> list[dict]:
    """
    Runs performance measurements for all combinations of models and prompts.

    Args:
        models: A list of model names to test.
        prompts: A list of prompts to test.

    Returns:
        A list of dictionaries, each containing the results for one experiment run.
    """
    results = []
    total_runs = len(models) * len(prompts)
    current_run = 0

    for prompt in prompts:
        for model in models:
            current_run += 1
            print(f"--- Running test {current_run}/{total_runs} ---")
            print(f"Model: {model}")
            print(f"Prompt: '{prompt[:50]}...'")  # Print truncated prompt

            result = measure_llm_performance(model, prompt)
            results.append(result)

            # Print summary for this run
            if result["error"]:
                print(f"Status: FAILED ({result['error']})")
            else:
                print(f"Status: SUCCESS")
                print(
                    f"  TTFT: {result['ttft_seconds']:.4f} s"
                    if result["ttft_seconds"] is not None
                    else "  TTFT: N/A"
                )
                print(
                    f"  TTLT: {result['ttlt_seconds']:.4f} s"
                    if result["ttlt_seconds"] is not None
                    else "  TTLT: N/A"
                )
                print(
                    f"  Completion Tokens: {result['total_completion_tokens']}")
                print(f"  Finish Reason: {result['finish_reason']}")
            print("-" * 20)

            # Add a delay to avoid hitting rate limits
            if current_run < total_runs:
                print(
                    f"Waiting {DELAY_BETWEEN_REQUESTS}s before next request...")
                time.sleep(DELAY_BETWEEN_REQUESTS)

    return results


def save_results_to_csv(results: list[dict], filename: str):
    """Saves the results to a CSV file."""
    if not results:
        print("No results to save.")
        return

    # Use the keys from the first result as headers (assuming consistency)
    fieldnames = results[0].keys()

    try:
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results successfully saved to {filename}")
    except IOError as e:
        print(f"Error writing to CSV file {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Define your models and prompts here ---
    models_to_test = [
        "google/gemini-2.5-pro-exp-03-25:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/learnlm-1.5-pro-experimental:free",
        # "google/gemma-3-1b-it:free",
        # "google/gemma-3-27b-it:free",
        # "huggingfaceh4/zephyr-7b-beta:free",
        # "meta-llama/llama-3.3-70b-instruct:free",
        # "google/gemini-2.0-pro-exp-02-05:free",
        # "nousresearch/deephermes-3-llama-3-8b-preview:free",
        # "google/gemma-3-27b-it:free",
        # "deepseek/deepseek-chat-v3-0324:free",
        # "mistralai/mistral-small-3.1-24b-instruct:free",
        # "huggingfaceh4/zephyr-7b-beta:free",
    ]

    prompts_to_test = [
        """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Please return this code in OCaml""",
        """Describe the colour yellow as if it were a texture""",
        """
I need you to create a timetable for me given the following facts: my plane takes off at 6:30am. I need to be at the airport 1h before take off. it will take 45mins to get to the airport. I need 1h to get dressed and have breakfast before we leave. The plan should include when to wake up and the time I need to get into the vehicle to get to the airport in time for my 6:30am flight , think through this step by step.
""",
        """
Give me a list of 10 natural numbers, such that at least one is prime, at least 6 are odd, at least 2 are powers of 2, and such that the 10 numbers have at minimum 25 digits between them.
""",
        """
I’d like you to provide an in-depth analysis of the potential socioeconomic impacts of widespread adoption of renewable energy sources, such as solar, wind, and hydropower, in a mid-sized developing country over the next two decades. Please consider the following aspects in your response: First, evaluate how the shift from fossil fuels to renewables might affect employment, including the types of jobs created (e.g., manufacturing of solar panels, wind turbine installation, hydropower plant maintenance) and those potentially lost (e.g., coal mining, oil extraction, and fossil fuel power plant operations). Assess the skill levels required for these new roles, the potential for retraining programs to transition workers from declining industries, and the geographic distribution of job gains versus losses, particularly in rural versus urban areas. Second, analyze the economic implications, such as changes in energy costs for households and businesses, impacts on GDP growth, and the role of foreign investment or domestic funding in scaling renewable infrastructure. Consider how reduced reliance on imported fossil fuels might affect trade balances, currency stability, and national energy security, as well as the potential for energy export opportunities if surplus renewable capacity is developed. Third, explore the effects on energy access and affordability, especially for underserved populations, including how decentralized renewable systems (e.g., off-grid solar) could bridge gaps in electrification and reduce energy poverty, and whether this might exacerbate or alleviate existing inequalities in urban slums versus remote villages. Fourth, examine the social dimensions, such as community displacement due to large-scale projects like hydropower dams, shifts in gender roles if women gain employment in renewable sectors, and public health improvements from reduced air pollution previously caused by coal or diesel generators. Fifth, consider the environmental justice perspective, evaluating how the transition might address or worsen disparities in exposure to environmental hazards (e.g., mining waste versus cleaner energy zones) and the involvement of indigenous or marginalized groups in decision-making processes for renewable projects. Sixth, assess the infrastructure challenges, including the need for grid upgrades, battery storage deployment, and land use conflicts, and how these might strain government budgets or require public-private partnerships. Finally, integrate these factors into a cohesive narrative, projecting how they could reshape the country’s socioeconomic landscape by 2045, including potential risks like economic dependence on a single renewable type (e.g., hydropower vulnerability to drought) or benefits like positioning the country as a regional green energy leader. Use a hypothetical mid-sized developing country (e.g., population 20-50 million, mixed urban-rural economy, moderate fossil fuel reliance) as a case study, and support your analysis with plausible quantitative estimates (e.g., job creation numbers, cost reductions) and qualitative insights (e.g., social cohesion effects), drawing on global trends and real-world examples where applicable, such as Kenya’s geothermal success or Vietnam’s solar expansion, while adapting them to the developing country context.

""",
    ]

    output_csv_filename = "llm_performance_benchmark.csv"

    # Check for API Key before starting
    print("Starting LLM benchmark...")
    all_results = run_experiments(models_to_test, prompts_to_test)
    save_results_to_csv(all_results, output_csv_filename)
    print("Benchmark finished.")
