import os
import time
import json
import uuid
from datetime import datetime, timezone
from google import genai
from google.genai import types

# Load GEMINI_API_KEY from a local .env file (not committed — see .env.example)
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not set. Copy .env.example to .env and fill in your key."
    )

# 1. Setup Local Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
output_dir = os.path.join(REPO_ROOT, "alphas", "raw")
os.makedirs(output_dir, exist_ok=True)

# 2. Initialize the Client
client = genai.Client(api_key=api_key)
model_name = 'gemini-3.5-flash'
temperature = 0.7

system_instruction = """
You are a quantitative researcher and algorithmic trading system developer specialising in US Equities. Your objective is to design one candidate quantitative trading alpha utilizing exclusively daily OHLCV market data.

The alpha must use only the following daily market variables:

open
high
low
close
volume
returns

You must output your response adhering strictly to the 5-section schema below. Do not include conversational filler, introductions, or conclusions. 

Prioritise internal consistency, economic plausibility, reproducibility, and precise implementation over apparent predictive performance. The alpha should be fully specified and reproducible without requiring additional assumptions.

# Section 1: Alpha Specification
Provide a structured summary containing:
- Formula Name: (Descriptive name)
- Holding Period: (Expected holding time in days)
- Orientation: (Long/Short or Long-only)
- Parameters: (List any lookback windows or thresholds)
- Required Inputs: (Must be restricted to OHLCV and Returns)

# Section 2: Mathematical Formulation
Provide the exact, step-by-step mathematical formulation of the alpha signal. You must use LaTeX for all mathematical notation. Clearly define all variables.
Avoid introducing unnecessary mathematical complexity solely to increase perceived novelty.

Prefer simple, interpretable constructions when they adequately represent the proposed market mechanism.

# Section 3: Causal Rationale
Provide a concise 3–5 sentence explanation linking the mathematical signal to a plausible economic or behavioural mechanism.

Do not justify the alpha using historical performance or hypothetical backtest results.

# Section 4: Structural Bounds
State explicit constraints regarding the execution of this strategy. You must confirm that the data relies solely on Point-in-Time (PIT) OHLCV data. Explicitly state the execution timing rule: signals generated at the market close on day `t` assume execution at the open or close of day `t+1`.

# Section 5: Python Implementation
Provide a fully vectorized Python code block using `pandas` that can be executed directly without defining custom helper functions or classes. 
Technical constraints:
- Wrap your logic inside a single function named `def generate_alpha(df):`
- The input is a single `pandas` DataFrame named `df` indexed by a MultiIndex of `(date, symbol)`.
- Available columns are strictly: `open`, `high`, `low`, `close`, `volume`, `returns`.
- Use `df.groupby(level='symbol')` for time-series rolling operations.
- Use `df.groupby(level='date')` for cross-sectional ranking or z-scoring.
- The function must return exactly one `pandas.Series` indexed identically to `df`.
- No future look-ahead bias is permitted; operations on row `t` must only use data up to `t`. Do not use `.shift(-1)`.

Output only a single executable Python code block.
No explanatory text should appear within Section 5. The final expression must evaluate to a single pandas. Series aligned to df. Do not include backtesting code, portfolio construction, transaction cost modelling, plotting, or performance evaluation.
"""

user_prompts = [
    "Generate one candidate quantitative trading alpha. Follow all requirements specified in the system prompt.",
] * 20  # 20 identical prompts, one alpha generated per call

print(f"Starting generation loop for {len(user_prompts)} alphas...")

# 3. The Generation Loop
for i, prompt in enumerate(user_prompts):
    alpha_id = f"alpha_{uuid.uuid4().hex[:8]}"
    print(f"[{i+1}/{len(user_prompts)}] Generating {alpha_id}...")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature
            )
        )

        # 4. Construct the Metadata
        metadata = {
            "alpha_id": alpha_id,
            "model": model_name,
            "provider": "Google",
            "theme": "Unconstrained Generation",
            "system_prompt_version": "v1.0",
            "user_prompt_version": "v1.0",
            "temperature": temperature,
            "seed": None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }

        # 5. Combine metadata and response
        payload = {
            "metadata": metadata,
            "raw_response": response.text
        }

        # 6. Save to disk
        file_path = os.path.join(output_dir, f"{alpha_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        print(f"   -> Saved successfully to {file_path}")

    except Exception as e:
        print(f"   -> ERROR generating {alpha_id}: {e}")

    # 7. Respect Free Tier Rate Limits
    if i < len(user_prompts) - 1:
        print("   -> Sleeping for 35 seconds to respect rate limits...\n")
        time.sleep(35)

print("\nGeneration pipeline complete!")
