
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.7):
    """
    Leaderboard-optimized hybrid routing.
    Keeps simple + confident cases on-device.
    Sends complex / low-confidence / malformed cases to cloud.
    """

    local = generate_cactus(messages, tools)

    # --- Basic safety ---
    local_calls = local.get("function_calls", [])
    local_conf = local.get("confidence", 0)
    local_time = local.get("total_time_ms", 0)

    valid_tool_names = {t["name"] for t in tools}

    # ---- Condition 1: No function calls → fallback ----
    if not local_calls:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback: no_calls)"
        cloud["local_confidence"] = local_conf
        cloud["total_time_ms"] += local_time
        return cloud

    # ---- Condition 2: Multiple function calls → fallback ----
    if len(local_calls) > 1:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback: multi_call)"
        cloud["local_confidence"] = local_conf
        cloud["total_time_ms"] += local_time
        return cloud

    call = local_calls[0]

    # ---- Condition 3: Invalid tool name ----
    if call["name"] not in valid_tool_names:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback: invalid_tool)"
        cloud["local_confidence"] = local_conf
        cloud["total_time_ms"] += local_time
        return cloud

    # ---- Condition 4: Missing required arguments ----
    required_params = {
        t["name"]: set(t["parameters"].get("required", []))
        for t in tools
    }

    required = required_params.get(call["name"], set())
    provided = set(call.get("arguments", {}).keys())

    if not required.issubset(provided):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback: missing_args)"
        cloud["local_confidence"] = local_conf
        cloud["total_time_ms"] += local_time
        return cloud

    # ---- Condition 5: Complex user query ----
    user_text = " ".join(
        m["content"] for m in messages if m["role"] == "user"
    ).lower()

    complex_keywords = [
        " and ",
        " or ",
        " if ",
        "compare",
        "difference",
        "between",
        "calculate",
        "then",
        "after",
    ]

    is_complex = any(k in user_text for k in complex_keywords)

    if is_complex:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback: complex_query)"
        cloud["local_confidence"] = local_conf
        cloud["total_time_ms"] += local_time
        return cloud

    # ---- Condition 6: Confidence check ----
    if local_conf < confidence_threshold:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback: low_conf)"
        cloud["local_confidence"] = local_conf
        cloud["total_time_ms"] += local_time
        return cloud

    # ---- Otherwise stay on-device ----
    local["source"] = "on-device"
    return local

def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")

############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
