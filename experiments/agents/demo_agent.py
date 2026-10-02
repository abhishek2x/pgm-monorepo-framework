import json
import sys


def main():
    request = json.load(sys.stdin)
    task = request["task"]
    response = {
        "steps": [
            {"action": "read project README and identify package entry points"}
        ],
        "messages": [],
        "files_touched": [],
        "token_usage": {},
        "tool_call_count": 1,
        "patch": "",
    }
    if not task.get("prompt"):
        raise ValueError("The task prompt is required.")
    print(json.dumps(response))


if __name__ == "__main__":
    main()
