# Helper utilities for HR Attrition project
def save_json(path, obj):
    import json
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
