import reasoning_gym as rg

def list_tasks_in_module(module_name):
    module = getattr(rg, module_name)
    print(f"\nTasks in module '{module_name}':")
    for attr in dir(module):
        if not attr.startswith("_"):
            task = getattr(module, attr)
            if callable(task):
                print(f"  - {attr}")

if __name__ == "__main__":
    list_tasks_in_module("arithmetic")
    list_tasks_in_module("logic")
