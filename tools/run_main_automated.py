import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import builtins
import src.main as app_main

# Monkeypatch input to simulate interactive answers
answers = iter(["TSLA", "4 months"])
_builtin_input = builtins.input

def fake_input(prompt=""):
    try:
        val = next(answers)
        print(prompt + val)
        return val
    except StopIteration:
        return _builtin_input(prompt)

builtins.input = fake_input

# Run main
try:
    app_main.main()
except SystemExit as e:
    print('Exited with', e)
except Exception:
    import traceback
    traceback.print_exc()
finally:
    builtins.input = _builtin_input
