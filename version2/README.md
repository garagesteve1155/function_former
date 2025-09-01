#FunctionFormer2

FunctionFormer2 is a self-healing autocoder that turns a high-level goal into a working Python script.
It plans an outline, generates the code section-by-section, runs pre-flight fixes, executes the program in a sandboxed subprocess, and iteratively repairs issues using your feedback or traceback errors.


This has 4 versions:

ff2_chatgpt.py — build targeting OpenAI ChatGPT models through the online API - Runs on ANY computer with internet.

ff2_mistral7b.py — local, offline build targeting Mistral 7B Instruct v0.3 - Requires around 10GB of RAM, no Graphics Card required. Runs faster if your computer has a Graphics Card, especially if the Graphics Card can fit everything instead of splitting the model between GPU and RAM. 

ff2_qwen14b.py - local, offline build targeting Qwen2.5 14B Instruct - Requires around 25GB of RAM, no Graphics Card required. Like the Mistral version, the more of the model that you can fit on a graphics card, the faster it will run.

ff2_qwen32b.py - local, offline build targeting Qwen2.5 32B Instruct - Requires around 70GB of RAM, no Graphics Card required. Like the Mistral version, the more of the model that you can fit on a graphics card, the faster it will run.


⚠️ You’ll also need one helper module file placed next to the ff2_*.py file you run:

LLM_Reasoning_Engine.py (small “planning/critique” helper)

This can be found in the LLM_Reasoning_Engine repo on my Github page. Drop the py file in the same folder as ff2_***.py.



✨ What it does

Interview → Outline → Code → Critique → Revise: asks a few clarifying questions, produces a structured outline, then generates imports, globals, functions, and main in small passes. Every section is critiqued across multiple aspects (spec match, structure, dependencies, readiness, style) and rewritten if needed.

After script has been created - 

Pre-run QA sweep: repairs sections against the outline before any execution.

Run & auto-fix loop: executes your script in a separate Python process, analyzes tracebacks, picks the smallest fix target, and patches just that part.

Human-in-the-loop: after each run, users also have the option to provide feedback before the next revision and it will apply one minimal change.

Snapshots: saves versions under ./script_snapshots/ so you can diff the evolution.


⚙️ Setup

Download your preferred launcher file (The ff2_***.py files).

Put LLM_Reasoning_Engine.py in the same folder as the launcher you plan to run (Can be found in the LLM_Reasoning_Engine repo on this same Github).


(Mistral Setup)
Download ALL of the files for Mistral 7B Instruct v0.3 to a local folder, e.g.:

C:/models/mistral-7b-instruct-v0.3


In a code editor, open ff2_mistral7b.py and set:

BASE_MODEL_PATH = "C:/models/mistral-7b-instruct-v0.3" (Or wherever your model folder is saved)


That path should contain all of the downloaded files.


(Qwen Setup)
Download ALL of the files for Qwen2.5 14B Instruct or Qwen2.5 32B Instruct to a local folder, e.g.:

C:/models/qwen2-5_14b_instruct
C:/models/qwen2-5_32b_instruct

In a code editor, open ff2_qwen14b.py or ff2_qwen32b.py and set:

BASE_MODEL_PATH = "C:/models/qwen2-5_14b_instruct"
or
BASE_MODEL_PATH = "C:/models/qwen2-5_32b_instruct"
(Or wherever your model folder is saved)


That path should contain all of the downloaded files.



OpenAI / ChatGPT build

For ff2_chatgpt.py:

In ff2_chatgpt.py, pick the model name you want to use (e.g., gpt-4o), and then provide your API key.




The app opens a Tkinter GUI with three columns:

Outline (editable) — live outline that seeds each generation pass

Script (editable) — the code being built & revised

Goal / Status — your current goal, plus live logs

Flow:

Enter your overall goal (e.g., “A Tkinter CSV deduper with preview & export”).

Answer the short interview (Ctrl+Enter to submit).

Review & confirm the outline (a popup will show it).

Watch it generate imports → globals → functions → main.

Automated pre-run QA sweep runs; then it asks permission to execute your script.

After each run, it will automatically fix any traceback errors. Users also have the option to give feedback or tell it to end testing.

Shortcuts: Ctrl+Enter submits text in popups.



🧩 Troubleshooting

Model won’t fit on my computer

Use the Mistral version (Requires only 10gb of ram) or use the ChatGPT version.


No GUI / Tkinter error

Install tk: sudo apt install python3-tk (Linux). On macOS/Windows, use the official Python.org installer.


Feedback is ignored

It applies one minimal change per run. Keep feedback short and specific (e.g., “rename foo to bar in main”, “call init_ui() before load_config()”). If there was a traceback error, best option *usually* is to leave the feedback blank and let it figure out how to fix the error on its own unless it gets into an endless loop.



🔒 Safety & execution

This tool executes generated Python in a subprocess you control. Only run it on projects and machines you trust. Review code in the Script pane before authorizing the first run.




⭐ Support

If you find this useful, star the repo and please consider becoming a member of the Patreon.
You can also browse my other projects here: github.com/garagesteve1155.
