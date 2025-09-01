import os, torch, re, threading, time, subprocess, sys, json, tempfile, tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import io
import tokenize
import LLM_Reasoning_Engine as rengine
import ast

def has_function(script_text: str, name: str) -> bool:
    try:
        tree = ast.parse(script_text)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body)

# === Configuration ===
BASE_MODEL_PATH = "C:/Path/To/Your/Model/Folder/mistral_7b_instruct_v0-3"

# === CUDA + model init (with Overload fallback) ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

# Tokenizer is always available
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

model = None
# === Snapshots ===
SNAPSHOT_ROOT = os.path.join(os.path.dirname(__file__), "script_snapshots")

def _slugify_for_fs(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip()).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if not s:
        s = "goal"
    return s[:maxlen]

def _init_hf_model():
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=quant_cfg,
        device_map="auto",
        offload_state_dict=True,
    ).eval()

model = _init_hf_model()

# === Framework Mode (templates with outline + code) ===
FRAMEWORK_MODE: bool = False
FRAMEWORK_INFO: dict | None = None
FRAMEWORKS_DIR = os.path.join(os.path.dirname(__file__), "frameworks")

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _parse_framework_blob(blob: str) -> tuple[str, str]:
    """
    Accepts either JSON: {"outline":"...","code":"..."} or text with OUTLINE/CODE markers.
    Returns (outline_text, code_text). Raises ValueError if not parseable.
    """
    s = blob.strip()
    # JSON first
    try:
        data = json.loads(s)
        if isinstance(data, dict) and "outline" in data and "code" in data:
            return str(data["outline"]).strip(), str(data["code"]).strip()
    except Exception:
        pass

    # Marker formats (robust): === OUTLINE === ... === CODE === ...
    # or lines starting with "OUTLINE:" and "CODE:" on their own.
    m = re.search(r"(?s)^\s*(?:=+\s*OUTLINE\s*=+|OUTLINE\s*:)\s*(.*?)\s*(?:=+\s*CODE\s*=+|CODE\s*:)\s*(.+)\s*$", s, flags=re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Fallback: try simple split tokens
    parts = re.split(r"(?i)^\s*CODE\s*:\s*$", s, maxsplit=1, flags=re.M)
    if len(parts) == 2:
        outline = re.sub(r"(?i)^\s*OUTLINE\s*:\s*$", "", parts[0], flags=re.M).strip()
        code = parts[1].strip()
        if outline and code:
            return outline, code

    raise ValueError("Framework file not in a recognized format. Use JSON with {'outline','code'} or OUTLINE/CODE markers.")

def discover_frameworks() -> list[dict]:
    """
    Scans ./frameworks for candidate files.
    Accepts: .json, .txt, .md, .pyfw (any text we can parse).
    Returns: [{"name","path","outline","code"}]
    """
    items: list[dict] = []
    if not os.path.isdir(FRAMEWORKS_DIR):
        return items
    for name in os.listdir(FRAMEWORKS_DIR):
        if not any(name.lower().endswith(ext) for ext in (".json", ".txt")):
            continue
        p = os.path.join(FRAMEWORKS_DIR, name)
        try:
            outline, code = _parse_framework_blob(_read_text(p))
            items.append({
                "name": os.path.splitext(name)[0],
                "path": p,
                "outline": outline,
                "code": code
            })
        except Exception:
            continue
    return items

# === Helpers ===
CHAT_HISTORY = []   # (prompt, final_answer) pairs – ONLY these go in history
SCRIPT_LINES = []   # accumulated script, section-by-section (we still store as lines)
GOAL_SPEC = None    # set once from the user's first prompt
FULL_OUTLINE = None # global outline text (machine-readable-ish), included in every subsequent prompt

SECTIONS_PLAN: list[tuple] = []  # e.g., [("imports", [...]), ("globals", [...]), ("function", {"name": "...", "desc": "..."}), ("main", "...")]
SECTION_INDEX: int = 0           # index of next section to generate

def auto_fix_param_shadowing_with_globals(code: str) -> str:
    """
    If a function declares 'global foo' or 'nonlocal foo' *and* also has a parameter named 'foo',
    Python raises: "SyntaxError: name 'foo' is parameter and global".
    This fixes it by renaming the parameter(s) to a safe name (e.g. foo_param) and updating uses
    in the function body (but NOT the 'global foo' statement nor attribute accesses like obj.foo).

    Assumes `code` contains exactly one top-level function definition (your generator's contract).
    Safe no-op if it can't confidently parse things.
    """
    try:
        # --- 1) Find header (spanning multiple lines) and grab the parameter list ---
        # Match: def name ( <params> ) :
        m = re.search(r'(?ms)^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\((.*?)\)\s*:', code)
        if not m:
            return code  # not a simple single function; leave unchanged

        params_span = (m.start(1), m.end(1))
        header_colon_end = m.end()  # position just after the ':' at end of header
        params_src = m.group(1)

        # Split params at top-level commas (tolerate newlines/annotations/defaults)
        chunks = []
        depth = 0
        buf = []
        for ch in params_src:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                chunks.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        if buf:
            chunks.append("".join(buf))

        def param_name_from_chunk(chunk: str) -> str | None:
            s = chunk.strip()
            if not s:
                return None
            # Skip bare * or /
            if s in {"*", "/"}:
                return None
            # *args / **kwargs
            if s.startswith("**"):
                s = s[2:]
            elif s.startswith("*"):
                s = s[1:]
            # Trim annotation ': ...' and default '= ...'
            s = s.split(":", 1)[0]
            s = s.split("=", 1)[0]
            s = s.strip()
            return s if re.match(r"^[A-Za-z_]\w*$", s) else None

        param_names = [p for p in (param_name_from_chunk(c) for c in chunks) if p]

        # --- 2) Collect globals/nonlocals in the body ---
        body_src = code[header_colon_end:]  # from colon to end
        g_names: set[str] = set()
        for gm in re.finditer(r'(?m)^\s*(?:global|nonlocal)\s+([A-Za-z0-9_,\s]+)', body_src):
            for nm in re.split(r'\s*,\s*', gm.group(1).strip()):
                if re.match(r'^[A-Za-z_]\w*$', nm):
                    g_names.add(nm)

        conflicts = [n for n in param_names if n in g_names]
        if not conflicts:
            return code  # nothing to fix

        # --- 3) Build rename map: old -> new (avoid collisions) ---
        used = set(param_names) | g_names
        rename: dict[str, str] = {}
        for old in conflicts:
            base = f"{old}_param"
            new = base
            i = 2
            while new in used:
                new = f"{base}{i}"
                i += 1
            rename[old] = new
            used.add(new)

        # --- 4) Rewrite the parameter list in the header only ---
        def replace_in_params(text: str) -> str:
            out = text
            for old, new in rename.items():
                # Replace bare identifier tokens in the header param list
                out = re.sub(rf'\b{re.escape(old)}\b', new, out)
            return out

        new_params_src = replace_in_params(params_src)
        new_header = code[:params_span[0]] + new_params_src + code[params_span[1]:header_colon_end]

        # --- 5) Token-wise rewrite of the body: change NAME tokens that match old param
        #         but NOT when preceded by '.' (obj.old) and NOT in 'global/nonlocal' lines.
        def rewrite_body(body: str) -> str:
            sio = io.StringIO(body)
            tokens = list(tokenize.generate_tokens(sio.readline))
            new_tokens = []
            prev_sig_tok = None
            in_global_line = False  # we're inside a "global ..." or "nonlocal ..." statement on this logical line

            for tok in tokens:
                tok_type, tok_str, start, end, line = tok

                # Track line context: when we encounter a NEWLINE/NL, reset the global-decl flag
                if tok_type in (tokenize.NEWLINE, tokenize.NL):
                    in_global_line = False

                # Entering a 'global'/'nonlocal' statement
                if tok_type == tokenize.NAME and tok_str in ("global", "nonlocal"):
                    in_global_line = True
                    new_tokens.append(tok)
                    prev_sig_tok = tok if tok_type != tokenize.NL else prev_sig_tok
                    continue

                # Candidate rename?
                if tok_type == tokenize.NAME and tok_str in rename and not in_global_line:
                    # If previous significant token was a dot, this is attribute 'obj.name' → don't rename
                    if prev_sig_tok and prev_sig_tok.type == tokenize.OP and prev_sig_tok.string == ".":
                        new_tokens.append(tok)  # keep attribute name
                    else:
                        # Replace with the new param name
                        tok = tokenize.TokenInfo(tok_type, rename[tok_str], start, end, line)
                        new_tokens.append(tok)
                else:
                    new_tokens.append(tok)

                # Update previous significant token (skip whitespace/indent/dedent)
                if tok_type not in (tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING):
                    prev_sig_tok = tok

            return tokenize.untokenize(new_tokens)

        new_body = rewrite_body(body_src)

        return new_header + new_body
    except Exception:
        # On any parsing hiccup, do nothing (fail-safe)
        return code
def _critique_is_ok(raw: str) -> bool:
    """
    Return True if the reviewer signaled OK, even if they added extra commentary.
    Accept if:
      • any line is exactly 'OK' (case-insensitive, optional punctuation), or
      • the very first non-empty token starts with 'OK'
    """
    if not raw:
        return False
    text = raw.strip()
    # Any standalone OK line?
    for ln in text.splitlines():
        if re.fullmatch(r"\s*ok[.!]?\s*", ln, flags=re.I):
            return True
    # Or starts with OK and then commentary
    if re.match(r"^\s*ok\b", text, flags=re.I):
        return True
    return False

def clean(txt: str) -> str:
    return re.sub(r"(# ?\d+)+", "", txt).strip()

def inst_first(msg: str) -> str:
    return f"<s>[INST] {msg.strip()} [/INST] "

def inst_later(msg: str) -> str:
    return f"[INST] {msg.strip()} [/INST] "


def join_script(lines: list[str]) -> str:
    return "\n".join(lines)

def wait_for_enter(reason: str) -> None:
    """
    Pause generation on any failed check/critique so you can inspect the console.
    Press Enter to resume the process.
    """
    try:
        print(f"\n[PAUSED] {reason}\nPress Enter to continue...", flush=True)
        #input()
    except Exception:
        # In environments without a real stdin, just continue.
        pass

# Strings/patterns the LLM sometimes appends that should be stripped from code blocks
STRIP_MARKERS: list[str] = [
    "(END OF SCRIPT)",
    # Add more exact strings here as needed, e.g. "# GLOBALS/CONFIG", "# IMPORTS", etc.
]
def _normalize_ws(s: str) -> str:
    """
    Normalize whitespace:
    - convert CRLF/CR to LF,
    - expand tabs to 4 spaces,
    - strip trailing spaces on each line.
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.expandtabs(4)
    s = re.sub(r"[ \t]+$", "", s, flags=re.M)
    return s

def _ensure_col0(line: str) -> str:
    """Remove all leading whitespace."""
    return re.sub(r"^\s+", "", line)

def auto_fix_indentation(code: str, section_kind: str, required_func: str | None = None) -> str:
    """
    Best-effort indentation fixer tailored to our sectioning rules:
    - IMPORTS/GLOBALS: force column 0.
    - FUNCTION: ensure 'def <required_func>:' at col 0; body lines are indented by exactly 4 spaces.
    - MAIN: ensure 'def main:' and the __main__ guard are at col 0; their bodies use 4 spaces.
    Always uses spaces (4-wide); removes tabs; trims trailing spaces.
    """
    code = _normalize_ws(code)
    lines = code.splitlines()

    if section_kind in {"imports", "globals"}:
        return "\n".join(ln.lstrip() for ln in lines).strip("\n")

    if section_kind == "function":
        # Find the function header; prefer exact match first
        pat_exact = rf"^\s*def\s+{re.escape(required_func or '')}\s*\("
        idx = None
        for i, ln in enumerate(lines):
            if re.match(pat_exact, ln):
                idx = i
                break
        if idx is None:
            # fallback to first def
            for i, ln in enumerate(lines):
                if re.match(r"^\s*def\s+[A-Za-z_]\w*\s*\(", ln):
                    idx = i
                    break
        if idx is None:
            return "\n".join(lines).strip("\n")  # nothing to fix

        # header at col 0
        lines[idx] = _ensure_col0(lines[idx])

        # body: until next top-level def/class/decorator/guard
        j = idx + 1
        while j < len(lines):
            ln = lines[j]
            if not ln.strip():
                j += 1
                continue
            if re.match(r"^\s*(def|class|@|if\s+__name__\s*==)", ln):
                break
            if not ln.startswith("    "):  # ensure 4
                lines[j] = "    " + ln.lstrip()
            j += 1

        return "\n".join(lines).strip("\n")

    if section_kind == "main":
        # def main at col 0
        for i, ln in enumerate(lines):
            if re.match(r"^\s*def\s+main\s*\(", ln):
                lines[i] = _ensure_col0(ln)
                # body until guard
                k = i + 1
                while k < len(lines):
                    ln2 = lines[k]
                    if re.match(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", ln2):
                        break
                    if ln2.strip() and not ln2.startswith("    "):
                        lines[k] = "    " + ln2.lstrip()
                    k += 1
                break

        # ensure guard at col 0 and body indented 4
        for i, ln in enumerate(lines):
            if re.match(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", ln):
                lines[i] = _ensure_col0(ln)
                # next non-empty line becomes exactly 4 spaces
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    lines[j] = "    " + lines[j].lstrip()
                break

        return "\n".join(lines).strip("\n")

    return "\n".join(lines).strip("\n")

def sanitize_candidate(code: str, section_kind: str | None = None) -> str:
    """Remove junk markers and normalize indentation/whitespace. Also auto-fixes outline-y GLOBALS."""
    if not code:
        return code

    cleaned = code
    for m in STRIP_MARKERS:
        cleaned = cleaned.replace(m, "")
    cleaned = re.sub(r"^\s*\((?:END|BEGIN)\b[^)]*\)\s*$", "", cleaned, flags=re.I | re.M)
    cleaned = re.sub(r"^\s*\[(?:END|BEGIN)\b[^\]]*\]\s*$", "", cleaned, flags=re.I | re.M)
    cleaned = _normalize_ws(cleaned)

    lines = [ln.rstrip() for ln in cleaned.splitlines()]

    # Imports: keep only import lines/ordering rules (existing behavior)
    if section_kind == "imports":
        lines = [ln.lstrip() for ln in lines if ln.strip()]
        future = [ln for ln in lines if ln.startswith("from __future__ import")]
        others = [ln for ln in lines if not ln.startswith("from __future__ import")]
        lines = future + others

    # Globals: convert outline bullets/headings into valid assignments to avoid compile failures
    elif section_kind == "globals":
        out_lines: list[str] = []
        for ln in lines:
            t = ln.strip()
            if not t:
                continue
            # Drop a "GLOBALS:" header line
            if re.match(r"^(#\s*)?globals\s*:?\s*$", t, flags=re.I):
                continue
            # Convert '- name - desc' or '- name' bullets into 'name = None  # desc'
            m = re.match(r"^[-*•]\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s*-\s*(.*))?$", t)
            if m:
                name = m.group(1)
                desc = (m.group(2) or "").strip()
                out_lines.append(f"{name} = None" + (f"  # {desc}" if desc else ""))
                continue
            # Convert a bare identifier line into 'name = None'
            m2 = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", t)
            if m2:
                out_lines.append(f"{m2.group(1)} = None")
                continue
            # Drop guidance like 'Initialize within MAIN'
            if re.search(r"initialize\s+.*main", t, flags=re.I):
                continue
            # Otherwise keep the line (e.g., real Python like `window = tkinter.Tk()`)
            out_lines.append(ln.lstrip())
        lines = out_lines

    else:
        lines = [ln for ln in lines]

    body = "\n".join(lines).strip("\n")

    req_func = None
    if section_kind == "function":
        # Determine the required function name
        m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", body, flags=re.M)
        req_func = m.group(1) if m else None

        if req_func:
            # Extract only the function block matching req_func
            pattern = rf"(?ms)^\s*def\s+{re.escape(req_func)}\s*\([^)]*\):.*?(?=^\s*def\s|\Z)"
            match = re.search(pattern, body)
            if match:
                body = match.group(0)

        # Fix param/global conflicts inside this single function
        body = auto_fix_param_shadowing_with_globals(body)

    return auto_fix_indentation(body, section_kind or "", required_func=req_func)


    # Final, section-aware indentation repair (enforce 4-space blocks, fix main guard)
    req_func = None
    if section_kind == "function":
        m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", body, flags=re.M)
        req_func = m.group(1) if m else None
    return auto_fix_indentation(body, section_kind or "", required_func=req_func)



def ensure_main_guard(code: str) -> str:
    """
    Ensure that a main() section includes:
        if __name__ == "__main__":
                main()
    - If guard exists but the call is missing, inserts the call right after the guard.
    - If guard is missing but def main(...) exists, appends the guard+call at top level.
    """
    text = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    guard_re = re.compile(r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*$')
    call_re  = re.compile(r'^\s*main\s*\(\s*\)\s*$')
    def_main_re = re.compile(r'^\s*(?:async\s+)?def\s+main\s*\(')

    has_def_main = any(def_main_re.match(ln) for ln in lines)

    # Find (last) guard line if present
    gidx = None
    for i, ln in enumerate(lines):
        if guard_re.match(ln):
            gidx = i

    if gidx is not None:
        # Ensure next non-empty line is the call
        j = gidx + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j >= len(lines) or not call_re.match(lines[j]):
            lines.insert(gidx + 1, "        main()")
        out = "\n".join(lines)
        return out if text.endswith("\n") else out
    elif has_def_main:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("        main()")
        out = "\n".join(lines)
        return out if text.endswith("\n") else out + "\n"

    return code






def extract_code_block(txt: str) -> str:
    """Return code inside the first fenced code block if present; otherwise return the whole text."""
    s = txt.strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", s, flags=re.I)
    return (m.group(1).strip() if m else s)
def extract_main_code(text: str) -> str:
    """
    Pull out exactly:
        def main(...):
            <indented body>
    
        if __name__ == "__main__":
            main()
    from a noisy LLM response (lists, prose, etc.). Returns "" if not found.
    """
    s = text.replace("\r\n", "\n")
    # Prefer fenced code if present
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", s, flags=re.I)
    if m:
        s = m.group(1)

    # Kill obvious list/bullet lines the LLM likes to emit
    s = re.sub(r"(?m)^\s*(?:\d+\)|\d+\.\s*|[-*•])\s+.*$", "", s)

    # State-machine extraction: capture def main block, then the guard and the main() call
    lines = s.splitlines()
    out = []
    in_main = False
    main_indent = None
    saw_guard = False
    saw_call = False

    def_indent_re = re.compile(r"^(\s*)def\s+main\s*\([^)]*\)\s*:")
    guard_re = re.compile(r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*$')
    call_re = re.compile(r'^\s*main\s*\(\s*\)\s*$')

    i = 0
    while i < len(lines):
        ln = lines[i]
        if not in_main:
            m = def_indent_re.match(ln)
            if m:
                in_main = True
                main_indent = len(m.group(1))
                out.append(ln.rstrip())
        else:
            out.append(ln.rstrip())
            # We’re still in main body while current line is blank or more-indented than def line
            # Once we hit a non-indented (top-level) line, check for the guard.
            if ln.strip() == "" or (len(ln) > 0 and (len(ln) - len(ln.lstrip())) > main_indent):
                pass
            else:
                # We’ve hit top-level; do NOT consume this line yet
                out.pop()  # remove the line we just appended (it's top-level)
                break
        i += 1

    # Now scan top-level for guard + call
    while i < len(lines):
        ln = lines[i].rstrip()
        if guard_re.match(ln):
            out.append(ln)
            saw_guard = True
            # expect the next non-empty line to be main()
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and call_re.match(lines[j]):
                out.append(lines[j].rstrip())
                saw_call = True
            break
        i += 1

    snippet = "\n".join(out).strip()
    if in_main and saw_guard and saw_call:
        return snippet
    return ""

def parse_outline(outline_text: str) -> dict:
    """
    Parse a structured outline into sections:
      - imports: list[str]
      - globals: list[str]
      - functions: list[{"name": str, "desc": str}]
      - main: str

    Robust FUNCTION parsing:
    - Accepts headers like "1) create_gui():", "create_gui() - ...", "button_click(button) Handles ...".
    - Accepts bare snake_case names on their own line (desc on following bullet lines).
    - Avoids treating description bullets like "- Initializes ..." as function headers.
    """
    def extract_code_block(txt: str) -> str:
        m = re.search(r"```(?:\w+)?\s*([\s\S]*?)```", txt.strip())
        return (m.group(1).strip() if m else txt)

    txt = extract_code_block(outline_text) or outline_text
    txt = txt.replace("—", "-").replace("–", "-")

    sections = {"imports": [], "globals": [], "functions": [], "main": ""}
    current: str | None = None

    def strip_ticks_and_parens(token: str) -> str:
        t = token.strip()
        if t.startswith("`") and t.endswith("`"):
            t = t[1:-1].strip()
        # drop trailing () if present
        t = re.sub(r"\(\)\s*$", "", t)
        return t

    # --- Multi-line function accumulation state ---
    pending_fn_name: str | None = None
    pending_desc_parts: list[str] = []

    def flush_pending():
        nonlocal pending_fn_name, pending_desc_parts
        if pending_fn_name:
            name = strip_ticks_and_parens(pending_fn_name)
            desc = " ".join(p.strip(" -:") for p in pending_desc_parts if p.strip()).strip()
            if not desc:
                desc = "No description provided."
            sections["functions"].append({"name": name, "desc": desc})
            pending_fn_name = None
            pending_desc_parts = []

    # --- Patterns ---
    split_at_number = re.compile(r"(?=\b\d+\)\s*)")  # split "1) ... 2) ..." lines
    bullet_prefix = re.compile(r"^\s*[-*•]\s*")
    header_candidate = re.compile(
        r"""^\s*
            (?:(?P<num>\d+)\)?[.)]?\s*)?      # optional leading numbering like '1) ' or '1.'
            (?:[-*•]\s*)?                     # optional bullet
            `?(?P<name>[A-Za-z_][A-Za-z0-9_]*)`?
            (?P<after>\s*(?:\([^)]*\))?\s*(?::|-)?)
            \s*(?P<rest>.*)$
        """,
        re.X,
    )

    def looks_like_bare_func_name(name: str) -> bool:
        # Accept bare names that look like standard Python function names (snake_case / lowercase)
        return bool(re.match(r"^[a-z_][a-z0-9_]*$", name))

    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue

        # Section headers (IMPORTS / GLOBALS / FUNCTIONS / MAIN)
        hdr = line.upper().rstrip(":").strip("` ")
        if hdr in {"IMPORTS", "GLOBALS", "FUNCTIONS", "MAIN"}:
            if current == "functions":
                flush_pending()
            current = hdr.lower()
            continue

        if current == "imports":
            if line[0:1] in {"-", "•", "*"}:
                sections["imports"].append(line.lstrip("-•* ").strip())
            elif re.search(r"[A-Za-z0-9_`]", line):
                sections["imports"].append(line)

        elif current == "globals":
            if line[0:1] in {"-", "•", "*"}:
                sections["globals"].append(line.lstrip("-•* ").strip())
            elif re.search(r"[A-Za-z0-9_`]", line):
                sections["globals"].append(line)

        elif current == "functions":
            # Support both one-per-line and compact multiple-per-line formats
            chunks = split_at_number.split(line) if re.search(r"\b\d+\)\s*", line) else [line]
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

                m = header_candidate.match(chunk)
                if m:
                    name = m.group("name") or ""
                    after = m.group("after") or ""
                    rest = (m.group("rest") or "").strip()
                    has_num = m.group("num") is not None
                    has_paren = "(" in after and ")" in after
                    has_sep = (":" in after) or ("-" in after)
                    starts_with_bullet = bool(bullet_prefix.match(chunk))

                    # Accept as a function header only if there's a strong signal:
                    accept = False
                    if has_num or has_paren or has_sep:
                        accept = True
                    elif not starts_with_bullet and looks_like_bare_func_name(name):
                        # allow bare snake_case names like "create_gui" as headers
                        accept = True

                    if accept and name:
                        flush_pending()
                        pending_fn_name = name
                        if rest:
                            pending_desc_parts.append(rest)
                        continue

                # Not a header: treat as continuation/description if inside a function block
                if pending_fn_name:
                    if chunk[0:1] in {"-", "•", "*"}:
                        pending_desc_parts.append(chunk.lstrip("-•* ").strip())
                    else:
                        pending_desc_parts.append(chunk)

        elif current == "main":
            sections["main"] = (sections["main"] + " " + line).strip() if sections["main"] else line

    # End of text: flush any last pending function
    if current == "functions":
        flush_pending()

    return sections





def build_sections_plan(parsed: dict) -> list[tuple]:
    """Create the ordered plan of generation passes based on the parsed outline."""
    plan: list[tuple] = []
    plan.append(("imports", parsed.get("imports", [])))
    plan.append(("globals", parsed.get("globals", [])))
    for f in parsed.get("functions", []):
        plan.append(("function", f))
    plan.append(("main", parsed.get("main", "")))
    return plan

def get_full_outline(goal: str, progress_cb=None) -> str:
    """Generate a structured, detailed layout outline (not prose) for the entire script."""
    outline_prompt = (
        "You are an expert Python software architect.\n"
        "Given the user's overall goal, produce a COMPLETE, DETAILED LAYOUT OUTLINE for the Python script.\n"
        "HARD REQUIREMENTS:\n"
        "- DO NOT include any Python code.\n"
        "- DO NOT wrap names in backticks.\n"
        "- For function names, DO NOT include parentheses (use create_gui, not create_gui()).\n"
        "- Return ONLY the outline in EXACTLY the following format (headings + bulleted/numbered lists):\n\n"
        "IMPORTS:\n"
        "- <module or package>\n"
        "- <module or package>\n\n"
        "GLOBALS:\n"
        "- <NAME> - <purpose of this item in 2 sentences>\n"
        "- <NAME> - <purpose of this item in 2 sentences>\n\n"
        "FUNCTIONS:\n"
        "1) <function_name> - <what the function does in 2 sentences, including variables used and other functions called and any other logic info>\n"
        "2) <function_name> - <what the function does in 2 sentences, including variables used and other functions called and any other logic info>\n"
        "...\n\n"
        "MAIN:\n"
        "- <what main() does at a high level>\n\n"
        "This outline will be referenced in all subsequent prompts.\n\n"
        f"User's goal:\n{goal}\n"
    )
    print("\n\n\n\nOUTLINE PROMPT:\n\n" + str(outline_prompt))
    outline_text = llm(outline_prompt, max_new_tokens=4096, temperature=0.25)
    print("\n\n\n\nOUTLINE RESPONSE:\n\n" + str(outline_text))
    outline_text = outline_text.strip()
    if callable(progress_cb):
        progress_cb(outline_text, "initial")
    return outline_text



def _outline_section_indices(outline_text: str, section_name: str) -> tuple[int, int, int]:
    """
    Return (heading_idx, start_idx, end_idx) over line indices for the given section.
    heading_idx: index of the 'SECTION:' line
    start_idx:   first content line after the heading
    end_idx:     index of the next heading line or len(lines)
    If not found, returns (-1, -1, -1).
    """
    headers = ["IMPORTS:", "GLOBALS:", "FUNCTIONS:", "MAIN:"]
    lines = outline_text.splitlines(True)  # keep line endings
    # map header -> line index
    positions = {}
    for i, ln in enumerate(lines):
        up = ln.strip().upper()
        if up in headers:
            positions[up] = i
    key = f"{section_name.upper()}:"
    if key not in positions:
        return (-1, -1, -1)
    heading_idx = positions[key]
    # find next heading
    subsequent = [positions[h] for h in headers if h in positions and positions[h] > heading_idx]
    next_heading_idx = min(subsequent) if subsequent else len(lines)
    start_idx = heading_idx + 1
    end_idx = next_heading_idx
    return heading_idx, start_idx, end_idx


def get_outline_section_body(outline_text: str, section_name: str) -> str:
    """Return only the body text (without the 'SECTION:' line) for the given section."""
    heading_idx, start_idx, end_idx = _outline_section_indices(outline_text, section_name)
    if heading_idx == -1:
        return ""
    lines = outline_text.splitlines(True)
    return "".join(lines[start_idx:end_idx]).strip()


def replace_outline_section(outline_text: str, section_name: str, new_body: str) -> str:
    """
    Replace the body of a section (keeping the 'SECTION:' heading line intact).
    new_body should NOT include the heading itself.
    """
    heading_idx, start_idx, end_idx = _outline_section_indices(outline_text, section_name)
    if heading_idx == -1:
        return outline_text
    lines = outline_text.splitlines(True)
    # Ensure new body ends with a newline (to keep formatting stable)
    new_body_text = (new_body.rstrip() + "\n") if new_body.strip() else ""
    new_lines = lines[: start_idx] + ([new_body_text] if new_body_text else []) + lines[end_idx:]
    return "".join(new_lines)
def add_function_to_outline(func_name: str, func_desc: str) -> None:
    """
    Append a new numbered function entry to the FUNCTIONS section of FULL_OUTLINE,
    then rebuild SECTIONS_PLAN from the updated outline.
    """
    global FULL_OUTLINE, SECTIONS_PLAN
    outline = FULL_OUTLINE or ""
    funcs_body = get_outline_section_body(outline, "FUNCTIONS")

    # If FUNCTIONS is missing, create it minimally
    if not funcs_body:
        # Build a minimal skeleton preserving other sections
        imports = get_outline_section_body(outline, "IMPORTS")
        globals_body = get_outline_section_body(outline, "GLOBALS")
        main_body = get_outline_section_body(outline, "MAIN")
        rebuilt = []
        rebuilt.append("IMPORTS:\n" + (imports.rstrip() + "\n" if imports else ""))
        rebuilt.append("GLOBALS:\n" + (globals_body.rstrip() + "\n" if globals_body else ""))
        rebuilt.append("FUNCTIONS:\n")
        rebuilt.append("MAIN:\n" + (main_body.rstrip() + "\n" if main_body else ""))
        outline = "\n".join(rebuilt)
        FULL_OUTLINE = outline
        funcs_body = ""

    # Determine next ordinal
    existing_nums = re.findall(r"(?m)^\s*(\d+)\)\s+[A-Za-z_][A-Za-z0-9_]*\s*-", funcs_body or "")
    next_num = int(existing_nums[-1]) + 1 if existing_nums else 1
    new_line = f"{next_num}) {func_name} - {func_desc.strip()}"

    new_body = (funcs_body.rstrip() + ("\n" if funcs_body and not funcs_body.endswith("\n") else "") + new_line + "\n")
    FULL_OUTLINE = replace_outline_section(outline, "FUNCTIONS", new_body)

    # Re-parse to refresh the plan
    parsed = parse_outline(FULL_OUTLINE)
    SECTIONS_PLAN[:] = build_sections_plan(parsed)

def critique_outline_section(goal: str, full_outline_text: str, section_name: str, section_body: str) -> str:
    """
    Run a 5-aspect critique over ONE outline section ('IMPORTS' | 'GLOBALS' | 'FUNCTIONS' | 'MAIN').
    Returns 'OK' only if ALL aspects pass; otherwise bullet points combining all failures.
    """
    aspects: list[tuple[str, str]] = [
        (
            "Format & Purity",
            "- No Python code; outline-only wording.\n"
            "- Section uses required structure for its type:\n"
            "  IMPORTS: one module/package per bullet line starting with '- '.\n"
            "  GLOBALS: '- NAME - purpose ...' lines; NAME is UPPER_SNAKE_CASE; no code.\n"
            "  FUNCTIONS: numbered list (1), 2), ...); snake_case names; no parentheses; each item has a 1–2 sentence purpose.\n"
            "  MAIN: bullet list of high-level steps; no code."
        ),
        (
            "Coverage vs Goal",
            "- Section content is sufficient given the user's goal.\n"
            "- FUNCTIONS section includes all helpers needed to implement the goal."
        ),
        (
            "Consistency Across Sections",
            "- Imports/globals implied by functions are present in their sections (conceptually; no specific code required).\n"
            "- No contradictions or duplicates."
        ),
        (
            "Clarity & Specificity",
            "- Items are concrete and unambiguous (avoid vague verbs like 'handle stuff').\n"
            "- Each function description states important inputs/outputs and other functions it calls (at outline level)."
        ),
        (
            "Minimality",
            "- No redundant items; no unrelated scope creep; stick to essentials required by the goal."
        ),
    ]

    def run_single(name: str, criteria: str) -> str:
        prompt = (
            "You are a strict Python software architect.\n"
            f"ASPECT: {name}\n"
            "Decide ONLY for this aspect if the given OUTLINE SECTION is acceptable.\n"
            "Write ONLY 'OK' if acceptable; otherwise list succinct bullet points of issues.\n\n"
            "CRITERIA:\n"
            f"{criteria}\n\n"
            f"USER GOAL:\n{goal}\n\n"
            f"FULL OUTLINE (reference):\n{full_outline_text}\n\n"
            f"SECTION UNDER REVIEW: {section_name}\n"
            f"{section_body}\n"
        )
        print("\n\n\n\nCRITIQUE (OUTLINE / ASPECT: " + name + ") PROMPT:\n\n" + str(prompt))
        resp = llm(prompt, max_new_tokens=4096, temperature=0.2)
        print("\n\n\n\nCRITIQUE (OUTLINE / ASPECT: " + name + ") RESPONSE:\n\n" + str(resp))
        return resp


    failures: list[str] = []
    for n, c in aspects:
        raw = run_single(n, c)
        if _critique_is_ok(raw):
            continue
        lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
        if not lines:
            lines = ["Unspecified issue."]
        for ln in lines:
            failures.append(f"- [{section_name} / {n}] {ln}")

    return "OK" if not failures else "\n".join(failures)


def rewrite_outline_section(goal: str, full_outline_text: str, section_name: str, critique_notes: str, old_body: str) -> str:
    """
    Rewrite ONLY the specified outline section body to address ALL critique notes.
    Output MUST be ONLY the section body (no section header).
    May add functions when section_name=='FUNCTIONS'. Change only what's necessary.
    """
    if section_name.upper() == "FUNCTIONS":
        instructions = (
            "Output ONLY the numbered FUNCTIONS list (no 'FUNCTIONS:' header).\n"
            "- Use snake_case names WITHOUT parentheses.\n"
            "- Each entry format: '1) name - one or two precise sentences ...'\n"
            "- Add any missing helpers required by the goal.\n"
            "- Do not include code; outline text only.\n"
            "- Change only what is necessary to satisfy the critique."
        )
    elif section_name.upper() == "IMPORTS":
        instructions = (
            "Output ONLY the IMPORTS bullets (no 'IMPORTS:' header).\n"
            "- One module/package per line starting with '- '.\n"
            "- No code; outline text only.\n"
            "- Change only what is necessary to satisfy the critique."
        )
    elif section_name.upper() == "GLOBALS":
        instructions = (
            "Output ONLY the GLOBALS bullets (no 'GLOBALS:' header).\n"
            "- Each line: '- NAME - purpose ...' with NAME in UPPER_SNAKE_CASE.\n"
            "- No code; outline text only.\n"
            "- Change only what is necessary to satisfy the critique."
        )
    else:  # MAIN
        instructions = (
            "Output ONLY the MAIN bullets (no 'MAIN:' header).\n"
            "- High-level orchestration steps only.\n"
            "- No code; outline text only.\n"
            "- Change only what is necessary to satisfy the critique."
        )

    prompt = (
        "You are refining ONE outline section based on critique notes.\n"
        "HARD REQUIREMENTS:\n"
        f"{instructions}\n\n"
        f"USER GOAL:\n{goal}\n\n"
        f"FULL OUTLINE (reference):\n{full_outline_text}\n\n"
        "CRITIQUE NOTES (must address all):\n"
        f"{critique_notes}\n\n"
        "CURRENT SECTION BODY:\n"
        f"{old_body}\n"
    )
    print("\n\n\n\nREWRITE (OUTLINE SECTION: " + section_name + ") PROMPT:\n\n" + str(prompt))
    raw = llm(prompt, max_new_tokens=4096, temperature=0.25)
    print("\n\n\n\nREWRITE (OUTLINE SECTION: " + section_name + ") RESPONSE:\n\n" + str(raw))
    return extract_code_block(raw).strip()


def refine_outline(goal: str, outline_text: str, progress_cb=None) -> str:
    """
    Critique/repair the OUTLINE one section at a time.
    Fires progress_cb(updated_outline, stage) after each section is OK/rewritten.
    """
    sections = ["IMPORTS", "GLOBALS", "FUNCTIONS", "MAIN"]
    refined = outline_text

    if progress_cb:
        progress_cb(refined, "refine:start")

    for name in sections:
        body = get_outline_section_body(refined, name)
        critique = critique_outline_section(goal, refined, name, body)

        if _critique_is_ok(critique):
            if progress_cb:
                progress_cb(refined, f"{name}:OK")
            continue

        new_body = rewrite_outline_section(goal, refined, name, critique, body)
        if new_body.strip():
            refined = replace_outline_section(refined, name, new_body)
        if progress_cb:
            progress_cb(refined, f"{name}:rewritten")

    if progress_cb:
        progress_cb(refined, "refine:done")

    return refined





def norm_line(s: str) -> str:
    """Normalize a line for duplicate detection (collapse whitespace, trim)."""
    return re.sub(r"\s+", " ", s.strip())


@torch.inference_mode()
def llm(prompt: str, **gen_kwargs) -> str:
    """Generate with proper v0.3 chat formatting (single <s>) and
       trim oldest history until ≤ 29000 tokens."""
    trimmed = CHAT_HISTORY.copy()

    def build_wrapped(pairs):
        segs = []
        for idx, (u, a) in enumerate(pairs):
            wrap = inst_first if idx == 0 else inst_later
            segs.append(f"{wrap(u)}{a.strip()} ")
        fw_note = ""
        if globals().get("FRAMEWORK_MODE", False):
            fw_note = (
                "FRAMEWORK MODE IS ACTIVE.\n"
                "- A framework baseline (outline + code) is already present in this conversation as the reference outline and current script.\n"
                "- Do NOT skip any section: when asked for a section, generate the complete section even if it ends up identical to the baseline; only that section, no extras.\n"
                "- Treat existing outline/code as a starting point that may be modified to fit the goal. You MAY add helpers inside of the function if required by the goal.\n"
            )

        # Base content: outline (if any) + prompt
        if FULL_OUTLINE:
            base = f"(REFERENCE OUTLINE for entire script, always consider this when answering):\n{FULL_OUTLINE}\n\n{prompt}"
        else:
            base = prompt

        # ✅ Only FW-mode change: concatenate note at the end of the SAME message
        user_content = base + (("\n\n" + fw_note) if fw_note else "")
        user_content += "\n\nDO NOT OVERENGINEER!!!! You must use the simplest, most logical ways to do stuff. Use as few imports, globals, and functions as possible to still accomplish the goal."
        segs.append(inst_later(user_content) if pairs else inst_first(user_content))


        return "".join(segs)

    wrapped = build_wrapped(trimmed)
    while len(tokenizer(wrapped, return_tensors="pt")["input_ids"][0]) > 29000 and trimmed:
        trimmed.pop(0)
        wrapped = build_wrapped(trimmed)

    if len(trimmed) < len(CHAT_HISTORY):
        CHAT_HISTORY[:] = trimmed

    inputs = tokenizer(wrapped, return_tensors="pt").to(model.device)
    defaults = dict(max_new_tokens=4096, temperature=0.7, top_p=0.9, do_sample=True)
    outputs  = model.generate(**inputs, **{**defaults, **gen_kwargs})
    prompt_len = inputs["input_ids"].shape[-1]
    new_ids    = outputs[0][prompt_len:]
    return clean(tokenizer.decode(new_ids, skip_special_tokens=True))


# Reasoning engine will reuse this exact generator
RENGINE_BACKEND = llm


# ---------- Section-by-section: CRITIQUE ----------
def critique_section(
    goal: str,
    script_so_far: str,
    section_kind: str,
    section_spec: dict | list | str,
    candidate_code: str,
    integration_mode: str = "append",  # "append" (default) or "replace"
) -> str:
    """
    Multi-aspect review of the proposed SECTION code.
    Returns 'OK' only if ALL aspects pass. Otherwise returns bullet points combining ALL failed aspects.

    integration_mode:
      - "append": reviewer prompt talks about appending the next section to the script.
      - "replace": reviewer prompt talks about replacing the existing section in the script.
    """
    import re

    # ---- details for prompts ----
    details = ""
    if section_kind == "function" and isinstance(section_spec, dict):
        details = f"Function name that MUST be defined: {section_spec.get('name','')}"
    elif section_kind in {"imports", "globals", "main"}:
        details = f"Section kind: {section_kind}"

    # Phrase the action correctly for this usage context
    if str(integration_mode).lower() == "replace":
        action_line = "Decide ONLY for this aspect if the proposed SECTION is acceptable as a replacement for the existing section in the script."
    else:
        action_line = "Decide ONLY for this aspect if the proposed SECTION is acceptable as the next append to the script."

    # ---- define five distinct critique aspects ----
    aspects: list[tuple[str, str]] = [
        (
            "Coverage & Spec Match",
            "- Verify the proposed code fully covers its outline spec for THIS section only.\n"
            "- IMPORTS: every listed import is present exactly once; nothing extra.\n"
            "- GLOBALS: all listed globals/consts are present; nothing outside scope.\n"
            "- FUNCTION: the exact required function is fully implemented (not a stub) and matches the intent described.\n"
            "- MAIN: implements the described high-level flow without drifting from the outline."
        ),
        (
            "Structure & Section Purity",
            "- The section contains ONLY code appropriate for its kind (imports/globals/single function/main).\n"
            "- No extra functions/classes in non-function sections; no imports inside functions unless explicitly necessary per outline.\n"
            "- For MAIN: includes only orchestration logic, not library/helper definitions."
        ),
        (
            "Interface & Dependencies",
            "- All referenced names are defined earlier or imported in this section when appropriate.\n"
            "- For FUNCTION: the function name matches EXACTLY; parameters/return align with usage implied by outline/script.\n"
            "- No reliance on undefined globals; no forward references that will break execution order."
        ),
        (
            "Production-Readiness (No Placeholders)",
            "- Absolutely no pseudocode, TODOs, ellipses (...), pass-only bodies, or raise NotImplementedError.\n"
            "- Has concrete logic; for FUNCTION includes a concise docstring.\n"
            "- No code fences/backticks; only real Python."
        ),
        (
            "Style, Indentation & Guardrails",
            "- Indentation is exactly 4 spaces; no tabs.\n"
            "- Reasonable naming (UPPER_CASE for constants in GLOBALS).\n"
            "- For MAIN: must include if __name__ == \"__main__\": main() guard.\n"
            "- No redundant imports; no duplicate definitions."
        ),
    ]

    def run_single_aspect(aspect_name: str, criteria: str) -> str:
        prompt = (
            "You are a strict Python reviewer.\n"
            f"ASPECT: {aspect_name}\n"
            f"{action_line}\n"
            "Write ONLY the word 'OK' if acceptable, WITH ABSOLUTELY NO EXTRA DESCRIPTION; otherwise, if the section is not acceptable, list issues succinctly in bullet points.\n\n"
            "CRITERIA:\n"
            f"{criteria}\n\n"
            "ALWAYS consider these contextual constraints as well:\n"
            "- Indentation must be exactly 4 spaces.\n"
            "- Respect section kind boundaries strictly.\n"
            "- Verify dependencies/order and required names.\n\n"
            f"User's goal:\n{goal}\n\n"
            f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
            f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n\n"
            f"Section details:\n{details}\n\n"
            f"Proposed section code:\n{candidate_code}\n"
        )
        print("\n\n\n\nCRITIQUE (ASPECT: " + aspect_name + ") PROMPT:\n\n" + str(prompt))
        resp = llm(prompt, max_new_tokens=4096, temperature=0.25)
        print("\n\n\n\nCRITIQUE (ASPECT: " + aspect_name + ") RESPONSE:\n\n" + str(resp))
        return resp

    failures: list[str] = []
    for name, crit in aspects:
        raw = run_single_aspect(name, crit)
        if _critique_is_ok(raw):
            continue
        # treat anything else as failure; keep the feedback
        txt = (raw or "").strip() or "No rationale provided."
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            lines = ["Unspecified issue."]
        for ln in lines:
            failures.append(f"- [{name}] {ln}")

    if not failures:
        return "OK"
    return "\n".join(failures)



# ---------- Section-by-section: PRE-REVISER (for non-critique failures) ----------
def pre_reviser_explain(goal: str, script_so_far: str, section_kind: str, section_spec: dict | list | str, candidate_code: str, checker_notes: str) -> str:
    """
    Explain non-critique basic failures (structure/validation/compilation) in clear, actionable bullet points.
    The output is plain text (no code fences).
    """
    details = ""
    if section_kind == "function" and isinstance(section_spec, dict):
        details = f"Function name that MUST be defined: {section_spec.get('name','')}"
    elif section_kind in {"imports", "globals", "main"}:
        details = f"Section kind: {section_kind}"

    prompt = (
        "You are a strict Python reviewer.\n"
        "The proposed SECTION failed a basic non-critique check (validation and/or compilation).\n"
        "Respond ONLY in sentences and paragraph style—no bullets, numbered lists, tables, or code. "
        "Explain precisely what should be changed so that a later code-writing pass can implement it. "
        "Do not include any code snippets or fences.\n\n"
        f"CHECKER NOTES:\n{checker_notes}\n\n"
        f"User's goal:\n{goal}\n\n"
        f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
        f"Current script (context):\n{script_so_far}\n\n(END OF SCRIPT)\n\n"
        f"Section details:\n{details}\n\n"
        f"Proposed section code:\n{candidate_code}\n"
    )

    print("\n\n\n\nPRE-REVISER EXPLANATION PROMPT:\n\n" + str(prompt))
    resp = llm(prompt, max_new_tokens=4096, temperature=0.2)
    print("\n\n\n\nPRE-REVISER EXPLANATION RESPONSE:\n\n" + str(resp))
    return resp.strip()

# ---------- Section-by-section: REWRITE ----------

def rewrite_section(goal: str, script_so_far: str, section_kind: str, section_spec: dict | list | str, critique: str, candidate_code: str) -> str:
    """
    Rewrite the proposed SECTION code to fix EVERY issue noted by the reviewer.
    Return ONLY a Python code block content (no fences, no prose).
    """
    details = ""
    if section_kind == "function" and isinstance(section_spec, dict):
        details = f"Function name to define: {section_spec.get('name','')}"
    elif section_kind in {"imports", "globals", "main"}:
        details = f"Section kind: {section_kind}"

    if section_kind == "imports":
        spec_text = "\n".join(f"- {x}" for x in (section_spec or []))
        revise_prompt = (
            "You are generating the IMPORTS section for a Python script.\n"
            "HARD REQUIREMENTS:\n"
            "- Output ONLY a Python code block with import statements needed for the entire script per the outline.\n"
            "- One import per line; avoid redundant imports; no other code or comments.\n"
            "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
            f"User's goal:\n{goal}\n\n"
            f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
            f"Items to cover:\n{spec_text}\n\n"
            + (f"\nReviewer notes:\n{critique}" if critique else "")
        )
    elif section_kind == "globals":
        spec_text = "\n".join(f"- {x}" for x in (section_spec or []))
        revise_prompt = (
            "You are generating the GLOBALS/CONFIG section for a Python script.\n"
            "HARD REQUIREMENTS:\n"
            "- Output ONLY a Python code block that declares constants/config/state.\n"
            "- No function or class definitions here. Keep it minimal and clear; add brief inline comments if helpful.\n"
            "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
            f"User's goal:\n{goal}\n\n"
            f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
            f"Items to cover:\n{spec_text}\n\n"
            f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
            + (f"\nReviewer notes:\n{critique}" if critique else "")
        )
    elif section_kind == "function":
        fname = section_spec.get("name", "")
        fdesc = section_spec.get("desc", "")
        revise_prompt = (
            "You are generating a SINGLE FUNCTION for a Python script.\n"
            "HARD REQUIREMENTS:\n"
            f"- Define exactly one function: `{fname}`.\n"
            "- Include a concise docstring that explains inputs/outputs/side-effects.\n"
            "- Use only built-in types in annotations to avoid extra imports.\n"
            "- Do NOT include other functions, classes, imports, or main().\n"
            "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
            f"User's goal:\n{goal}\n\n"
            f"Function purpose:\n{fdesc}\n\n"
            f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
            f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
            + (f"\nReviewer notes:\n{critique}" if critique else "")
        )

    else:  # main
        revise_prompt = (
            "You are generating ONLY the MAIN section for a Python script.\n"
            "Output ONLY valid Python code (no prose, no lists, no comments, no fences).\n"
            "It must consist of exactly two top-level statements in this order:\n"
            "def main():\n"
            "        # orchestrate previously defined helpers only\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "        main()\n"
            "Rules:\n"
            "- STRICT: Call only functions that ALREADY EXIST in the 'Current script' shown below. Do NOT invent or guess names from the outline. If a needed helper is missing, omit that call.\n"
            "- Prefer the minimal sequence to start the program (e.g., GUI setup then event loop if present).\n"
            "- Do not re-import or redefine anything.\n"
            "- Do not create nested defs/classes.\n"
            "- Keep main() short and deterministic.\n"
            "- All block indentation must be 4 spaces.\n\n"
            f"User's goal:\n{goal}\n\n"
            f"Main high-level plan from outline:\n{section_spec}\n\n"
            f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
            f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
            + (f"\nReviewer notes:\n{critique}" if critique else "")
        )



    print("\n\n\n\nREVISE (SECTION) PROMPT:\n\n"+str(revise_prompt))
    raw = llm(revise_prompt, max_new_tokens=4096, temperature=0.25)
    print("\n\n\n\nREVISE (SECTION) RESPONSE:\n\n"+str(raw))
    return extract_code_block(raw)

def _extract_json_blob(text: str) -> str:
    """Robustly grab the biggest plausible JSON object/array from text (strips code fences)."""
    import re
    s = text.strip()

    # strip leading/trailing code fences if present
    s = re.sub(r"^```(?:json|js|javascript)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)

    # try to find the largest {...} first
    if "{" in s and "}" in s:
        start = s.find("{")
        end = s.rfind("}")
        if end > start:
            return s[start:end+1]

    # else try the largest [...]
    if "[" in s and "]" in s:
        start = s.find("[")
        end = s.rfind("]")
        if end > start:
            return s[start:end+1]

    return ""

def _json_loads_lenient(blob: str):
    """Try strict JSON, then lightly 'repair' (remove trailing commas) and try again."""
    import json, re
    try:
        return json.loads(blob)
    except Exception:
        # remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

def parse_interview_questions(raw: str) -> list[dict]:
    """
    Returns list of {'id': int, 'key': str, 'question': str}.
    Tries strict JSON -> lenient JSON -> regex for "question": "..." -> lines ending with '?'.
    """
    import re

    # 1) JSON path
    blob = _extract_json_blob(raw)
    if blob:
        data = _json_loads_lenient(blob)
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            out = []
            for i, q in enumerate(data["questions"], 1):
                if isinstance(q, dict):
                    question = str(q.get("question", "")).strip()
                    if question:
                        qid = int(q.get("id", i))
                        key = str(q.get("key", f"q{i}"))
                        out.append({"id": qid, "key": key, "question": question})
            if out:
                return out

    # 2) Regex pull of "question": "..."
    hits = re.findall(r'"question"\s*:\s*"([^"]+)"', raw)
    if not hits:
        # also try single quotes
        hits = re.findall(r"'question'\s*:\s*'([^']+)'", raw)
    if hits:
        return [{"id": i+1, "key": f"q{i+1}", "question": h.strip()} for i, h in enumerate(hits)]

    # 3) Last resort: pick lines that look like actual questions (end with '?')
    #    (ignore braces/brackets and obvious JSON syntax lines)
    text = re.sub(r"```.*?```", "", raw, flags=re.S)  # drop fenced code
    lines = [ln.strip(" -•\t") for ln in text.splitlines()]
    cand = [ln for ln in lines
            if ln.endswith("?") and not any(t in ln for t in ["{", "}", "[", "]", "questions", "id", "key", ":"])]
    if cand:
        return [{"id": i+1, "key": f"q{i+1}", "question": ln} for i, ln in enumerate(cand[:15])]

    return []

def _safe_json_loads(text: str, fallback=None):
    import json
    try:
        return json.loads(text)
    except Exception:
        return fallback

class ChatGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Function Former 2.0 from Making Made Easy LLC")

        # --- provide default theme values early (used by _cool_scrolled_text etc.) ---
        self.BG = "#0a0f14"
        self.BG_PANEL = "#0f1620"
        self.BG_DARK = "#091017"
        self.FG = "#e6edf3"
        self.MUTED = "#9aa4b2"
        self.ACCENT = "#00e5ff"
        self.SUCCESS = "#6fffb0"
        self.OUTLINE = "#1f2a34"
        self.SELECTION = "#1b2838"
        self.ACCENT_BG = "#102832"
        # ---------------------------------------------------------------------------

        # === LAYOUT: three vertical columns (left = outline, middle = code, right = sidebar) ===
        self.left_col = tk.Frame(self.root)
        self.mid_col = tk.Frame(self.root)
        self.right_col = tk.Frame(self.root)

        # Left and middle expand; right is a sidebar
        self.left_col.pack(side="left", fill="both", expand=True)
        self.mid_col.pack(side="left", fill="both", expand=True)
        self.right_col.pack(side="left", fill="y")  # grid is used INSIDE this frame

        # --- LEFT: Outline (takes the left side) ---
        self.outline_label = tk.Label(self.left_col, text="Outline (editable)")
        self.outline_label.pack(anchor="w", padx=5)

        self.outline_frame, self.outline_editor = self._cool_scrolled_text(
            self.left_col, wrap="word", height=40, width=60
        )
        self.outline_frame.pack(padx=5, pady=(0, 5), fill="both", expand=True)

        # Keep FULL_OUTLINE synced to whatever is typed here
        def _sync_outline(event=None, self=self):
            try:
                self.outline_editor.edit_modified(False)
            except Exception:
                pass
            text = self.outline_editor.get("1.0", "end").strip()
            global FULL_OUTLINE
            FULL_OUTLINE = text
        self.outline_editor.bind("<<Modified>>", _sync_outline)

        # --- MIDDLE: Script (code editor) ---
        self.script_label = tk.Label(self.mid_col, text="Script (editable)")
        self.script_label.pack(anchor="w", padx=5)

        self.script_frame, self.script_editor = self._cool_scrolled_text(
            self.mid_col, wrap="none", height=40, width=80
        )
        self.script_frame.pack(padx=5, pady=(0, 5), fill="both", expand=True)

        # Keep SCRIPT_LINES synced to whatever is typed here
        def _sync_script(event=None, self=self):
            try:
                self.script_editor.edit_modified(False)
            except Exception:
                pass
            text = self.script_editor.get("1.0", "end")
            global SCRIPT_LINES
            SCRIPT_LINES = text.splitlines()
        self.script_editor.bind("<<Modified>>", _sync_script)

        # --- RIGHT: Goal + Conversation boxes fill the entire column dynamically ---
        # Use grid in the right column so the 2 text boxes split vertical space equally.
        self.right_col.grid_columnconfigure(0, weight=1)
        # Make rows 1 and 3 (the scrolled text frames) share *all* extra height equally.
        for r in (1, 3):
            self.right_col.grid_rowconfigure(r, weight=1, uniform="rightboxes")
        # All other rows (labels, hidden input, button) do not consume stretch.
        for r in (0, 2, 4, 5, 6):
            self.right_col.grid_rowconfigure(r, weight=0)

        # Goal
        self.goal_label = tk.Label(self.right_col, text="Goal")
        self.goal_label.grid(row=0, column=0, sticky="w", padx=5, pady=(0, 2))
        self.goal_frame, self.goal_box = self._cool_scrolled_text(
            self.right_col, wrap="word", height=1, width=42  # height=1 so grid stretch controls size
        )
        self.goal_box.configure(state="disabled")
        self.goal_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))

        # Status
        self.display_label = tk.Label(self.right_col, text="Status")
        self.display_label.grid(row=2, column=0, sticky="w", padx=5, pady=(0, 2))
        self.display_frame, self.display = self._cool_scrolled_text(
            self.right_col, wrap="word", height=1, width=42  # height=1 so grid stretch controls size
        )
        self.display.configure(state="disabled")
        self.display_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=(0, 5))

        # Input (kept for compatibility; hidden and takes no stretch)
        self.input_label = tk.Label(self.right_col, text="Input")
        self.input_label.grid(row=4, column=0, sticky="w", padx=5, pady=(0, 2))
        self.input_frame, self.input_box = self._cool_scrolled_text(
            self.right_col, wrap="word", height=1, width=42
        )
        self.input_frame.grid(row=5, column=0, sticky="nsew", padx=5, pady=(0, 5))

        # Action button (kept for compatibility; hidden)
        self.send_btn = tk.Button(self.right_col, text="Set Goal (Ctrl+Enter)", command=self.on_send)
        self.send_btn.grid(row=6, column=0, sticky="e", padx=5, pady=(0, 5))

        # Hide the inline input and inline button; popups will be used instead
        self.input_label.grid_remove()
        self.input_frame.grid_remove()
        self.send_btn.grid_remove()

        # Section Progress — removed
        self.progress_label = None
        self.progress = None

        # counters for the reasoning phase
        self._rprog_total = 0
        self._rprog_value = 0

        # Bind Ctrl+Enter to send
        self.input_box.bind("<Control-Return>", lambda e: self.on_send())

        # Tag colors
        self.display.tag_configure("prompt", foreground="#0080ff")
        self.display.tag_configure("response", foreground="#00aa00")
        self.display.tag_configure("script", foreground="#333333")

        # Auto-run state (hardcoded ON)
        self.auto_running = True
        self.gen_lock = threading.Lock()
        self.auto_thread = threading.Thread(target=self.auto_loop, daemon=True)
        self.auto_thread.start()

        # Post-generation run/fix state
        self.post_run_started = False

        # Run feedback state
        self.awaiting_run_feedback = False
        self.feedback_event = threading.Event()
        self.latest_feedback = None

        # --- Interview state (goal-refinement via reasoning engine) ---
        self.interview_active = False
        self.interview_questions = []   # list[{"id": int, "key": str, "question": str}]
        self.interview_idx = 0
        self.working_goal = None        # live-updated, starts from GOAL_SPEC
        self.interview_history = []     # list[{"q": str, "a": str}]
        # Hook into the Reasoning Engine's progress events
        try:
            rengine.register_progress_sink(self.rengine_progress)
            self.append_display("Reasoning progress sink registered.", "response")
        except Exception as e:
            self.append_display(f"Could not register global progress sink: {e}", "response")
        self.snapshot_dir = None
        self.snapshot_counter = 0
    def ensure_goal_snapshot_dir(self):
        """
        Create a unique folder for this goal and store goal/outline for reference.
        Path: ./script_snapshots/<timestamp>__<goal-slug>/
        """
        if self.snapshot_dir:
            return  # already set

        try:
            goal = (GOAL_SPEC or self.working_goal or "goal").strip()
        except Exception:
            goal = "goal"
        slug = _slugify_for_fs(goal)
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(SNAPSHOT_ROOT, f"{ts}__{slug}")
        os.makedirs(path, exist_ok=True)
        self.snapshot_dir = path
        self.snapshot_counter = 0

        # Save metadata for convenience
        try:
            with open(os.path.join(path, "goal.txt"), "w", encoding="utf-8") as f:
                f.write((GOAL_SPEC or self.working_goal or "").strip())
            with open(os.path.join(path, "outline.txt"), "w", encoding="utf-8") as f:
                f.write((FULL_OUTLINE or "").strip())
        except Exception:
            pass

        self.append_display(f"Snapshot folder: {path}", "response")


    def save_script_snapshot(self, tag: str):
        """
        Save the current script into the per-goal snapshot folder with an
        incrementing prefix to keep order (00_initial.py, 01_after_run_1.py, ...).
        """
        try:
            if not self.snapshot_dir:
                self.ensure_goal_snapshot_dir()
            fname = f"{self.snapshot_counter:02d}_{tag}.py"
            path = os.path.join(self.snapshot_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(join_script(SCRIPT_LINES))
            self.snapshot_counter += 1
            self.append_display(f"Saved snapshot: {path}", "response")
        except Exception as e:
            self.append_display(f"Snapshot save failed: {e}", "response")

    def open_outline_review_popup(self, initial_outline: str):
        """
        Modal popup shown AFTER outline refinement is complete.
        Lets the user review/edit the outline text, and on Submit we use
        exactly what they typed to build the section plan and start generation.
        """
        win = tk.Toplevel(self.root)
        win.title("Review Outline")
        try:
            win.configure(bg=self.BG)
        except Exception:
            pass
        win.transient(self.root)
        win.grab_set()  # modal

        # Message
        msg = ("Outline created.\n\n"
               "Please review the outline and make any necessary changes. "
               "When you press 'Use This Outline', we'll use your edited version to generate the code.")
        lbl = tk.Label(win, text=msg, justify="left")
        try:
            lbl.configure(bg=self.BG, fg=self.FG, font=("Consolas", 10, "bold"), wraplength=640)
        except Exception:
            pass
        lbl.pack(padx=12, pady=(12, 8), anchor="w")

        # Editable outline box (prefilled with the refined outline)
        frame, txt = self._cool_scrolled_text(win, wrap="word", width=84, height=26)
        frame.pack(padx=12, pady=(0, 10), fill="both", expand=True)
        txt.insert("1.0", initial_outline or "")
        txt.focus_set()

        # Buttons row
        row = tk.Frame(win)
        try:
            row.configure(bg=self.BG)
        except Exception:
            pass
        row.pack(padx=12, pady=(0, 12), anchor="e")

        def _submit():
            edited = txt.get("1.0", "end").strip()
            if not edited:
                # require a non-empty outline; keep focus if empty
                txt.focus_set()
                return
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            self._apply_user_outline_and_start(edited)

        def _cancel():
            # If user cancels, proceed with the refined outline as-is
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            self._apply_user_outline_and_start(initial_outline)

        # Primary button
        ok_btn = tk.Button(row, text="Use This Outline", command=_submit)
        try:
            ok_btn.configure(
                bg=self.ACCENT_BG, fg=self.FG, activebackground=self.ACCENT,
                activeforeground=self.BG, relief="flat", bd=0, padx=12, pady=6,
                font=("Consolas", 10, "bold")
            )
        except Exception:
            pass
        ok_btn.pack(side="right")

        # Secondary (optional) cancel button
        cancel_btn = tk.Button(row, text="Use As-Is", command=_cancel)
        try:
            cancel_btn.configure(
                bg=self.BG_PANEL, fg=self.FG, activebackground=self.ACCENT_BG,
                activeforeground=self.FG, relief="flat", bd=0, padx=12, pady=6,
                font=("Consolas", 10)
            )
        except Exception:
            pass
        cancel_btn.pack(side="right", padx=(0, 8))

        # Enter = submit
        win.bind("<Return>", lambda e: _submit())
        # Window close = cancel
        win.protocol("WM_DELETE_WINDOW", _cancel)

        # Center-ish
        win.update_idletasks()
        try:
            x = self.root.winfo_rootx() + max(20, (self.root.winfo_width() - win.winfo_width()) // 2)
            y = self.root.winfo_rooty() + max(20, (self.root.winfo_height() - win.winfo_height()) // 3)
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass


    def _apply_user_outline_and_start(self, outline_text: str):
        """
        Persist the (possibly edited) outline, refresh the Outline panel,
        rebuild the section plan, and start code generation.
        """
        global FULL_OUTLINE, SECTIONS_PLAN, SECTION_INDEX
        FULL_OUTLINE = (outline_text or "").strip()

        # Reflect into the left Outline editor and log
        self.publish_outline(FULL_OUTLINE, "user_reviewed")
        self.append_display("Outline confirmed. Starting code generation...", "response")

        # Build plan and kick off generation
        parsed = parse_outline(FULL_OUTLINE)
        SECTIONS_PLAN[:] = build_sections_plan(parsed)
        SECTION_INDEX = 0
        self._ensure_auto_on()
        self.prog_reset(len(SECTIONS_PLAN))

        # ✅ run off the Tk thread to avoid UI freeze
        threading.Thread(target=self.generate_next_section, daemon=True).start()


    def open_input_popup(self, prompt_text: str, submit_label: str, on_submit):
        """
        Show a modal popup asking the user for text.
        - prompt_text: what we're asking the user (shown on the popup)
        - submit_label: text for the popup button (e.g., 'Submit Answer (Ctrl+Enter)')
        - on_submit(value): callback receiving the text the user entered
        """
        win = tk.Toplevel(self.root)
        win.title("Input required")
        try:
            win.configure(bg=self.BG)
        except Exception:
            pass
        win.transient(self.root)
        win.grab_set()  # modal

        # Prompt message (exactly what we need from the user)
        lbl = tk.Label(win, text=prompt_text, justify="left")
        try:
            lbl.configure(bg=self.BG, fg=self.FG, font=("Consolas", 10, "bold"), wraplength=520)
        except Exception:
            pass
        lbl.pack(padx=12, pady=(12, 6), anchor="w")

        # Text entry (re-use themed scroller)
        frame, txt = self._cool_scrolled_text(win, wrap="word", width=64, height=8)
        frame.pack(padx=12, pady=(0, 10), fill="both", expand=True)
        txt.focus_set()

        # Submit action
        def _submit():
            value = txt.get("1.0", "end").strip()
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            if callable(on_submit):
                on_submit(value)

        btn = tk.Button(win, text=submit_label, command=_submit)
        try:
            btn.configure(
                bg=self.ACCENT_BG, fg=self.FG, activebackground=self.ACCENT,
                activeforeground=self.BG, relief="flat", bd=0, padx=12, pady=6,
                font=("Consolas", 10, "bold")
            )
        except Exception:
            pass
        btn.pack(padx=12, pady=(0, 12), anchor="e")

        # Ctrl+Enter submits (like before)
        txt.bind("<Control-Return>", lambda e: _submit())

        # Simple centering
        win.update_idletasks()
        try:
            x = self.root.winfo_rootx() + max(20, (self.root.winfo_width() - win.winfo_width()) // 2)
            y = self.root.winfo_rooty() + max(20, (self.root.winfo_height() - win.winfo_height()) // 3)
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass
    def open_framework_picker(self, on_done=None):
        """
        Modal chooser shown right after the user submits a goal.
        Lets them pick a framework (outline+code) or continue with no framework.
        """
        fw_list = discover_frameworks()
        win = tk.Toplevel(self.root)
        win.title("Choose a Framework (optional)")
        try:
            win.configure(bg=self.BG)
        except Exception:
            pass
        win.transient(self.root)
        win.grab_set()

        info = tk.Label(win, text="Optionally select a framework. It preloads an outline and matching code.\nYou can still add/modify sections normally.",
                        justify="left")
        try:
            info.configure(bg=self.BG, fg=self.FG, font=("Consolas", 10, "bold"), wraplength=640)
        except Exception:
            pass
        info.pack(padx=12, pady=(12, 8), anchor="w")

        # list + preview
        frame = tk.Frame(win)
        try:
            frame.configure(bg=self.BG)
        except Exception:
            pass
        frame.pack(padx=12, pady=(0, 10), fill="both", expand=True)

        lb = tk.Listbox(frame, height=10)
        try:
            lb.configure(bg=self.BG_PANEL, fg=self.FG, selectbackground=self.SELECTION, relief="flat")
        except Exception:
            pass
        lb.grid(row=0, column=0, sticky="nswe", padx=(0, 8))

        pv_frame, pv = self._cool_scrolled_text(frame, wrap="word", width=72, height=16)
        pv_frame.grid(row=0, column=1, sticky="nswe")

        frame.grid_columnconfigure(0, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        names = ["(No framework)"] + [fw["name"] for fw in fw_list]
        for n in names:
            lb.insert("end", n)
        lb.selection_set(0)

        def refresh_preview(_evt=None):
            idxs = lb.curselection()
            if not idxs:
                return
            idx = idxs[0]
            pv.configure(state="normal")
            pv.delete("1.0", "end")
            if idx == 0:
                pv.insert("1.0", "No framework. The tool will build from a blank outline and empty script.")
            else:
                fw = fw_list[idx - 1]
                preview = (
                    "=== OUTLINE (preview) ===\n" + fw["outline"][:2000] +
                    ("\n...\n" if len(fw["outline"]) > 2000 else "") +
                    "\n\n=== CODE (preview) ===\n" + fw["code"][:2000] +
                    ("\n...\n" if len(fw["code"]) > 2000 else "")
                )
                pv.insert("1.0", preview)
            pv.configure(state="disabled")

        lb.bind("<<ListboxSelect>>", refresh_preview)
        refresh_preview()

        btn_row = tk.Frame(win)
        try:
            btn_row.configure(bg=self.BG)
        except Exception:
            pass
        btn_row.pack(padx=12, pady=(0, 12), anchor="e")

        def _use():
            idxs = lb.curselection()
            if not idxs:
                return
            idx = idxs[0]
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            if idx == 0:
                # no framework
                self.append_display("Framework: none selected. Proceeding normally.", "response")
                if callable(on_done): on_done()
                return

            fw = fw_list[idx - 1]
            self.apply_framework(fw)
            if callable(on_done): on_done()

        def _skip():
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            self.append_display("Framework selection skipped.", "response")
            if callable(on_done): on_done()

        use_btn = tk.Button(btn_row, text="Use Selected Framework", command=_use)
        skip_btn = tk.Button(btn_row, text="Continue Without", command=_skip)
        for b in (use_btn, skip_btn):
            try:
                b.configure(bg=self.ACCENT_BG, fg=self.FG, activebackground=self.ACCENT,
                            activeforeground=self.BG, relief="flat", bd=0, padx=12, pady=6,
                            font=("Consolas", 10, "bold"))
            except Exception:
                pass
        skip_btn.pack(side="right")
        use_btn.pack(side="right", padx=(0, 8))

        # center-ish
        win.update_idletasks()
        try:
            x = self.root.winfo_rootx() + max(20, (self.root.winfo_width() - win.winfo_width()) // 2)
            y = self.root.winfo_rooty() + max(20, (self.root.winfo_height() - win.winfo_height()) // 3)
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def apply_framework(self, fw: dict):
        """
        Activate framework mode and preload outline + code into the editors.
        """
        global FRAMEWORK_MODE, FRAMEWORK_INFO, FULL_OUTLINE, SCRIPT_LINES
        FRAMEWORK_MODE = True
        FRAMEWORK_INFO = fw
        FULL_OUTLINE = (fw.get("outline") or "").strip()
        SCRIPT_LINES = (fw.get("code") or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
        self.publish_outline(FULL_OUTLINE, "framework:loaded")
        self.refresh_script_editor()
        self.append_display(f"Framework selected: {fw.get('name','(unnamed)')}", "response")

    def open_confirm_popup(self, title: str, message: str, button_label: str, on_confirm, on_cancel=None):
        """
        Simple modal confirm dialog with a single primary button and optional cancel.
        Calls on_confirm() or on_cancel() and then closes.
        """
        win = tk.Toplevel(self.root)
        win.title(title)
        try:
            win.configure(bg=self.BG)
        except Exception:
            pass
        win.transient(self.root)
        win.grab_set()  # modal

        # Message
        lbl = tk.Label(win, text=message, justify="left")
        try:
            lbl.configure(bg=self.BG, fg=self.FG, font=("Consolas", 10, "bold"), wraplength=520)
        except Exception:
            pass
        lbl.pack(padx=12, pady=(12, 8), anchor="w")

        # Buttons row
        btn_row = tk.Frame(win)
        try:
            btn_row.configure(bg=self.BG)
        except Exception:
            pass
        btn_row.pack(padx=12, pady=(0, 12), anchor="e")

        def _confirm():
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            if callable(on_confirm):
                on_confirm()

        def _cancel():
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()
            if callable(on_cancel):
                on_cancel()

        ok_btn = tk.Button(btn_row, text=button_label, command=_confirm)
        try:
            ok_btn.configure(
                bg=self.ACCENT_BG, fg=self.FG, activebackground=self.ACCENT,
                activeforeground=self.BG, relief="flat", bd=0, padx=12, pady=6,
                font=("Consolas", 10, "bold")
            )
        except Exception:
            pass
        ok_btn.pack(side="right")

        cancel_btn = tk.Button(btn_row, text="Cancel", command=_cancel)
        try:
            cancel_btn.configure(
                bg=self.BG_PANEL, fg=self.FG, activebackground=self.ACCENT_BG,
                activeforeground=self.FG, relief="flat", bd=0, padx=12, pady=6,
                font=("Consolas", 10)
            )
        except Exception:
            pass
        cancel_btn.pack(side="right", padx=(0, 8))

        # Enter = confirm
        win.bind("<Return>", lambda e: _confirm())
        # Window close = cancel
        win.protocol("WM_DELETE_WINDOW", _cancel)

        # Center-ish
        win.update_idletasks()
        try:
            x = self.root.winfo_rootx() + max(20, (self.root.winfo_width() - win.winfo_width()) // 2)
            y = self.root.winfo_rooty() + max(20, (self.root.winfo_height() - win.winfo_height()) // 3)
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass


    def wait_for_run_confirmation(self) -> bool:
        """
        Modal pause before starting the run & repair routine.
        Returns True only if the user explicitly confirms via the popup.
        """
        evt = threading.Event()
        result = {"go": False}

        def _show():
            msg = (
                "About to test-run your generated code.\n\n"
                "This will execute the script in a separate Python process so we can check for errors "
                "and (if needed) auto-repair them.\n\n"
                "Press 'Run & Test Now' to commence."
            )
            self.open_confirm_popup(
                title="Ready to run & test",
                message=msg,
                button_label="Run & Test Now",
                on_confirm=lambda: (result.update(go=True), evt.set()),
                on_cancel=lambda: (result.update(go=False), evt.set()),
            )

        # Show popup on the UI thread, wait here in the worker thread
        self.root.after(0, _show)
        evt.wait()

        if result["go"]:
            self.append_display("User confirmed: starting run & repair.", "response")
        else:
            self.append_display("Run cancelled by user.", "response")

        return result["go"]

    def show_goal_popup(self):
        # Same button label text as before; prompt text simply describes what's needed
        self.open_input_popup(
            "Please enter your overall goal:",
            "Set Goal (Ctrl+Enter)",
            self._handle_goal_submit
        )

    def _handle_goal_submit(self, text: str):
        global GOAL_SPEC
        if not text.strip():
            # Re-open until we get something non-empty
            self.show_goal_popup()
            return
        GOAL_SPEC = text.strip()
        self.append_display("Goal received.", "prompt")
        self.update_goal_box()

        # Show framework picker now; when it closes, proceed to interview.
        def _after_fw():
            self.send_btn.config(text="Submit Answer (Ctrl+Enter)", state="disabled")
            self.start_goal_interview(GOAL_SPEC)

        self.open_framework_picker(on_done=_after_fw)
        return



    def _init_scrollbar_styles(self, style: ttk.Style):
        # Arrowless, slim, neon thumb on dark trough
        style.layout(
            "Cyber.Vertical.TScrollbar",
            [("Vertical.Scrollbar.trough",
              {"children": [("Vertical.Scrollbar.thumb",
                             {"expand": "1", "sticky": "nswe"})],
               "sticky": "ns"})]
        )
        style.layout(
            "Cyber.Horizontal.TScrollbar",
            [("Horizontal.Scrollbar.trough",
              {"children": [("Horizontal.Scrollbar.thumb",
                             {"expand": "1", "sticky": "nswe"})],
               "sticky": "we"})]
        )

        # Base colors
        style.configure(
            "Cyber.Vertical.TScrollbar",
            troughcolor=self.BG_DARK,
            background=self.ACCENT,       # thumb color
            darkcolor=self.ACCENT,
            lightcolor=self.ACCENT,
            bordercolor=self.OUTLINE,
            gripcount=0
        )
        style.configure(
            "Cyber.Horizontal.TScrollbar",
            troughcolor=self.BG_DARK,
            background=self.ACCENT,
            darkcolor=self.ACCENT,
            lightcolor=self.ACCENT,
            bordercolor=self.OUTLINE,
            gripcount=0
        )

        # Simple “active/pressed” glow shift
        style.map(
            "Cyber.Vertical.TScrollbar",
            background=[("active", self.SUCCESS), ("pressed", self.SUCCESS)]
        )
        style.map(
            "Cyber.Horizontal.TScrollbar",
            background=[("active", self.SUCCESS), ("pressed", self.SUCCESS)]
        )

    def _cool_scrolled_text(self, parent, *, wrap="word", width=60, height=10):
        """
        Returns (container_frame, text_widget) with themed ttk scrollbars.
        Horizontal bar is added automatically for wrap='none'.
        """
        outer = tk.Frame(parent, bg=self.BG)
        txt = tk.Text(outer, wrap=wrap, undo=True, width=width, height=height)

        # Vertical scrollbar
        vbar = ttk.Scrollbar(
            outer, orient="vertical",
            style="Cyber.Vertical.TScrollbar",
            command=txt.yview
        )
        txt.configure(yscrollcommand=vbar.set)

        # Horizontal if needed
        if wrap == "none":
            hbar = ttk.Scrollbar(
                outer, orient="horizontal",
                style="Cyber.Horizontal.TScrollbar",
                command=txt.xview
            )
            txt.configure(xscrollcommand=hbar.set)

        # Layout (use grid inside the container)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        txt.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        if wrap == "none":
            hbar.grid(row=1, column=0, sticky="ew")

        # Match your editor look
        try:
            txt.configure(
                bg=self.BG_PANEL, fg=self.FG, insertbackground=self.ACCENT,
                relief="flat", bd=0, highlightthickness=1,
                highlightbackground=self.OUTLINE, highlightcolor=self.ACCENT,
                selectbackground=self.SELECTION
            )
        except Exception:
            pass

        def _hover_expand(sb):
            def enter(_): 
                try: sb.configure(width=16)
                except: pass
            def leave(_): 
                try: sb.configure(width=10)
                except: pass
            sb.bind("<Enter>", enter)
            sb.bind("<Leave>", leave)

        _hover_expand(vbar)
        if wrap == "none":
            _hover_expand(hbar)

        return outer, txt

            
    def _ensure_auto_on(self):
        self.auto_running = True
        if not getattr(self, "auto_thread", None) or not self.auto_thread.is_alive():
            self.auto_thread = threading.Thread(target=self.auto_loop, daemon=True)
            self.auto_thread.start()

    def start_goal_interview(self, initial_goal: str):
        """Kick off the requirements interview: generate questions via reasoning engine."""
        # ✅ IMPORTANT: disable auto-run during the interview so the answer button works
        self.auto_running = False

        self.interview_active = True
        self.working_goal = initial_goal
        self.append_display("Creating clarifying questions...", "prompt")
        self.send_btn.config(state="disabled")

        threading.Thread(target=self._interview_worker, args=(initial_goal,), daemon=True).start()


    def _interview_worker(self, goal: str):
        """
        1) Ask the reasoning engine to emit a strictly JSON question list.
        2) Present the first question to the user.
        """
        prompt = (
            "You are a requirements interviewer. Given the user's initial goal for the python script that they want to create, "
            "ask the minimum set of specific questions needed to fully specify the deliverable working Python script.\n\n"
            f"INITIAL_GOAL (This goal is for creating a python script only):\n{goal}\n\n"
            "Return ONLY JSON with this exact schema (no extra text):\n"
            "{{\"questions\": [{{\"id\": 1, \"key\": \"snake_case_short_key\", \"question\": \"<clear single question>\"}}]}}\n"
            "- 5–15 questions.\n"
            "- Each question should be answerable in one short sentence.\n"
            "- Order questions logically, most blocking info first."
        )
        raw = rengine.reason(prompt, add_to_history=False, llm_backend=RENGINE_BACKEND, progress_cb=self.rengine_progress)

        qs = parse_interview_questions(raw)

        if not qs:
            qs = [{"id": 1, "key": "goal_clarification", "question": "What’s the most important outcome you want from this tool?"}]

        self.interview_questions = qs
        self.interview_idx = 0
        self.root.after(0, self.ask_next_question)



    def ask_next_question(self):
        """Show the next question in a modal popup; if none left, synthesize the refined goal."""
        if self.interview_idx >= len(self.interview_questions):
            # Finished -> synthesize as before
            global GOAL_SPEC
            self.append_display("Synthesizing refined goal from full Q&A...", "response")
            self.send_btn.config(state="disabled")  # harmless; button is hidden
            def _finalize():
                try:
                    original_goal = self.working_goal
                    questions_list = "\n".join(
                        f"{i+1}. {qobj['question']}"
                        for i, qobj in enumerate(self.interview_questions)
                    )
                    qa_transcript = "\n".join(
                        f"Q{i+1}: {pair['q']}\nA{i+1}: {pair['a']}"
                        for i, pair in enumerate(self.interview_history)
                    )
                    prompt = (
                        "You are refining a project goal based on a completed requirements interview.\n"
                        "Rewrite the goal to include every concrete detail gathered. Keep it concise but fully specified.\n"
                        "IRRELEVANCE RULE: If an answer indicates the question is irrelevant or out of scope, or expresses no preference "
                        "(e.g., 'n/a', 'not applicable', 'none', 'no preference', 'either', 'doesn't matter', 'skip', 'default', or is empty), "
                        "IGNORE the data from that question entirely. Do not invent or assume values for ignored items; prefer leaving them "
                        "unspecified rather than guessing.\n\n"
                        f"ORIGINAL_GOAL:\n{original_goal}\n\n"
                        "QUESTIONS_ASKED:\n"
                        f"{questions_list}\n\n"
                        "QA_TRANSCRIPT:\n"
                        f"{qa_transcript}\n\n"
                        "Return ONLY JSON:\n"
                        "{\"updated_goal\":\"<one paragraph refined goal including concrete choices/values and excluding any ignored items>\","
                        "\"notes\":[\"<optional short note>\"]}"
                    )
                    raw = rengine.reason(
                        prompt,
                        add_to_history=False,
                        llm_backend=RENGINE_BACKEND,
                        progress_cb=self.rengine_progress,
                    )
                    blob = _extract_json_blob(raw)
                    data = _safe_json_loads(blob, {})
                    updated = (data.get("updated_goal") or raw).strip()

                    # Use ONLY the rewritten goal from this point forward
                    global GOAL_SPEC, CHAT_HISTORY
                    self.working_goal = updated
                    GOAL_SPEC = self.working_goal
                    CHAT_HISTORY.clear()  # prevent any prior prompts containing the original goal from being reused

                    self.root.after(0, lambda: self.append_display("Updated goal:\n" + self.working_goal, "response"))
                    self.root.after(0, self.update_goal_box)
                    self.interview_active = False
                    self._ensure_auto_on()
                    self.root.after(0, self.start_outline_from_goal)
                except Exception as e:
                    msg = f"Failed to synthesize refined goal: {e}"
                    self.root.after(0, self.append_display, msg, "response")


            threading.Thread(target=_finalize, daemon=True).start()
            return

        # Ask the next question (unchanged text), but in a popup
        q = self.interview_questions[self.interview_idx]["question"]
        self.append_display(f"Question {self.interview_idx + 1}/{len(self.interview_questions)}:\n{q}", "prompt")

        def _submit_answer(answer_text: str):
            if not answer_text:
                # If blank, re-open the same question
                self.root.after(50, lambda: self.open_input_popup(
                    f"{q}\n\n(Please provide an answer.)",
                    "Submit Answer (Ctrl+Enter)",
                    _submit_answer
                ))
                return
            # Mirror the old UI log and continue
            self.append_display(f"Answer:\n{answer_text}", "prompt")
            self.process_answer(answer_text)

        # Popup shows the exact question; button belongs to the popup
        self.open_input_popup(q, "Submit Answer (Ctrl+Enter)", _submit_answer)




    def process_answer(self, answer_text: str):
        """Record the user's answer; defer goal synthesis until all questions are answered."""
        self.send_btn.config(state="disabled")
        qobj = self.interview_questions[self.interview_idx]
        q = qobj["question"]

        # Save transcript only; no incremental rewriting
        self.interview_history.append({"q": q, "a": answer_text})

        # Advance to next question
        self.interview_idx += 1
        self.root.after(0, self.ask_next_question)


    def start_outline_from_goal(self):
        """
        Begin the existing outline → refine → sections pipeline,
        using the (possibly updated) GOAL_SPEC.
        After refinement, show a popup to let the user review/edit the outline.
        """
        def _outline_job():
            global FULL_OUTLINE, SECTIONS_PLAN, SECTION_INDEX, GOAL_SPEC
            self.root.after(0, lambda: self.append_display("Generating outline...", "prompt"))
            initial_outline = get_full_outline(GOAL_SPEC, progress_cb=self.publish_outline)

            self.root.after(0, lambda: self.append_display("Refining outline...", "prompt"))
            refined_outline = refine_outline(GOAL_SPEC, initial_outline, progress_cb=self.publish_outline)
            FULL_OUTLINE = refined_outline

            # Show review popup instead of immediately starting generation
            self.root.after(
                0,
                lambda: self.append_display(
                    "Outline created. Please review it in the popup and submit your final version.",
                    "response"
                )
            )
            self.root.after(0, lambda: self.open_outline_review_popup(refined_outline))

        threading.Thread(target=_outline_job, daemon=True).start()


    def publish_outline(self, text: str, stage: str | None = None):
        """
        Thread-safe: updates the global FULL_OUTLINE and refreshes the Outline editor.
        Optionally logs a small status line in the conversation pane.
        """
        global FULL_OUTLINE
        FULL_OUTLINE = text

        def _update():
            try:
                # reflect latest outline in the editable Outline box
                self.outline_editor.delete("1.0", "end")
                self.outline_editor.insert("1.0", text)
                self.outline_editor.edit_modified(False)
                if stage:
                    self.append_display(f"Outline update: {stage}", "response")
            except Exception:
                pass

        self.root.after(0, _update)


    def append_display(self, text: str, tag: str):
        self.display.configure(state="normal")
        self.display.insert("end", text + "\n\n", tag)
        self.display.configure(state="disabled")
        self.display.see("end")

    def update_goal_box(self):
        """Refresh the right-most goal box with the current goal and re-written goal (if any)."""
        try:
            self.goal_box.configure(state="normal")
            self.goal_box.delete("1.0", "end")
            base = (GOAL_SPEC or "") or (self.working_goal or "")
            rewritten = self.working_goal or ""
            if rewritten and base and rewritten != base:
                content = f"Goal:\n{base}\n\nRe-written Goal:\n{rewritten}"
            else:
                content = f"Goal:\n{base}"
            self.goal_box.insert("1.0", content.strip())
            self.goal_box.configure(state="disabled")
        except Exception:
            pass

    def refresh_script_editor(self):
        """Refresh the middle Script editor to show current SCRIPT_LINES."""
        try:
            full_script_text = join_script(SCRIPT_LINES)

            def _set_script_text(s=full_script_text):
                try:
                    self.script_editor.delete("1.0", "end")
                    self.script_editor.insert("1.0", s)
                    self.script_editor.edit_modified(False)
                except Exception:
                    pass

            self.root.after(0, _set_script_text)
        except Exception:
            pass


    # ---------- reasoning UI helpers ----------
    def append_reason(self, text: str):
        # Route reasoning updates to the Status display
        self.append_display(text, "response")


    def _reason_reset(self, total: int):
        # track progress internally; no separate progressbar
        self._rprog_total = max(1, int(total))
        self._rprog_value = 0
        # optional: a simple line in the status area
        self.append_display("Reasoning started.", "response")

    def _reason_inc(self, n: int = 1):
        self._rprog_value = min(self._rprog_total, self._rprog_value + n)
        # no progressbar updates anymore
        # uncomment if you want a textual ticker:
        # self.append_display(f"Reasoning progress: {self._rprog_value}/{self._rprog_total}", "response")



    def rengine_progress(self, phase: str, payload):
        """
        Receives progress events from LLM_Reasoning_Engine.
        Phases:
          - plan: payload = list[str] steps
          - step: payload = {'index','total','label'}
          - synth: payload = draft str
          - critique: payload = critique str (often 'OK')
          - final: payload = final str
        """
        def _shorten(x, width=800):
            s = str(x or "").strip()
            return s if len(s) <= width else (s[:width] + " …")

        def _do():
            if phase == "plan":
                steps = list(payload or [])
                # steps + synth + critique + final
                self._reason_reset(len(steps) + 3)
                self.append_reason("Plan:")
                if steps:
                    for i, s in enumerate(steps, 1):
                        self.append_reason(f"  {i}. {s}")
                else:
                    self.append_reason("  (no steps returned)")
            elif phase == "step":
                try:
                    idx = payload.get("index")
                    total = payload.get("total")
                    label = payload.get("label") or ""
                    prefix = f"Step {idx}/{total}" if (idx and total) else "Step"
                except Exception:
                    label = str(payload)
                    prefix = "Step"
                self._reason_inc(1)
                self.append_reason(f"{prefix}: {label}")
            elif phase == "synth":
                self._reason_inc(1)
                self.append_reason("Synthesized draft:")
                self.append_reason(_shorten(payload))
            elif phase == "critique":
                self._reason_inc(1)
                txt = str(payload or "").strip()
                if txt.lower() == "ok":
                    self.append_reason("Critique: OK")
                else:
                    self.append_reason("Critique: issues found")
                    self.append_reason(_shorten(txt))
            elif phase == "final":
                self._reason_inc(1)
                self.append_reason("Final answer ready.")
                self.append_reason(_shorten(payload))
            else:
                self.append_reason(f"{phase}: {payload}")

        # ensure UI thread
        self.root.after(0, _do)

    # ---------- progress helpers ----------
    def prog_reset(self, total: int):
        """Progress bar removed — no-op."""
        return

    def prog_inc(self):
        """Progress bar removed — no-op."""
        return


    # ---------- auto-run ----------
    def toggle_auto(self):
        """Toggle automatic generation of the next section repeatedly."""
        global GOAL_SPEC

        if not self.auto_running:
            if GOAL_SPEC is None:
                return
            self.auto_running = True
            self.auto_btn.config(text="Auto: ON (Stop)")
            self.send_btn.config(state="disabled")
            self.auto_thread = threading.Thread(target=self.auto_loop, daemon=True)
            self.auto_thread.start()
        else:
            self.auto_running = False
            self.auto_btn.config(text="Auto: OFF")

    def auto_loop(self):
        """Background loop that keeps generating the next section until stopped."""
        while self.auto_running:
            self.generate_next_section()
            if not self.auto_running:
                break
            time.sleep(0.05)


    def on_send(self):
        global GOAL_SPEC, FULL_OUTLINE, SECTIONS_PLAN, SECTION_INDEX

        if self.auto_running and not (self.interview_active or self.awaiting_run_feedback or GOAL_SPEC is None):
            return

        user_text = self.input_box.get("1.0", "end").strip()

        if self.interview_active:
            answer = user_text
            if not answer:
                return
            self.input_box.delete("1.0", "end")
            self.append_display(f"Answer:\n{answer}", "prompt")
            self.process_answer(answer)
            return

        if self.awaiting_run_feedback:
            self.latest_feedback = user_text
            self.input_box.delete("1.0", "end")
            self.awaiting_run_feedback = False
            self.append_display("Feedback received. Continuing.", "response")
            self.feedback_event.set()
            return

        if GOAL_SPEC is None:
            if not user_text:
                return
            GOAL_SPEC = user_text
            self.append_display("Goal received.", "prompt")
            self.input_box.delete("1.0", "end")
            # update the right-most goal box immediately
            self.update_goal_box()

            self.send_btn.config(text="Submit Answer (Ctrl+Enter)", state="disabled")
            self.start_goal_interview(GOAL_SPEC)
            return

        self.input_box.delete("1.0", "end")
        self.send_btn.config(state="disabled")
        threading.Thread(target=self.generate_next_section, daemon=True).start()





    def generate_next_section(self):
        if not self.gen_lock.acquire(blocking=False):
            return
        try:
            global GOAL_SPEC, FULL_OUTLINE, SCRIPT_LINES, SECTIONS_PLAN, SECTION_INDEX

            if SECTION_INDEX >= len(SECTIONS_PLAN):
                # Nothing left to generate
                self.root.after(0, self.finish_cycle)
                return

            goal = GOAL_SPEC or ""
            # Use the current editable script text (SCRIPT_LINES stays in sync via editor binding)
            script_so_far = join_script(SCRIPT_LINES)
            section_kind, section_spec = SECTIONS_PLAN[SECTION_INDEX]

            # --- utilities ---
            def compiles_with_script(block: str) -> tuple[bool, str, str]:
                test_src = (script_so_far + "\n\n" if script_so_far else "") + block
                try:
                    compile(test_src, "<next-section-check>", "exec")
                    return True, "", block
                except SyntaxError as e:
                    if "indent" in str(e).lower():
                        fixed = auto_fix_indentation(
                            block,
                            section_kind,
                            section_spec.get("name") if (section_kind == "function" and isinstance(section_spec, dict)) else None,
                        )
                        test_src2 = (script_so_far + "\n\n" if script_so_far else "") + fixed
                        try:
                            compile(test_src2, "<next-section-check-fixed>", "exec")
                            return True, "", fixed
                        except SyntaxError as e2:
                            return False, f"{type(e2).__name__}: {e2}", block
                    return False, f"{type(e).__name__}: {e}", block

            # --- NEW: selection helper if we exhaust attempts ---
            def _llm_pick_best_section_candidate(
                section_kind_local: str,
                section_spec_local,
                script_ctx: str,
                goal_ctx: str,
                labeled_candidates: list[tuple[str, str]],
            ) -> str:
                """
                Ask the LLM to pick the best candidate by LABEL.
                Returns the chosen label (must be one of the provided labels).
                """
                labels = [lab for lab, _ in labeled_candidates]
                # Build a clearly labeled, strictly formatted comparison prompt
                header = (
                    "You are selecting the BEST candidate to use for the next section of a Python script.\n"
                    "Return ONLY ONE label from the allowed list EXACTLY as written. No code, no explanations.\n"
                    "Choose the candidate that is most likely to compile and fit the outline and current script.\n"
                    "Section kind rules:\n"
                    "- imports: only import lines.\n"
                    "- globals: only top-level assignments, no defs/classes.\n"
                    "- function: exactly one def <name>(...): with a concise docstring, no extra defs/imports.\n"
                    "- main: defines main() and includes the __main__ guard; orchestrates existing helpers only.\n\n"
                    f"ALLOWED_LABELS: {', '.join(labels)}\n\n"
                    f"SECTION_KIND: {section_kind_local}\n"
                    f"SECTION_SPEC: {section_spec_local}\n"
                    f"USER_GOAL:\n{goal_ctx}\n\n"
                    "CANDIDATES (each wrapped with BEGIN/END markers and a unique LABEL):\n"
                )
                blocks = []
                for lab, code in labeled_candidates:
                    blocks.append(
                        f"=== BEGIN_CANDIDATE {lab} ===\n```python\n{code}\n```\n=== END_CANDIDATE {lab} ==="
                    )
                prompt = (
                    header
                    + "\n\n".join(blocks)
                    + "\n\nCURRENT_SCRIPT_CONTEXT (read-only):\n"
                    + script_ctx
                    + "\n\n"
                    "Respond with ONLY one of the ALLOWED_LABELS. No extra text."
                )
                resp = llm(prompt, max_new_tokens=4096, temperature=0.1, top_p=0.95, do_sample=False).strip()
                # Normalize and validate
                # Prefer exact match; otherwise extract the first valid label substring.
                if resp in labels:
                    return resp
                for lab in labels:
                    if lab in resp:
                        return lab
                # Fallback to last label if parsing failed
                return labels[-1]

            # --- synthesize candidate section ---
            max_attempts = 5
            last_candidate_code = ""
            # NEW: collect both synth and rewrite candidates for late selection
            collected_candidates: list[tuple[str, str]] = []

            for attempt in range(1, max_attempts + 1):
                if section_kind == "imports":
                    spec_text = "\n".join(f"- {x}" for x in (section_spec or []))
                    synth_prompt = (
                        "You are generating the IMPORTS section for a Python script.\n"
                        "HARD REQUIREMENTS:\n"
                        "- Output ONLY a Python code block with import statements needed for the entire script per the outline.\n"
                        "- One import per line; avoid redundant imports; no other code or comments.\n"
                        "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
                        f"User's goal:\n{goal}\n\n"
                        f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
                        f"Items to cover:\n{spec_text}\n\n"
                        f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
                    )
                elif section_kind == "globals":
                    spec_text = "\n".join(f"- {x}" for x in (section_spec or []))
                    synth_prompt = (
                        "You are generating the GLOBALS/CONFIG section for a Python script.\n"
                        "HARD REQUIREMENTS:\n"
                        "- Output ONLY a Python code block that declares constants/config/state.\n"
                        "- No function or class definitions here. Keep it minimal and clear; add brief inline comments if helpful.\n"
                        "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
                        f"User's goal:\n{goal}\n\n"
                        f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
                        f"Items to cover:\n{spec_text}\n\n"
                        f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
                    )
                elif section_kind == "function":
                    fname = section_spec.get("name", "")
                    fdesc = section_spec.get("desc", "")
                    synth_prompt = (
                        "You are generating a SINGLE FUNCTION for a Python script.\n"
                        "HARD REQUIREMENTS:\n"
                        f"- Define exactly one function: `{fname}`.\n"
                        "- Include a concise docstring that explains inputs/outputs/side-effects.\n"
                        "- Use only built-in types in annotations to avoid extra imports.\n"
                        "- Do NOT include other functions, classes, imports, or main().\n"
                        "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
                        f"User's goal:\n{goal}\n\n"
                        f"Function purpose:\n{fdesc}\n\n"
                        f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
                        f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
                    )
                else:  # main
                    synth_prompt = (
                        "You are generating ONLY the MAIN section for a Python script.\n"
                        "Output ONLY valid Python code (no prose, no lists, no comments, no fences).\n"
                        "It must consist of exactly two top-level statements in this order:\n"
                        "def main():\n"
                        "        # orchestrate previously defined helpers only\n"
                        "\n"
                        "if __name__ == \"__main__\":\n"
                        "        main()\n"
                        "Rules:\n"
                        "- STRICT: Call only functions that ALREADY EXIST in the 'Current script' shown below. Do NOT invent or guess names from the outline. If a needed helper is missing, omit that call.\n"
                        "- Prefer the minimal sequence to start the program (e.g., GUI setup then event loop if present).\n"
                        "- Do not re-import or redefine anything.\n"
                        "- Do not create nested defs/classes.\n"
                        "- Keep main() short and deterministic.\n"
                        "- All block indentation must be 4 spaces.\n\n"
                        f"User's goal:\n{goal}\n\n"
                        f"Main high-level plan from outline:\n{section_spec}\n\n"
                        f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
                        f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
                    )

                print("\n\n\n\nSYNTH (SECTION) PROMPT (attempt " + str(attempt) + "):\n\n" + str(synth_prompt))
                raw_response = llm(synth_prompt, max_new_tokens=4096, temperature=0.25, top_p=0.9, do_sample=True)
                candidate_code = sanitize_candidate(extract_code_block(raw_response), section_kind)
                last_candidate_code = candidate_code
                print("\n\n\n\nSYNTH (SECTION) RESPONSE (attempt " + str(attempt) + "):\n\n" + str(raw_response))
                if section_kind == "main":
                    candidate_code = ensure_main_guard(candidate_code)

                # Collect the synthesized candidate
                collected_candidates.append((f"SYNTH_{attempt}", candidate_code))

                ok_basic = True
                basic_issues: list[str] = []

                if section_kind == "function":
                    fname = section_spec.get("name", "")
                    if not re.search(rf"^\s*def\s+{re.escape(fname)}\s*\(", candidate_code, flags=re.M):
                        ok_basic = False
                        basic_issues.append(f"Missing or misnamed function definition: def {fname}(...).")

                elif section_kind == "imports":
                    bad_lines = []
                    for ln in candidate_code.splitlines():
                        s = ln.strip()
                        if not s:
                            continue
                        if not (s.startswith("import ") or s.startswith("from ")):
                            bad_lines.append(ln)
                    if bad_lines:
                        ok_basic = False
                        basic_issues.append("Imports section contains non-import lines:")
                        basic_issues.extend(f"  • {bl}" for bl in bad_lines)

                elif section_kind == "globals":
                    if re.search(r"^\s*(def|class)\s+", candidate_code, flags=re.M):
                        ok_basic = False
                        basic_issues.append("Globals section contains a function/class definition.")

                elif section_kind == "main":
                    has_def = re.search(r"^\s*(?:async\s+)?def\s+\w+\s*\(", candidate_code, flags=re.M)
                    has_guard = re.search(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", candidate_code, flags=re.M)
                    if not has_def:
                        basic_issues.append("MAIN section is missing a top-level def main(...): block.")
                    if not has_guard:
                        basic_issues.append('MAIN section is missing the __main__ guard: if __name__ == "__main__": main()')
                    ok_basic = bool(has_def and has_guard)

                compile_ok, compile_err, maybe_fixed = compiles_with_script(candidate_code)
                candidate_code = maybe_fixed

                if not ok_basic or not compile_ok:
                    print("❌ Basic validation/compilation failed for candidate code.")
                    wait_for_enter("Basic validation/compilation failed")

                    checker_notes = ""
                    if basic_issues:
                        checker_notes += "- Structural/validation issues:\n" + "\n".join(f"  {x}" for x in basic_issues) + "\n"
                    if not compile_ok and compile_err:
                        checker_notes += f"- Compilation error:\n  {compile_err}\n"

                    detailed_expl = pre_reviser_explain(
                        goal=goal,
                        script_so_far=script_so_far,
                        section_kind=section_kind,
                        section_spec=section_spec,
                        candidate_code=candidate_code,
                        checker_notes=checker_notes or "No additional notes."
                    )
                    print("\n\n[CHECKER NOTES]\n" + checker_notes)
                    print("\n\n[PRE-REVISER EXPLANATION]\n" + detailed_expl)
                    critique = f"CHECKER NOTES:\n{checker_notes}\nDETAILED EXPLANATION:\n{detailed_expl}"
                else:
                    critique = critique_section(goal, script_so_far, section_kind, section_spec, candidate_code)
                    if (critique or "").strip().lower() != "ok":
                        print("❌ Critique failed. See details above.")
                        wait_for_enter("Critique failed")

                if (critique or "").strip().lower() == "ok":
                    final_code = candidate_code
                else:
                    rewrite_candidate = sanitize_candidate(
                        rewrite_section(goal, script_so_far, section_kind, section_spec, critique, candidate_code),
                        section_kind,
                    )
                    if section_kind == "main":
                        rewrite_candidate = ensure_main_guard(rewrite_candidate)

                    # Collect the rewrite candidate regardless of success
                    collected_candidates.append((f"REWRITE_{attempt}", rewrite_candidate))

                    compile_ok2, compile_err2, maybe_fixed2 = compiles_with_script(rewrite_candidate)

                    if not compile_ok2:
                        print("❌ Rewritten code failed compilation.")
                        wait_for_enter("Rewritten code failed compilation")
                        continue
                    else:
                        final_code = maybe_fixed2
                    recheck = critique_section(goal, script_so_far, section_kind, section_spec, final_code)
                    if (recheck or "").strip().lower() != "ok":
                        print("❌ Recheck after rewrite failed. See details above.")
                        wait_for_enter("Recheck after rewrite failed")
                        continue

                # Success
                break
            else:
                # Exhausted attempts → ask LLM to pick the best among all collected candidates
                if collected_candidates:
                    chosen_label = _llm_pick_best_section_candidate(
                        section_kind_local=section_kind,
                        section_spec_local=section_spec,
                        script_ctx=script_so_far,
                        goal_ctx=goal,
                        labeled_candidates=collected_candidates,
                    )
                    # Find the chosen code, fallback to last candidate if not found
                    chosen_code = None
                    for lab, code in collected_candidates:
                        if lab == chosen_label:
                            chosen_code = code
                            break
                    final_code = chosen_code or (last_candidate_code or "")
                else:
                    final_code = last_candidate_code or ""

                if section_kind == "main":
                    final_code = ensure_main_guard(final_code)
                # Final safety compile try; if it fails, keep as-is (existing behavior was to use last candidate)
                _ok, _err, maybe_fixed_final = compiles_with_script(final_code)
                if _ok:
                    final_code = maybe_fixed_final

            if final_code:
                if globals().get("FRAMEWORK_MODE", False):
                    # Replace the corresponding section in-place instead of appending
                    updated_script = script_so_far
                    if section_kind == "imports":
                        updated_script = self.replace_imports_block(updated_script, final_code)
                        label = "IMPORTS"
                    elif section_kind == "globals":
                        updated_script = self.replace_globals_block(updated_script, final_code)
                        label = "GLOBALS"
                    elif section_kind == "function":
                        fname = section_spec.get("name", "") if isinstance(section_spec, dict) else ""
                        if fname:
                            existing = self.extract_function_block(updated_script, fname)
                            if existing:
                                updated_script = self.replace_function_block(updated_script, fname, final_code)
                            else:
                                # Insert before main() if present, else append
                                main_blk = self.extract_main_block(updated_script)
                                if main_blk and main_blk in updated_script:
                                    idx = updated_script.find(main_blk)
                                    updated_script = updated_script[:idx].rstrip() + "\n\n" + final_code.rstrip() + "\n\n" + updated_script[idx:]
                                else:
                                    updated_script = updated_script.rstrip() + "\n\n" + final_code.rstrip() + "\n"
                        label = fname or "function"
                    else:  # main
                        updated_script = self.replace_main_block(updated_script, ensure_main_guard(final_code))
                        label = "MAIN"

                    SCRIPT_LINES = updated_script.splitlines()
                    self.refresh_script_editor()
                    self.root.after(0, lambda: self.append_display(f"Generated (replaced) section: {label}", "response"))
                    # ADVANCE TO NEXT SECTION
                    SECTION_INDEX += 1
                    self.prog_inc()

                else:
                    # Original behavior: append
                    cleaned_lines = []
                    for ln in final_code.splitlines():
                        stripped = ln.strip()
                        if stripped.startswith("#") and not stripped.lstrip("#").strip() == "":
                            continue
                        cleaned_lines.append(ln)
                    SCRIPT_LINES.extend(cleaned_lines)

                    full_script_text = join_script(SCRIPT_LINES)
                    def _set_script_text(s=full_script_text):
                        try:
                            self.script_editor.delete("1.0", "end")
                            self.script_editor.insert("1.0", s)
                            self.script_editor.edit_modified(False)
                        except Exception:
                            pass
                    self.root.after(0, _set_script_text)

                    label = (section_spec.get("name") if section_kind == "function" and isinstance(section_spec, dict)
                             else section_kind.upper())
                    self.root.after(0, lambda: self.append_display(f"Generated and integrated section: {label}", "response"))
                    # ADVANCE TO NEXT SECTION
                    SECTION_INDEX += 1
                    self.prog_inc()

            self.root.after(0, self.finish_cycle)


        finally:
            self.gen_lock.release()




    def finish_cycle(self):
        # Keep Next Section button disabled during auto-run
        finished = SECTIONS_PLAN and SECTION_INDEX >= len(SECTIONS_PLAN)

        if finished:
            self.send_btn.config(state="disabled")
            if not self.post_run_started:
                self.post_run_started = True
                # Run 5-pass full-script critique & targeted fixes BEFORE any execution
                threading.Thread(target=self.pre_run_critique_and_fix, daemon=True).start()
        else:
            self.send_btn.config(state="normal")

    def create_new_function_via_normal_process(self, func_name: str, func_desc: str) -> bool:
        """
        Generate a new function using the same single-function prompt,
        critique it, possibly rewrite, then insert it before main().
        Returns True if the function was integrated and compiles with the script.
        """
        global GOAL_SPEC, FULL_OUTLINE, SCRIPT_LINES
        script_so_far = join_script(SCRIPT_LINES)

        # Synthesize (same spec as your normal single-function generation)
        prompt = (
            "You are generating a SINGLE FUNCTION for a Python script.\n"
            "HARD REQUIREMENTS:\n"
            f"- Define exactly one function: `{func_name}`.\n"
            "- Include a concise docstring that explains inputs/outputs/side-effects.\n"
            "- Use only built-in types in annotations to avoid extra imports.\n"
            "- Do NOT include other functions, classes, imports, or main().\n"
            "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
            f"User's goal:\n{GOAL_SPEC or ''}\n\n"
            f"Function purpose:\n{func_desc}\n\n"
            f"Full Script's Outline (reference):\n{FULL_OUTLINE or ''}\n\n"
            f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
        )
        raw = llm(prompt, max_new_tokens=4096, temperature=0.25, top_p=0.9, do_sample=True)
        candidate = sanitize_candidate(extract_code_block(raw), "function")

        # Basic check: correct function name
        if not re.search(rf"^\s*def\s+{re.escape(func_name)}\s*\(", candidate, flags=re.M):
            # Seed a rewrite with an explicit note
            critique_blob = f"Function header must be exactly: def {func_name}(...):"
            candidate = sanitize_candidate(
                rewrite_section(GOAL_SPEC or "", script_so_far, "function",
                                {"name": func_name, "desc": func_desc},
                                critique_blob, candidate),
                "function",
            )

        # Focused critique → rewrite if needed
        reviewed = critique_section(GOAL_SPEC or "", script_so_far, "function",
                                    {"name": func_name, "desc": func_desc}, candidate)
        if (reviewed or "").strip().lower() != "ok":
            candidate = sanitize_candidate(
                rewrite_section(GOAL_SPEC or "", script_so_far, "function",
                                {"name": func_name, "desc": func_desc},
                                reviewed, candidate),
                "function",
            )

        # Insert new function BEFORE main()
        full_script = join_script(SCRIPT_LINES)
        main_blk = self.extract_main_block(full_script)
        if main_blk and main_blk in full_script:
            idx = full_script.find(main_blk)
            new_script = full_script[:idx].rstrip() + "\n\n" + candidate.rstrip() + "\n\n" + full_script[idx:]
        else:
            new_script = full_script.rstrip() + "\n\n" + candidate.rstrip() + "\n"

        # Compile check
        try:
            compile(new_script, "<new-function-integration>", "exec")
        except SyntaxError as e:
            msg = f"Failed to compile new function {func_name}: {e}"
            self.root.after(0, self.append_display, msg, "response")
            return False


        SCRIPT_LINES = new_script.splitlines()
        self.root.after(0, lambda: self.append_display(f"Created missing function: {func_name}", "response"))
        return True

    def handle_missing_function_during_run(self, func_name: str, error_text: str) -> bool:
        """
        Derive a concise outline description for the missing function from the traceback + code,
        append it to the outline, then create the function via the normal process.
        """
        global FULL_OUTLINE
        script_so_far = join_script(SCRIPT_LINES)
        desc_prompt = (
            "You are updating a Python script outline.\n"
            f"The program crashed because a function named `{func_name}` was referenced but not defined.\n"
            "Write ONE or TWO precise sentences describing what this function should do,\n"
            "including inputs (parameters or event object), outputs/side-effects, and any GUI/keybinding context if relevant.\n"
            "OUTPUT ONLY the sentences—no code, no bullets, no headings."
            f"\n\nTRACEBACK:\n{error_text}\n\nCURRENT SCRIPT:\n{script_so_far}\n"
        )
        desc = clean(extract_code_block(llm(desc_prompt, max_new_tokens=4096, temperature=0.2))).strip()
        if not desc:
            desc = "Auto-added function required by runtime error. Handles the specific event and updates the GUI as needed."

        # Append to outline + refresh plan
        add_function_to_outline(func_name, desc)
        # reflect the FUNCTIONS change in the Outline panel right away
        self.publish_outline(FULL_OUTLINE, f"FUNCTIONS: added {func_name}")

        # Create the function using the normal creation pipeline
        ok = self.create_new_function_via_normal_process(func_name, desc)
        return ok
    def run_full_script_critiques(self, goal: str, outline: str, script_text: str) -> str:
        """
        Run five whole-script critique passes. Returns 'OK' iff ALL passes are OK,
        otherwise returns bullet points prefixed with the aspect name.
        """
        aspects: list[tuple[str, str]] = [
            (
                "Completeness vs Outline",
                "- Every function listed in the outline exists and is fully implemented.\n"
                "- MAIN exists and includes the __main__ guard.\n"
                "- IMPORTS and GLOBALS are sufficient for the referenced names.\n"
                "- No TODO/pass/ellipsis placeholders."
            ),
            (
                "Dependency Resolution",
                "- All referenced identifiers are defined or imported before use.\n"
                "- Function calls use correct arity/names; no forward refs that break execution.\n"
                "- No missing modules or symbols implied by the outline."
            ),
            (
                "Structure & Cohesion",
                "- No duplicate function definitions or conflicting globals.\n"
                "- No helper/function/class definitions inside MAIN.\n"
                "- Reasonable separation: imports at top, globals next, then defs, then main guard."
            ),
            (
                "Runtime Readiness",
                "- Code is concrete (no pseudocode) and likely to execute without NameError/AttributeError.\n"
                "- Threading/locks used consistently; no obvious dead references in callbacks.\n"
                "- GUI startup path is reachable from main()."
            ),
            (
                "Style & Hygiene",
                "- No duplicate imports; avoid unused imports if evident.\n"
                "- Constants in GLOBALS use UPPER_CASE when applicable.\n"
                "- One and only one __main__ guard."
            ),
        ]

        def run_aspect(name: str, criteria: str) -> str:
            prompt = (
                "You are a strict Python reviewer.\n"
                f"ASPECT: {name}\n"
                "Decide ONLY for this aspect if the FULL SCRIPT is acceptable.\n"
                "Write ONLY 'OK' if acceptable; otherwise list succinct bullet issues.\n\n"
                "CRITERIA:\n"
                f"{criteria}\n\n"
                f"User's goal:\n{goal}\n\n"
                f"Full Script's Outline (reference):\n{outline}\n\n"
                f"Full script under review:\n{script_text}\n"
            )
            print("\n\n\n\nFULL-SCRIPT CRITIQUE (ASPECT: " + name + ") PROMPT:\n\n" + str(prompt))
            resp = llm(prompt, max_new_tokens=4096, temperature=0.2)
            print("\n\n\n\nFULL-SCRIPT CRITIQUE (ASPECT: " + name + ") RESPONSE:\n\n" + str(resp))
            return resp

        failures: list[str] = []
        for name, criteria in aspects:
            raw = run_aspect(name, criteria)
            if _critique_is_ok(raw):
                continue
            lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
            if not lines:
                lines = ["Unspecified issue."]
            for ln in lines:
                failures.append(f"- [{name}] {ln}")

        return "OK" if not failures else "\n".join(failures)
    def request_run_feedback(self, round_idx: int):
        """
        Prompt the user for run feedback using a modal popup. Blocks the worker thread
        on an Event until the user submits (empty allowed).
        """
        # Prepare to receive feedback
        self.latest_feedback = None
        self.feedback_event.clear()
        self.awaiting_run_feedback = True

        msg = f"Run {round_idx} completed. Describe any issues and press Submit. (Or you can tell the AI to end testing.)"
        self.root.after(0, lambda: self.append_display(msg, "prompt"))

        def _submit_fb(text: str):
            self.latest_feedback = text
            self.awaiting_run_feedback = False
            try:
                self.send_btn.config(text="Next Section (Ctrl+Enter)", state="disabled")  # hidden anyway
            except Exception:
                pass
            self.feedback_event.set()

        # Show the SAME message on the popup and collect feedback there
        self.root.after(0, lambda: self.open_input_popup(
            msg,
            "Submit Feedback (Ctrl+Enter)",
            _submit_fb
        ))

        # Wait (background thread) until user submits in the popup
        self.feedback_event.wait()

        # Record feedback into history (unchanged behavior)
        fb = (self.latest_feedback or "").strip()
        if fb:
            CHAT_HISTORY.append((f"User feedback after run {round_idx}:", fb))
            self.root.after(0, lambda: self.append_display("Feedback noted.", "response"))
    def llm_should_end_testing(self, feedback: str, run_ctx: dict | None = None) -> bool:
        """
        Returns True if the user's run feedback clearly signals we're done.
        Uses a quick heuristic first, then asks the LLM with a strict JSON schema.
        """
        if not feedback or not feedback.strip():
            return False

        text = feedback.strip()

        # Fast-path heuristic (common phrases); ignore obvious negations.
        neg_hits = re.search(r"\b(not|isn['’]t|ain['’]t|don['’]t|doesn['’]t|no|almost|nearly|except)\b", text, flags=re.I)
        done_hits = re.search(
            r"\b(done|finished|complete|good\s*to\s*go|g2g|looks\s*(good|great|perfect)|"
            r"all\s*set|ship\s*it|ready\s*to\s*(ship|go|publish|use)|"
            r"no\s*further\s*(changes|edits|fixes)|works\s*for\s*me|LGTM|👍)\b",
            text,
            flags=re.I
        )
        if done_hits and not neg_hits:
            return True  # confident enough without calling the model

        # Ask the LLM to decide (strict JSON)
        run_ctx = run_ctx or {}
        ctx_json = json.dumps(run_ctx, ensure_ascii=False, default=str)

        prompt = (
            "You are a precise classifier.\n"
            "Decide if the user's feedback indicates testing should END NOW.\n"
            "Return ONLY JSON like: {\"end_testing\": true/false, \"reason\": \"<short>\"}\n"
            "Guidelines:\n"
            "- end_testing = true for phrases like: done, finished, good to go, ship it, all set, LGTM, looks good, ready to publish.\n"
            "- end_testing = false if they ask for any change, mention issues, or express uncertainty (even mildly).\n"
            "- Negations like 'not done', 'almost done', 'looks good but...' => false.\n\n"
            f"FEEDBACK:\n{text}\n\n"
            "RUN_CONTEXT (optional):\n"
            f"{ctx_json}\n"
        )

        raw = llm(prompt, max_new_tokens=4096, temperature=0.1)
        blob = _extract_json_blob(raw) or raw
        data = _safe_json_loads(blob, {}) or {}
        return bool(data.get("end_testing") is True)



    def llm_select_fix_target_from_critique(self, critique_text: str, full_script: str, outline: str) -> dict:
        """
        Choose the smallest single target to modify based on critique feedback.
        Returns JSON with:
          - target_kind: one of ["function","main","imports","globals"]
          - name: function name if target_kind == "function"
        """
        prompt = (
            "You are a Python maintenance expert. Given a full script, its outline, and critique notes,\n"
            "choose the ONE smallest target to edit so that the critique is satisfied.\n"
            "Respond ONLY as JSON with keys:\n"
            '  "target_kind": one of ["function","main","imports","globals"]\n'
            '  "name": the function name if target_kind=="function" (omit otherwise)\n'
            "Pick the highest-leverage minimal fix; avoid regenerating the whole script.\n\n"
            f"OUTLINE:\n{outline}\n\n"
            f"SCRIPT:\n{full_script}\n\n"
            f"CRITIQUE:\n{critique_text}\n"
        )
        print("\n\n\n\nSELECT FIX TARGET FROM CRITIQUE PROMPT:\n\n" + str(prompt))
        raw = llm(prompt, max_new_tokens=4096, temperature=0.2)
        print("\n\n\n\nSELECT FIX TARGET FROM CRITIQUE RESPONSE:\n\n" + str(raw))

        txt = extract_code_block(raw) or raw
        try:
            return json.loads(txt.strip())
        except Exception:
            m = re.search(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"', txt)
            if m:
                return {"target_kind": "function", "name": m.group(1)}
            # Fallback heuristic
            if re.search(r"\bimport\b|\bmissing module\b|IMPORT", critique_text, flags=re.I):
                return {"target_kind": "imports"}
            return {"target_kind": "main"}

    def llm_generate_imports_fix(self, current_imports: str, outline: str, full_script: str, critique_text: str) -> str:
        """
        Produce ONLY the corrected imports block (one import per line, no comments).
        """
        prompt = (
            "You are fixing ONLY the imports block at the top of a Python script.\n"
            "HARD REQUIREMENTS:\n"
            "- Output ONLY import statements to replace the current imports block.\n"
            "- One import per line; avoid duplicates; no comments or blank lines.\n"
            "- Include everything required by names referenced in the script and outline.\n\n"
            f"CURRENT IMPORTS:\n{current_imports}\n\n"
            f"OUTLINE (reference):\n{outline}\n\n"
            f"SCRIPT (reference):\n{full_script}\n\n"
            f"CRITIQUE NOTES:\n{critique_text}\n"
        )
        print("\n\n\n\nGENERATE IMPORTS FIX PROMPT:\n\n" + str(prompt))
        raw = llm(prompt, max_new_tokens=4096, temperature=0.2)
        print("\n\n\n\nGENERATE IMPORTS FIX RESPONSE:\n\n" + str(raw))

        return extract_code_block(raw).strip()

    def llm_generate_globals_fix(self, current_globals: str, outline: str, full_script: str, critique_text: str) -> str:
        """
        Produce ONLY the corrected globals/config block (top-level assignments/constants).
        """
        prompt = (
            "You are fixing ONLY the GLOBALS/CONFIG block of a Python script.\n"
            "HARD REQUIREMENTS:\n"
            "- Output ONLY top-level assignments/constants to replace the current globals block.\n"
            "- No function/class defs here; keep it minimal and sufficient.\n"
            "- Use UPPER_CASE for constants where appropriate.\n\n"
            f"CURRENT GLOBALS:\n{current_globals}\n\n"
            f"OUTLINE (reference):\n{outline}\n\n"
            f"SCRIPT (reference):\n{full_script}\n\n"
            f"CRITIQUE NOTES:\n{critique_text}\n"
        )
        print("\n\n\n\nGENERATE GLOBALS FIX PROMPT:\n\n" + str(prompt))
        raw = llm(prompt, max_new_tokens=4096, temperature=0.2)
        print("\n\n\n\nGENERATE GLOBALS FIX RESPONSE:\n\n" + str(raw))

        return extract_code_block(raw).strip()

    def extract_imports_block(self, script_text: str) -> str:
        lines = script_text.splitlines(True)
        n = len(lines)
        # Skip initial comments/blank lines
        i = 0
        while i < n and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
            i += 1
        start = None
        j = i
        while j < n:
            s = lines[j].lstrip()
            if s.startswith("import ") or s.startswith("from "):
                if start is None:
                    start = j
                j += 1
                continue
            if start is not None and (s.strip() == "" or s.startswith("#")):
                j += 1
                continue
            break
        if start is None:
            return ""
        end = j
        return "".join(lines[start:end])

    def replace_imports_block(self, script_text: str, new_imports: str) -> str:
        lines = script_text.splitlines(True)
        n = len(lines)
        # Locate existing imports block or insertion point
        i = 0
        while i < n and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
            i += 1
        start = None
        j = i
        while j < n:
            s = lines[j].lstrip()
            if s.startswith("import ") or s.startswith("from "):
                if start is None:
                    start = j
                j += 1
                continue
            if start is not None and (s.strip() == "" or s.startswith("#")):
                j += 1
                continue
            break
        if start is None:
            # Insert imports at position i
            new_block = (new_imports.rstrip() + "\n")
            return "".join(lines[:i]) + new_block + "".join(lines[i:])
        end = j
        new_block = (new_imports.rstrip() + "\n")
        return "".join(lines[:start]) + new_block + "".join(lines[end:])

    def extract_globals_block(self, script_text: str) -> str:
        # Identify after imports block
        lines = script_text.splitlines(True)
        n = len(lines)
        # Find end of imports region
        i = 0
        while i < n and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
            i += 1
        started = False
        while i < n:
            s = lines[i].lstrip()
            if s.startswith("import ") or s.startswith("from ") or s.strip() == "" or s.startswith("#"):
                i += 1
                started = True or started
                continue
            break
        start = i
        # Globals continue until first def/class/if __name__ guard/decorator
        j = start
        while j < n:
            s = lines[j].lstrip()
            if s.startswith("def ") or s.startswith("class ") or s.startswith("@") or re.match(r'if\s+__name__\s*==', s):
                break
            j += 1
        if j <= start:
            return ""
        return "".join(lines[start:j]).rstrip("\n")

    def replace_globals_block(self, script_text: str, new_globals: str) -> str:
        lines = script_text.splitlines(True)
        n = len(lines)
        # Find start after imports
        i = 0
        while i < n and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
            i += 1
        while i < n and (lines[i].lstrip().startswith("import ") or lines[i].lstrip().startswith("from ") or lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
            i += 1
        start = i
        j = start
        while j < n and not (lines[j].lstrip().startswith("def ") or lines[j].lstrip().startswith("class ") or lines[j].lstrip().startswith("@") or re.match(r'\s*if\s+__name__\s*==', lines[j])):
            j += 1
        new_block = (new_globals.rstrip() + "\n") if new_globals.strip() else ""
        return "".join(lines[:start]) + new_block + "".join(lines[j:])

    def pre_run_critique_and_fix(self, max_rounds: int = 8):
        """
        BEFORE any execution: sweep the script SECTION-BY-SECTION using the OUTLINE.
        For each section (imports, globals, EACH function, then main):
          - If the section is missing, run the NORMAL CREATION process for that section.
          - If the section exists, run the NORMAL REVISION process (critique_section -> rewrite_section)
            focused ONLY on that section, while providing the FULL SCRIPT as context.

        Update: this version uses an AST-based existence check for functions to avoid
        false "missing" detections that caused duplicate re-creation. It also de-dups
        function entries in SECTIONS_PLAN for the sweep.

        NEW: as part of the pre-run checks, remove any duplicate top-level function
        definitions that are exactly the same (same name AND identical body).
        """
        import ast
        import re

        global SCRIPT_LINES, FULL_OUTLINE, GOAL_SPEC, SECTIONS_PLAN
        self.root.after(0, lambda: self.append_display("Starting pre-run checks (QA sweep)...", "response"))

        # ---------- helpers ----------
        def build_script_with(section_kind: str, section_spec: dict | list | str, block: str, base_script: str) -> str:
            """Return a new script text with `block` placed/replaced for the given section."""
            if section_kind == "imports":
                return self.replace_imports_block(base_script, block)
            if section_kind == "globals":
                return self.replace_globals_block(base_script, block)
            if section_kind == "function":
                fname = section_spec.get("name", "") if isinstance(section_spec, dict) else ""
                # If function exists, replace; else insert before main() guard (or append).
                if ast_has_function(base_script, fname):
                    # Prefer AST-based block extraction; fall back to regex-based extractor.
                    existing_ast = ast_get_function_block(base_script, fname)
                    if existing_ast:
                        return base_script.replace(existing_ast, block.rstrip() + "\n", 1)
                    if self.extract_function_block(base_script, fname):
                        return self.replace_function_block(base_script, fname, block)
                # Insert before main if present, else append
                main_blk = self.extract_main_block(base_script)
                if main_blk and main_blk in base_script:
                    idx = base_script.find(main_blk)
                    return base_script[:idx].rstrip() + "\n\n" + block.rstrip() + "\n\n" + base_script[idx:]
                return base_script.rstrip() + "\n\n" + block.rstrip() + "\n"
            # main
            fixed = ensure_main_guard(block)
            return self.replace_main_block(base_script, fixed)

        def compiles_script(new_script_text: str) -> tuple[bool, str]:
            try:
                compile(new_script_text, "<pre-run-sweep>", "exec")
                return True, ""
            except SyntaxError as e:
                return False, f"{type(e).__name__}: {e}"

        def synthesize_section(section_kind: str, section_spec: dict | list | str, script_so_far: str) -> str:
            """Run the same NORMAL CREATION prompt style used during generation."""
            goal = GOAL_SPEC or ""
            if section_kind == "imports":
                spec_text = "\n".join(f"- {x}" for x in (section_spec or []))
                prompt = (
                    "You are generating the IMPORTS section for a Python script.\n"
                    "HARD REQUIREMENTS:\n"
                    "- Output ONLY a Python code block with import statements needed for the entire script per the outline.\n"
                    "- One import per line; avoid redundant imports; no other code or comments.\n"
                    "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
                    f"User's goal:\n{goal}\n\n"
                    f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
                    f"Items to cover:\n{spec_text}\n\n"
                    f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
                )
            elif section_kind == "globals":
                spec_text = "\n".join(f"- {x}" for x in (section_spec or []))
                prompt = (
                    "You are generating the GLOBALS/CONFIG section for a Python script.\n"
                    "HARD REQUIREMENTS:\n"
                    "- Output ONLY a Python code block that declares constants/config/state.\n"
                    "- No function or class definitions here. Keep it minimal and clear; add brief inline comments if helpful.\n"
                    "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
                    f"User's goal:\n{goal}\n\n"
                    f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
                    f"Items to cover:\n{spec_text}\n\n"
                    f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
                )
            elif section_kind == "function":
                fname = section_spec.get("name", "")
                fdesc = section_spec.get("desc", "")
                prompt = (
                    "You are generating a SINGLE FUNCTION for a Python script.\n"
                    "HARD REQUIREMENTS:\n"
                    f"- Define exactly one function: `{fname}`.\n"
                    "- Include a concise docstring that explains inputs/outputs/side-effects.\n"
                    "- Use only built-in types in annotations to avoid extra imports.\n"
                    "- Do NOT include other functions, classes, imports, or main().\n"
                    "- DO NOT include ANY info from the outline in your response. Do not include any extra letter, numbers, characters, labels, titles, or anything other than literally only the new code section that you are tasked to create. Do not include any extra labels or characters that are not directly the code itself.\n\n"
                    f"User's goal:\n{goal}\n\n"
                    f"Function purpose:\n{fdesc}\n\n"
                    f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
                    f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
                )
            else:  # main
                prompt = (
                    "You are generating ONLY the MAIN section for a Python script.\n"
                    "Output ONLY valid Python code (no prose, no lists, no comments, no fences).\n"
                    "It must consist of exactly two top-level statements in this order:\n"
                    "def main():\n"
                    "        # orchestrate previously defined helpers only\n"
                    "\n"
                    "if __name__ == \"__main__\":\n"
                    "        main()\n"
                    "Rules:\n"
                    "- STRICT: Call only functions that ALREADY EXIST in the 'Current script' shown below. Do NOT invent or guess names from the outline. If a needed helper is missing, omit that call.\n"
                    "- Prefer the minimal sequence to start the program.\n"
                    "- Do not re-import or redefine anything.\n"
                    "- Do not create nested defs/classes.\n"
                    "- Keep main() short and deterministic.\n"
                    "- All block indentation must be 4 spaces.\n\n"
                    f"User's goal:\n{goal}\n\n"
                    f"Main high-level plan from outline:\n{section_spec}\n\n"
                    f"Full Script's Outline (reference):\n{FULL_OUTLINE}\n\n"
                    f"Current script:\n{script_so_far}\n\n(END OF SCRIPT)\n"
                )
            raw = llm(prompt, max_new_tokens=4096, temperature=0.25, top_p=0.9, do_sample=True)
            code = sanitize_candidate(extract_code_block(raw), section_kind)
            if section_kind == "main":
                code = ensure_main_guard(code)
            return code

        def review_and_optionally_rewrite(section_kind: str, section_spec: dict | list | str, candidate_code: str, script_so_far: str) -> str:
            """Run critique_section; if not OK, run rewrite_section to get a corrected section."""
            critique = critique_section(GOAL_SPEC or "", script_so_far, section_kind, section_spec, candidate_code, integration_mode="replace")
            if (critique or "").strip().lower() == "ok":
                return candidate_code
            fixed = sanitize_candidate(
                rewrite_section(GOAL_SPEC or "", script_so_far, section_kind, section_spec, critique, candidate_code),
                section_kind,
            )
            if section_kind == "main":
                fixed = ensure_main_guard(fixed)
            return fixed

        # ---------- robust function existence helpers (AST-first) ----------
        def ast_has_function(script_text: str, name: str) -> bool:
            """True if a top-level function with `name` exists (decorators allowed)."""
            try:
                tree = ast.parse(script_text)
            except SyntaxError:
                return False
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return True
            return False

        def ast_get_function_block(script_text: str, name: str) -> str:
            """
            Return the exact source block for a top-level function using AST
            line spans when available. Falls back to '' if not found.
            """
            try:
                tree = ast.parse(script_text)
            except SyntaxError:
                return ""
            lines = script_text.splitlines(True)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    # Python 3.8+ provides end_lineno; if absent, try regex fallback
                    if hasattr(node, "lineno") and hasattr(node, "end_lineno") and node.end_lineno:
                        start = max(1, node.lineno) - 1
                        # include decorators if present
                        if getattr(node, "decorator_list", None):
                            start = min(start, min(max(1, d.lineno) - 1 for d in node.decorator_list))
                        end = max(node.end_lineno, node.lineno)  # end is inclusive
                        return "".join(lines[start:end])
                    break
            # Fallback: a simple header regex (looser than the full-block extractor)
            m = re.search(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(name)}\s*\(", script_text)
            if not m:
                return ""
            # Use the class's regex extractor if header exists
            block = self.extract_function_block(script_text, name)
            return block or ""

        # ---------- NEW: remove duplicate functions that are exactly identical ----------
        def _dedupe_identical_functions(script_text: str) -> tuple[str, int]:
            """
            Remove duplicate top-level function definitions that are textually identical
            (after whitespace normalization), keeping the first occurrence.
            Duplicates must have the SAME function name and identical normalized source.
            Returns (new_script_text, removed_count).
            """
            try:
                tree = ast.parse(script_text)
            except SyntaxError:
                return script_text, 0

            lines = script_text.splitlines(True)
            blocks = []  # (name, start_idx, end_idx, src)

            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    if hasattr(node, "lineno"):
                        start = max(1, node.lineno) - 1
                        if getattr(node, "decorator_list", None):
                            start = min(start, min(max(1, d.lineno) - 1 for d in node.decorator_list))
                        if hasattr(node, "end_lineno") and node.end_lineno:
                            end = max(node.end_lineno, node.lineno)
                        else:
                            # fallback: regex slice if end_lineno missing
                            blk = self.extract_function_block(script_text, node.name)
                            if not blk:
                                continue
                            # locate blk span
                            joined = "".join(lines)
                            idx = joined.find(blk)
                            if idx < 0:
                                continue
                            upto = joined[:idx]
                            start = upto.count("\n")
                            end = start + blk.count("\n") + (0 if blk.endswith("\n") else 1)
                        src = "".join(lines[start:end])
                        blocks.append((node.name, start, end, src))

            seen: dict[tuple[str, str], tuple[int, int]] = {}
            to_remove: list[tuple[int, int]] = []
            for name, start, end, src in blocks:
                key = (name, _normalize_ws(src))
                if key in seen:
                    to_remove.append((start, end))
                else:
                    seen[key] = (start, end)

            if not to_remove:
                return script_text, 0

            # delete from bottom to top to preserve indices
            to_remove.sort(key=lambda p: p[0], reverse=True)
            for start, end in to_remove:
                del lines[start:end]
            return "".join(lines), len(to_remove)

        # Apply deduplication before the sweep
        _before = join_script(SCRIPT_LINES)
        _after, _removed = _dedupe_identical_functions(_before)
        if _removed:
            SCRIPT_LINES = _after.splitlines()
            self.refresh_script_editor()
            self.root.after(0, lambda rc=_removed: self.append_display(f"Removed {rc} duplicate function definition(s).", "response"))


        # ---------- ensure we have a plan & de-dup functions locally ----------
        if not SECTIONS_PLAN:
            parsed = parse_outline(FULL_OUTLINE or "")
            SECTIONS_PLAN = build_sections_plan(parsed)

        deduped_plan = []
        seen_funcs = set()
        for kind, spec in SECTIONS_PLAN:
            if kind == "function":
                fname = spec.get("name", "") if isinstance(spec, dict) else ""
                if fname and fname in seen_funcs:
                    continue
                if fname:
                    seen_funcs.add(fname)
            deduped_plan.append((kind, spec))

        # Sweep rounds
        for round_idx in range(1, max_rounds + 1):
            made_change = False

            for section_kind, section_spec in deduped_plan:
                script_text = join_script(SCRIPT_LINES)

                # --- Determine existing block robustly ---
                if section_kind == "imports":
                    existing = self.extract_imports_block(script_text)

                elif section_kind == "globals":
                    existing = self.extract_globals_block(script_text)

                elif section_kind == "function":
                    fname = section_spec.get("name", "") if isinstance(section_spec, dict) else ""
                    if not fname:
                        continue

                    # AST-first presence check
                    if ast_has_function(script_text, fname):
                        # Prefer AST-based exact block; fall back to regex
                        existing = ast_get_function_block(script_text, fname) or self.extract_function_block(script_text, fname)
                        # As a last resort, treat presence as "existing" even if we failed to slice the block,
                        # to avoid creating a duplicate. In that case, just skip creation.
                        if not existing:
                            self.root.after(0, lambda n=fname: self.append_display(
                                f"Detected function `{n}` via AST but could not slice its block; skipping creation to avoid duplicates.", "response"))
                            continue
                    else:
                        existing = ""

                else:  # main
                    existing = self.extract_main_block(script_text)

                # --- Missing -> NORMAL CREATION ---
                if not existing:
                    candidate = synthesize_section(section_kind, section_spec, script_text)

                    # Minimal validation like in generate_next_section
                    basic_issues = []
                    if section_kind == "function":
                        fname = section_spec.get("name", "")
                        if not re.search(rf"^\s*def\s+{re.escape(fname)}\s*\(", candidate, flags=re.M):
                            basic_issues.append(f"Missing or misnamed function definition: def {fname}(...).")
                    elif section_kind == "imports":
                        bad_lines = [ln for ln in candidate.splitlines() if ln.strip() and not (ln.strip().startswith("import ") or ln.strip().startswith("from "))]
                        if bad_lines:
                            basic_issues.append("Imports section contains non-import lines:\n" + "\n".join(f"  • {bl}" for bl in bad_lines))
                    elif section_kind == "globals":
                        if re.search(r"^\s*(def|class)\s+", candidate, flags=re.M):
                            basic_issues.append("Globals section contains a function/class definition.")
                    else:  # main
                        if not re.search(r"^\s*(?:async\s+)?def\s+main\s*\(", candidate, flags=re.M):
                            basic_issues.append("MAIN section is missing def main(...):")
                        if not re.search(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", candidate, flags=re.M):
                            basic_issues.append('MAIN section is missing the __main__ guard.')

                    # If basic validation complained, run pre_reviser_explain + rewrite_section once
                    if basic_issues:
                        notes = "- Structural/validation issues:\n" + "\n".join(f"  {x}" for x in basic_issues) + "\n"
                        explanation = pre_reviser_explain(GOAL_SPEC or "", script_text, section_kind, section_spec, candidate, notes)
                        critique_blob = f"CHECKER NOTES:\n{notes}\nDETAILED EXPLANATION:\n{explanation}"
                        candidate = sanitize_candidate(
                            rewrite_section(GOAL_SPEC or "", script_text, section_kind, section_spec, critique_blob, candidate),
                            section_kind,
                        )
                        if section_kind == "main":
                            candidate = ensure_main_guard(candidate)

                    # Critique -> possible rewrite
                    candidate = review_and_optionally_rewrite(section_kind, section_spec, candidate, script_text)

                    # Try compiling with candidate placed
                    new_script_text = build_script_with(section_kind, section_spec, candidate, script_text)
                    ok, err = compiles_script(new_script_text)
                    if not ok:
                        # One more rewrite pass seeded with the compile error
                        checker = f"- Compilation error:\n  {err}\n"
                        explanation = pre_reviser_explain(GOAL_SPEC or "", script_text, section_kind, section_spec, candidate, checker)
                        critique_blob = f"CHECKER NOTES:\n{checker}\nDETAILED EXPLANATION:\n{explanation}"
                        candidate = sanitize_candidate(
                            rewrite_section(GOAL_SPEC or "", script_text, section_kind, section_spec, critique_blob, candidate),
                            section_kind,
                        )
                        if section_kind == "main":
                            candidate = ensure_main_guard(candidate)
                        new_script_text = build_script_with(section_kind, section_spec, candidate, script_text)
                        ok, err = compiles_script(new_script_text)
                        if not ok:
                            # Give up on this section for this round; move on
                            continue

                    SCRIPT_LINES = new_script_text.splitlines()
                    self.refresh_script_editor()
                    label = (section_spec.get("name") if section_kind == "function" and isinstance(section_spec, dict) else section_kind.upper())
                    self.root.after(0, lambda l=label: self.append_display(f"Created missing section: {l}", "response"))
                    made_change = True
                    continue  # move to next section


                # --- Exists -> NORMAL REVISION (focused critique on this section only) ---
                revised = review_and_optionally_rewrite(section_kind, section_spec, existing, script_text)
                if revised.strip() != existing.strip():
                    test_script = build_script_with(section_kind, section_spec, revised, script_text)
                    ok, err = compiles_script(test_script)
                    if not ok:
                        # Try one compile-seeded rewrite
                        checker = f"- Compilation error:\n  {err}\n"
                        explanation = pre_reviser_explain(GOAL_SPEC or "", script_text, section_kind, section_spec, revised, checker)
                        critique_blob = f"CHECKER NOTES:\n{checker}\nDETAILED EXPLANATION:\n{explanation}"
                        revised = sanitize_candidate(
                            rewrite_section(GOAL_SPEC or "", script_text, section_kind, section_spec, critique_blob, revised),
                            section_kind,
                        )
                        if section_kind == "main":
                            revised = ensure_main_guard(revised)
                        test_script = build_script_with(section_kind, section_spec, revised, script_text)
                        ok, _ = compiles_script(test_script)
                        if not ok:
                            continue

                    SCRIPT_LINES = test_script.splitlines()
                    self.refresh_script_editor()
                    label = (section_spec.get("name") if section_kind == "function" and isinstance(section_spec, dict) else section_kind.upper())
                    self.root.after(0, lambda l=label: self.append_display(f"Revised section: {l}", "response"))
                    made_change = True


            if not made_change:
                self.root.after(0, lambda: self.append_display("Pre-run section sweeps: OK", "response"))
                break

        final_script_text = join_script(SCRIPT_LINES)

        # NEW: require explicit confirmation before running the code
        if self.wait_for_run_confirmation():
            self.run_and_autofix_until_clean()
        else:
            # User chose not to proceed; just stop here gracefully.
            return





    def run(self):
        def apply_futuristic_theme():
            # --- palette ---
            self.BG = "#0a0f14"
            self.BG_PANEL = "#0f1620"
            self.BG_DARK = "#091017"
            self.FG = "#e6edf3"
            self.MUTED = "#9aa4b2"
            self.ACCENT = "#00e5ff"
            self.SUCCESS = "#6fffB0"
            self.OUTLINE = "#1f2a34"
            self.SELECTION = "#1b2838"
            self.ACCENT_BG = "#102832"

            # window + global options
            self.root.configure(bg=self.BG)
            self.root.option_add("*Background", self.BG)
            self.root.option_add("*Foreground", self.FG)
            self.root.option_add("*Font", "Consolas 10")
            self.root.option_add("*activeBackground", self.ACCENT_BG)
            self.root.option_add("*activeForeground", self.FG)
            self.root.option_add("*insertBackground", self.ACCENT)
            self.root.option_add("*selectBackground", self.SELECTION)

            # ttk theme + widgets
            style = ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure(".", background=self.BG, foreground=self.FG, fieldbackground=self.BG)
            style.configure("TFrame", background=self.BG)
            style.configure("TLabel", background=self.BG, foreground=self.FG)
            style.configure("TButton", background=self.BG_PANEL, foreground=self.FG)
            style.map("TButton",
                      background=[("active", self.ACCENT_BG)],
                      foreground=[("active", self.FG)])
            style.configure("Cyber.Horizontal.TProgressbar",
                            troughcolor=self.BG_DARK,
                            background=self.ACCENT)
            self._init_scrollbar_styles(style)
            try:
                self.progress.configure(style="Cyber.Horizontal.TProgressbar")
            except Exception:
                pass

            # frames
            for f in (self.left_col, self.mid_col, self.right_col):
                try:
                    f.configure(bg=self.BG)
                except Exception:
                    pass

            # labels
            for lbl in (self.outline_label, self.script_label, self.goal_label,
                        self.display_label, self.input_label):
                try:
                    lbl.configure(bg=self.BG, fg=self.FG, font=("Consolas", 10, "bold"))
                except Exception:
                    pass


            # text areas
            for st in (self.outline_editor, self.script_editor, self.goal_box, self.display, self.input_box):
                try:
                    st.configure(
                        bg=self.BG_PANEL,
                        fg=self.FG,
                        insertbackground=self.ACCENT,
                        relief="flat",
                        borderwidth=1,
                        highlightthickness=1,
                        highlightbackground=self.OUTLINE,
                        highlightcolor=self.ACCENT,
                        selectbackground=self.SELECTION
                    )
                except Exception:
                    pass

            # send button
            try:
                self.send_btn.configure(
                    bg=self.ACCENT_BG,
                    fg=self.FG,
                    activebackground=self.ACCENT,
                    activeforeground=self.BG,
                    relief="flat",
                    bd=0,
                    highlightthickness=0,
                    padx=12,
                    pady=6,
                    font=("Consolas", 10, "bold")
                )
            except Exception:
                pass

            # recolor tags in the status console
            try:
                self.display.tag_configure("prompt", foreground=self.ACCENT)
                self.display.tag_configure("response", foreground=self.SUCCESS)
                self.display.tag_configure("script", foreground=self.MUTED)
            except Exception:
                pass

        apply_futuristic_theme()
        # Pop the goal prompt immediately (since input is needed to proceed)
        self.root.after(100, self.show_goal_popup)
        self.root.mainloop()


    # ========= Post-generation: run & auto-fix loop =========
    def run_and_autofix_until_clean(self, max_rounds: int = 8):
        """
        Save the current script to a temp file, run it, and if it errors:
        1) LLM decides which function/section to fix based on traceback.
        2) LLM proposes a fixed version of that function/section.
        3) Replace it in SCRIPT_LINES, then re-run.

        Additionally:
        - After EVERY run, prompt the user for feedback in the GUI and wait for submission.
        - If the run succeeded and the user provided feedback, APPLY A SINGLE, MINIMAL EDIT
          to the script based on that feedback (imports/globals/function/main), update the
          Script panel, and stop. (Previously it only said "Feedback noted" and did nothing.)
        - After EACH run-edit (code replacement), print the full script to the GUI.
        """
        global FULL_OUTLINE, SCRIPT_LINES, GOAL_SPEC
        self.ensure_goal_snapshot_dir()
        self.save_script_snapshot("initial")  # 00_initial.py

        def _compile_ok(src: str) -> bool:
            try:
                compile(src, "<feedback-fix>", "exec")
                return True
            except SyntaxError as e:
                msg = f"Feedback edit failed to compile: {e}"
                self.root.after(0, self.append_display, msg, "response")
                return False

        for round_idx in range(1, max_rounds + 1):
            # Write script to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tf:
                path = tf.name
            try:
                self.write_current_script_to_path(path)
                result = self.run_script_once(path)
            finally:
                pass  # keep the temp file for this loop; overwritten next round

            if result["returncode"] == 0:
                self.root.after(0, lambda: self.append_display(f"Run OK (round {round_idx}).", "response"))

                # Ask for feedback and WAIT here
                self.request_run_feedback(round_idx)

                # ✅ FIX: define `fb` BEFORE first use
                fb = (self.latest_feedback or "").strip()

                # Allow the model to end testing based on feedback
                if fb and self.llm_should_end_testing(fb, {
                    "returncode": result["returncode"],
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", "")
                }):
                    self.root.after(0, lambda: self.append_display("Ending testing per user feedback (LLM decision).", "response"))
                    self.save_script_snapshot(f"after_run_{round_idx}_end_testing")
                    break

                # If user provided non-empty feedback, apply one minimal change based on it
                if fb:
                    full_script_text = join_script(SCRIPT_LINES)
                    outline_text = FULL_OUTLINE or ""
                    target = self.llm_select_fix_target_from_critique(fb, full_script_text, outline_text)

                    target_kind = (target or {}).get("target_kind", "")
                    target_name = (target or {}).get("name", "")

                    new_script_text = full_script_text  # default no-op

                    if target_kind == "imports":
                        current_imports = self.extract_imports_block(full_script_text)
                        fixed_imports = self.llm_generate_imports_fix(current_imports, outline_text, full_script_text, fb)
                        if fixed_imports.strip():
                            candidate = self.replace_imports_block(full_script_text, fixed_imports)
                            if _compile_ok(candidate):
                                new_script_text = candidate
                                self.root.after(0, lambda: self.append_display("Applied feedback to IMPORTS.", "response"))

                    elif target_kind == "globals":
                        current_globals = self.extract_globals_block(full_script_text)
                        fixed_globals = self.llm_generate_globals_fix(current_globals, outline_text, full_script_text, fb)
                        candidate = self.replace_globals_block(full_script_text, fixed_globals)
                        if _compile_ok(candidate):
                            new_script_text = candidate
                            self.root.after(0, lambda: self.append_display("Applied feedback to GLOBALS.", "response"))

                    elif target_kind == "function" and target_name:
                        # Find function description from outline (if present)
                        func_desc = ""
                        try:
                            parsed = parse_outline(outline_text) if outline_text else {"functions": []}
                            for f in (parsed.get("functions") or []):
                                if (f.get("name") or "").strip() == target_name:
                                    func_desc = f.get("desc") or ""
                                    break
                        except Exception:
                            func_desc = ""

                        bad_fn_src = self.extract_function_block(full_script_text, target_name)
                        if not bad_fn_src:
                            # If missing, create via normal process and then re-extract
                            desc = func_desc or ("Auto-added per user feedback: " + fb[:160])
                            self.root.after(0, lambda n=target_name: self.append_display(
                                f"Function `{n}` not found. Creating it based on feedback...", "response"))
                            ok_created = self.create_new_function_via_normal_process(target_name, desc)
                            if ok_created:
                                full_script_text = join_script(SCRIPT_LINES)
                                bad_fn_src = self.extract_function_block(full_script_text, target_name)

                        if bad_fn_src:
                            # Use the section rewriter with the feedback as the critique
                            fixed_fn = sanitize_candidate(
                                rewrite_section(
                                    GOAL_SPEC or "",
                                    full_script_text,
                                    "function",
                                    {"name": target_name, "desc": func_desc or ("Revised per user feedback.")},
                                    fb,
                                    bad_fn_src
                                ),
                                "function",
                            )
                            candidate = self.replace_function_block(full_script_text, target_name, fixed_fn)
                            if _compile_ok(candidate):
                                new_script_text = candidate
                                self.root.after(0, lambda n=target_name: self.append_display(f"Applied feedback to function `{n}`.", "response"))

                    elif target_kind == "main":
                        bad_main = self.extract_main_block(full_script_text)
                        if bad_main:
                            fixed_main = self.llm_generate_main_fix(
                                bad_main_src=bad_main,
                                full_script=full_script_text,
                                outline=FULL_OUTLINE or "",
                                error_text=fb,
                            )
                            if not fixed_main:
                                self.root.after(0, lambda: self.append_display("No main() fix produced.", "response"))
                                self.save_script_snapshot(f"after_run_{round_idx}")
                                break
                            new_script_text = self.replace_main_block(full_script_text, fixed_main)
                            SCRIPT_LINES = new_script_text.splitlines()
                            self.root.after(0, lambda: self.append_display("Replaced main() section.", "response"))
                            full_after_edit = join_script(SCRIPT_LINES)
                            self.save_script_snapshot(f"after_run_{round_idx}")
                            continue

                    applied_change = False
                    if new_script_text != full_script_text:
                        SCRIPT_LINES = new_script_text.splitlines()
                        applied_change = True
                    else:
                        self.root.after(0, lambda: self.append_display("Feedback noted, but no actionable edit was produced.", "response"))

                    self.save_script_snapshot(f"after_run_{round_idx}")

                    if applied_change:
                        self.root.after(0, lambda: self.append_display("Feedback edits applied; re-running...", "response"))
                        continue  # run another round with the updated script

                break  # no feedback or no actionable change → we're done

            # ---------- Error case: analyze + fix ----------
            err_text = (result.get("stderr") or "").strip()
            out_text = (result.get("stdout") or "").strip()
            trace_text = (err_text if err_text else out_text) or "(no output)"

            self.root.after(0, lambda t=trace_text: self.append_display(f"Traceback (round {round_idx}):\n{t}", "script"))

            # Ask for feedback BEFORE attempting an edit (so user can hint the fix)
            self.request_run_feedback(round_idx)
            fb = (self.latest_feedback or "").strip()
            # Even on errors, allow user to say "that's fine / ship it"
            if fb and self.llm_should_end_testing(fb, {
                "returncode": result["returncode"],
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
            }):
                self.root.after(0, lambda: self.append_display("Ending testing per user feedback (LLM decision).", "response"))
                self.save_script_snapshot(f"after_run_{round_idx}_end_testing")
                break

            full_script_text = join_script(SCRIPT_LINES)
            # Fast-path for a missing function NameError (deterministic, no LLM needed)
            m = re.search(r"NameError: name '([A-Za-z_]\w*)' is not defined", trace_text)
            if m:
                missing_name = m.group(1)
                if not self.extract_function_block(full_script_text, missing_name):
                    self.root.after(0, lambda n=missing_name: self.append_display(
                        f"Function `{n}` not found. Creating it via the normal process...", "response"))
                    ok_created = self.handle_missing_function_during_run(missing_name, trace_text)
                    if not ok_created:
                        self.root.after(0, lambda n=missing_name: self.append_display(
                            f"Could not create function `{n}`.", "response"))
                        break
                    full_after_edit = join_script(SCRIPT_LINES)
                    self.save_script_snapshot(f"after_run_{round_idx}")
                    continue

            # Otherwise, fall back to LLM selection
            target = self.llm_select_fix_target(trace_text, full_script_text, FULL_OUTLINE or "")

            target_kind = target.get("target_kind", "")
            target_name = target.get("name", "")

            if target_kind == "function" and target_name:
                bad_fn_src = self.extract_function_block(full_script_text, target_name)
                if not bad_fn_src:
                    # Missing -> create it via the normal process
                    self.root.after(0, lambda n=target_name: self.append_display(
                        f"Function `{n}` not found. Creating it via the normal process...", "response"))
                    ok_created = self.handle_missing_function_during_run(target_name, trace_text)
                    if not ok_created:
                        self.root.after(0, lambda n=target_name: self.append_display(
                            f"Could not create function `{n}`.", "response"))
                        break
                    full_after_edit = join_script(SCRIPT_LINES)
                    continue
                else:
                    # Exists -> fix it in place using the function fixer
                    fixed_fn = sanitize_candidate(
                        self.llm_generate_function_fix(
                            target_name,
                            bad_fn_src,
                            full_script_text,
                            FULL_OUTLINE or "",
                            trace_text,
                        ),
                        "function",
                    )
                    if not fixed_fn.strip():
                        self.root.after(0, lambda n=target_name: self.append_display(
                            f"No fix generated for `{n}`.", "response"))
                        break
                    new_script_text = self.replace_function_block(full_script_text, target_name, fixed_fn)
                    try:
                        compile(new_script_text, "<run-edit>", "exec")
                    except SyntaxError as e:
                        self.root.after(0, lambda e=e: self.append_display(
                            f"Function fix didn't compile: {e}", "response"))
                        break
                    SCRIPT_LINES = new_script_text.splitlines()
                    self.root.after(0, lambda n=target_name: self.append_display(
                        f"Replaced function `{n}`.", "response"))
                    full_after_edit = join_script(SCRIPT_LINES)
                    self.save_script_snapshot(f"after_run_{round_idx}")
                    continue

            elif target_kind == "main":
                bad_main = self.extract_main_block(full_script_text)
                if not bad_main:
                    self.root.after(0, lambda: self.append_display(
                        "Could not locate main() section.", "response"))
                    break
                fixed_main = self.llm_generate_main_fix(
                    bad_main_src=bad_main,
                    full_script=full_script_text,
                    outline=FULL_OUTLINE or "",
                    error_text=trace_text,
                )
                if not fixed_main:
                    self.root.after(0, lambda: self.append_display(
                        "No main() fix produced.", "response"))
                    break
                new_script_text = self.replace_main_block(full_script_text, fixed_main)
                SCRIPT_LINES = new_script_text.splitlines()
                self.root.after(0, lambda: self.append_display("Replaced main() section.", "response"))
                full_after_edit = join_script(SCRIPT_LINES)
                self.save_script_snapshot(f"after_run_{round_idx}")
                continue

            elif target_kind in ("imports", "globals"):
                if target_kind == "imports":
                    current_imports = self.extract_imports_block(full_script_text)
                    fixed_imports = self.llm_generate_imports_fix(
                        current_imports, FULL_OUTLINE or "", full_script_text, trace_text)
                    if fixed_imports.strip():
                        candidate = self.replace_imports_block(full_script_text, fixed_imports)
                        try:
                            compile(candidate, "<imports-fix>", "exec")
                            SCRIPT_LINES = candidate.splitlines()
                            self.root.after(0, lambda: self.append_display("Applied imports fix.", "response"))
                            full_after_edit = join_script(SCRIPT_LINES)
                            self.save_script_snapshot(f"after_run_{round_idx}")
                            continue
                        except SyntaxError as e:
                            self.root.after(0, lambda e=e: self.append_display(
                                f"Imports fix didn't compile: {e}", "response"))
                            break
                else:
                    current_globals = self.extract_globals_block(full_script_text)
                    fixed_globals = self.llm_generate_globals_fix(
                        current_globals, FULL_OUTLINE or "", full_script_text, trace_text)
                    candidate = self.replace_globals_block(full_script_text, fixed_globals)
                    try:
                        compile(candidate, "<globals-fix>", "exec")
                        SCRIPT_LINES = candidate.splitlines()
                        self.root.after(0, lambda: self.append_display("Applied globals fix.", "response"))
                        full_after_edit = join_script(SCRIPT_LINES)
                        self.save_script_snapshot(f"after_run_{round_idx}")
                        continue
                    except SyntaxError as e:
                        self.root.after(0, lambda e=e: self.append_display(
                            f"Globals fix didn't compile: {e}", "response"))
                        break

            # Fallback
            self.root.after(0, lambda: self.append_display(f"Unrecognized target: {target}", "response"))
            break





    def write_current_script_to_path(self, path: str):
        src = join_script(SCRIPT_LINES)
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)

    def run_script_once(self, path: str, timeout: int = 120) -> dict:
        try:
            cp = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
            return {
                "returncode": cp.returncode,
                "stdout": cp.stdout,
                "stderr": cp.stderr,
            }
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": f"{type(e).__name__}: {e}"}

    def llm_select_fix_target(self, error_text: str, full_script: str, outline: str) -> dict:
        """
        Ask the LLM which function/section to fix.
        Returns a dict like: {"target_kind": "function", "name": "foo"} or {"target_kind":"main"}.
        """
        prompt = (
            "You are a Python debugging expert.\n"
            "Given the full script, the structured outline, and the Python error traceback, decide the SINGLE best place to fix.\n"
            "Respond ONLY as JSON with keys:\n"
            '  - "target_kind": one of ["function","main"]\n'
            '  - "name": the function name if target_kind=="function" (omit for main)\n'
            "No prose, no code fences, just a JSON object.\n\n"
            f"OUTLINE:\n{outline}\n\n"
            f"SCRIPT:\n{full_script}\n\n"
            f"TRACEBACK:\n{error_text}\n"
        )
        print("\n\n\n\nSELECT FIX TARGET (TRACEBACK) PROMPT:\n\n" + str(prompt))
        raw = llm(prompt, max_new_tokens=4096, temperature=0.2)
        print("\n\n\n\nSELECT FIX TARGET (TRACEBACK) RESPONSE:\n\n" + str(raw))

        txt = extract_code_block(raw) or raw
        try:
            return json.loads(txt.strip())
        except Exception:
            # Fallback: try to extract a plausible function name
            m = re.search(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)"', txt)
            if m:
                return {"target_kind": "function", "name": m.group(1)}
            return {"target_kind": "main"}

    def llm_generate_function_fix(
        self,
        function_name: str,
        bad_function_src: str,
        full_script: str,
        outline: str,
        error_text: str,
    ) -> str:
        """
        Ask the LLM to output ONLY the corrected function definition for `function_name`.
        """
        prompt = (
            "You are fixing a SINGLE Python function.\n"
            "HARD REQUIREMENTS:\n"
            f"- Output ONLY the corrected definition of the function `{function_name}` as plain Python code (no fences, no extra text).\n"
            "- Keep the function focused; include a concise docstring.\n"
            "- Preserve the function signature unless the traceback proves it must change.\n\n"
            f"FUNCTION TO FIX:\n{bad_function_src}\n\n"
            f"ERROR TRACEBACK:\n{error_text}\n\n"
            f"OUTLINE (reference):\n{outline}\n\n"
            f"FULL SCRIPT (reference):\n{full_script}\n"
        )
        print("\n\n\n\nFUNCTION FIX PROMPT (" + function_name + "):\n\n" + str(prompt))
        raw = llm(prompt, max_new_tokens=4096, temperature=0.3)
        print("\n\n\n\nFUNCTION FIX RESPONSE (" + function_name + "):\n\n" + str(raw))

        return extract_code_block(raw).strip()

    def llm_generate_main_fix(
        self,
        bad_main_src: str,
        full_script: str,
        outline: str,
        error_text: str,
    ) -> str:
        """
        Ask the LLM to output ONLY a corrected main() section including the guard.
        """
        prompt = (
            "You are fixing the MAIN section of a Python script.\n"
            "HARD REQUIREMENTS:\n"
            '- Output ONLY Python code that defines main() and includes the guard: if __name__ == \"__main__\": main()\n'
            "- Do not redefine already-existing helper functions; orchestrate them.\n\n"
            f"CURRENT MAIN SECTION:\n{bad_main_src}\n\n"
            f"ERROR TRACEBACK:\n{error_text}\n\n"
            f"OUTLINE (reference):\n{outline}\n\n"
            f"FULL SCRIPT (reference):\n{full_script}\n"
        )
        print("\n\n\n\nMAIN FIX PROMPT:\n\n" + str(prompt))
        raw = llm(prompt, max_new_tokens=4096, temperature=0.3)
        print("\n\n\n\nMAIN FIX RESPONSE:\n\n" + str(raw))

        return ensure_main_guard(extract_code_block(raw).strip())


    # ---------- text replace helpers ----------
    @staticmethod
    def extract_function_block(script_text: str, func_name: str) -> str:
        pat = rf"(?ms)^\s*(?:async\s+)?def\s+{re.escape(func_name)}\s*\([^)]*\)\s*:[\s\S]*?(?=^\s*(?:async\s+)?def\s+\w+\s*\(|^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:|\Z)"

        m = re.search(pat, script_text)
        return m.group(0) if m else ""

    @staticmethod
    def replace_function_block(script_text: str, func_name: str, new_func_src: str) -> str:
        pat = rf"(?ms)^\s*(?:async\s+)?def\s+{re.escape(func_name)}\s*\([^)]*\)\s*:[\s\S]*?(?=^\s*(?:async\s+)?def\s+\w+\s*\(|^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:|\Z)"

        if not re.search(pat, script_text):
            return script_text  # no change
        return re.sub(pat, new_func_src.rstrip() + "\n", script_text)

    @staticmethod
    def extract_main_block(script_text: str) -> str:
        pat = r"(?ms)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*:[\s\S]*?(?:^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*\n\s*\1\s*\(\s*\)\s*\n?)"

        m = re.search(pat, script_text)
        return m.group(0) if m else ""

    @staticmethod
    def replace_main_block(script_text: str, new_main_src: str) -> str:
        pat = r"(?ms)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*:[\s\S]*?(?:^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*\n\s*\1\s*\(\s*\)\s*\n?)"

        if not re.search(pat, script_text):
            # If main() wasn't found, just append the provided main at the end.
            return script_text.rstrip() + "\n\n" + new_main_src.rstrip() + "\n"
        return re.sub(pat, new_main_src.rstrip() + "\n", script_text)



if __name__ == "__main__":
    ChatGUI().run()


