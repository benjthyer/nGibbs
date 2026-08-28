#!/usr/bin/env python3
"""PreToolUse(Bash) guardrail: block git commands that change repository state.

Reads the hook JSON from stdin. If a Bash command runs a state-changing git
subcommand (add, commit, push, reset, rebase, checkout, clean, ...), this emits a
PreToolUse "deny" decision so the call never runs. Read-only git (status, diff,
log, show, blame, rev-parse, ...) passes straight through, as does every
non-git command.

Design choices:
  * Fails OPEN. Any parse error or unexpected input -> exit 0 with no decision,
    so a bug here can never wedge an unattended run. The deny-list in
    .claude/settings.local.json still backstops the common `git commit`/`git
    push`/... spellings even if this hook no-ops.
  * When a git subcommand is ambiguous (`git branch <name>`, `git config x y`)
    it is DENIED, not prompted -- "avoid dangerous ambiguity" per project policy.
  * The shell line is split on ; && || | and newlines; each segment is checked
    independently, and env-assignment / common wrapper prefixes (env, sudo, nice,
    time, nohup, xargs, ...) are stripped before looking for `git`.
"""

import json
import shlex
import sys

# Subcommands that only ever READ repository state.
READ_ONLY = {
    "status", "diff", "log", "show", "blame", "shortlog", "reflog",
    "rev-parse", "rev-list", "ls-files", "ls-tree", "ls-remote", "cat-file",
    "describe", "name-rev", "merge-base", "show-ref", "show-branch",
    "for-each-ref", "symbolic-ref", "var", "version", "help", "grep",
    "whatchanged", "cherry", "range-diff", "diff-tree", "diff-index",
    "diff-files", "count-objects", "verify-commit", "verify-tag", "fsck",
    "check-ignore", "check-attr", "check-ref-format", "annotate", "instaweb",
    "get-tar-commit-id", "column", "interpret-trailers",
}

# Hard-mutating verbs -- always denied, and also the set scanned for when `git`
# is not the head token of a segment (e.g. `xargs git commit`).
HARD_MUTATING = {
    "add", "commit", "push", "pull", "fetch", "clone", "init", "reset",
    "restore", "checkout", "switch", "rm", "mv", "clean", "merge", "rebase",
    "cherry-pick", "revert", "am", "apply", "stash", "gc", "prune", "repack",
    "filter-branch", "filter-repo", "update-ref", "update-index", "write-tree",
    "commit-tree", "hash-object", "mktag", "mktree", "replace", "fast-import",
    "pack-objects", "unpack-objects", "symbolic-ref", "notes", "bisect",
    "format-patch", "request-pull", "send-email", "p4", "svn", "daemon",
    "sparse-checkout", "maintenance", "gui", "citool", "difftool", "mergetool",
    "rerere", "worktree", "submodule", "subtree", "range-diff",
}

# Value-taking git *global* options to skip while locating the subcommand.
GLOBAL_OPTS_WITH_VALUE = {
    "-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix",
    "--exec-path", "--config-env",
}

WRAPPERS = {
    "env", "nice", "time", "nohup", "stdbuf", "ionice", "chrt", "setsid",
    "xargs", "sudo", "doas", "command", "builtin", "exec", "then", "do",
    "else", "eval",
}

CONDITIONAL_VALUE_FLAGS = {
    "--contains", "--no-contains", "--merged", "--no-merged", "--points-at",
    "--format", "--sort", "--color", "--column", "-n",
}


def _emit_deny(reason):
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
            "suppressOutput": True,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    sys.exit(0)


def _split_segments(command):
    """Tokenize the shell line and split into segments on ; && || | newline."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    tokens = list(lexer)
    segments, current = [], []
    for tok in tokens:
        if tok in (";", "&&", "||", "|", "|&", "&", "\n"):
            if current:
                segments.append(current)
                current = []
        elif tok in ("(", ")", "{", "}"):
            continue
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


def _strip_prefix(tokens):
    """Drop leading env-assignments and wrapper commands (env FOO=bar, sudo, ...)."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if "=" in tok and tok.split("=", 1)[0].replace("_", "a").isalnum() \
                and (tok[0].isalpha() or tok[0] == "_"):
            i += 1
            continue
        if tok in WRAPPERS:
            i += 1
            # after a wrapper, also skip any immediately-following option flags
            # and env-assignments (best effort; not a full getopt)
            while i < len(tokens) and (
                tokens[i].startswith("-") or "=" in tokens[i]
            ):
                i += 1
            continue
        break
    return tokens[i:]


def _locate_subcommand(git_args):
    """Given tokens after `git`, skip global options and return (subcmd, rest)."""
    i = 0
    while i < len(git_args):
        tok = git_args[i]
        if tok == "--":
            i += 1
            break
        if tok.startswith("-"):
            key = tok.split("=", 1)[0]
            if key in GLOBAL_OPTS_WITH_VALUE and "=" not in tok:
                i += 2  # option consumes the next token as its value
            else:
                i += 1
            continue
        return tok, git_args[i + 1:]
    return None, []


def _conditional_ok(subcmd, rest):
    """True if a CONDITIONAL subcommand is in an unambiguously read-only form."""
    non_flag = []
    skip_next = False
    for idx, tok in enumerate(rest):
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("-"):
            if "=" not in tok and tok in CONDITIONAL_VALUE_FLAGS:
                skip_next = True
            continue
        non_flag.append(tok)

    if subcmd == "branch":
        bad = {"-d", "-D", "--delete", "-m", "-M", "--move", "-c", "-C",
               "--copy", "-u", "--set-upstream-to", "--unset-upstream",
               "--edit-description", "-f", "--force"}
        if any(t.split("=", 1)[0] in bad for t in rest):
            return False
        return not non_flag  # a bare arg = branch name to create/rename

    if subcmd == "tag":
        bad = {"-a", "-s", "-m", "-d", "--delete", "-f", "--force", "-u",
               "--sign", "--annotate", "--create-reflog"}
        if any(t.split("=", 1)[0] in bad for t in rest):
            return False
        return not non_flag

    if subcmd == "config":
        bad = {"--add", "--unset", "--unset-all", "--replace-all", "-e",
               "--edit", "--remove-section", "--rename-section", "--set",
               "--set-all"}
        if any(t.split("=", 1)[0] in bad for t in rest):
            return False
        read_flags = {"--get", "--get-all", "--get-regexp", "--get-urlmatch",
                      "-l", "--list", "--name-only", "--show-origin",
                      "--show-scope", "--get-color", "--get-colorbool"}
        if any(t.split("=", 1)[0] in read_flags for t in rest):
            return True
        return len(non_flag) <= 1  # `git config a.b` reads; `git config a.b x` writes

    if subcmd in ("remote", "stash", "submodule", "worktree", "notes",
                  "bundle", "sparse-checkout"):
        read_subs = {
            "remote": {"-v", "--verbose", "show", "get-url", "get-url"},
            "stash": {"list", "show"},
            "submodule": {"status", "summary"},
            "worktree": {"list"},
            "notes": {"list", "show"},
            "bundle": {"list-heads", "verify"},
            "sparse-checkout": {"list"},
        }[subcmd]
        if not non_flag:
            # bare `git remote`, `git submodule` etc. just print status
            return subcmd in ("remote", "submodule")
        return non_flag[0] in read_subs

    return False


CONDITIONAL = {"branch", "tag", "config", "remote", "stash", "submodule",
               "worktree", "notes", "bundle", "sparse-checkout"}


def _check_segment(tokens):
    tokens = _strip_prefix(tokens)
    if not tokens:
        return

    head = tokens[0]
    if head == "git":
        subcmd, rest = _locate_subcommand(tokens[1:])
        if subcmd is None:
            return  # bare `git` / `git --help`
        if subcmd in READ_ONLY:
            return
        if subcmd in CONDITIONAL:
            if _conditional_ok(subcmd, rest):
                return
            _emit_deny(
                f"Blocked: `git {subcmd}` here is not an unambiguously read-only "
                f"form (project policy: no git state changes). Use an explicit "
                f"read form or ask the user to run it."
            )
        _emit_deny(
            f"Blocked: `git {subcmd}` changes repository state and is disabled "
            f"by project policy (.claude/hooks/block-git-mutations.py). "
            f"Read-only git (status/diff/log/show/blame/rev-parse) is allowed."
        )

    # `git` not the head token: conservative scan for `git <hard-mutating verb>`
    for a, b in zip(tokens, tokens[1:]):
        if a == "git" and b in HARD_MUTATING:
            _emit_deny(
                f"Blocked: `git {b}` changes repository state and is disabled by "
                f"project policy (.claude/hooks/block-git-mutations.py)."
            )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail open

    if payload.get("tool_name") != "Bash":
        sys.exit(0)
    command = (payload.get("tool_input") or {}).get("command")
    if not isinstance(command, str) or "git" not in command:
        sys.exit(0)

    try:
        for segment in _split_segments(command):
            _check_segment(segment)
    except SystemExit:
        raise
    except Exception:
        sys.exit(0)  # fail open on any parsing surprise

    sys.exit(0)


if __name__ == "__main__":
    main()
