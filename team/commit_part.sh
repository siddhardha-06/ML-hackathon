#!/usr/bin/env bash
# =============================================================================
# commit_part.sh <p1|p2|p3|p4> ["commit message" | --watch [minutes]]
#
# Commits and pushes ONLY the files owned by one team member, under that member's
# own git name/email, to the team repository. Called by team/commit_p1.sh ... commit_p4.sh.
#
# Safe to run any number of times, in any order:
#   * syncs with GitHub first (teammates' commits are pulled in)
#   * local untracked copies of files that are already on GitHub with IDENTICAL content
#     are removed before pulling, so git can bring in the tracked versions (nothing is lost)
#   * never commits the dataset, the virtualenv, parquet caches or files over 90 MB
# =============================================================================
set -uo pipefail

ROLE="${1:-}"
MSG="${2:-}"
UROLE=$(printf %s "$ROLE" | tr a-z A-Z)

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

case "$ROLE" in
  p1) TITLE="Data processing, labels, candidate generation & features"
      PATHS=(common/__init__.py common/config.py common/geo.py data features results/p1 docs/PART_P1.md) ;;
  p2) TITLE="Evaluation metrics, model experimentation & Bayesian optimisation"
      PATHS=(models tuning common/metrics.py results/p2 artifacts/final_model.joblib artifacts/final_model_card.json
             artifacts/feature_columns.json artifacts/models docs/PART_P2.md) ;;
  p3) TITLE="HMM/Viterbi route reconstruction & Folium maps"
      PATHS=(hmm viz results/p3 artifacts/hmm_params.json docs/PART_P3.md) ;;
  p4) TITLE="EDA, FastAPI backend, evaluator UI, deployment & report"
      PATHS=(api ui team eda results/eda report README.md requirements.txt .gitignore .gitattributes
             results/p4 docs/PART_P4.md docs/README.md) ;;
  *)  die "usage: bash team/commit_part.sh <p1|p2|p3|p4> [\"message\"]" ;;
esac

command -v git >/dev/null || die "git is not installed (https://git-scm.com/download/win)"
cd "$(dirname "$0")/.." || die "cannot open project folder"
git rev-parse --show-toplevel >/dev/null 2>&1 || die "This folder is not inside your cloned team repo. Copy the project files into your clone first."
cd "$(git rev-parse --show-toplevel)"
git remote get-url origin >/dev/null 2>&1 || die "No 'origin' remote - clone the team repository first."

if [ -z "$(git config user.name || true)" ]; then read -r -p "Your name (for commits): " n; git config --global user.name "$n"; fi
if [ -z "$(git config user.email || true)" ]; then read -r -p "Your GitHub email: " e; git config --global user.email "$e"; fi
say "Committing part $UROLE ($TITLE) as $(git config user.name) <$(git config user.email)>"

# local-only ignore rules (in case .gitignore is not pushed yet)
mkdir -p .git/info
for p in "Student folder/" ".venv/" "__pycache__/" "artifacts/*.parquet" "*.pyc" ".ipynb_checkpoints/" "*.log" "results_p2_log.txt"; do
  grep -qxF "$p" .git/info/exclude 2>/dev/null || echo "$p" >> .git/info/exclude
done

git fetch -q origin || die "Cannot reach GitHub (internet? logged in? added as collaborator?)"
BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##')
[ -n "$BRANCH" ] || BRANCH=$(git ls-remote --symref origin HEAD 2>/dev/null | awk '/^ref:/{sub("refs/heads/","",$2);print $2;exit}')
[ -n "$BRANCH" ] || BRANCH=main
REMOTE_HAS_BRANCH=0
git rev-parse -q --verify "origin/$BRANCH" >/dev/null && REMOTE_HAS_BRANCH=1

if git rev-parse -q --verify HEAD >/dev/null; then
  [ "$(git branch --show-current)" = "$BRANCH" ] || git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH"
else
  git symbolic-ref HEAD "refs/heads/$BRANCH"
fi

is_mine() {  # is this path inside one of this member's owned paths?
  local p
  for p in "${PATHS[@]}"; do
    [ "$1" = "$p" ] && return 0
    case "$1" in "$p"/*) return 0 ;; esac
  done
  return 1
}

sync_with_remote() {
  [ "$REMOTE_HAS_BRANCH" = 1 ] || return 0
  if ! git rev-parse -q --verify HEAD >/dev/null; then
    # fresh clone of a repo that was empty: adopt the remote history without touching local
    # files, then write out any remote files that are missing locally
    git reset -q "origin/$BRANCH"
    git checkout-index -a 2>/dev/null || true
    say "Joined existing history on origin/$BRANCH"
    return 0
  fi
  # Local untracked copies of files that are already on GitHub would block the pull:
  #  * byte-identical copy            -> removed (git brings back the same file)
  #  * a TEAMMATE's file that differs -> moved to .git/replaced-by-remote/ (the owner's pushed
  #                                      version wins; your old copy is kept there as a backup)
  #  * one of YOUR files that differs -> left alone (git will stop and tell you)
  # (ls-tree -z: avoids Git Bash path mangling of "rev:path")
  local line meta f removed=0 replaced=0
  while IFS= read -r -d '' line; do
    meta=${line%%$'\t'*}; f=${line#*$'\t'}
    [ -f "$f" ] || continue
    git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 && continue
    if [ "$(git hash-object -- "$f")" = "${meta##* }" ]; then
      rm -f -- "$f"; removed=$((removed + 1))
    elif ! is_mine "$f"; then
      mkdir -p ".git/replaced-by-remote/$(dirname -- "$f")"
      mv -f -- "$f" ".git/replaced-by-remote/$f"; replaced=$((replaced + 1))
    fi
  done < <(git ls-tree -r -z "origin/$BRANCH")
  [ "$removed" -gt 0 ] && say "Re-syncing $removed file(s) that teammates already pushed"
  [ "$replaced" -gt 0 ] && say "Using teammates' pushed version of $replaced file(s) (old copies in .git/replaced-by-remote/)"
  git pull -q --rebase --autostash origin "$BRANCH" || {
    git rebase --abort >/dev/null 2>&1
    die "Could not merge teammates' work (same file edited differently). Ask the owner of that file, then re-run."; }
}
commit_once() {
  local msg="$1" p f mb N STAGED=0
  git fetch -q origin || die "Cannot reach GitHub"
  git rev-parse -q --verify "origin/$BRANCH" >/dev/null && REMOTE_HAS_BRANCH=1
  sync_with_remote

  # stage only this member's paths
  for p in "${PATHS[@]}"; do
    [ -e "$p" ] || continue
    git add -- "$p" && STAGED=1
  done
  [ "$STAGED" = 1 ] || die "None of your files were found here: ${PATHS[*]}"

  # drop anything too large for GitHub
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    mb=$(( $(wc -c < "$f") / 1048576 ))
    if [ "$mb" -ge 90 ]; then git reset -q -- "$f"; warn "Not committed (${mb} MB, too large for GitHub): $f"; fi
  done < <(git diff --cached --name-only --diff-filter=AM)

  if git diff --cached --quiet; then
    say "Nothing new in part $UROLE - already up to date."
  else
    N=$(git diff --cached --name-only | wc -l | tr -d ' ')
    git commit -q -m "${msg:-$UROLE: $TITLE}" && say "Committed $N file(s)"
  fi

  for attempt in 1 2 3; do
    if git push -q -u origin "HEAD:$BRANCH"; then say "Pushed to origin/$BRANCH"; return 0; fi
    warn "Push rejected (a teammate pushed meanwhile) - syncing and retrying ($attempt/3)"
    git fetch -q origin; REMOTE_HAS_BRANCH=1; sync_with_remote
  done
  die "Push failed. Check you are logged in to GitHub and added as a collaborator."
}

if [ "$MSG" = "--watch" ]; then
  MIN="${3:-10}"
  say "Watch mode: committing part $UROLE every $MIN min (Ctrl+C to stop)"
  while true; do
    commit_once "$UROLE: update $(date '+%Y-%m-%d %H:%M')"
    sleep $(( MIN * 60 ))
  done
else
  commit_once "$MSG"
fi
