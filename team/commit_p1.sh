#!/usr/bin/env bash
# Member P1: run this from Git Bash to commit + push YOUR part of PathSnap under YOUR git name.
#   bash team/commit_p1.sh                 (default message)
#   bash team/commit_p1.sh "your message"
#   bash team/commit_p1.sh --watch          (auto-commit every 10 min; --watch 5 = every 5 min)
exec bash "$(dirname "$0")/commit_part.sh" p1 "$@"
