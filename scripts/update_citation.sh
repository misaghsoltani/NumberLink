#!/usr/bin/env bash
set -euo pipefail

# This script generates the BibTeX output from CITATION.cff into a temporary
# directory, post-processes it there, injects it into README.md between the
# markers, and stages README.md

# create temp dirs
if tmpdir=$(mktemp -d 2>/dev/null); then
    :
    elif tmpdir=$(mktemp -d -t citation 2>/dev/null); then
    :
else
    tmpdir="/tmp/citation.$$"
    mkdir -p "$tmpdir"
fi
if tmpdir2=$(mktemp -d 2>/dev/null); then
    :
    elif tmpdir2=$(mktemp -d -t readme 2>/dev/null); then
    :
else
    tmpdir2="/tmp/readme_tmp.$$"
    mkdir -p "$tmpdir2"
fi
trap 'rm -rf "${tmpdir}" "${tmpdir2}"' EXIT

# generate bibtex into the tempdir
generated="$tmpdir/citation.bib"
if command -v cffconvert >/dev/null 2>&1; then
    cffconvert -f bibtex -o "$generated"
else
    echo "cffconvert not found; please install cffconvert" >&2
    exit 2
fi

# Post-process the generated bib in tempdir
processed="$tmpdir/citation.processed.bib"
in_entry=0
have_year=0
while IFS= read -r line; do
    if [[ $line =~ ^@misc\{ ]]; then
        printf '%s\n' "@misc{numberlinkenv2025soltani," >> "$processed"
        in_entry=1
        have_year=0
        continue
    fi
    
    if [[ $in_entry -eq 1 ]]; then
        if [[ $line =~ ^[[:space:]]*\}$ ]]; then
            if [[ $have_year -eq 0 ]]; then
                printf '  %-6s = %s\n' "year" "{2025}" >> "$processed"
            fi
            printf '%s\n' "$line" >> "$processed"
            in_entry=0
            have_year=0
            continue
        fi
        # trim leading/trailing whitespace
        trimmed="$line"
        trimmed="${trimmed#"${trimmed%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
        if [[ -n $trimmed ]]; then
            # find year field
            if [[ $trimmed =~ ^year[[:space:]]*= ]]; then
                have_year=1
                field_name="${trimmed%%=*}"
                field_value="${trimmed#*=}"
                field_name="${field_name%" "}"
                field_name="${field_name#" "}"
                field_value="${field_value%" "}"
                field_value="${field_value#" "}"
                printf '  %-6s = %s\n' "$field_name" "$field_value" >> "$processed"
            else
                # ensure trailing comma for non-year fields
                if [[ $trimmed != *, ]]; then
                    trimmed="${trimmed},"
                fi
                field_name="${trimmed%%=*}"
                field_value="${trimmed#*=}"
                field_name="${field_name%" "}"
                field_name="${field_name#" "}"
                field_value="${field_value%" "}"
                field_value="${field_value#" "}"
                printf '  %-6s = %s\n' "$field_name" "$field_value" >> "$processed"
            fi
        fi
        continue
    fi
    
    printf '%s\n' "$line" >> "$processed"
done < "$generated"

echo 'Post-processed generated bib'

# Inject into README between markers using the processed temp file
START="<!-- CITATION-BIBTEX:START -->"
END="<!-- CITATION-BIBTEX:END -->"

tmp="$tmpdir2/readme_tmp"
awk -v start="$START" -v end="$END" -v bibfile="$processed" '
    $0 ~ start {
        print
        print "```bibtex"
        n = 0
        while ((getline line < bibfile) > 0) {
            n++
            a[n] = line
        }
        close(bibfile)
        # trim trailing blank lines
        while (n > 0 && a[n] == "") { n-- }
        for (i = 1; i <= n; i++) print a[i]
        print "```"
        inside = 1
        next
    }
    $0 ~ end { inside = 0; print; next }
    !inside { print }
' README.md > "$tmp" && mv "$tmp" README.md

git add README.md
