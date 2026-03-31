#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggressive second pass: Replace ALL remaining Chinese characters"""
import re
from pathlib import Path

files_to_skip = {".git", "__pycache__", ".pytest_cache", "checkpoints_v2", "outputs", "tmp"}

def should_skip(path):
    path_str = str(path).lower()
    for skip in files_to_skip:
        if skip in path_str:
            return True
    return False

# Chinese Unicode ranges
# CJK Unified Ideographs: \u4e00-\u9fff
# CJK Compatibility Ideographs: \uf900-\ufaff
chinese_pattern = re.compile(r'[\u4e00-\u9fff\uf900-\ufaff]+')

def replace_chinese_with_comments(text):
    """Replace Chinese with [REMOVED_CHINESE] marker in comments"""

    # Find all Chinese character sequences
    def replace_func(match):
        zh_text = match.group(0)
        # In comments, just mark it
        return f"[REMOVED_ZH:{len(zh_text)}]"

    result = chinese_pattern.sub(replace_func, text)
    return result

py_files = [f for f in Path(".").rglob("*.py") if not should_skip(f)]

print("Second pass: Removing ALL remaining Chinese characters...")
print("=" * 70)

modified_count = 0

for py_file in sorted(py_files):
    try:
        content = py_file.read_text(encoding="utf-8")

        if chinese_pattern.search(content):
            new_content = replace_chinese_with_comments(content)
            py_file.write_text(new_content, encoding="utf-8")
            modified_count += 1
            print(f"[FIXED] {py_file}")
        else:
            print(f"[OK]    {py_file}")
    except Exception as e:
        print(f"[ERR]   {py_file} - {type(e).__name__}: {e}")

print("=" * 70)
print(f"\nModified: {modified_count} files")
EOF
