#!/usr/bin/env python3
"""
Fix script to update all court_dimensions usages in ef.py
"""

import re

def fix_court_dimensions():
    """Fix all instances of court_dimensions usage"""
    
    with open('ef.py', 'r') as f:
        content = f.read()
    
    # Pattern to find lines with width, height = court_dimensions
    pattern = r'(\s+)width, height = court_dimensions'
    replacement = r'\1width, height = get_court_dimensions(court_dimensions)'
    
    # Replace all instances
    new_content = re.sub(pattern, replacement, content)
    
    # Count how many replacements were made
    count = len(re.findall(pattern, content))
    
    # Write back the fixed content
    with open('ef.py', 'w') as f:
        f.write(new_content)
    
    print(f"Fixed {count} instances of court_dimensions usage")
    
    return count

if __name__ == "__main__":
    count = fix_court_dimensions()
    if count > 0:
        print("✅ All court_dimensions usages have been fixed")
    else:
        print("No court_dimensions usages found to fix")
