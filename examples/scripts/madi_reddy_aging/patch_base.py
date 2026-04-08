import sys
filename = '/Users/mohitmehta/.gemini/antigravity/playground/white-planetary/madi_reddy_sei/.venv/lib/python3.12/site-packages/pybamm/models/full_battery_models/base_battery_model.py'
with open(filename, 'r') as f:
    content = f.read()

# 1. Update docstring
doc_target = '"VonKolzenberg2020", "tunnelling limited",'
if doc_target in content:
    content = content.replace(doc_target, '"VonKolzenberg2020", "tunnelling limited", "madi-reddy",')

# 2. Update options list
opt_target = '"tunnelling limited",'
# We want to add "madi-reddy" after "tunnelling limited" in the SEI options list
if opt_target in content:
    # First occurrence is documentation, second is the actual options list
    # Let's find where SEI options are defined
    sei_marker = '"SEI": [\n'
    sei_pos = content.find(sei_marker)
    if sei_pos != -1:
        target_pos = content.find(opt_target, sei_pos)
        if target_pos != -1:
            insertion = '\n                "madi-reddy",'
            content = content[:target_pos + len(opt_target)] + insertion + content[target_pos + len(opt_target):]

with open(filename, 'w') as f:
    f.write(content)
print('Successfully patched base_battery_model.py')
