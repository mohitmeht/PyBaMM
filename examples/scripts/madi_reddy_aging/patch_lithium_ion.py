import sys
filename = '/Users/mohitmehta/.gemini/antigravity/playground/white-planetary/madi_reddy_sei/.venv/lib/python3.12/site-packages/pybamm/models/full_battery_models/lithium_ion/base_lithium_ion_model.py'
with open(filename, 'r') as f:
    content = f.read()

# 1. Update set_sei_submodel
target_sei = '                elif sei_option == "constant":\n                    submodel = pybamm.sei.ConstantSEI(\n                        self.param, domain, self.options, phase\n                    )'
replacement_sei = target_sei + '\n                elif sei_option == "madi-reddy":\n                    submodel = pybamm.sei.MadiReddySEI(\n                        self.param, domain, self.options, phase\n                    )'

if target_sei in content:
    content = content.replace(target_sei, replacement_sei)
else:
    print('SEI target not found')

# 2. Update set_sei_on_cracks_submodel
target_cracks = '                        sei_option in ["none", "constant"]\n                        or sei_on_cracks_option == "false"\n                    ):'
replacement_cracks = '                        sei_option in ["none", "constant", "madi-reddy"]\n                        or sei_on_cracks_option == "false"\n                    ):'
# Actually, the user might want MadiReddy on cracks too, but let's keep it simple for now
# per standard SEIGrowth behavior.

if target_cracks in content:
    content = content.replace(target_cracks, replacement_cracks)
else:
    print('Cracks target not found')

with open(filename, 'w') as f:
    f.write(content)
print('Successfully patched base_lithium_ion_model.py')
