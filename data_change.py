from astropy.table import Table
import numpy as np

colnames = ['source_id', 'AS', 'SG', 'AA_str', 'AA_int', 'ra', 'dec', 'd', 'g', 'M', 'V', 'Li', 'F', 'P_age_lt50', 'P_spatial', 'P_sp', 'P_fin']

rows = []
rows = []
with open('data.txt', 'r') as f:
    next(f)  # skip header
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # parts[0]=source_id, parts[1]=AS, parts[2]=SG
        # parts[3] is either AA_str (when AA present) or ra (when absent)
        try:
            float(parts[3])
            # ra — AA is absent, insert two empty placeholders
            parts.insert(3, 'nan')  # AA_str
            parts.insert(4, 'nan')  # AA_int
        except ValueError:
            # AA_str is present, AA_int is parts[4] — leave as is
            pass
        while len(parts) < len(colnames):
            parts.append('nan')
        rows.append(parts[:len(colnames)])

t = Table(rows=rows, names=colnames)

str_cols   = ['AS', 'AA_str']
int_cols   = ['source_id', 'SG']
float_cols = ['AA_int', 'ra', 'dec', 'd', 'g', 'M', 'V', 'Li', 'F', 'P_age_lt50', 'P_spatial', 'P_sp', 'P_fin']

for col in float_cols:
    t[col] = t[col].astype(float)
    
for col in int_cols:
    t[col] = t[col].astype(int)

t.write('CepHerGroups_SPYGLASS.fits', format='fits', overwrite=True)
print(t)
