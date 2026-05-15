from astropy.table import Table
import numpy as np

cat = Table.read('data/crossmatch/CepHerGroups_SPYGLASS.fits')

assoc = np.array([a.strip() for a in np.array(cat['AS']).astype(str)])
sg    = np.array([s.strip() for s in np.array(cat['SG']).astype(str)])

# remove stars not assigned to any subgroup (SG == -1)
valid = sg != '-1'
cat   = cat[valid]
assoc = assoc[valid]
sg    = sg[valid]

print(f"Removed {(~valid).sum()} unassigned stars (SG=-1)")
print(f"Kept {valid.sum()} stars with valid subgroup assignments")

cat['SUBGROUP'] = np.array([f'{a}_{s}' for a, s in zip(assoc, sg)])

cat.write('data/crossmatch/CepHerGroups_SPYGLASS.fits',
          format='fits', overwrite=True)
print(f"Saved with SUBGROUP column: {cat['SUBGROUP'][:5]}")
