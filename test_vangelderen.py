import numpy as np


# sanity check by reproducing values from
# Resolution limit of cylinder diameter estimation by diffusion MRI: The impact of gradient waveform and orientation dispersion
D_invivo = 2e-9
scheme_clinical = np.array([[0.08, 40e-3, 40e-3]])
scheme_connectom = np.array([[0.3, 40e-3, 40e-3]])

# 1% signal drop for D_min = 3.3 um with scheme_clinical
print(vangelderen_cylinder_perp(D_invivo, 0.5*3.3e-6, scheme_clinical, m_max=10))
# 5% signal drop for D_min = 4.9 um with scheme_clinical
print(vangelderen_cylinder_perp(D_invivo, 0.5*4.9e-6, scheme_clinical, m_max=10))
# 1% signal drop for D_min = 1.7 um with scheme_connectom
print(vangelderen_cylinder_perp(D_invivo, 0.5*1.7e-6, scheme_connectom, m_max=10))

