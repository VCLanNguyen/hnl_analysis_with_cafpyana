"""Physical and detector constants for SBND."""

__all__ = ['RHO', 'N_A', 'M_AR', 'V_SBND', 'NTARGETS']

RHO = 1.3836        # g/cm3, liquid Ar density
N_A = 6.02214076e23 # Avogadro's number
M_AR = 40           # g, molar mass of argon
# x cm (drift) * z cm (width) * y cm (height), excluding 90 cm of y-dimension at high z
V_SBND = (190)*2 * ((250 - 10)*(190*2) + (450-250)*(100 + 190))
NTARGETS = RHO * V_SBND * N_A / M_AR
