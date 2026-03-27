"""Configurations and constants."""
import seaborn as sns
import uproot

#------------------------------------------------------------------#
# Sheffield color palette, used for plotting
#------------------------------------------------------------------#
col_dict = {
        'Teal':("#005A8F")
        , 'Aqua':("#00BBCC")
        , 'SkyBlue':("#64CBE8")
        , 'MintGreen':("#00CE7C")
        , 'Spearmint':("#3BD4AE")
        , 'PastelGreen':("#A1DED2")
        , 'Mauve':("#663DB3")
        , 'Purple':("#981F92")
        , 'Lavender':("#DAA8E2")
        , 'Coral':("#E7004C")
        , 'Flamingo':("#FF6371")
        , 'Peach':("#FF9664")
        , 'DeepViolet':("#440099")
        , 'PowderBlue':("#9ADBE8")
        , 'MidnightBlack':("#131E29")
        , 'RosyBrown4':("#8B6969")
        , 'SlateGray':("#708090")
        , 'LightGray':("#D0D2D4")
    }
#------------------------------------------------------------------#
# Generic Pi0 analysis
#------------------------------------------------------------------#
# dictionary mapping signal to ints. Signal == 0 is assumed to be the desired topology. 
signal_pi0_dict = {"CCpi0":0, "NCpi0": 1, "othernumuCC":2, "otherNC": 3, "CCnue": 4, "nonFV":5, "dirt":6, "cosmic":7, "offbeam":8}
signal_pi0_labels = [
                 r"CC$\nu$$\pi^0$",
                 r"NC$\nu$$\pi^0$",
                 r"Other CC $\nu_\mu$",
                 r"Other NC $\nu$",
                 r"CC $\nu_e$",
                 r"Non-FV $\nu$",
                 r"Dirt $\nu$",
                 "Cosmic",
                 "Offbeam",
                 "HNL"
                 ]
signal_pi0_colors = [
                col_dict['MintGreen'] #ccpi0
                ,col_dict['Coral'] #ncpi0
                ,col_dict['Teal'] #othercc
                ,col_dict['Lavender'] #othernc
                ,col_dict['Aqua'] #ccnue
                ,col_dict['Peach'] #nonfv
                ,col_dict['RosyBrown4'] #dirt
                ,col_dict['LightGray'] #cosmic  
                ,col_dict['SlateGray'] #offbeam
                ]
#------------------------------------------------------------------#
# HNL analysis
#------------------------------------------------------------------#
# dictionary mapping signal to ints. Signal == 0 is assumed to be the desired topology. 
signal_hnl_dict = {"CCpi0":0, "NCpi0": 1, "othernumuCC":2, "otherNC": 3, "CCnue": 4, "nonFV":5, "dirt":6, "cosmic":7, "offbeam":8, "hnl":9}
signal_hnl_labels = [
                 r"CC$\nu$$\pi^0$",
                 r"NC$\nu$$\pi^0$",
                 r"Other CC $\nu_\mu$",
                 r"Other NC $\nu$",
                 r"CC $\nu_e$",
                 r"Non-FV $\nu$",
                 r"Dirt $\nu$",
                 "Cosmic",
                 "Offbeam",
                 "HNL"
                 ]
signal_hnl_colors = [
                col_dict['MintGreen'] #ccpi0
                ,col_dict['Coral'] #ncpi0
                ,col_dict['Teal'] #othercc
                ,col_dict['Lavender'] #othernc
                ,col_dict['Aqua'] #ccnue
                ,col_dict['Peach'] #nonfv
                ,col_dict['RosyBrown4'] #dirt
                ,col_dict['LightGray'] #cosmic  
                ,col_dict['SlateGray'] #offbeam
                ,col_dict['DeepViolet'] #hnl
                ]


#------------------------------------------------------------------#
# CCnue analysis
#------------------------------------------------------------------#
signal_ccnue_dict = {"nueCC":0,
               "numuCCpi0":1,
               "NCpi0":2,
               "othernumuCC":3,
               "othernueCC": 4,
               "otherNC":5, 
               "nonFV":6 ,
               "dirt":7,
               "cosmic":8,
               "offbeam":9}

signal_ccnue_labels = [r"CC $\nu_e$",
                 r"CC $\nu_\mu\pi^0$",
                 r"NC$\nu$$\pi^0$",
                 r"other CC $\nu_\mu$",
                 r"other CC $\nu_e$",
                 r"other NC $\nu$",
                 r"Non-FV $\nu$",
                 r"Dirt $\nu$",
                 "cosmic",
                 "offbeam"]

# default colors used for plotting 
signal_ccnue_colors = ["C0", "C1", "C2", "C3", "darkslateblue", "C4","C6","C5","darkgray","lightgray"]

#------------------------------------------------------------------#
# Generic analysis
#------------------------------------------------------------------#

generic_dict = {"CCnu":0,"NCnu":1,"nonFV":2,"dirt":3,"cosmic":4}
generic_labels = [r"CC $\nu$",r"NC $\nu$",r"Non-FV $\nu$",r"Dirt $\nu$","cosmic"]
generic_colors = ["C3", "darkslateblue", "C5", "C6","C7"]

#------------------------------------------------------------------#
# Specifiy which dictionary to use for plotting
#------------------------------------------------------------------#

signal_dict = signal_hnl_dict
signal_labels = signal_hnl_labels
signal_colors = signal_hnl_colors

#------------------------------------------------------------------#
# dictionary mapping particle to pdg code, used for plotting
pdg_dict = {
    r"$e$":        {"pdg":11,   },
    r"$\mu$":      {"pdg":13,   },
    r"$\gamma$":   {"pdg":22,   },
    r"$p$":        {"pdg":2212, },
    r"$\pi^{+/-}$":{"pdg":211,  },
    # "pi0": {"pdg":111, "mass":0.134976},
    # "n": {"pdg":2112, "mass":0.939565},
    # "other": {"pdg":0, "mass":0}
}

#------------------------------------------------------------------#
# flux file, units: /m^2/10^6 POT, 50 MeV bins
fluxfile = "/exp/sbnd/data/users/lynnt/xsection/flux/sbnd_original_flux.root"
with uproot.open(fluxfile) as f:
    nue_flux = f["flux_sbnd_nue"].to_numpy()
    flux_vals = nue_flux[0]
integrated_flux = flux_vals.sum()/1e4 # to cm2