import numpy as np
import pyhf

pyhf.set_backend("numpy")

#-----------------------------------------------------------------------------------------------------#
#pyhf expects lists, not numpy arrays, so we need to convert the numpy arrays to lists
def to_pylist(x):
    if isinstance(x, list):
        return [to_pylist(i) for i in x]
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return x

#-----------------------------------------------------------------------------------------------------#
def make_model_stats_only(signal_dict, bkg_dict, dt_dict=None, ifData=False):
    model = pyhf.Model(
        {
      "channels": [
        {
          "name": "singlechannel",
          "samples": [
            {
              "name": "signal",
              "data": signal_dict['counts'],
              "modifiers": [
                {"name": "mu", "type": "normfactor", "data": None},
                {"name": "signal_stat", "type": "shapesys", "data": signal_dict['stat_err']},
              ]
            },
            {
              "name": "background",
              "data": bkg_dict['counts'],
              "modifiers": [
                {"name": "bkg_stat", "type": "staterror", "data": bkg_dict['stat_err']},
              ]
            }
          ]
        }
      ]
    }
    )

    if ifData:
        data = dt_dict['counts'] + model.config.auxdata
    else:
        #For MC study: we assume data is the same as predicted background
        data = bkg_dict['counts'] + model.config.auxdata
    
    return model, data