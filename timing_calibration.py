from datetime import datetime
import numpy as np


#=======================================================================#
# Function to apply timing correction to the input dataframe BFM flash time
#=======================================================================#

def timing_correction(indf, period=18.936, t0_offset=0, prefix='', ifData=False):

    var_name = 'flashTime_'+prefix
    
    if ifData:
        #also correct T0
        indf['slc', 'barycenterFM', var_name, '', '', ''] = indf.slc.barycenterFM.flashTime + indf.frameApplyAtCaf/1e3
    else:
        indf['slc', 'barycenterFM', var_name, '', '', ''] = indf.slc.barycenterFM.flashTime

    # convert us to ns
    # align to detector front face z=0
    # align to t=0 so first peak is at period/2
    indf['slc', 'barycenterFM', var_name, '', '', ''] = indf.slc.barycenterFM[var_name]*1000 - indf.slc.vertex.z/29.97 + t0_offset + period/2

    #apply modulo to fold into one period
    indf['slc', 'barycenterFM', var_name+'_mod', '', '', ''] = indf.slc.barycenterFM[var_name]%period
    return indf

#=======================================================================#
# Drop bad period data
#=======================================================================#
def data_filter_bad(indf, bad_dict):

    nfilter = 0

    for k,v in bad_dict.items():
        tmin = v[0]
        tmax = v[1]

        mask = (indf['tdcRwm'].astype(np.int64) >= tmin) & (indf['tdcRwm'].astype(np.int64) <= tmax)
        nfilter += (len(indf[mask]))
        indf = indf[~mask]

    print(' Remove {:.0f} slices'.format(nfilter))

    return indf

#=======================================================================#
# Correct good period-by-period
#=======================================================================#
def data_correct_good(indf, good_dict, odict, pdict):

    ncorrect = 0
    df_list = []

    for k,v in good_dict.items():
        tmin = v[0]
        tmax = v[1]

        mask = (indf['tdcRwm'].astype(np.int64) >= tmin) & (indf['tdcRwm'].astype(np.int64) <= tmax)

        df_chunk = indf[mask]
        ncorrect += (len(df_chunk))

        data_first_peak = odict[k]
        data_period = pdict[k] 

        dfchunk = tc.timing_correction(df_chunk, period=data_period, t0_offset=data_first_peak, prefix='calib', ifData=True)
        
        df_list.append(df_chunk)

    print(' Correct {:.0f} slices'.format(ncorrect))

    return pd.concat(df_list)

#=======================================================================#
# Run 1 Timeline
#=======================================================================#

###Run 1 Total Duration
tmin_run1 = datetime(2025, 2, 10, 0, 0, 0)
tmax_run1 = datetime(2025, 7, 8, 23, 59, 59)

#-----------------------------------------------------------------------#
###Duration 1: Not much happened -- LLRF test happened at some points in March and was not noted
tmin_period1a = tmin_run1.timestamp() * 1e9 #Run 1 begins
tmax_period1a = datetime(2025,2,16, 9,43,00).timestamp()*1e9 #RF tripped + vacuum issues

tmin_period1b = tmax_period1a
tmax_period1b = datetime(2025,2,16, 19,00,00).timestamp()*1e9 #Beam recovered after several resets

tmin_period1c = tmax_period1b
tmax_period1c = datetime(2025,2,20, 2,47,00).timestamp()*1e9 #Booster injection tuning

tmin_period1d = tmax_period1c 
tmax_period1d = datetime(2025, 4, 2, 16, 57, 0).timestamp() * 1e9
#-----------------------------------------------------------------------#
###Duration 2: Bunch Rotation On
#After beam resumed operation -- bunch rotation was turned on
tmin_rotation = tmax_period1d
#Bunch rotation was turned off per experiment request
tmax_rotation = datetime(2025, 4, 7, 16, 1, 00).timestamp() * 1e9 #BWh

#-----------------------------------------------------------------------#
###Duration 3: Tuned + Various LLRF Switches
#After bunch rotation was turned off, the beam was retuned
tmin_period3a = tmax_rotation
#Switch from analog to digital LLRF
tmax_period3a = datetime(2025, 4,23, 17,50,0).timestamp() *1e9

#datetime(2025,4,28,20,49,19)) #BRF tripped and on 28 April night -- various resets follow until 29 April
#datetime(2025,4,29,12,19,00) #Digital LLRF tripped again
#datetime(2025,4,29,16,09,00) #Switched back to analog LLRF back take an hour to ramp back to nominal condition
tmin_period3b = tmax_period3a
tmax_period3b = datetime(2025,4,28,20,49,19).timestamp()*1e9 #When BRF first tripped

tmin_period3c = tmax_period3b
tmax_period3c = datetime(2025, 4, 29, 17, 00, 00).timestamp() *1e9 #Switched to analog LLRF

tmin_period3d = tmax_period3c
tmax_period3d = datetime(2025, 5, 6, 9, 30,0).timestamp() *1e9 #8:53am CT -- short test with digital llrf

#llrf3a = datetime(2025, 5, 6, 9, 30,0).timestamp() *1e9 #8:53am CT -- stopped for switching analog to digital LLRF
#llrf3b = datetime(2025, 5, 6, 15, 20,0).timestamp() *1e9 #15:20pm CT
tmin_period3e = tmax_period3d
tmax_period3e = datetime(2025, 5, 6, 15, 20,0).timestamp() *1e9 

#llrf4 = datetime(2025, 6, 2, 15, 0,0) #8:53am CT -- stopped for switching analog to digital LLRF
tmin_period3f = tmax_period3e
tmax_period3f = datetime(2025, 6, 2, 15, 0,0).timestamp() *1e9
#-----------------------------------------------------------------------#
###Duration 4: Post switched to digital LLRF
tmin_period4 = tmax_period3f
tmax_period4 = datetime(2025, 6, 23, 18, 53, 0).timestamp() * 1e9 #Extended beam downtime

#-----------------------------------------------------------------------#
###Duration 5: Post extended beam downtime
tmin_period5a = tmax_period4
tmax_period5a = datetime(2025,6,26,13,00,00).timestamp() *1e9

tmin_period5b = tmax_period5a
tmax_period5b = datetime(2025,6,26,21,40,00).timestamp() *1e9 #booster study for 1 day

tmin_period5c = tmax_period5b
tmax_period5c = tmax_run1.timestamp() * 1e9
#-----------------------------------------------------------------------#

#=======================================================================#
#Tag v10_14_02 Calibration Contants
#=======================================================================#

#MC neutrino
mcbnb_offset_calib = -368.945 #ns
mcbnb_period_calib = 18.936 #ns
#-----------------------------------------------------------------------#

#MC HNL
mchnl_offset_calib =  mcbnb_offset_calib
mchnl_period_calib = mcbnb_period_calib
#-----------------------------------------------------------------------#

#Data Offbeam
offbeam_offset_calib = -525 #ns
offbeam_period_calib = 18.936 #ns
#-----------------------------------------------------------------------#

#Data BNB
bad_period_dict = {
        "1b": [tmin_period1b, tmax_period1b]
        , "3c": [tmin_period3c, tmax_period3c]
        , "3e": [tmin_period3e, tmax_period3e]
        , "5b": [tmin_period5b, tmax_period5b]
     }
#------------------------------------------------------------#

good_period_dict = { 
        "1ac": [tmin_period1a, tmax_period1c]
         ,  "1d": [tmin_period1d, tmax_period1d]
         , "rotation": [tmin_rotation, tmax_rotation]
         , "3a": [tmin_period3a, tmax_period3a]
         , "3b": [tmin_period3b, tmax_period3b] 
         , "3d": [tmin_period3d, tmax_period3d]
         , "3f": [tmin_period3f, tmax_period3f]
         , "4": [tmin_period4, tmax_period4]
         , "5ac": [tmin_period5a, tmax_period5c]
        }

pdict = {
        "1ac": np.float64(18.931)
         , "1d": np.float64(18.933) #1ac, 1d
         , "rotation": np.float64(18.938) #rotation
         , "3a": np.float64(18.936) #3a
         , "3b": np.float64(18.937) #3b
         , "3d": np.float64(18.936) #3d
         , "3f": np.float64(18.936) #3f
         , "4": np.float64(18.935) #4
         , "5ac": np.float64(18.937) #5ac
}

odict = { 
        "1ac": -525.165
         , "1d": -524.987 #1ac/ 1d
         , "rotation": -524.700 #rotation
         , "3a": -524.858#-524.77 #3a
         , "3b": -516.380#-516.05 #3b
         , "3d": -524.780 #3d
         , "3f": -513.820# -513.79 #3f
         , "4": -524.365 #-524.22 #4
         , "5ac": -522.738 #-522.848#-522.57 #5ac
         } 
#=======================================================================#
#Tag v10_14_02 BugFix
#=======================================================================#

def bugfix_mcbnb_bfm_flashtime(indf):

    #flash time = flash time - 135 ns
    #offset by exactly 2 period of beam cycles?
    mc_pds_cable_length = 0.135 #us
    period = 18.936/1000 #ns to us
    indf[('slc', 'barycenterFM', 'flashTime', '', '', '')] = indf[('slc', 'barycenterFM', 'flashTime', '', '', '')] + mc_pds_cable_length + period*2
    return indf

def bugfix_mchnl_bfm_flashtime(indf):

    #flash time = flash time - 135 ns
    mc_pds_cable_length = 0.135 #us
    indf[('slc', 'barycenterFM', 'flashTime', '', '', '')] = indf[('slc', 'barycenterFM', 'flashTime', '', '', '')] - mc_pds_cable_length
    return indf