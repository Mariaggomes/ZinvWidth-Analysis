# Import essential libraries for data analysis and visualization
import numpy as np                  
import matplotlib.pylab as plt      
import pandas as pd                 
import awkward as ak                
import uproot                       

import vector                       
vector.register_awkward()           

import requests                     
import os                           
import time                        
import json                         

# Import utility functions from a custom module for handling specific tasks
from geometry import deltar, radial_to_cartesian2d, cartesian_to_radial2d, bound_phi
import dpoa_utilities      
from dpoa_utilities import get_files_for_dataset  
from dpoa_utilities import pretty_print  
from dpoa_utilities import build_lumi_mask  
import glob

# Find all .txt files inside ../raw
input_files = [file for file in glob.glob("../raw/*.txt")]

# List to store all .root file paths
filenames = []

# Read each .txt file and extract the .root file paths
for txt_file in input_files:
    print(f"\n📄 Reading list: {txt_file}")
    with open(txt_file, "r") as file:
        root_files = [line.strip() for line in file.readlines() if line.strip()]
        filenames.extend(root_files)
    dataset_name = os.path.splitext(os.path.basename(txt_file))[0]

# Print all collected .root file paths
print("\nList of all .root files to be processed:")
for fname in filenames:
    print(fname)


# Lists to store the data of all variables
jet_pt, jet_eta, jet_phi, jet_mass = [], [], [], []
jet_btag_csvv2, jet_ch_hef = [], []
muon_iso, muon_tightId, muon_pt, muon_eta, muon_phi, muon_mass, muon_looseId = [], [], [], [], [], [], []
met_pt, met_eta, met_phi, calo_met_pt, met_pt_pf, met_phi_pf = [], [], [], [], [], []
electron_pt, electron_eta, electron_phi, electron_mass = [], [], [], []
tau_pt, tau_eta, tau_phi, tau_mass = [], [], [], []
photon_pt, photon_eta, photon_phi, photon_mass = [], [], [], []
gen_weights, pileup = [], []
HLT_PFMET170_HBHECleaned, HLT_PFMET170_HBHE_BeamHaloCleaned = [], []
HLT_PFMETNoMu120_PFMHTNoMu120_IDTight, HLT_MET75_IsoTrk50 = [], []

# Process each file
for filename in filenames:
    try:
        f = uproot.open(filename)
        if "Events" in f:
            events = f["Events"]

            # Jets
            jet_pt.append(events["Jet_pt"].array())
            jet_eta.append(events["Jet_eta"].array())
            jet_phi.append(events["Jet_phi"].array())
            jet_mass.append(events["Jet_mass"].array())
            jet_btag_csvv2.append(events["Jet_btagCSVV2"].array())
            jet_ch_hef.append(events["Jet_chHEF"].array())

            # Muons
            muon_iso.append(events["Muon_miniIsoId"].array())
            muon_tightId.append(events["Muon_tightId"].array())
            muon_pt.append(events["Muon_pt"].array())
            muon_eta.append(events["Muon_eta"].array())
            muon_phi.append(events["Muon_phi"].array())
            muon_mass.append(events["Muon_mass"].array())
            muon_looseId.append(events["Muon_looseId"].array())

            # MET
            met_pt.append(events["PuppiMET_pt"].array())
            met_phi.append(events["PuppiMET_phi"].array())
            met_eta.append(0*events['PuppiMET_pt'].array())
            calo_met_pt.append(events["CaloMET_pt"].array())
            met_pt_pf.append(events["MET_pt"].array())
            met_phi_pf.append(events["MET_phi"].array())

            # Electrons
            electron_pt.append(events["Electron_pt"].array())
            electron_eta.append(events["Electron_eta"].array())
            electron_phi.append(events["Electron_phi"].array())
            electron_mass.append(events["Electron_mass"].array())

            # Taus
            tau_pt.append(events["Tau_pt"].array())
            tau_eta.append(events["Tau_eta"].array())
            tau_phi.append(events["Tau_phi"].array())
            tau_mass.append(events["Tau_mass"].array())

            # Photons
            photon_pt.append(events["Photon_pt"].array())
            photon_eta.append(events["Photon_eta"].array())
            photon_phi.append(events["Photon_phi"].array())
            photon_mass.append(events["Photon_mass"].array())

            gen_weights.append(events['genWeight'].array())
            pileup.append(events['Pileup_nTrueInt'].array())

            # Triggers of MET 
            HLT_PFMET170_HBHECleaned.append(events['HLT_PFMET170_HBHECleaned'].array())
            HLT_PFMET170_HBHE_BeamHaloCleaned.append(events['HLT_PFMET170_HBHE_BeamHaloCleaned'].array())
            HLT_PFMETNoMu120_PFMHTNoMu120_IDTight.append(events['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight'].array())
            HLT_MET75_IsoTrk50.append(events['HLT_MET75_IsoTrk50'].array())
            
            # Print success when opening the file
            print(f"Successfully processed {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Concatenate all variables in single arrays
jet_pt = ak.concatenate(jet_pt, axis=0)
jet_eta = ak.concatenate(jet_eta, axis=0)
jet_phi = ak.concatenate(jet_phi, axis=0)
jet_mass = ak.concatenate(jet_mass, axis=0)
jet_btag_csvv2 = ak.concatenate(jet_btag_csvv2, axis=0)
jet_ch_hef = ak.concatenate(jet_ch_hef, axis=0)

muon_iso = ak.concatenate(muon_iso, axis=0)
muon_tightId = ak.concatenate(muon_tightId, axis=0)
muon_pt = ak.concatenate(muon_pt, axis=0)
muon_eta = ak.concatenate(muon_eta, axis=0)
muon_phi = ak.concatenate(muon_phi, axis=0)
muon_mass = ak.concatenate(muon_mass, axis=0)
muon_looseId = ak.concatenate(muon_looseId, axis=0)

met_pt = ak.concatenate(met_pt, axis=0)
met_eta = ak.concatenate(met_eta, axis=0)
met_phi = ak.concatenate(met_phi, axis=0)
calo_met_pt = ak.concatenate(calo_met_pt, axis=0)
met_pt_pf = ak.concatenate(met_pt_pf, axis=0)
met_phi_pf = ak.concatenate(met_phi_pf, axis=0)

electron_pt = ak.concatenate(electron_pt, axis=0)
electron_eta = ak.concatenate(electron_eta, axis=0)
electron_phi = ak.concatenate(electron_phi, axis=0)
electron_mass = ak.concatenate(electron_mass, axis=0)

tau_pt = ak.concatenate(tau_pt, axis=0)
tau_eta = ak.concatenate(tau_eta, axis=0)
tau_phi = ak.concatenate(tau_phi, axis=0)
tau_mass = ak.concatenate(tau_mass, axis=0)

photon_pt = ak.concatenate(photon_pt, axis=0)
photon_eta = ak.concatenate(photon_eta, axis=0)
photon_phi = ak.concatenate(photon_phi, axis=0)
photon_mass = ak.concatenate(photon_mass, axis=0)

gen_weights = ak.concatenate(gen_weights, axis=0)
pileup = ak.concatenate(pileup, axis=0)

HLT_PFMET170_HBHECleaned = ak.concatenate(HLT_PFMET170_HBHECleaned, axis=0) 
HLT_PFMET170_HBHE_BeamHaloCleaned = ak.concatenate(HLT_PFMET170_HBHE_BeamHaloCleaned, axis=0)
HLT_PFMETNoMu120_PFMHTNoMu120_IDTight = ak.concatenate(HLT_PFMETNoMu120_PFMHTNoMu120_IDTight, axis=0)
HLT_MET75_IsoTrk50 = ak.concatenate(HLT_MET75_IsoTrk50, axis=0)


# Decompose MET into x and y components
met_px = met_pt * np.cos(met_phi)
met_py = met_pt * np.sin(met_phi)

# Decompose the transverse momentum (pT) of muons, electrons, and taus into x and y components
muon_px = muon_pt * np.cos(muon_phi)
muon_py = muon_pt * np.sin(muon_phi)

electron_px = electron_pt * np.cos(electron_phi)
electron_py = electron_pt * np.sin(electron_phi)

tau_px = tau_pt * np.cos(tau_phi)
tau_py = tau_pt * np.sin(tau_phi)

# Combine all px and py components using ak.concatenate
combined_px = ak.concatenate([muon_px, electron_px, tau_px], axis=1)
combined_py = ak.concatenate([muon_py, electron_py, tau_py], axis=1)

# Sum the x and y components to calculate the recoil vector U
recoil_px = met_px + ak.sum(combined_px, axis=1)
recoil_py = met_py + ak.sum(combined_py, axis=1)

# Create a structured array representing the recoil vector U with x and y components
recoil_vector_u = ak.zip({"px": recoil_px, "py": recoil_py})

# Calculate the magnitude of the recoil vector U
recoil_magnitude_u = np.sqrt(recoil_px**2 + recoil_py**2)

jets_sorted = ak.sort(jet_pt, axis=1, ascending=False)
eta_sorted = ak.sort(jet_eta, axis=1, ascending=False)
phi_sorted = ak.sort(jet_phi, axis=1, ascending=False)
mass_sorted = ak.sort(jet_mass, axis=1, ascending=False)
btag_sorted = ak.sort(jet_btag_csvv2, axis=1, ascending=False)
ch_hef_sorted = ak.sort(jet_ch_hef, axis=1, ascending=False)

mask_4jets = ak.num(jet_pt) >= 4

# Apply mask to all arrays
jets_sorted = jets_sorted[mask_4jets]
eta_sorted = eta_sorted[mask_4jets]
phi_sorted = phi_sorted[mask_4jets]
mass_sorted = mass_sorted[mask_4jets]
btag_sorted = btag_sorted[mask_4jets]
ch_hef_sorted = ch_hef_sorted[mask_4jets]
 
def save_file(dataset_name='default', IS_DATA=False):
    """
    Processes and saves the data in CSV, dealing with irregular structures for correct export.
    """

    N_gen = -999
    gw_pos = -999
    gw_neg = -999

    tmpval_events = np.ones(len(met_pt[mask_4jets]))
    tmpval = np.ones_like(met_pt[mask_4jets]) 

    if not IS_DATA:
        gen_weights_per_candidate = tmpval * gen_weights[mask_4jets]
        pileup_per_candidate = tmpval * pileup[mask_4jets]
        gw_pos = ak.sum(gen_weights > 0)
        gw_neg = ak.sum(gen_weights < 0)
        N_gen = gw_pos - gw_neg
    else:
        pileup_per_candidate = -999 * tmpval
        gen_weights_per_candidate = -999 * tmpval

    # Dictionary of output
    mydict = {
        'pileup': pileup_per_candidate,
        'weight': gen_weights_per_candidate,
        'N_gen': N_gen * tmpval_events,
        'gw_pos': gw_pos * tmpval_events,
        'gw_neg': gw_neg * tmpval_events,

        # MET
        'met_pt': met_pt[mask_4jets],
        'met_phi': met_phi[mask_4jets],
        'met_eta': met_eta[mask_4jets],
        'recoil_magnitude_u': recoil_magnitude_u[mask_4jets],

        # MET alternativos
        'calo_met_pt': calo_met_pt[mask_4jets],
        'met_pt_pf': met_pt_pf[mask_4jets],
        'met_phi_pf': met_phi_pf[mask_4jets],

        # Jets (4 principais)
        'lead_jet_pt': jets_sorted[:, 0],
        'lead_jet_eta': eta_sorted[:, 0],
        'lead_jet_phi': phi_sorted[:, 0],
        'lead_jet_mass': mass_sorted[:, 0],
        'lead_jet_btag_csvv2': btag_sorted[:, 0],
        'lead_jet_ch_hef': ch_hef_sorted[:, 0],

        'lead_jet2_pt': jets_sorted[:, 1],
        'lead_jet2_eta': eta_sorted[:, 1],
        'lead_jet2_phi': phi_sorted[:, 1],
        'lead_jet2_mass': mass_sorted[:, 1],
        'lead_jet2_btag_csvv2': btag_sorted[:, 1],
        'lead_jet2_ch_hef': ch_hef_sorted[:, 1],

        'lead_jet3_pt': jets_sorted[:, 2],
        'lead_jet3_eta': eta_sorted[:, 2],
        'lead_jet3_phi': phi_sorted[:, 2],
        'lead_jet3_mass': mass_sorted[:, 2],
        'lead_jet3_btag_csvv2': btag_sorted[:, 2],
        'lead_jet3_ch_hef': ch_hef_sorted[:, 2],

        'lead_jet4_pt': jets_sorted[:, 3],
        'lead_jet4_eta': eta_sorted[:, 3],
        'lead_jet4_phi': phi_sorted[:, 3],
        'lead_jet4_mass': mass_sorted[:, 3],
        'lead_jet4_btag_csvv2': btag_sorted[:, 3],
        'lead_jet4_ch_hef': ch_hef_sorted[:, 3],
    }

    # Complete list with all irregular events
    irregular_keys = [
        # Muons
        'muon_pt', 'muon_eta', 'muon_phi', 'muon_mass',
        'muon_iso', 'muon_tightId', 'muon_looseId',

        # Electrons
        'electron_pt', 'electron_eta', 'electron_phi', 'electron_mass',

        # Taus
        'tau_pt', 'tau_eta', 'tau_phi', 'tau_mass',

        # Photons
        'photon_pt', 'photon_eta', 'photon_phi', 'photon_mass',

        # MET e recoil 
        'recoil_magnitude_u',

        # others
        'pileup', 
    ]

    # Adds the converted irregular fields to standard lists
    for key in irregular_keys:
        try:
            array_masked = eval(f"{key}[mask_4jets]")
            mydict[key] = ak.to_list(array_masked)
        except Exception as e:
            print(f"[Warning] '{key}' variable could not be converted: {e}")

    # Creates the DataFrame
    df = pd.DataFrame.from_dict(mydict)

    # Export to CSV
    df.to_csv(f"../training_data/Z2JetsToNuNu/{dataset_name}.csv", index=False)
    print(f"Saved data: {dataset_name}.csv")

# Call the function by passing the dataset name and whether it is data or Monte Carlo
save_file(dataset_name=dataset_name, IS_DATA=False)

