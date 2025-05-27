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

pretty_print(events.keys(), fmt='40s', require=['HLT_PFMET170'])

pretty_print(events.keys(), fmt='40s', require=['HLT_PFMETNoMu'])

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

# Lists to store the variables of events that have passed through the cuts
selected_jet_pt = []
selected_jet_eta = []
selected_jet_phi = []
selected_jet_mass = []
selected_jet_btag_csvv2 = []
selected_jet_ch_hef = []

selected_muon_pt = []
selected_muon_eta = []
selected_muon_phi = []
selected_muon_mass = []

selected_met_pt = []
selected_met_phi = []
selected_met_eta = []

selected_calo_met_pt = []

selected_met_pt_pf = []
selected_met_phi_pf = []

selected_recoil_magnitude_u = []

selected_electron_pt = []
selected_electron_eta = []
selected_electron_phi = []
selected_electron_mass = []

selected_tau_pt = []
selected_tau_eta = []
selected_tau_phi = []
selected_tau_mass = []

selected_photon_pt = []
selected_photon_eta = []
selected_photon_phi = []
selected_photon_mass = []

selected_lead_jet_pt = []
selected_lead_jet_eta = []
selected_lead_jet_phi = []
selected_lead_jet_mass = []
selected_lead_jet_btag_csvv2 = []
selected_lead_jet_ch_hef = []

selected_lead_jet2_pt = []
selected_lead_jet2_eta = []
selected_lead_jet2_phi = []
selected_lead_jet2_mass = []
selected_lead_jet2_btag_csvv2 = []
selected_lead_jet2_ch_hef = []

selected_lead_jet3_pt = []
selected_lead_jet3_eta = []
selected_lead_jet3_phi = []
selected_lead_jet3_mass = []
selected_lead_jet3_btag_csvv2 = []
selected_lead_jet3_ch_hef = []

selected_lead_jet4_pt = []
selected_lead_jet4_eta = []
selected_lead_jet4_phi = []
selected_lead_jet4_mass = []
selected_lead_jet4_btag_csvv2 = []
selected_lead_jet4_ch_hef = []

gen_weights_selected = []
pileup_selected = []

# Calculate the number of jets for all events before the loop
num_jets = ak.num(jet_pt, axis=1)  

#Define the threshold for btag
medium_btag_threshold = 0.8

mask_lumi = build_lumi_mask('Cert_278820-284044_13TeV_Legacy2016_Collisions16GH_JSON.txt', events)

# Define the processing function
def process_file(dataset='default', IS_DATA=False):
    """
    Function to process data or Monte Carlo files.
    Parameters:
    - dataset: Name of the dataset (string)
    - IS_DATA: Boolean indicating if the input is real data (True) or Monte Carlo (False)
    """
    
    # Initialize event counters
    total_events = 0
    events_passed_trigger = 0
    events_passed_recoil_magnitude_u = 0
    events_passed_lead_jet = 0
    

    # Iterate over the events and apply cuts one by one
    for i in range(len(recoil_magnitude_u)):
        total_events += 1  

        # Select events that pass MET triggers
        if (HLT_PFMET170_HBHECleaned[i] or 
                HLT_PFMET170_HBHE_BeamHaloCleaned[i] or
                HLT_PFMETNoMu120_PFMHTNoMu120_IDTight[i] or 
                HLT_MET75_IsoTrk50[i]):
            events_passed_trigger += 1
        else: 
             continue

        # Cuts: Baseline 
        if recoil_magnitude_u[i] > 200:
            continue
        else:
            events_passed_recoil_magnitude_u += 1


        # Get the jets and sort them by pt
        jets_sorted_indices = ak.argsort(jet_pt[i], ascending=False)  

        if len(jet_pt[i]) > 0:
            jets_sorted = jet_pt[i][jets_sorted_indices]  
            eta_sorted = jet_eta[i][jets_sorted_indices]  
            phi_sorted = jet_phi[i][jets_sorted_indices]  
            mass_sorted = jet_mass[i][jets_sorted_indices]  
            ch_hef_sorted = jet_ch_hef[i][jets_sorted_indices]  
            btag_sorted = jet_btag_csvv2[i][jets_sorted_indices]  

            # Lead jet cut
            lead_jet_pt = jets_sorted[0]
            lead_jet_eta = eta_sorted[0]
            lead_jet_ch_hef = ch_hef_sorted[0]
            
            # Aplica o corte lead_jet_pt > 200
            if lead_jet_pt > 200:
                continue
            else:
                events_passed_lead_jet += 1

        # Cuts: Jets + pt miss region
        if not IS_DATA:
            gen_weights_selected.append(gen_weights[i])
            pileup_selected.append(pileup[i])
                           
        for j in range(len(jets_sorted)):  
            selected_jet_pt.append(jets_sorted[j])
            selected_jet_eta.append(eta_sorted[j])
            selected_jet_phi.append(phi_sorted[j])
            selected_jet_mass.append(mass_sorted[j])
            selected_jet_btag_csvv2.append(btag_sorted[j])
            selected_jet_ch_hef.append(ch_hef_sorted[j])

        selected_muon_pt.append(muon_pt[i])
        selected_muon_eta.append(muon_eta[i])
        selected_muon_phi.append(muon_phi[i])
        selected_muon_mass.append(muon_mass[i])

        selected_met_pt.append(met_pt[i])
        selected_met_phi.append(met_phi[i])
        selected_met_eta.append(met_eta[i])  

        selected_calo_met_pt.append(calo_met_pt[i])

        selected_met_pt_pf.append(met_pt_pf[i])
        selected_met_phi_pf.append(met_phi_pf[i])

        selected_recoil_magnitude_u.append(recoil_magnitude_u[i])

        selected_electron_pt.append(electron_pt[i])
        selected_electron_eta.append(electron_eta[i])
        selected_electron_phi.append(electron_phi[i])
        selected_electron_mass.append(electron_mass[i])

        selected_tau_pt.append(tau_pt[i])
        selected_tau_eta.append(tau_eta[i])
        selected_tau_phi.append(tau_phi[i])
        selected_tau_mass.append(tau_mass[i])

        selected_photon_pt.append(photon_pt[i])
        selected_photon_eta.append(photon_eta[i])
        selected_photon_phi.append(photon_phi[i])
        selected_photon_mass.append(photon_mass[i])
        
        if len(jets_sorted) > 0:
            selected_lead_jet_pt.append(jets_sorted[0])
            selected_lead_jet_eta.append(eta_sorted[0])
            selected_lead_jet_phi.append(phi_sorted[0])
            selected_lead_jet_mass.append(mass_sorted[0])
            selected_lead_jet_btag_csvv2.append(btag_sorted[0])
            selected_lead_jet_ch_hef.append(ch_hef_sorted[0])

        if len(jets_sorted) > 1:        
            selected_lead_jet2_pt.append(jets_sorted[1])
            selected_lead_jet2_eta.append(eta_sorted[1])
            selected_lead_jet2_phi.append(phi_sorted[1])
            selected_lead_jet2_mass.append(mass_sorted[1])
            selected_lead_jet2_btag_csvv2.append(btag_sorted[1])
            selected_lead_jet2_ch_hef.append(ch_hef_sorted[1])

        if len(jets_sorted) > 2:
            selected_lead_jet3_pt.append(jets_sorted[2])
            selected_lead_jet3_eta.append(eta_sorted[2])
            selected_lead_jet3_phi.append(phi_sorted[2])
            selected_lead_jet3_mass.append(mass_sorted[2])
            selected_lead_jet3_btag_csvv2.append(btag_sorted[2])
            selected_lead_jet3_ch_hef.append(ch_hef_sorted[2])

        if len(jets_sorted) > 3:
            selected_lead_jet4_pt.append(jets_sorted[3])
            selected_lead_jet4_eta.append(eta_sorted[3])
            selected_lead_jet4_phi.append(phi_sorted[3])
            selected_lead_jet4_mass.append(mass_sorted[3])
            selected_lead_jet4_btag_csvv2.append(btag_sorted[3])
            selected_lead_jet4_ch_hef.append(ch_hef_sorted[3])

    print(f"Processing summary:")
    print(f"Total events processed: {total_events}")
    print(f"Events after trigger selection: {events_passed_trigger}")
    print(f"Events after recoil magnitude U cut: {events_passed_recoil_magnitude_u}")
    print(f"Events after lead jet cut: {events_passed_lead_jet}")


#Calling the process_file function with the chosen name
process_file(dataset=dataset_name, IS_DATA=False)

def save_file(dataset_name='default', IS_DATA=False):
    """
    Function to process and save real data or Monte Carlo in CSV file.
    Parameters:
    - dataset_name: Name of the dataset (string)
    - IS_DATA: Boolean that indicates whether it is given real (True) or Monte Carlo (False)
    """

    # Initialize variables
    N_gen = -999
    pileup = -999
    gw_pos = -999
    gw_neg = -999

    pileup_per_candidate = None

    tmpval_events = np.ones(len(selected_met_pt))
    tmpval = np.ones_like(selected_met_pt) 

    # Conditional for Monte Carlo or real data
    if not IS_DATA:
        # If it is Monte Carlo, calculate gen_weights and pileup 
        gen_weights_per_candidate = tmpval * gen_weights_selected
        pileup_per_candidate = tmpval * pileup_selected

        # Calculate gw_pos, gw_neg and N_gen
        gw_pos = ak.count(gen_weights[gen_weights > 0])
        gw_neg = ak.count(gen_weights[gen_weights < 0])
        N_gen = gw_pos - gw_neg
    else:
        # If real data, use placeholder values
        pileup_per_candidate = -999 * tmpval
        gen_weights_per_candidate = -999 * tmpval

    # Create the dictionary with the results
    mydict = {}
    mydict['pileup'] = pileup_per_candidate
    mydict['weight'] = gen_weights_per_candidate
    mydict['N_gen'] = N_gen * tmpval_events
    mydict['gw_pos'] = gw_pos * tmpval_events
    mydict['gw_neg'] = gw_neg * tmpval_events

    # Muons
    mydict['muon_pt'] = selected_muon_pt
    mydict['muon_eta'] = selected_muon_eta
    mydict['muon_phi'] = selected_muon_phi
    mydict['muon_mass'] = selected_muon_mass

    # MET
    mydict['met_pt'] = selected_met_pt
    mydict['met_phi'] = selected_met_phi
    mydict['met_eta'] = selected_met_eta

    #U 
    mydict['recoil_magnitude_u'] = selected_recoil_magnitude_u

    # CaloMET e MET PF
    mydict['calo_met_pt'] = selected_calo_met_pt
    mydict['met_pt_pf'] = selected_met_pt_pf
    mydict['met_phi_pf'] = selected_met_phi_pf

    # Electrons
    mydict['electron_pt'] = selected_electron_pt
    mydict['electron_eta'] = selected_electron_eta
    mydict['electron_phi'] = selected_electron_phi
    mydict['electron_mass'] = selected_electron_mass

    # Taus
    mydict['tau_pt'] = selected_tau_pt
    mydict['tau_eta'] = selected_tau_eta
    mydict['tau_phi'] = selected_tau_phi
    mydict['tau_mass'] = selected_tau_mass

    # Photons
    mydict['photon_pt'] = selected_photon_pt
    mydict['photon_eta'] = selected_photon_eta
    mydict['photon_phi'] = selected_photon_phi
    mydict['photon_mass'] = selected_photon_mass

    # Lead Jet 1
    mydict['lead_jet_pt'] = selected_lead_jet_pt
    mydict['lead_jet_eta'] = selected_lead_jet_eta
    mydict['lead_jet_phi'] = selected_lead_jet_phi
    mydict['lead_jet_mass'] = selected_lead_jet_mass
    mydict['lead_jet_btag_csvv2'] = selected_lead_jet_btag_csvv2
    mydict['lead_jet_ch_hef'] = selected_lead_jet_ch_hef

    # Lead Jet 2
    mydict['lead_jet2_pt'] = selected_lead_jet2_pt
    mydict['lead_jet2_eta'] = selected_lead_jet2_eta
    mydict['lead_jet2_phi'] = selected_lead_jet2_phi
    mydict['lead_jet2_mass'] = selected_lead_jet2_mass
    mydict['lead_jet2_btag_csvv2'] = selected_lead_jet2_btag_csvv2
    mydict['lead_jet2_ch_hef'] = selected_lead_jet2_ch_hef

    # Lead Jet 3
    mydict['lead_jet3_pt'] = selected_lead_jet3_pt
    mydict['lead_jet3_eta'] = selected_lead_jet3_eta
    mydict['lead_jet3_phi'] = selected_lead_jet3_phi
    mydict['lead_jet3_mass'] = selected_lead_jet3_mass
    mydict['lead_jet3_btag_csvv2'] = selected_lead_jet3_btag_csvv2
    mydict['lead_jet3_ch_hef'] = selected_lead_jet3_ch_hef

    # Lead Jet 4
    mydict['lead_jet4_pt'] = selected_lead_jet4_pt
    mydict['lead_jet4_eta'] = selected_lead_jet4_eta
    mydict['lead_jet4_phi'] = selected_lead_jet4_phi
    mydict['lead_jet4_mass'] = selected_lead_jet4_mass
    mydict['lead_jet4_btag_csvv2'] = selected_lead_jet4_btag_csvv2
    mydict['lead_jet4_ch_hef'] = selected_lead_jet4_ch_hef

    max_length = max(len(lst) for lst in mydict.values())  

    for key in mydict:
        if isinstance(mydict[key], list):  
            while len(mydict[key]) < max_length:
                mydict[key].append(None)  

    # Convert the dictionary into a DataFrame pandas
    df = pd.DataFrame.from_dict(mydict)

    # Save the DataFrame as a CSV file, using the name provided by the user
    df.to_csv(f"../training_data/Z2JetsToNuNu_Negative/{dataset_name}.csv", index=False)

    print(f"Data processed and saved as {dataset_name}.csv")

# Call the function by passing the dataset name and whether it is given or Monte Carlo
save_file(dataset_name=dataset_name, IS_DATA=False)
