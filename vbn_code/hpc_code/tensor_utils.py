import pandas as pd
import numpy as np

def get_tensor_unit_table(unit_table, tensor_unit_ids):
    '''Returns unit table with units ordered as they are in the tensor
    
    INPUTS:
        unit_table: unit dataframe with unit metadata
        tensor_unit_ids: the unit ids stored in the session tensor (ie session_tensor['unitIds'][()])
    
    OUTPUTS:
        tensor_unit_table: unit table filtered for the units in the tensor and reordered for convenient indexing
    '''
    
    units = unit_table.set_index('unit_id').loc[tensor_unit_ids].reset_index()

    return units
    

def get_unit_ids_by_area(unit_table, areaname):
    '''Get the ids for units in a particular area
    
    INPUTS:
        unit_table: unit dataframe
        areaname: name of area as it appears in the units table 'structure_with_layer' column
        
    OUTPUTS:
        list of unit ids for units in the area of interest
    '''
    
    unit_ids = unit_table[unit_table['structure_with_layer']==areaname]['unit_id'].values
    
    return unit_ids
    

def get_unit_indices_by_area(unit_table, tensor_unit_ids, areaname, method='equal'):
    '''
    Get the indices for the unit dimension of the tensor for only those units in a given area
    
    INPUTS:
        unit_table: unit dataframe for session
        tensor_unit_ids: the unit ids stored in the session tensor (ie session_tensor['unitIds'][()])
        areaname: the area of interest for which you would like to filter units
        method: 
            if 'equal' only grab the units with an exact match to the areaname
            if 'contains' grab all units that contain the areaname in the string. This can be useful to, for example,
                grab all of the units in visual cortex regardless of area or layer (areaname would be 'VIS')
    
    OUTPUT
        the indices of the tensor for the units of interest
    '''
    
    units = get_tensor_unit_table(unit_table, tensor_unit_ids)
    if method == 'equal':
        unit_indices = units[units['structure_acronym']==areaname].index.values
    
    elif method == 'contains':
        unit_indices = units[units['structure_with_layer'].str.contains(areaname)].index.values
    
    return unit_indices
    

def get_tensor_for_unit_selection(unit_indices, spikes, dtype=bool):
    '''
    Subselect a portion of the tensor for a particular set of units. You might try to do this
    with something like spikes[unit_indices] but this ends up being very slow. When the H5 file is saved,
    the data is chunked by units, so reading it out one unit at a time is much faster
    
    INPUTS:
        unit_indices: the indices of the array along the unit dimension that you'd like to extract
        spikes: the spikes tensor (ie tensor['spikes'] from the h5 file)
        
    OUTPUT:
        the subselected spikes tensor for only the units of interest
    '''
    
    s = np.zeros((len(unit_indices), spikes.shape[1], spikes.shape[2]), dtype=dtype)
    for i,u in enumerate(unit_indices):
        s[i]=spikes[u]
    
    return s