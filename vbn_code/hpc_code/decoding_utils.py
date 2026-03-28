import math
import os
import warnings
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import sklearn
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
import h5py
import argparse

import pathlib


def getTrainTestSplits(y,nSplits,hasClasses=True):
    # cross validation using stratified shuffle split
    # each split preserves the percentage of samples of each class
    # all samples used in one test set
    if isinstance(y[0], str):
        notNan = np.array([True]*len(y))
    else:
        notNan = ~np.isnan(y)

    y = np.array(y)
    if hasClasses:
        classVals = np.unique(y[notNan])
        nClasses = len(classVals)
        nSamples = notNan.sum()
        samplesPerClass = [np.sum(y==val) for val in classVals]
        if any(n < nSplits for n in samplesPerClass):
            return None,None
    else:
        classVals = [None]
        samplesPerClass = [notNan.sum()] 
    samplesPerSplit = [round(n/nSplits) for n in samplesPerClass]
    shuffleInd = np.random.permutation(np.where(notNan)[0])
    trainInd = []
    testInd = []
    for k in range(nSplits):
        testInd.append([])
        for val,n in zip(classVals,samplesPerSplit):
            start = int(k*n)
            end = int(start+n)
            ind = shuffleInd[y[shuffleInd]==val] if hasClasses else shuffleInd
            testInd[-1].extend(ind[start:end] if k+1<nSplits else ind[start:])
        trainInd.append(np.setdiff1d(shuffleInd,testInd[-1]))
    return trainInd,testInd


def trainDecoder(model,X,y,nSplits):
    classVals = np.unique(y)
    nClasses = len(classVals)
    nSamples = len(y)
    cv = {'estimator': [sklearn.base.clone(model) for _ in range(nSplits)]}
    cv['train_score'] = []
    cv['test_score'] = []
    cv['predict'] = np.full(nSamples, '', dtype='O')
    cv['predict_proba'] = np.full((nSamples,nClasses),np.nan)
    cv['decision_function'] = np.full((nSamples,nClasses),np.nan) if nClasses>2 else np.full(nSamples,np.nan)
    cv['feature_importance'] = []
    cv['coef'] = []
    modelMethods = dir(model)
    trainInd,testInd = getTrainTestSplits(y,nSplits)
    for estimator,train,test in zip(cv['estimator'],trainInd,testInd):
        estimator.fit(X[train],y[train])
        cv['train_score'].append(estimator.score(X[train],y[train]))
        cv['test_score'].append(estimator.score(X[test],y[test]))
        cv['predict'][test] = estimator.predict(X[test])
        for method in ('predict_proba','decision_function'):
            if method in modelMethods:
                cv[method][test] = getattr(estimator,method)(X[test])
        for attr in ('feature_importance_','coef_'):
            if attr in estimator.__dict__:
                cv[attr[:-1]].append(getattr(estimator,attr))
    return cv

def trainModel(model,X,y,nSplits):
    classVals = np.unique(y)
    nClasses = len(classVals)
    nSamples = len(y)
    cv = {'estimator': [sklearn.base.clone(model) for _ in range(nSplits)]}
    cv['train_balanced_accuracy'] = []
    cv['test_balanced_accuracy'] = []
    cv['predict'] = np.full(nSamples, '', dtype='O')
    cv['predict_proba'] = np.full((nSamples,nClasses),np.nan)
    cv['coef'] = []
    modelMethods = dir(model)
    trainInd,testInd = getTrainTestSplits(y,nSplits,hasClasses=False)
    for num_iter, (estimator,train,test) in enumerate(zip(cv['estimator'],trainInd,testInd)):
        estimator.fit(X[train],y[train])
        cv['train_balanced_accuracy'].append(balanced_accuracy_score(y[train], estimator.predict(X[train])))
        cv['test_balanced_accuracy'].append(balanced_accuracy_score(y[test], estimator.predict(X[test])))
        cv['predict'][test] = estimator.predict(X[test])
        for method in ('predict_proba',):
            if method in modelMethods:
                cv[method][test] = getattr(estimator,method)(X[test])
        for attr in ('coef_',):
            if attr in estimator.__dict__:
                cv[attr[:-1]].append(getattr(estimator,attr))
    return cv


def getUnitsInRegion(units,region,layer=None,rs=False,fs=False, cell_type=None):
    if region == 'all':
        return np.array([True]*len(units))
    if region in ('SC/MRN cluster 1','SC/MRN cluster 2'):
        clust = 1 if '1' in region else 2
        dirPath = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/VBN_video_analysis')
        clustId = np.load(os.path.join(dirPath,'sc_mrn_clusterId.npy'))
        clustUnitId = np.load(os.path.join(dirPath,'sc_mrn_clusterUnitId.npy'))
        u = clustUnitId[np.in1d(clustUnitId,units.index) & (clustId==clust)]
        inRegion = np.in1d(units.index,u)
    else:
        if region=='VISall':
            reg = ('VISp','VISl','VISrl','VISal','VISpm','VISam')
        elif region in ('SC', 'SCm'):
            reg = ('SCig','SCiw')
        elif region=='Hipp':
            reg = ('HPF','DG','CA1','CA3')
        elif region=='Sub':
            reg = ('SUB','ProS','PRE','POST')
        elif region=='midbrain':
            reg = ('SCig', 'SCiw', 'APN', 'MRN', 'MB')
        elif region=='SCMRN':
            reg = ('SCig', 'SCiw', 'MRN')
        elif '_' in region:
            reg = tuple(region.split('_'))
        else:
            reg = region
        inRegion = np.in1d(units['structure_acronym'], reg)
        if np.any([a in region for a in ('VISp','VISl','VISrl','VISal','VISpm','VISam', 'VISall')]):
            if layer is not None and not layer == 'all':
                if layer in ['4', '5']:
                    inRegion = inRegion & np.in1d(units['cortical_layer'], layer)
                elif layer == '6':
                    inRegion = inRegion & (units['cortical_layer'].isin(['6a', '6b', '6'])).values
                elif layer == '2/3':
                    inRegion = inRegion & (units['cortical_layer'].isin(['1', '2/3'])).values
                elif layer not in ['1', '2/3', '4', '5', '6a', '6b', '6']:
                    raise ValueError(f'layer {layer} not recognized')
            if rs or fs:
                rsUnits = np.array(units['waveform_duration'] > 0.4)
                if rs and not fs:
                    inRegion = inRegion & rsUnits
                elif fs and not rs:
                    inRegion = inRegion & ~rsUnits
            if cell_type is not None and not cell_type == 'all':


                units['SST'] =  (units['genotype'].str.contains('Sst')) & \
                                (units['pulse_high_mean_evoked_rate_zscored']>2) & \
                                (units['pulse_high_first_spike_latency']<0.008) & \
                                (units['pulse_high_first_spike_jitter']<0.002) & \
                                (units['raised_cosine_high_fraction_time_responsive']>0.3)
                

                units['VIP'] =  (units['genotype'].str.contains('Vip')) & \
                                (units['pulse_high_mean_evoked_rate_zscored']>2) & \
                                (units['pulse_high_first_spike_latency']<0.008) & \
                                (units['pulse_high_first_spike_jitter']<0.002) & \
                                (units['raised_cosine_high_fraction_time_responsive']>0.3)

                units['RS'] =   (units['waveform_duration']>0.4)&(~units['SST'])&(~units['VIP'])
                units['FS'] =   (units['waveform_duration']<0.4)&(~units['SST'])&(~units['VIP'])

                inRegion = inRegion & units[cell_type].values

        if 'cluster' in region:
            clustTable = pd.read_csv(pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/units_with_fast_slow_cluster_ids.csv'))
            clustId = np.array(clustTable['fast_slow_cluster_id'])
            clustUnitId = np.array(clustTable['unit_id'])
            clust = 1 if 'cluster 1' in region else 2
            u = clustUnitId[np.in1d(clustUnitId,units.index) & (clustId==clust)]
            inRegion = np.in1d(units.index,u)
    return inRegion

# def getUnitsInRegion(units,region,layer=None,rs=False,fs=False):
#     if region == 'all':
#         inRegion = [True]*len(units)
#     elif region in ('SC/MRN cluster 1','SC/MRN cluster 2'):
#         clust = 1 if '1' in region else 2
#         dirPath = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/VBN_video_analysis')
#         clustId = np.load(os.path.join(dirPath,'sc_mrn_clusterId.npy'))
#         clustUnitId = np.load(os.path.join(dirPath,'sc_mrn_clusterUnitId.npy'))
#         u = clustUnitId[np.in1d(clustUnitId,units.index) & (clustId==clust)]
#         inRegion = np.in1d(units.index,u)
#     else:
#         if region=='VISall':
#             reg = ('VISp','VISl','VISrl','VISal','VISpm','VISam')
#         elif region=='SC':
#             reg = ('SCig','SCiw')
#         elif region=='Hipp':
#             reg = ('HPF','DG','CA1','CA3')
#         elif region=='Sub':
#             reg = ('SUB','ProS','PRE','POST')
#         elif region=='midbrain':
#             reg = ('SCig', 'SCiw', 'APN', 'MRN', 'MB')
#         elif region=='SCMRN':
#             reg = ('SCig', 'SCiw', 'MRN')
#         else:
#             reg = region
#         inRegion = np.in1d(units['structure_acronym'],reg)
#         if 'VIS' in region:
#             if layer is not None:
#                 inRegion = inRegion & np.in1d(units['cortical_layer'],layer)
#             if rs or fs:
#                 rsUnits = np.array(units['waveform_duration'] > 0.4)
#                 if rs and not fs:
#                     inRegion = inRegion & rsUnits
#                 elif fs and not rs:
#                     inRegion = inRegion & ~rsUnits
#         if 'cluster' in region:
#             clustTable = pd.read_csv(pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/units_with_fast_slow_cluster_ids.csv'))
#             clustId = np.array(clustTable['fast_slow_cluster_id'])
#             clustUnitId = np.array(clustTable['unit_id'])
#             clust = 1 if 'cluster 1' in region else 2
#             u = clustUnitId[np.in1d(clustUnitId,units.index) & (clustId==clust)]
#             inRegion = np.in1d(units.index,u)
#     return inRegion


def get_units_in_cluster(unit_table, *cluster_ids, clustering='new'):

    if cluster_ids[0] == 'all':
        return np.array([True]*len(unit_table))

    if clustering == 'old':
        cluster_column = 'cluster_labels'
    elif clustering == 'new':
        cluster_column = 'cluster_labels_new'
    else:
        raise(ValueError(f'Invalid clustering value: {clustering}. Must be "old" or "new"'))
    
    # print(clustering)
    return (unit_table[cluster_column].isin(cluster_ids)).values


def apply_unit_quality_filter(unit_table, no_abnorm=True):
    
    if no_abnorm:
        no_abnorm_filter = (unit_table['abnormal_activity'].isnull() & unit_table['abnormal_histology'].isnull())
    else:
        no_abnorm_filter = [True]*len(unit_table)

    qc_filter = [(unit_table['isi_violations']<0.5)&
                (unit_table['amplitude_cutoff']<0.1)&
                (unit_table['presence_ratio']>0.9)&
                (unit_table['quality']=='good')&
                (no_abnorm_filter)]
    
    return qc_filter[0].values


def get_imagematched_lick_nolicks(stim):

    flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    image_ids = stim[~stim['omitted']]['image_name'].unique()
    im_dict = {im:{} for im in np.unique(image_ids)}
    image_lick_inds = []
    image_nolick_inds = []
    for im in np.unique(image_ids):

        lickflashes = np.where(lick & nonChangeFlashes & (stim['image_name']==im))[0]
        nolickflashes = np.where(~lick & nonChangeFlashes & (stim['image_name']==im))[0]

        minflashes = min(len(lickflashes), len(nolickflashes))
        if minflashes>0:
            image_lick_inds.append(np.random.choice(lickflashes, minflashes, replace=False))
            image_nolick_inds.append(np.random.choice(nolickflashes, minflashes, replace=False))


    image_licks = np.array([False]*len(stim))
    image_nolicks = np.array([False]*len(stim))
    for lick_inds, nolick_inds in zip(image_lick_inds, image_nolick_inds):
        image_licks[lick_inds] = True
        image_nolicks[nolick_inds] = True

    return image_licks, image_nolicks


def getBehavData(stim):
    # stim = stimulus table or index of
    flashTimes = np.array(stim['start_time'])
    image_ids = np.array(stim['image_name'])
    changeTimes = flashTimes[stim['is_change']]
    hit = np.array(stim['hit'])
    engaged = np.array([np.sum(hit[stim['is_change']][(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in flashTimes])
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlashes = np.array(stim['is_change'] & ~autoRewarded & engaged)
    catch = stim['catch'].copy()
    catch[catch.isnull()] = False
    catch = np.array(catch).astype(bool) & engaged
    catchFlashes = np.zeros(catch.size,dtype=bool)
    catchFlashes[np.searchsorted(flashTimes,np.unique(stim['change_time_no_display_delay'][catch]))] = True
    omittedFlashes = np.array(stim['omitted']) & engaged
    prevOmittedFlashes = np.array(stim['previous_omitted']) & engaged
    nonChangeFlashes = np.array(engaged &
                                (~stim['is_change']) & 
                                (~stim['omitted']) & 
                                (~stim['previous_omitted']) & 
                                (stim['flashes_since_change']>5) &
                                (stim['flashes_since_last_lick']>1))
    novelFlashes = stim['novel_image'].copy()
    novelFlashes[novelFlashes.isnull()] = False
    novelFlashes = np.array(novelFlashes).astype(bool) & engaged
    lick = np.array(stim['lickbout_for_flash_during_response_window'])
    lickTimes = np.array(stim['lick_time'])
    lickLatency = lickTimes - flashTimes
    earlyLick = lickLatency < 0.15
    lateLick = lickLatency > 0.75
    lick[earlyLick | lateLick] = False
    lickTimes[earlyLick | lateLick] = np.nan
    return flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes


def decodeImage(sessionId, unitTable, unitData, stimTable, regions, unitSampleSize, decodeWindowEnd, 
                decode_full_timecourse_index=-1, use_nonchange=True, class_weight=None, cell_type='all'):
    '''
    sessionID: ecephys session id
    unitTable: df made from units_with_cortical_layers.csv
    uniData: tensor from 'vbnAllUnitSpikeTensor.hdf5'
    stimTable: master stim table
    '''
    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=class_weight)
    nCrossVal = 5
    #unitSampleSize = [1,5,10,15,20,25,30,40,50,60]
    decodeWindowSize = 10
    #decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)


    baseWin = slice(680,750)
    respWin = slice(30,100)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    flashes_to_use = nonChangeFlashes if use_nonchange else changeFlashes
    nFlashes = flashes_to_use.sum()
    
    d = {region: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','prediction','confidence', 'balanced_accuracy',
                                                     'imagewise_recall', 'imagewise_precision')}
         for sampleSize in unitSampleSize} for region in regions}
    d['decodeWindows'] = decodeWindows
    d['hit'] = np.array(stim['hit'])[changeFlashes]

    image_id_order = [im for im in np.unique(image_ids[flashes_to_use]) if im not in ['im083_r', 'im111_r']] + ['im083_r', 'im111_r']
    image_counts = [np.sum(flashes_to_use & (image_ids==im)) for im in image_id_order]

    d['image_order'] = image_id_order

    y = []
    for im in image_id_order:
        num_im = (flashes_to_use&(image_ids==im)).sum()
        y.extend([im]*num_im)
    y = np.array(y)

    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = getUnitsInRegion(units,region, cell_type=cell_type)
        if not any(inRegion):
            continue

        highQuality = apply_unit_quality_filter(units)
        final_unit_filter = highQuality & inRegion

        sp = np.zeros((final_unit_filter.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(final_unit_filter)[0]):
            sp[i]=spikes[u,:,:]
        

        # changeSp = sp[:,changeFlashes,:]
        # nonChangeSp = sp[:,np.where(nonChangeFlashes)[0]-1,:]
      
        image_sp = [sp[:, (flashes_to_use & (image_ids==im)), :] for im in image_id_order]

        #hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
        nUnits = final_unit_filter.sum()
        image_sp = [s[:,:,:decodeWindows[-1]].reshape((nUnits,im_count,len(decodeWindows),decodeWindowSize)).sum(axis=-1) for s, im_count in zip(image_sp, image_counts)]
        #changeSp,preChangeSp = [s[:,:,:decodeWindows[-1]].reshape((nUnits,nChange,len(decodeWindows),decodeWindowSize)).sum(axis=-1) for s in (changeSp,preChangeSp)]
        
        decodeWindowSampleSize = unitSampleSize[decode_full_timecourse_index]
        #decodeWindowSampleSize = 10 if region=='SC/MRN cluster 1' else 20
        for sampleSize in unitSampleSize:
            if nUnits < sampleSize:
                continue
            if sampleSize>1:
                if sampleSize==nUnits:
                    nSamples = 1
                    unitSamples = [np.arange(nUnits)]
                else:
                    # >99% chance each neuron is chosen at least once
                    nSamples = int(math.ceil(math.log(0.01)/math.log(1-sampleSize/nUnits)))
                    unitSamples = [np.random.choice(nUnits,sampleSize,replace=False) for _ in range(nSamples)]
            else:
                nSamples = nUnits
                unitSamples = [[i] for i in range(nUnits)]

            for winEnd in decodeWindows:
                # if sampleSize!=decodeWindowSampleSize and winEnd!=decodeWindows[-1]:
                #     continue
                winEnd = int(winEnd/decodeWindowSize)
                for metric in d[region][sampleSize]:
                    d[region][sampleSize][metric].append([])
                for unitSamp in unitSamples:
                    X = np.concatenate([s[unitSamp,:,:winEnd].transpose(1,0,2).reshape((im_count,-1)) for s, im_count in zip(image_sp, image_counts)])                       
                    cv = trainDecoder(model,X,y,nCrossVal)
                    d[region][sampleSize]['trainAccuracy'][-1].append(np.mean(cv['train_score']))
                    d[region][sampleSize]['featureWeights'][-1].append(np.mean(cv['coef'],axis=0).squeeze())
                    d[region][sampleSize]['accuracy'][-1].append(np.mean(cv['test_score']))
                    d[region][sampleSize]['prediction'][-1].append(cv['predict'])
                    d[region][sampleSize]['confidence'][-1].append(cv['decision_function'])
                    d[region][sampleSize]['balanced_accuracy'][-1].append(sklearn.metrics.balanced_accuracy_score(y, cv['predict']))

                    im_wise_recall = []
                    im_wise_precision = []
                    predictions = cv['predict']
                    for im in image_id_order:
                        tps = (y==im)&(predictions==im)
                        fps = (y!=im)&(predictions==im)
                        tns = (y!=im)&(predictions!=im)
                        fns = (y==im)&(predictions!=im)
                        
                        im_recall = np.sum(tps)/(np.sum(y==im))
                        im_wise_recall.append(im_recall)
                        im_precision = np.sum(tps)/(np.sum(tps)+np.sum(fps))
                        im_wise_precision.append(im_precision)
                    d[region][sampleSize]['imagewise_recall'][-1].append(im_wise_recall)
                    d[region][sampleSize]['imagewise_precision'][-1].append(im_wise_precision)

                for metric in d[region][sampleSize]:
                    if metric == 'prediction':
                        d[region][sampleSize][metric][-1] = scipy.stats.mode(d[region][sampleSize][metric][-1],axis=0)[0][0]
                    else:
                        d[region][sampleSize][metric][-1] = np.median(d[region][sampleSize][metric][-1],axis=0)
    warnings.filterwarnings('default')
    # if not output_dir is None:
    #     np.save(os.path.join(output_dir,'unitChangeDecoding','unitChangeDecoding_'+str(sessionId)+'.npy'),d)
    return d


def decodeImageSlidingWindow(sessionId, unitTable, unitData, stimTable, 
                            regions, unitSampleSize, decodeWindowEnd, decodeWindowSize, decodeWindowBinSize, decodeWindowSlidingStep,
                            use_nonchange=True, through_omission=False, class_weight=None, rs=False, cluster='all',
                            ):
    '''
    Very similar to decodeImage, but now sliding a window across time (rather than accumulating)
    sessionID: ecephys session id
    unitTable: df made from units_with_cortical_layers.csv
    uniData: tensor from 'vbnAllUnitSpikeTensor.hdf5'
    stimTable: master stim table
    '''

    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
    nCrossVal = 5

    # decodeWindowSize = 50
    # decodeWindowBinSize = 10
    # decodeWindowSlidingStep = 20
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSlidingStep,decodeWindowSlidingStep)

    baseWin = slice(680,750)
    respWin = slice(30,100)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']

    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    flashes_to_use = nonChangeFlashes if use_nonchange else changeFlashes
    if through_omission:
        flashes_to_use = np.append(omittedFlashes[1:], False)
        #flashes_to_use = np.append(changeFlashes[2:], [False, False]) #test
    
    #flashes_to_use = np.append(changeFlashes[1:], False)
    nFlashes = flashes_to_use.sum()

    d = {region: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','prediction','confidence', 'balanced_accuracy',
                                                        'imagewise_recall', 'imagewise_precision', 'image_order', 'unit_ids')}
            for sampleSize in unitSampleSize} for region in regions}
    d['decodeWindows'] = decodeWindows
    d['hit'] = np.array(stim['hit'])[changeFlashes]

    if decodeWindowEnd>750:
        flashes_to_use = flashes_to_use[:-1]
        image_ids = image_ids[:-1]

    image_id_order = [im for im in np.unique(image_ids[flashes_to_use]) if im not in ['im083_r', 'im111_r', 'omitted']] + ['im083_r', 'im111_r']
    y = []
    for im in image_id_order:
        num_im = (flashes_to_use&(image_ids==im)).sum()
        y.extend([im]*num_im)

    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = getUnitsInRegion(units,region, rs=rs)
        highQuality = apply_unit_quality_filter(units, no_abnorm=False)
        inCluster = get_units_in_cluster(units, *cluster, clustering='new')

        final_unit_filter = highQuality & inRegion & inCluster
        if np.sum(final_unit_filter) < np.min(unitSampleSize):
            continue
        
        if decodeWindowEnd<=750:
            sp = np.zeros((final_unit_filter.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
        else:
            sp = np.zeros((final_unit_filter.sum(),spikes.shape[1]-1,spikes.shape[2]*2),dtype=bool)

        for i,u in enumerate(np.where(final_unit_filter)[0]):
            if decodeWindowEnd<=750:
                sp[i]=spikes[u,:,:]
            else:
                sp[i] = np.concatenate((spikes[u,:-1,:], spikes[u,1:,:]), axis=1)
        
        image_sp = [sp[:, (flashes_to_use & (image_ids==im)), :] for im in image_id_order]
        image_counts = [np.sum(flashes_to_use & (image_ids==im)) for im in image_id_order]
        nUnits = final_unit_filter.sum()
        image_sp = [s[:,:,:decodeWindows[-1]].reshape((nUnits,im_count,-1,decodeWindowBinSize)).sum(axis=-1) for s, im_count in zip(image_sp, image_counts)]

        y = np.array(y)
        # np.random.shuffle(y)
        decodeWindowSampleSize = 10 if region=='SC/MRN cluster 1' else 20
        for sampleSize in unitSampleSize:
            if nUnits < sampleSize:
                continue
            if sampleSize>1:
                if sampleSize==nUnits:
                    nSamples = 1
                    unitSamples = [np.arange(nUnits)]
                else:
                    # >99% chance each neuron is chosen at least once
                    nSamples = int(math.ceil(math.log(0.01)/math.log(1-sampleSize/nUnits)))
                    unitSamples = [np.random.choice(nUnits,sampleSize,replace=False) for _ in range(nSamples)]
            else:
                nSamples = nUnits
                unitSamples = [[i] for i in range(nUnits)]

            for winEnd in decodeWindows:
                # if region in ('VISall', 'Hipp') and sampleSize<50:
                #     continue
                winEnd = int(winEnd/decodeWindowBinSize)
                winStart = winEnd-int(decodeWindowSize/decodeWindowBinSize)
                
                for metric in d[region][sampleSize]:
                    if not metric == 'image_order':
                        d[region][sampleSize][metric].append([])
                for unitSamp in unitSamples:
                    X = np.concatenate([s[unitSamp,:,winStart:winEnd].transpose(1,0,2).reshape((im_count,-1)) for s, im_count in zip(image_sp, image_counts)])                       
                    cv = trainDecoder(model,X,y,nCrossVal)
                    d[region][sampleSize]['trainAccuracy'][-1].append(np.mean(cv['train_score']))
                    d[region][sampleSize]['featureWeights'][-1].append(np.mean(cv['coef'],axis=0).squeeze())
                    d[region][sampleSize]['accuracy'][-1].append(np.mean(cv['test_score']))
                    d[region][sampleSize]['prediction'][-1].append(cv['predict'])
                    d[region][sampleSize]['confidence'][-1].append(cv['decision_function'])
                    d[region][sampleSize]['balanced_accuracy'][-1].append(sklearn.metrics.balanced_accuracy_score(y, cv['predict']))
                    d[region][sampleSize]['unit_ids'][-1].append(units[final_unit_filter].iloc[unitSamp].index.values)
                    im_wise_recall = []
                    im_wise_precision = []
                    predictions = cv['predict']
                    for im in image_id_order:
                        tps = (y==im)&(predictions==im)
                        fps = (y!=im)&(predictions==im)
                        tns = (y!=im)&(predictions!=im)
                        fns = (y==im)&(predictions!=im)
                        
                        im_recall = np.sum(tps)/(np.sum(y==im))
                        im_wise_recall.append(im_recall)
                        im_precision = np.sum(tps)/(np.sum(tps)+np.sum(fps))
                        im_wise_precision.append(im_precision)
                    d[region][sampleSize]['imagewise_recall'][-1].append(im_wise_recall)
                    d[region][sampleSize]['imagewise_precision'][-1].append(im_wise_precision)

                for metric in d[region][sampleSize]:
                    if metric == 'prediction':
                        d[region][sampleSize][metric][-1] = scipy.stats.mode(d[region][sampleSize][metric][-1],axis=0)[0][0]
                    elif metric == 'image_order':
                        d[region][sampleSize][metric] = image_id_order
                    elif metric not in ('featureWeights', 'unit_ids'):
                        d[region][sampleSize][metric][-1] = np.median(d[region][sampleSize][metric][-1],axis=0)
    warnings.filterwarnings('default')
    # if not output_dir is None:
    #     np.save(os.path.join(output_dir,'unitChangeDecoding','unitChangeDecoding_'+str(sessionId)+'.npy'),d)
    return d


def sessionDecoding(sessionId, label, cluster, unitTable, unitData, stimTable, regions, unitSampleSize, decodeWindowEnd, 
                class_weight=None, rs=False, outputDir="/Volumes/programs/mindscope/workgroups/np-behavior/VBN_decoding_from_sensory_action_clusters",
                clustering='new'):
    '''
    sessionID: ecephys session id
    unitTable: df made from units_with_cortical_layers.csv
    uniData: tensor from 'vbnAllUnitSpikeTensor.hdf5'
    stimTable: master stim table
    '''
    base_sub = True
    base_sub_suffix = '_basesub' if base_sub else ''

    if cluster == 'sensory':
        if clustering == 'old':
            clusters = np.arange(6)
        else:
            clusters = [1,2,3,4,5]
    elif cluster == 'action':
        if clustering == 'old':
            clusters = [6,7,9,10,11,12]
        else:
            clusters = [6,7,8]
    elif cluster=='change':
        clusters = [6,]
    elif cluster =='transient':
        clusters = [1,]
    elif cluster == 'sustained':
        clusters = [0,]
    else:
        if not isinstance(cluster, list):
            clusters = [int(cluster),]
        else:
            clusters = cluster

    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=class_weight)
    nCrossVal = 5
    #unitSampleSize = [1,5,10,15,20,25,30,40,50,60]
    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
            
    imageName = np.array(stim['image_name'])
    if label == 'change':
        flashInd = (nonChangeFlashes & lick, changeFlashes & lick)
    elif label == 'change_no_lick_matching':
        flashInd = (changeFlashes, stim['is_prechange'].values)
    elif label == 'lick':
        flashInd = (nonChangeFlashes & ~lick, nonChangeFlashes & lick)
    elif label == 'hit':
        flashInd = (changeFlashes & ~lick, changeFlashes & lick)
    elif label == 'image':   
        flashInd = tuple(nonChangeFlashes & ~lick & (imageName==img) for img in np.unique(imageName) if img != 'omitted')
    elif label == 'visual_response':
        flashInd = (omittedFlashes & ~lick, nonChangeFlashes & ~lick)
    elif label == 'reaction_time':
        flashInd = tuple(nonChangeFlashes & stim['reaction_time'].notna().values & (stim['rt_quintiles']==rtq).values for rtq in np.arange(5))
    
    if any(flashes.sum() < 20 for flashes in flashInd):
        return

    # d = {region: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','prediction','confidence', 'balanced_accuracy',
    #                                                  'imagewise_recall', 'imagewise_precision')}
    #      for sampleSize in unitSampleSize} for region in regions}
    # d['decodeWindows'] = decodeWindows
    
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    qualityUnits = apply_unit_quality_filter(units)
    spikes = unitData[str(sessionId)]['spikes']

    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = qualityUnits if region=='all' else qualityUnits & getUnitsInRegion(units,region)
        unitsToUse = inRegion if cluster=='all' else inRegion & get_units_in_cluster(units, *clusters, clustering=clustering)
        nUnits = unitsToUse.sum()
        # print(f'num units: {nUnits}')
             
        sp = np.zeros((nUnits,spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(unitsToUse)[0]):
            sp[i] = spikes[u,:,:]
        
        total_flashes = np.sum([flashes.sum() for flashes in flashInd])
        flashSpikes = np.zeros((nUnits, total_flashes, len(decodeWindows)))
        print(f'flashSpikes: {flashSpikes.shape}')
        y = np.zeros(total_flashes)
        flash_index = 0
        for f,flashes in enumerate(flashInd):
            response = sp[:,flashes,:decodeWindows[-1]]
            if base_sub:
                baseline = np.mean(sp[:, np.where(flashes)[0]-1, -100:], axis=2)
                response = response - baseline[:,:,None]
            flashSpikes[:, flash_index:flash_index+flashes.sum(), :] = (response.reshape((nUnits,flashes.sum(),len(decodeWindows),decodeWindowSize)).sum(axis=-1))
            y[flash_index:flash_index+flashes.sum()] = f
            flash_index = flash_index + flashes.sum()

        y = y.astype(int)

        for sampleSize in unitSampleSize:
            if nUnits < sampleSize:
                continue
            if sampleSize>1:
                if sampleSize==nUnits:
                    nSamples = 1
                    unitSamples = [np.arange(nUnits)]
                else:
                    # >99% chance each neuron is chosen at least once
                    nSamples = int(math.ceil(math.log(0.01)/math.log(1-sampleSize/nUnits)))
                    unitSamples = [np.random.choice(nUnits,sampleSize,replace=False) for _ in range(nSamples)]
            else:
                nSamples = nUnits
                unitSamples = [[i] for i in range(nUnits)]
            
            accuracy = np.zeros((len(unitSamples),len(decodeWindows)))
            for j, winEnd in enumerate(decodeWindows):
                # if sampleSize!=decodeWindowSampleSize and winEnd!=decodeWindows[-1]:
                #     continue
                winEnd = int(winEnd/decodeWindowSize)
                for i, unitSamp in enumerate(unitSamples):
                    X = flashSpikes[unitSamp, :, :winEnd].transpose(1,0,2).reshape(len(y), -1)
                    cv = trainDecoder(model,X,y,nCrossVal)
                    predicted = cv['predict']
                    accuracy[i, j] = balanced_accuracy_score(y, predicted.astype(int))
                #     d[region][sampleSize]['trainAccuracy'][-1].append(np.mean(cv['train_score']))
                #     d[region][sampleSize]['featureWeights'][-1].append(np.mean(cv['coef'],axis=0).squeeze())
                #     d[region][sampleSize]['accuracy'][-1].append(np.mean(cv['test_score']))
                #     d[region][sampleSize]['prediction'][-1].append(cv['predict'])
                #     d[region][sampleSize]['confidence'][-1].append(cv['decision_function'])
                #     d[region][sampleSize]['balanced_accuracy'][-1].append(sklearn.metrics.balanced_accuracy_score(y, cv['predict']))

                # for metric in d[region][sampleSize]:
                #     if metric == 'prediction':
                #         d[region][sampleSize][metric][-1] = scipy.stats.mode(d[region][sampleSize][metric][-1],axis=0)[0][0]
                #     else:
                #         d[region][sampleSize][metric][-1] = np.median(d[region][sampleSize][metric][-1],axis=0)
            if label == 'change':
                dirName = 'sessionChangeDecoding'+ base_sub_suffix
            elif label == 'lick':
                dirName = 'sessionLickDecoding' + base_sub_suffix
            elif label == 'hit':
                dirName = 'sessionHitDecoding'+ base_sub_suffix
            elif label == 'image':
                dirName = 'sessionImageDecoding' + base_sub_suffix
            elif label == 'visual_response':
                dirName = 'sessionVisualResponseDecoding' + base_sub_suffix
            elif label == 'reaction_time':
                dirName = 'sessionReactionTimeDecoding' + base_sub_suffix
            elif label == 'change_no_lick_matching':
                dirName = 'sessionChangePrechangeDecoding' + base_sub_suffix

            savedir = os.path.join(outputDir, dirName)
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            
            np.save(os.path.join(savedir, dirName.split('_')[0]+'_'+str(sessionId)+'_'+region+'_'+cluster+'_'+str(unitSampleSize)+'.npy'),accuracy)

    warnings.filterwarnings('default')
    # if not output_dir is None:
    #     np.save(os.path.join(output_dir,'unitChangeDecoding','unitChangeDecoding_'+str(sessionId)+'.npy'),d)


def pooledDecoding(label, region, cluster, unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', clustering='new', ):
    stimTableFile = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
    stimTable = pd.read_csv(stimTableFile)
    baseDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables')
    outputDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/VBN_decoding_from_sensory_action_clusters/with_unitsamp_replacement_and_same_splits_for_sessionunits')
    unitTable = pd.read_csv(os.path.join(baseDir,'master_units_with_responsiveness.csv'))
    if condition == 'active':
        unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')
    elif condition == 'passive':
        unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor_passive.hdf5'),mode='r')


    base_sub = True
    base_sub_suffix = '_basesub' if base_sub else ''
    condition_suffix = '_' + condition

    if cluster == 'sensory':
        if clustering == 'old':
            clusters = np.arange(6)
        else:
            clusters = [1,2,3,4,5]
    elif cluster == 'action':
        if clustering == 'old':
            clusters = [6,7,9,10,11,12]
        else:
            clusters = [6,7,8]
    elif cluster=='change':
        clusters = [6,]
    elif cluster =='transient':
        clusters = [1,]
    elif cluster == 'sustained':
        clusters = [0,]
    else:
        if not isinstance(cluster, list):
            clusters = [int(cluster),]
        else:
            clusters = cluster

    print(f'cluster: {clusters}')

    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)
    minFlashes = 20
    #unitSampleSize = 50
    #nUnitSamples = 100
    #nPseudoFlashes = 100

    flashSpikes = [] # binned spikes for all flashes of each label for each session with neurons in region and cluster
    unitIndex = [] # nUnits x (session, unit in session)
    unitIDs = [] # unit ids corresponding to unitIndex
    sessionIndex = 0
    for sessionId in stimTable['session_id'].unique():
        stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
        # if stim.iloc[0]['experience_level']=='Novel':
        #     continue
        flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
        
        imageName = np.array(stim['image_name'])
        if label == 'change':
            flashInd = (nonChangeFlashes & lick, changeFlashes & lick)
        elif label == 'change_no_lick_matching':
            flashInd = (changeFlashes, stim['is_prechange'].values)
        elif label == 'lick':
            flashInd = (nonChangeFlashes & ~lick, nonChangeFlashes & lick)
        elif label == 'lick_imagematched':
            flashInd = get_imagematched_lick_nolicks(stim)
        elif label == 'hit':
            flashInd = (changeFlashes & ~lick, changeFlashes & lick)
        elif label == 'image':   
            flashInd = tuple(nonChangeFlashes & ~lick & (imageName==img) for img in np.unique(imageName) if img != 'omitted')
        elif label == 'visual_response':
            flashInd = (omittedFlashes & ~lick, nonChangeFlashes & ~lick)
        elif label == 'reaction_time':
            flashInd = tuple(nonChangeFlashes & stim['reaction_time'].notna().values & (stim['rt_quintiles']==rtq).values for rtq in np.arange(5))
        elif label == 'flashes_since_lick':
            flashInd = tuple(stim['engaged'] &
                                (~stim['is_change']) & 
                                (~stim['omitted']) & 
                                (~stim['previous_omitted']) & 
                                (stim['flashes_since_change']>5) & 
                                (~stim['lickbout_for_flash_during_response_window']) &
                                (stim['flashes_since_last_lick']==fsl).values for fsl in np.arange(1,10))
        elif label == 'change_eligible':
            flashInd = (stim['engaged'] &
                            (~stim['is_change']) & 
                            (~stim['omitted']) & 
                            (~stim['previous_omitted']) & 
                            (stim['flashes_since_change']>5) & 
                            (~stim['lickbout_for_flash_during_response_window']) &
                            (stim['flashes_since_last_lick'].isin((5,6,7)).values), 
                            stim['engaged'] &
                            (~stim['is_change']) & 
                            (~stim['omitted']) & 
                            (~stim['previous_omitted']) & 
                            (stim['flashes_since_change']>5) & 
                            (~stim['lickbout_for_flash_during_response_window']) &
                            (stim['flashes_since_last_lick'].isin((2,3,4)).values))
        
        # passflash = any(flashes.sum() < minFlashes for flashes in flashInd)
        # print(f'pass flash num: {passflash}')
        if any(flashes.sum() < minFlashes for flashes in flashInd):
            continue

        units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
        qualityUnits = apply_unit_quality_filter(units)
        inRegion = qualityUnits if region=='all' else qualityUnits & getUnitsInRegion(units,region)
        inCluster = get_units_in_cluster(units, *clusters, clustering=clustering)
        unitsToUse = inRegion if cluster=='all' else inRegion & inCluster
        nUnits = unitsToUse.sum()
        # print(f'num units: {nUnits}')
        if nUnits < 1:
            continue
        spikes = unitData[str(sessionId)]['spikes']     
        sp = np.zeros((nUnits,spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(unitsToUse)[0]):
            sp[i] = spikes[u,:,:]
            # unitIDs.append(units.iloc[u]['unit_id'])
        for f,flashes in enumerate(flashInd):
            if sessionIndex == 0:
                flashSpikes.append([])
            response = sp[:,flashes,:decodeWindows[-1]]
            if base_sub:
                baseline = np.mean(sp[:, np.where(flashes)[0]-1, -100:], axis=2)
                response = response - baseline[:,:,None]
            flashSpikes[f].append(response.reshape((nUnits,flashes.sum(),len(decodeWindows),decodeWindowSize)).sum(axis=-1))
        
        unitIndex.append(np.stack((np.full(nUnits,sessionIndex),np.arange(nUnits))).T)
        sessionIndex += 1
    if sum(len(i) for i in unitIndex) < unitSampleSize:
        return
    unitIndex = np.concatenate(unitIndex)

    unitSamples = [np.random.choice(unitIndex.shape[0],unitSampleSize,replace=True) for _ in range(nUnitSamples)] #changed to True 10/15/25 to test
    # unitSampleIDs = [[unitIDs[i] for i in samp] for samp in unitSamples]
    y = np.repeat(np.arange(len(flashInd)),nPseudoFlashes)
    accuracy = np.zeros((nUnitSamples,len(decodeWindows)))
    featureWeights = []
    warnings.filterwarnings('ignore')
    for i,unitSamp in enumerate(unitSamples):
        pseudoTrain = [np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))]
        pseudoTest = [np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))]
        for f,flashes in enumerate(flashInd):
            for k,(s,u) in enumerate(unitIndex[unitSamp]):
                np.random.seed(i * s + 1) #set seed so that units from same session have same train/test split, but different across different unit samples 10/15/25
                n = flashSpikes[f][s].shape[1]
                r = np.random.permutation(n)
                train = r[n//2:]
                test = r[:n//2]
                pseudoTrain[f][k] = flashSpikes[f][s][u,np.random.choice(train,nPseudoFlashes,replace=True)] 
                pseudoTest[f][k] = flashSpikes[f][s][u,np.random.choice(test,nPseudoFlashes,replace=True)] 
        pseudoTrain = np.concatenate(pseudoTrain,axis=1)
        pseudoTest = np.concatenate(pseudoTest,axis=1)
        for j,winEnd in enumerate((decodeWindows/decodeWindowSize).astype(int)):
            Xtrain = pseudoTrain[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
            Xtest = pseudoTest[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
            decoder = LinearSVC(C=1.0,max_iter=int(1e4), class_weight='balanced')
            decoder.fit(Xtrain,y)
            accuracy[i,j] = decoder.score(Xtest,y)
            if winEnd==100: #at end of decision window, save feature weights
                featureWeights.append(decoder.coef_)

    warnings.filterwarnings('default')

    if label == 'change':
        dirName = 'pooledChangeDecoding'+ base_sub_suffix + condition_suffix
    elif label == 'lick':
        dirName = 'pooledLickDecoding' + base_sub_suffix + condition_suffix
    elif label == 'hit':
        dirName = 'pooledHitDecoding'+ base_sub_suffix + condition_suffix
    elif label == 'image':
        dirName = 'pooledImageDecoding' + base_sub_suffix + condition_suffix
    elif label == 'visual_response':
        dirName = 'pooledVisualResponseDecoding' + base_sub_suffix + condition_suffix
    elif label == 'reaction_time':
        dirName = 'pooledReactionTimeDecoding' + base_sub_suffix + condition_suffix
    elif label == 'change_no_lick_matching':
        dirName = 'pooledChangePrechangeDecoding' + base_sub_suffix + condition_suffix
    elif label == 'flashes_since_lick':
        dirName = 'pooledFlashesSinceLick' + base_sub_suffix + condition_suffix
    elif label == 'change_eligible':
        dirName = 'pooledChangeEligible' + base_sub_suffix + condition_suffix
    elif label == 'lick_imagematched':
        dirName = 'pooledLickImageMatched' + base_sub_suffix + condition_suffix

    savedir = os.path.join(outputDir, dirName)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    np.save(os.path.join(savedir,dirName.split('_')[0]+'_'+region+'_'+cluster+'_'+str(unitSampleSize)+'_'+str(nPseudoFlashes)+'_'+str(nUnitSamples)+'_'+ str(decodeWindowSize)+'binsize.npy'),accuracy)
    # np.save(os.path.join(savedir,dirName.split('_')[0]+'_'+region+'_'+cluster+'_'+str(unitSampleSize)+'_'+str(nPseudoFlashes)+'_'+str(nUnitSamples)+'_'+ str(decodeWindowSize)+'binsize_unitIDs.npy'),unitSampleIDs)


#outputDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/VBN_decoding_from_sensory_action_clusters_regionmatched')

def pooledDecoding_match_areas(label, region1, region2, cluster, unitSampleSize, nPseudoFlashes, nUnitSamples):
    stimTableFile = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
    stimTable = pd.read_csv(stimTableFile)
    baseDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables')
    outputDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/VBN_decoding_from_sensory_action_clusters_regionmatched/unit_resamp_with_replacement_and_same_session_same_splits')
    unitTable = pd.read_csv(os.path.join(baseDir,'master_units_with_responsiveness.csv'))
    unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')
    clusterTable = pd.read_csv(os.path.join(baseDir,'unit_cluster_labels.csv'))
    if cluster == 'sensory':
        clusters = np.arange(6)
    elif cluster == 'action':
        clusters = [6,7,9,10,11,12]
    elif cluster=='change':
        clusters = [6,]

    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)
    minFlashes = 20
    #unitSampleSize = 50
    #nUnitSamples = 100
    #nPseudoFlashes = 100

    flashSpikes = {r:[] for r in [region1, region2]} # binned spikes for all flashes of each label for each session with neurons in region and cluster
    unitIndex = {r:[] for r in [region1, region2]} # nUnits x (session, unit in session)
    sessionIndex = 0
    for sessionId in stimTable['session_id'].unique():
        stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
        # if stim.iloc[0]['experience_level']=='Novel':
        #     continue
        flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
        
        imageName = np.array(stim['image_name'])
        if label == 'change':
            flashInd = (nonChangeFlashes & lick, changeFlashes & lick)
        elif label == 'change_no_lick_matching':
            flashInd = (changeFlashes, stim['is_prechange'].values)
        elif label == 'lick':
            flashInd = (nonChangeFlashes & ~lick, nonChangeFlashes & lick)
        elif label == 'hit':
            flashInd = (changeFlashes & ~lick, changeFlashes & lick)
        elif label == 'image':   
            flashInd = tuple(nonChangeFlashes & ~lick & (imageName==img) for img in np.unique(imageName) if img != 'omitted')
        elif label == 'visual_response':
            flashInd = (omittedFlashes & ~lick, nonChangeFlashes & ~lick)
        elif label == 'reaction_time':
            flashInd = tuple(nonChangeFlashes & stim['reaction_time'].notna().values & (stim['rt_quintiles']==rtq).values for rtq in np.arange(5))
        
        # passflash = any(flashes.sum() < minFlashes for flashes in flashInd)
        # print(f'pass flash num: {passflash}')
        if any(flashes.sum() < minFlashes for flashes in flashInd):
            continue

        units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
        qualityUnits = apply_unit_quality_filter(units)
        unitsToUse = {}
        nUnits_per_region = []
        for ir, region in enumerate([region1, region2]):
            inRegion = qualityUnits if region=='all' else qualityUnits & getUnitsInRegion(units,region)
            unitsToUse[region] = inRegion if cluster=='all' else inRegion & get_units_in_cluster(units, *clusters, clustering='new')
            nUnits_per_region.append(unitsToUse[region].sum())
        
        # print(f'num units: {nUnits}')
        if np.min(nUnits_per_region)<1:
            continue

        nUnits = np.min(nUnits_per_region)

        spikes = unitData[str(sessionId)]['spikes']
        for region in [region1, region2]:
            sp = np.zeros((nUnits,spikes.shape[1],spikes.shape[2]),dtype=bool)
            uindices = np.random.choice(np.where(unitsToUse[region])[0], nUnits, replace=False)
            for i,u in enumerate(uindices):
                sp[i] = spikes[u,:,:]
            
            for f,flashes in enumerate(flashInd):
                if sessionIndex == 0:
                    flashSpikes[region].append([])
                flashSpikes[region][f].append(sp[:,flashes,:decodeWindows[-1]].reshape((nUnits,flashes.sum(),len(decodeWindows),decodeWindowSize)).sum(axis=-1))
            
            unitIndex[region].append(np.stack((np.full(nUnits,sessionIndex),np.arange(nUnits))).T)
        sessionIndex += 1
        print(f'Session {sessionId} {sessionIndex}')

    if sum(len(i) for i in unitIndex[region1]) < unitSampleSize:
        return

    for region in [region1, region2]:
        unitIndex[region] = np.concatenate(unitIndex[region])


    unitSamples = [np.random.choice(unitIndex[region1].shape[0],unitSampleSize,replace=True) for _ in range(nUnitSamples)]
    y = np.repeat(np.arange(len(flashInd)),nPseudoFlashes)
    accuracy = {r: np.zeros((nUnitSamples,len(decodeWindows))) for r in [region1, region2]}
    warnings.filterwarnings('ignore')
    for i,unitSamp in enumerate(unitSamples):
        pseudoTrain = {r: [np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))] for r in [region1, region2]}
        pseudoTest = {r:[np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))] for r in [region1, region2]}
        for f,flashes in enumerate(flashInd):
            for k,uind in enumerate(unitSamp):
                s = unitIndex[region1][uind][0]
                np.random.seed(i * s + 1) #set seed so that units from same session have same train/test split, but different across different unit samples 10/15/25
                n = flashSpikes[region1][f][s].shape[1]
                r = np.random.permutation(n)
                train = r[n//2:]
                test = r[:n//2]
                sess_train = np.random.choice(train,nPseudoFlashes,replace=True)
                sess_test = np.random.choice(test,nPseudoFlashes,replace=True)
                for region in [region1, region2]:
                    pseudoTrain[region][f][k] = flashSpikes[region][f][s][unitIndex[region][uind][1], sess_train]
                    pseudoTest[region][f][k] = flashSpikes[region][f][s][unitIndex[region][uind][1], sess_test]
        for region in [region1, region2]:
            pseudoTrainr = np.concatenate(pseudoTrain[region],axis=1)
            pseudoTestr = np.concatenate(pseudoTest[region],axis=1)
            for j,winEnd in enumerate((decodeWindows/decodeWindowSize).astype(int)):
                Xtrain = pseudoTrainr[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
                Xtest = pseudoTestr[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
                decoder = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
                decoder.fit(Xtrain,y)
                accuracy[region][i,j] = decoder.score(Xtest,y) 
    warnings.filterwarnings('default')

    if label == 'change':
        dirName = 'pooledChangeDecoding'
    elif label == 'lick':
        dirName = 'pooledLickDecoding'
    elif label == 'hit':
        dirName = 'pooledHitDecoding'
    elif label == 'image':
        dirName = 'pooledImageDecoding'
    elif label == 'visual_response':
        dirName = 'pooledVisualResponseDecoding'
    elif label == 'reaction_time':
        dirName = 'pooledReactionTimeDecoding'
    elif label == 'change_no_lick_matching':
        dirName = 'pooledChangePrechangeDecoding'

    savedir = os.path.join(outputDir, dirName)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for region in [region1, region2]:
        otherregion = [r for r in [region1, region2] if not r==region]
        np.save(os.path.join(savedir,dirName+'_'+region+'_'+cluster+'_'+str(unitSampleSize)+'_'+str(nPseudoFlashes)+'_'+str(nUnitSamples)+f'_match_to_{otherregion}.npy'),accuracy)


def pooledDecoding_match_areas_hierarchical_bootstrap(label, region1, region2, cluster, unitSampleSize, nPseudoFlashes, nUnitSamples):
    stimTableFile = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
    stimTable = pd.read_csv(stimTableFile)
    baseDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables')
    outputDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/VBN_decoding_from_sensory_action_clusters_regionmatched_hierarchical_bootstrap')
    unitTable = pd.read_csv(os.path.join(baseDir,'master_units_with_responsiveness.csv'))
    unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')
    clusterTable = pd.read_csv(os.path.join(baseDir,'unit_cluster_labels.csv'))
    if cluster == 'sensory':
        clusters = np.arange(6)
    elif cluster == 'action':
        clusters = [6,7,9,10,11,12]
    elif cluster=='change':
        clusters = [6,]
    elif cluster =='transient':
        clusters = [1,]

    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)
    minFlashes = 20
    #unitSampleSize = 50
    #nUnitSamples = 100
    #nPseudoFlashes = 100

    flashSpikes = {r:[] for r in [region1, region2]} # binned spikes for all flashes of each label for each session with neurons in region and cluster
    unitIndex = {r:[] for r in [region1, region2]} # nUnits x (session, unit in session)
    sessionIndex = 0
    sessionIds = []
    for sessionId in stimTable['session_id'].unique():
        stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
        # if stim.iloc[0]['experience_level']=='Novel':
        #     continue
        flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
        
        imageName = np.array(stim['image_name'])
        if label == 'change':
            flashInd = (nonChangeFlashes & lick, changeFlashes & lick)
        elif label == 'change_no_lick_matching':
            flashInd = (changeFlashes, stim['is_prechange'].values)
        elif label == 'lick':
            flashInd = (nonChangeFlashes & ~lick, nonChangeFlashes & lick)
        elif label == 'hit':
            flashInd = (changeFlashes & ~lick, changeFlashes & lick)
        elif label == 'image':   
            flashInd = tuple(nonChangeFlashes & ~lick & (imageName==img) for img in np.unique(imageName) if img != 'omitted')
        elif label == 'visual_response':
            flashInd = (omittedFlashes & ~lick, nonChangeFlashes & ~lick)
        elif label == 'reaction_time':
            flashInd = tuple(nonChangeFlashes & stim['reaction_time'].notna().values & (stim['rt_quintiles']==rtq).values for rtq in np.arange(5))
        
        # passflash = any(flashes.sum() < minFlashes for flashes in flashInd)
        # print(f'pass flash num: {passflash}')
        if any(flashes.sum() < minFlashes for flashes in flashInd):
            continue

        units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
        qualityUnits = apply_unit_quality_filter(units)
        unitsToUse = {}
        nUnits_per_region = []
        for ir, region in enumerate([region1, region2]):
            inRegion = qualityUnits if region=='all' else qualityUnits & getUnitsInRegion(units,region)
            unitsToUse[region] = inRegion if cluster=='all' else inRegion & get_units_in_cluster(units, *clusters, clustering='new')
            nUnits_per_region.append(unitsToUse[region].sum())
        
        
        if np.min(nUnits_per_region)<1:
            continue

        nUnits = np.min(nUnits_per_region)
        print(f'num units: {nUnits}')
        
        spikes = unitData[str(sessionId)]['spikes']
        for region in [region1, region2]:
            sp = np.zeros((nUnits,spikes.shape[1],spikes.shape[2]),dtype=bool)
            uindices = np.random.choice(np.where(unitsToUse[region])[0], nUnits, replace=False)
            for i,u in enumerate(uindices):
                sp[i] = spikes[u,:,:]
            
            for f,flashes in enumerate(flashInd):
                if sessionIndex == 0:
                    flashSpikes[region].append([])
                flashSpikes[region][f].append(sp[:,flashes,:decodeWindows[-1]].reshape((nUnits,flashes.sum(),len(decodeWindows),decodeWindowSize)).sum(axis=-1))
            
            unitIndex[region].append(np.stack((np.full(nUnits,sessionIndex),np.arange(nUnits))).T)
        sessionIndex += 1
        sessionIds.append(sessionId)
        print(f'Session {sessionId} {sessionIndex}')

    if sum(len(i) for i in unitIndex[region1]) < unitSampleSize:
        return

    for session_iteration in range(100):
        chosen_sessions = np.random.choice(np.arange(sessionIndex), sessionIndex, replace=True)
        iteration_unit_index = {r:[] for r in [region1, region2]}
        for region in [region1, region2]:
            iteration_unit_index[region] = np.concatenate([unitIndex[region][c] for c in chosen_sessions])

        unitSamples = [np.random.choice(iteration_unit_index[region1].shape[0],unitSampleSize,replace=True) for _ in range(nUnitSamples)]
        y = np.repeat(np.arange(len(flashInd)),nPseudoFlashes)
        accuracy = {r: np.zeros((nUnitSamples,len(decodeWindows))) for r in [region1, region2]}
        warnings.filterwarnings('ignore')
        for i,unitSamp in enumerate(unitSamples):
            pseudoTrain = {r: [np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))] for r in [region1, region2]}
            pseudoTest = {r:[np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))] for r in [region1, region2]}
            for f,flashes in enumerate(flashInd):
                for k,uind in enumerate(unitSamp):
                    s = iteration_unit_index[region1][uind][0]
                    n = flashSpikes[region1][f][s].shape[1]
                    r = np.random.permutation(n)
                    train = r[n//2:]
                    test = r[:n//2]
                    sess_train = np.random.choice(train,nPseudoFlashes,replace=True)
                    sess_test = np.random.choice(test,nPseudoFlashes,replace=True)
                    for region in [region1, region2]:
                        pseudoTrain[region][f][k] = flashSpikes[region][f][s][iteration_unit_index[region][uind][1], sess_train]
                        pseudoTest[region][f][k] = flashSpikes[region][f][s][iteration_unit_index[region][uind][1], sess_test]
            for region in [region1, region2]:
                pseudoTrainr = np.concatenate(pseudoTrain[region],axis=1)
                pseudoTestr = np.concatenate(pseudoTest[region],axis=1)
                for j,winEnd in enumerate((decodeWindows/decodeWindowSize).astype(int)):
                    Xtrain = pseudoTrainr[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
                    Xtest = pseudoTestr[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
                    decoder = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
                    decoder.fit(Xtrain,y)
                    accuracy[region][i,j] = decoder.score(Xtest,y) 
        warnings.filterwarnings('default')

        if label == 'change':
            dirName = 'pooledChangeDecoding'
        elif label == 'lick':
            dirName = 'pooledLickDecoding'
        elif label == 'hit':
            dirName = 'pooledHitDecoding'
        elif label == 'image':
            dirName = 'pooledImageDecoding'
        elif label == 'visual_response':
            dirName = 'pooledVisualResponseDecoding'
        elif label == 'reaction_time':
            dirName = 'pooledReactionTimeDecoding'
        elif label == 'change_no_lick_matching':
            dirName = 'pooledChangePrechangeDecoding'

        savedir = os.path.join(outputDir, dirName)
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        for region in [region1, region2]:
            otherregion = [r for r in [region1, region2] if not r==region]
            np.save(os.path.join(savedir,dirName+'_'+region+'_'+cluster+'_'+str(unitSampleSize)+'_'+str(nPseudoFlashes)+'_'+str(nUnitSamples)+f'_match_to_{otherregion}_hierachicalbootstrap_{session_iteration}.npy'),accuracy)
            np.save(os.path.join(savedir,dirName+'_'+region+'_'+cluster+'_'+str(unitSampleSize)+'_'+str(nPseudoFlashes)+'_'+str(nUnitSamples)+f'_match_to_{otherregion}_hierachicalbootstrapUnitSamples_{session_iteration}.npy'),unitSamples)


def pooledDecoding_unit_subsets(label, unit_ids, unit_subset_name, unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', experience='all', use_max_sample_size_available=False, outputDir=None):
    stimTableFile = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
    stimTable = pd.read_csv(stimTableFile)
    stimTable = stimTable[stimTable['no_abnorm']]
    baseDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables')
    
    if outputDir is None:
        outputDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/VBN_revision_decoding_dropouts')
    
    # unitTable = pd.read_csv(os.path.join(baseDir,'master_units_with_responsiveness.csv'))
    if condition == 'active':
        unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')
    elif condition == 'passive':
        unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor_passive.hdf5'),mode='r')

    base_sub = True
    base_sub_suffix = '_basesub' if base_sub else ''
    condition_suffix = '_' + condition

    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)
    minFlashes = 20
    
    if experience == 'Novel':
        stimTable = stimTable[stimTable['experience_level']=='Novel']
    elif experience == 'Familiar':
        stimTable = stimTable[stimTable['experience_level']=='Familiar']
    elif experience == 'all':
        pass
    else:
        raise ValueError('experience must be Novel, Familiar, or all')

    flashSpikes = [] # binned spikes for all flashes of each label for each session with neurons in region and cluster
    unitIndex = [] # nUnits x (session, unit in session)
    sessionIndex = 0
    for sessionId in stimTable['session_id'].unique():
        stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
        # if stim.iloc[0]['experience_level']=='Novel':
        #     continue
        flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
        
        imageName = np.array(stim['image_name'])
        if label == 'change':
            flashInd = (nonChangeFlashes & lick, changeFlashes & lick)
        elif label == 'change_no_lick_matching':
            flashInd = (changeFlashes, stim['is_prechange'].values)
        elif label == 'lick':
            flashInd = (nonChangeFlashes & ~lick, nonChangeFlashes & lick)
        elif label == 'lick_imagematched':
            flashInd = get_imagematched_lick_nolicks(stim)
        elif label == 'hit':
            flashInd = (changeFlashes & ~lick, changeFlashes & lick)
        elif label == 'image':
            if experience=='all':   
                flashInd = tuple(nonChangeFlashes & ~lick & (imageName==img) for img in np.unique(imageName) if img != 'omitted')
            else:
                flashInd = tuple(nonChangeFlashes & ~lick & (imageName==img) for img in np.unique(imageName) if (img != 'omitted') and (img not in ['im083_r', 'im111_r']))
        elif label == 'visual_response':
            flashInd = (omittedFlashes & ~lick, nonChangeFlashes & ~lick)
        elif label == 'reaction_time':
            flashInd = tuple(nonChangeFlashes & stim['reaction_time'].notna().values & (stim['rt_quintiles']==rtq).values for rtq in np.arange(5))
        elif label == 'flashes_since_lick':
            flashInd = tuple(stim['engaged'] &
                                (~stim['is_change']) & 
                                (~stim['omitted']) & 
                                (~stim['previous_omitted']) & 
                                (stim['flashes_since_change']>5) & 
                                (~stim['lickbout_for_flash_during_response_window']) &
                                (stim['flashes_since_last_lick']==fsl).values for fsl in np.arange(1,10))
        elif label == 'change_eligible':
            flashInd = (stim['engaged'] &
                            (~stim['is_change']) & 
                            (~stim['omitted']) & 
                            (~stim['previous_omitted']) & 
                            (stim['flashes_since_change']>5) & 
                            (~stim['lickbout_for_flash_during_response_window']) &
                            (stim['flashes_since_last_lick'].isin((5,6,7)).values), 
                            stim['engaged'] &
                            (~stim['is_change']) & 
                            (~stim['omitted']) & 
                            (~stim['previous_omitted']) & 
                            (stim['flashes_since_change']>5) & 
                            (~stim['lickbout_for_flash_during_response_window']) &
                            (stim['flashes_since_last_lick'].isin((2,3,4)).values))
        
        # passflash = any(flashes.sum() < minFlashes for flashes in flashInd)
        # print(f'pass flash num: {passflash}')
        if any(flashes.sum() < minFlashes for flashes in flashInd):
            continue

        session_unit_ids = unitData[str(sessionId)]['unitIds'][:]
        unitsToUse = np.isin(session_unit_ids, unit_ids)
        
        nUnits = unitsToUse.sum()
        # print(f'num units: {nUnits}')
        if nUnits < 1:
            continue
        spikes = unitData[str(sessionId)]['spikes']     
        sp = np.zeros((nUnits,spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(unitsToUse)[0]):
            sp[i] = spikes[u,:,:]
        
        for f,flashes in enumerate(flashInd):
            if sessionIndex == 0:
                flashSpikes.append([])
            response = sp[:,flashes,:decodeWindows[-1]]
            if base_sub:
                baseline = np.mean(sp[:, np.where(flashes)[0]-1, -100:], axis=2)
                response = response - baseline[:,:,None]
            flashSpikes[f].append(response.reshape((nUnits,flashes.sum(),len(decodeWindows),decodeWindowSize)).sum(axis=-1))
        
        unitIndex.append(np.stack((np.full(nUnits,sessionIndex),np.arange(nUnits))).T)
        sessionIndex += 1
    
    #Make sure there are enough units to sample, if use_max_sample_size_available is True, then run with as many units as are available
    if sum(len(i) for i in unitIndex) < unitSampleSize:
        if use_max_sample_size_available:
            if sum(len(i) for i in unitIndex) < 1:
                return
            unitSampleSize = sum(len(i) for i in unitIndex)
        else:
            return
        
    unitIndex = np.concatenate(unitIndex)

    unitSamples = [np.random.choice(unitIndex.shape[0],unitSampleSize,replace=True) for _ in range(nUnitSamples)] #changed to True 10/15/25 to test
    y = np.repeat(np.arange(len(flashInd)),nPseudoFlashes)
    accuracy = np.zeros((nUnitSamples,len(decodeWindows)))
    warnings.filterwarnings('ignore')
    for i,unitSamp in enumerate(unitSamples):
        pseudoTrain = [np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))]
        pseudoTest = [np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))]
        for f,flashes in enumerate(flashInd):
            for k,(s,u) in enumerate(unitIndex[unitSamp]):
                np.random.seed(i * s + 1) #set seed so that units from same session have same train/test split, but different across different unit samples 10/15/25
                n = flashSpikes[f][s].shape[1]
                r = np.random.permutation(n)
                train = r[n//2:]
                test = r[:n//2]
                pseudoTrain[f][k] = flashSpikes[f][s][u,np.random.choice(train,nPseudoFlashes,replace=True)] 
                pseudoTest[f][k] = flashSpikes[f][s][u,np.random.choice(test,nPseudoFlashes,replace=True)] 
        pseudoTrain = np.concatenate(pseudoTrain,axis=1)
        pseudoTest = np.concatenate(pseudoTest,axis=1)
        for j,winEnd in enumerate((decodeWindows/decodeWindowSize).astype(int)):
            Xtrain = pseudoTrain[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
            Xtest = pseudoTest[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
            decoder = LinearSVC(C=1.0,max_iter=int(1e4), class_weight='balanced')
            decoder.fit(Xtrain,y)
            accuracy[i,j] = decoder.score(Xtest,y) 
    warnings.filterwarnings('default')

    if label == 'change':
        dirName = 'pooledChangeDecoding'+ base_sub_suffix + condition_suffix
    elif label == 'lick':
        dirName = 'pooledLickDecoding' + base_sub_suffix + condition_suffix
    elif label == 'hit':
        dirName = 'pooledHitDecoding'+ base_sub_suffix + condition_suffix
    elif label == 'image':
        dirName = 'pooledImageDecoding' + base_sub_suffix + condition_suffix
    elif label == 'visual_response':
        dirName = 'pooledVisualResponseDecoding' + base_sub_suffix + condition_suffix
    elif label == 'reaction_time':
        dirName = 'pooledReactionTimeDecoding' + base_sub_suffix + condition_suffix
    elif label == 'change_no_lick_matching':
        dirName = 'pooledChangePrechangeDecoding' + base_sub_suffix + condition_suffix
    elif label == 'flashes_since_lick':
        dirName = 'pooledFlashesSinceLick' + base_sub_suffix + condition_suffix
    elif label == 'change_eligible':
        dirName = 'pooledChangeEligible' + base_sub_suffix + condition_suffix
    elif label == 'lick_imagematched':
        dirName = 'pooledLickImageMatched' + base_sub_suffix + condition_suffix

    savedir = os.path.join(outputDir, dirName)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    np.save(os.path.join(savedir,dirName.split('_')[0]+'_'+ unit_subset_name +'_'+str(unitSampleSize)+'_'+
                         str(nPseudoFlashes)+'_'+str(nUnitSamples)+'_' + str(decodeWindowSize)+'binsize.npy'),accuracy)


def findNearest(array,values):
    ind = np.searchsorted(array,values,side='left')
    for i,j in enumerate(ind):
        if j > 0 and (j == len(array) or math.fabs(values[i] - array[j-1]) < math.fabs(values[i] - array[j])):
            ind[i] = j-1
    return ind

def decodeFromFacemap(sessionId, basesub=True):
    baseDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables')
    outputDir = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/VBN_video_analysis')
    stimTableFile = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
    stimTable = pd.read_csv(stimTableFile)
    videoTable = pd.read_excel(os.path.join(baseDir,'vbn_video_paths_full_validation.xlsx'))
    videoTable.insert(0,'session_id',[int(s[:s.find('_')]) for s in videoTable['exp_id']])
    
    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
    nCrossVal = 5
    decodeWindowStart = 0
    decodeWindowEnd = 0.75
    frameInterval = 1/60
    decodeWindows = np.arange(decodeWindowStart,decodeWindowEnd+frameInterval/2,frameInterval)
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    
    flashSvd = []
    videoIndex = np.where(videoTable['session_id'] == sessionId)[0][0]
    for videoType in ('side',): # ('side','face')
        videoPath = videoTable.loc[videoIndex,videoType+'_video'].replace('\\','/')
        videoName = os.path.basename(videoPath)
        facemapDataPath = os.path.join(outputDir,'facemapData',videoName[:-4]+'_proc.hdf5')
        with h5py.File(facemapDataPath) as facemapData:
            svd = facemapData['motSVD'][()]
        frameTimesPath = videoTable.loc[videoIndex,videoType+'_timestamp_path'].replace('\\','/')
        frameTimes = np.load(frameTimesPath)
        flashSvd.append([])
        for flashTime in flashTimes:
            frameIndex = findNearest(frameTimes,decodeWindows+flashTime)
            svd_flash = svd[frameIndex]
            if basesub:
                baselineIndex = findNearest(frameTimes, np.arange(-0.1, 0, frameInterval) + flashTime)
                baseline_svd = svd[baselineIndex]
                svd_flash = svd_flash - np.mean(baseline_svd, axis=0)[None, :]
            flashSvd[-1].append(svd_flash)
    flashSvd = np.concatenate(flashSvd,axis=2)
    
    decoderLabels = ('non-change lick','change lick novel','change no lick novel','non-change lick novel','non-change no lick novel')
    d = {metric: {lbl: [] for lbl in decoderLabels} for metric in ('trainAccuracy','featureWeights','accuracy','balancedAccuracy','prediction','confidence')}
    d['decodeWindows'] = decodeWindows
    d['changeFlashes'] = changeFlashes
    d['nonChangeFlashes'] = nonChangeFlashes
    d['novelFlashes'] = novelFlashes
    d['lick'] = lick
    
    for lbl,flashes,ind in zip(decoderLabels,
                               (nonChangeFlashes,changeFlashes & lick,changeFlashes & ~lick,nonChangeFlashes & lick,nonChangeFlashes & ~lick),
                               (lick,novelFlashes,novelFlashes,novelFlashes,novelFlashes)):
        y = ind[flashes]
        if np.sum(y) >= 10 and np.sum(~y) >= 10:
            warnings.filterwarnings('ignore')
            for i in range(len(decodeWindows)):
                X = flashSvd[flashes,:i+1].reshape(flashes.sum(),-1)   
                cv = trainDecoder(model,X,y,nCrossVal)
                d['trainAccuracy'][lbl].append(np.mean(cv['train_score']))
                d['featureWeights'][lbl].append(np.mean(cv['coef'],axis=0).squeeze())
                d['accuracy'][lbl].append(np.mean(cv['test_score']))
                d['balancedAccuracy'][lbl].append(sklearn.metrics.balanced_accuracy_score(y.astype(bool),cv['predict'].astype(bool)))
                d['prediction'][lbl].append(cv['predict'])
                d['confidence'][lbl].append(cv['decision_function'])
            warnings.filterwarnings('default')
    
    np.save(os.path.join(outputDir,'facemapDecoding_basesub','facemapDecoding_'+str(sessionId)+'.npy'),d)




