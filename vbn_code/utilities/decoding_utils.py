import math
import os
import warnings
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import sklearn
from sklearn.svm import LinearSVC
import pathlib
from sklearn.metrics import roc_curve, auc
import concurrent.futures
import tqdm
import h5py


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


def getUnitsInRegion(units,region,layer=None,rs=False,fs=False, cell_type=None):
    
    if region in ('SC/MRN cluster 1','SC/MRN cluster 2'):
        clust = 1 if '1' in region else 2
        dirPath = pathlib.Path('/Volumes/programs/mindscope/workgroups/np-behavior/VBN_video_analysis')
        clustId = np.load(os.path.join(dirPath,'sc_mrn_clusterId.npy'))
        clustUnitId = np.load(os.path.join(dirPath,'sc_mrn_clusterUnitId.npy'))
        u = clustUnitId[np.in1d(clustUnitId,units.index) & (clustId==clust)]
        inRegion = np.in1d(units.index,u)
    
    elif region=='all':
        inRegion = [True]*len(units)

    else:
        if region=='VISall':
            reg = ('VISp','VISl','VISrl','VISal','VISpm','VISam')
        elif region=='VISlateral':
            reg = ('VISp', 'VISl', 'VISal')
        elif region == 'VISmedial':
            reg = ('VISpm', 'VISam')
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
        else:
            reg = region
        inRegion = np.in1d(units['structure_acronym'], reg)
        if np.any([a in region for a in ('VISp','VISl','VISrl','VISal','VISpm','VISam','VISall', 'VISlateral', 'VISmedial', 'Hipp')]):
            if layer is not None and not layer == 'all':
                if layer in ['4', '5']:
                    inRegion = inRegion & np.in1d(units['cortical_layer'], layer)
                elif layer == '6':
                    inRegion = inRegion & (units['cortical_layer'].isin(['6a', '6b'])).values
                elif layer == '2/3':
                    inRegion = inRegion & (units['cortical_layer'].isin(['1', '2/3'])).values
                elif layer not in ['1', '2/3', '4', '5', '6a', '6b']:
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


def get_units_in_cluster(unit_table, *cluster_ids, clustering='old'):

    if cluster_ids[0] == 'all':
        return np.array([True]*len(unit_table))

    if clustering == 'old':
        cluster_column = 'cluster_labels'
    elif clustering == 'new':
        cluster_column = 'cluster_labels_new'
    else:
        raise(ValueError(f'Invalid clustering value: {clustering}. Must be "old" or "new"'))
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


def apply_condition_filter(units, condition, trajectory='all'):
    '''
    Applies condition filter depending on training trajectory as follows:
    
    condition: Familiar or Novel

    trajectory:
        all: grab all units matching condition regardless of training trajectory
        GGH: grab units only from mice trained on G, with G as first recording day set
        GHG: grab units only from mice trained on G, with H as first recording day set
        HHG: grab units only from mice trained on H, with H as first recording day set
    '''
    cond = condition.split('_')[0]

    if trajectory == 'all':
        incondition = (units['experience_level']==cond).values
    
    else:
        familiar_set = trajectory[0]
        familiar_day = 1 if trajectory[1]==familiar_set else 2
        novel_day = 2 if trajectory[1]==familiar_set else 1
        novel_set = 'H' if familiar_set=='G' else 'G'
        
        set_to_filter_on = familiar_set if cond=='Familiar' else novel_set
        day_to_filter_on = familiar_day if cond=='Familiar' else novel_day

        incondition = ((units['image_set']==set_to_filter_on)&(units['session_number']==day_to_filter_on)&(units['experience_level']==cond)).values

    return incondition


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
    lick = np.array(stim['lick_for_flash'])
    lickTimes = np.array(stim['lick_time'])
    lickLatency = lickTimes - flashTimes
    earlyLick = lickLatency < 0.15
    lateLick = lickLatency > 0.75
    lick[earlyLick | lateLick] = False
    lickTimes[earlyLick | lateLick] = np.nan
    return flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes


def decodeImage(sessionId, unitTable, unitData, stimTable, regions, unitSampleSize, decodeWindowEnd, use_nonchange=True, class_weight=None):
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
        inRegion = getUnitsInRegion(units,region)
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
                if sampleSize!=decodeWindowSampleSize and winEnd!=decodeWindows[-1]:
                    continue
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

def decodeSingleClass(sessionId, unitTable, unitData, stimTable, regions, unitSampleSize, decodeWindowEnd, use_nonchange=True, class_weight=None):
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
        inRegion = getUnitsInRegion(units,region)
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
                if sampleSize!=decodeWindowSampleSize and winEnd!=decodeWindows[-1]:
                    continue
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
                            use_nonchange=True, class_weight=None,
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
    flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = du.getBehavData(stim)
    flashes_to_use = nonChangeFlashes if use_nonchange else changeFlashes
    
    #flashes_to_use = np.append(changeFlashes[1:], False)
    nFlashes = flashes_to_use.sum()

    d = {region: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','prediction','confidence', 'balanced_accuracy',
                                                        'imagewise_recall', 'imagewise_precision', 'image_order')}
            for sampleSize in unitSampleSize} for region in regions}
    d['decodeWindows'] = decodeWindows
    d['hit'] = np.array(stim['hit'])[changeFlashes]

    image_id_order = [im for im in np.unique(image_ids[flashes_to_use]) if im not in ['im083_r', 'im111_r', 'omitted']] + ['im083_r', 'im111_r']
    y = []
    for im in image_id_order:
        num_im = (flashes_to_use&(image_ids==im)).sum()
        y.extend([im]*num_im)

    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = getUnitsInRegion(units,region)
        highQuality = apply_unit_quality_filter(units, no_abnorm=False)
        #inCluster = du.get_units_in_cluster()

        final_unit_filter = highQuality & inRegion
        sp = np.zeros((final_unit_filter.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(final_unit_filter)[0]):
            sp[i]=spikes[u,:,:]
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
                # if sampleSize!=decodeWindowSampleSize and winEnd!=decodeWindows[-1]:
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
                    else:
                        d[region][sampleSize][metric][-1] = np.median(d[region][sampleSize][metric][-1],axis=0)
    warnings.filterwarnings('default')
    # if not output_dir is None:
    #     np.save(os.path.join(output_dir,'unitChangeDecoding','unitChangeDecoding_'+str(sessionId)+'.npy'),d)
    return d


def calc_auroc(x1, x2):
    fpr, tpr, thresh = roc_curve([0]*len(x1) + [1]*len(x2), np.concatenate([x1, x2]))
    return auc(fpr, tpr)


def unit_decoding(session_id, tensor_file, unit_ids, cond1_flash_indices, cond2_flash_indices, baseline_length=None, 
                    response_window_start=20, response_window_end=150):
        
    tensor = h5py.File(tensor_file, 'r')
    session_tensor = tensor[str(session_id)]
    unit_indices = [ind for ind, u in enumerate(session_tensor['unitIds'][()]) if u in unit_ids]
    spikes = session_tensor['spikes']

    norm_factor = len(cond1_flash_indices)*len(cond2_flash_indices)

    response_slice = slice(response_window_start, response_window_end)
    unit_auroc = np.full((len(unit_indices)), np.nan)
    unit_pval = np.full((len(unit_indices)), np.nan)
    for ucount, uind in enumerate(unit_indices):
        unit_cond_spikes = []
        for icond, cond_flash_indices in enumerate([cond1_flash_indices, cond2_flash_indices]):
            resp = np.mean(spikes[uind][cond_flash_indices, response_slice], axis=1)

            if baseline_length is not None:
                baseline = np.mean(spikes[uind][cond_flash_indices-1, -baseline_length:], axis=1)
                unit_resp = resp-baseline
            else:
                unit_resp = resp

            unit_cond_spikes.append(unit_resp)
        
        #unit_auroc[ucount] = calc_auroc(*unit_cond_spikes)
        mann_u_stats = scipy.stats.mannwhitneyu(*unit_cond_spikes)
        unit_auroc[ucount] = 1-mann_u_stats[0]/norm_factor
        unit_pval[ucount] = mann_u_stats[1]

    return unit_auroc, unit_pval, session_tensor['unitIds'][unit_indices]


def run_unit_decoding(tensor_file, stim_file, session_list, unit_ids, stim_filter_cond1, stim_filter_cond2, 
                        baseline_length=50, response_window_start=20, response_window_end=150, shift=0):
    
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=None)        
    future_to_session = {}
    for session in session_list:
        stims = stim_file[stim_file['session_id']==int(session)].reset_index()
        
        flash_indices = []
        for stim_filter in [stim_filter_cond1, stim_filter_cond2]:
            chained_query = ' & '.join(stim_filter)
            stims_subset = stims.query(chained_query)
            flash_indices.append(stims_subset.index.values+shift)
        
        fut = pool.submit(unit_decoding, 
                            session, 
                            tensor_file,
                            unit_ids, 
                            flash_indices[0], flash_indices[1], 
                            baseline_length = baseline_length, 
                            response_window_start = response_window_start,
                            response_window_end = response_window_end
                            )

        future_to_session[fut] = session
    
    unit_auroc_data = []
    unit_pvals = []
    unit_ids = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):

        session = future_to_session[future]
        try:
            data = future.result()
            unit_auroc_data.append(data[0])
            unit_pvals.append(data[1])
            unit_ids.append(data[2])
            session_ids.append(session)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return unit_auroc_data, unit_pvals, unit_ids