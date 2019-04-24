import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import copy
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from numba import jit
from sklearn.model_selection import StratifiedKFold
from tsfresh.feature_extraction import extract_features
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from functools import partial, wraps
from datetime import datetime as dt
import gc; gc.enable()
cpu_num = 4

@jit
def haversine_plus(lon1, lat1, lat2, lon2, hgs, hgp, dmd): #hgs, hgp, dmd追加
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) from 
    #https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    #Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    #Implementing Haversine Formula: 
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),  
                          np.multiply(np.cos(lat1), 
                                      np.multiply(np.cos(lat2), 
                                                  np.power(np.sin(np.divide(dlon, 2)), 2))))
    
    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))
    
    # distance_modulus from https://www.kaggle.com/c/PLAsTiCC-2018/discussion/71871
    return {
        'haversine': haversine, 
        'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2))
   }

@jit
def process_flux(df):
    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({
        'flux_ratio_sq': flux_ratio_sq, 
        'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq,}, 
        index=df.index)
    
    return pd.concat([df, df_flux], axis=1)

@jit
def process_flux_agg(df):
    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_min'].values
    
    df_flux_agg = pd.DataFrame({
        'flux_w_mean': flux_w_mean,
        'flux_diff1': flux_diff,
        'flux_diff2': flux_diff / df['flux_mean'].values,       
        'flux_diff3': flux_diff /flux_w_mean,
        }, index=df.index)
    
    return pd.concat([df, df_flux_agg], axis=1)

@jit
def added_features(df):
    hpb_by_dsm = np.divide(df['hostgal_photoz_by_err'].values, df['distmod'].values)
    hpd_by_dsm = np.divide(df['hostgal_photoz_div_err'].values, df['distmod'].values)
    df_added = pd.DataFrame({'hpb_by_dsm': hpb_by_dsm, 'hpd_by_dsm': hpd_by_dsm}, index=df.index)
    
    flux_pass_col = ['__fft_coefficient__coeff_0__attr_"abs"', '__fft_coefficient__coeff_1__attr_"abs"', '__fft_coefficient__coeff_2__attr_"abs"', 
                    '__kurtosis', '__maximum', '__mean', '__median', '__minimum', '__skewness', '__standard_deviation', '__variance']
    dict_added = {}
    for i in flux_pass_col:
        for j in list(itertools.combinations(range(6), 2)):
            dict_added['{}{}'.format(j[0],j[1])+i] = np.divide(df['{}'.format(j[0])+i].values, df['{}'.format(j[1])+i].values)
    df_added =pd.DataFrame(dict_added, index=df.index)
    
    for i in flux_pass_col:
        for j in range(6):
            df = df.drop('{}'.format(j)+i, axis=1)
    
    return pd.concat([df, df_added], axis=1)

def featurize(df, df_meta, aggs, fcp, n_jobs=4):
    
    """
    Extracting Features from train set
    Features from olivier's kernel
    very smart and powerful feature that is generously given here 
    https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    per passband features with tsfresh library. fft features added to capture periodicity 
    https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70346#415506
    """
    
    df = process_flux(df)

    agg_df = df.groupby('object_id').agg(aggs)
    agg_df.columns = [ '{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
    agg_df = process_flux_agg(agg_df) # new feature to play with tsfresh

    # Add more features with
    agg_df_ts_flux_passband = extract_features(df, 
                                               column_id='object_id', 
                                               column_sort='mjd', 
                                               column_kind='passband', 
                                               column_value='flux', 
                                               default_fc_parameters=fcp['flux_passband'], n_jobs=n_jobs)

    agg_df_ts_flux = extract_features(df, 
                                      column_id='object_id', 
                                      column_value='flux', 
                                      default_fc_parameters=fcp['flux'], n_jobs=n_jobs)

    agg_df_ts_flux_by_flux_ratio_sq = extract_features(df, 
                                      column_id='object_id', 
                                      column_value='flux_by_flux_ratio_sq', 
                                      default_fc_parameters=fcp['flux_by_flux_ratio_sq'], n_jobs=n_jobs)

    # Add smart feature that is suggested here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected']==1].copy()
    agg_df_mjd = extract_features(df_det, 
                                  column_id='object_id', 
                                  column_value='mjd', 
                                  default_fc_parameters=fcp['mjd'], n_jobs=n_jobs)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']
    
    agg_df_ts_flux_passband.index.rename('object_id', inplace=True) 
    agg_df_ts_flux.index.rename('object_id', inplace=True) 
    agg_df_ts_flux_by_flux_ratio_sq.index.rename('object_id', inplace=True) 
    agg_df_mjd.index.rename('object_id', inplace=True)      
    agg_df_ts = pd.concat([agg_df, 
                           agg_df_ts_flux_passband, 
                           agg_df_ts_flux, 
                           agg_df_ts_flux_by_flux_ratio_sq, 
                           agg_df_mjd], axis=1).reset_index()
    
    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')
    result = added_features(result) #追加
    return result

def process_meta(filename):
    meta_df = pd.read_csv(filename)
    
    meta_dict = dict()
    # distance
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values, 
                   meta_df['gal_l'].values, meta_df['gal_b'].values,
                   meta_df['hostgal_specz'].values, meta_df['hostgal_photoz'].values, meta_df['distmod'].values)) #追加
    #
    meta_dict['hostgal_photoz_certain'] = np.multiply(
            meta_df['hostgal_photoz'].values, 
             np.exp(meta_df['hostgal_photoz_err'].values))
            
    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)

    meta_df['hostgal_photoz_by_err'] = meta_df['hostgal_photoz'] * meta_df['hostgal_photoz_err']
    meta_df['hostgal_photoz_div_err'] = meta_df['hostgal_photoz'] / meta_df['hostgal_photoz_err']
    return meta_df

def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def lgbm_multi_weighted_logloss_gal(y_true, y_preds):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """  
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 16, 53, 65, 92]
    class_weights = {6: 1, 16: 1, 53: 1, 65: 1, 92: 1}

    loss = multi_weighted_logloss(y_true, y_preds, classes, class_weights)
    return 'wloss', loss, False

def lgbm_multi_weighted_logloss_extra(y_true, y_preds):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """  
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [15, 42, 52, 62, 64, 67, 88, 90, 95]
    class_weights = {15: 2, 42: 1, 52: 1, 62: 1, 64: 2, 67: 1, 88: 1, 90: 1, 95: 1}

    loss = multi_weighted_logloss(y_true, y_preds, classes, class_weights)
    return 'wloss', loss, False

def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    return importances_

def lgbm_modeling_cross_validation_gal(params,
                                   full_train, 
                                   y, 
                                   classes, 
                                   class_weights, 
                                   nr_fold=5, 
                                   random_state=1):

    # Compute weights
    w = y.value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}
    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(n_splits=nr_fold, 
                            shuffle=True, 
                            random_state=random_state)
    
    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]
    
        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgbm_multi_weighted_logloss_gal,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        print('no {}-fold loss: {}'.format(fold_ + 1, 
              multi_weighted_logloss(val_y, oof_preds[val_, :], 
                                     classes, class_weights)))
    
        imp_df = pd.DataFrame({
                'feature': full_train.columns,
                'gain': clf.feature_importances_,
                'fold': [fold_ + 1] * len(full_train.columns),
                })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(y_true=y, y_preds=oof_preds, 
                                   classes=classes, class_weights=class_weights)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))
    df_importances = save_importances(importances_=importances)
#     df_importances.to_csv('lgbm_importances_gal.csv', index=False)
    
    return clfs, score, df_importances

def lgbm_modeling_cross_validation_extra(params,
                                   full_train, 
                                   y, 
                                   classes, 
                                   class_weights, 
                                   nr_fold=5, 
                                   random_state=1):

    # Compute weights
    w = y.value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}
    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(n_splits=nr_fold, 
                            shuffle=True, 
                            random_state=random_state)
    
    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]
    
        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgbm_multi_weighted_logloss_extra,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        print('no {}-fold loss: {}'.format(fold_ + 1, 
              multi_weighted_logloss(val_y, oof_preds[val_, :], 
                                     classes, class_weights)))
    
        imp_df = pd.DataFrame({
                'feature': full_train.columns,
                'gain': clf.feature_importances_,
                'fold': [fold_ + 1] * len(full_train.columns),
                })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(y_true=y, y_preds=oof_preds, 
                                   classes=classes, class_weights=class_weights)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))
    df_importances = save_importances(importances_=importances)
#     df_importances.to_csv('lgbm_importances_extra.csv', index=False)
    
    return clfs, score, df_importances

def predict_chunk(df_, clfs_gal_, clfs_extra_, meta_, feat_gal_cols_, feat_extra_cols_, featurize_configs, train_mean_gal,  train_mean_extra):
    
    # process all features
    full_test = featurize(df_, meta_, 
                          featurize_configs['aggs'], 
                          featurize_configs['fcp'])
    test_mask = full_test['distmod'].isnull().values
    full_test.fillna(0, inplace=True)
    
    # Make predictions
    preds_gal_ = None
    for clf in clfs_gal_:
        if preds_gal_ is None:
            preds_gal_ = clf.predict_proba(full_test.loc[test_mask, feat_gal_cols_])
        else:
            preds_gal_ += clf.predict_proba(full_test.loc[test_mask, feat_gal_cols_])
    preds_gal_ = preds_gal_ / len(clfs_gal_)
    
    preds_extra_ = None
    for clf in clfs_extra_:
        if preds_extra_ is None:
            preds_extra_ = clf.predict_proba(full_test.loc[~test_mask, feat_extra_cols_])
        else:
            preds_extra_ += clf.predict_proba(full_test.loc[~test_mask, feat_extra_cols_])
    preds_extra_ = preds_extra_ / len(clfs_extra_)
    
    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99_gal = np.ones(preds_gal_.shape[0])
    for i in range(preds_gal_.shape[1]):
        preds_99_gal *= (1 - preds_gal_[:, i])

    preds_99_extra = np.ones(preds_extra_.shape[0])
    for i in range(preds_extra_.shape[1]):
        preds_99_extra *= (1 - preds_extra_[:, i])
    
    # Create DataFrame from predictions
    gal_colunms=['class_{}'.format(s) for s in clfs_gal_[0].classes_]
    extra_columns=['class_{}'.format(s) for s in clfs_extra_[0].classes_]
    preds_df_= pd.DataFrame(np.zeros([preds_gal_.shape[0]+preds_extra_.shape[0], 15]),
                                    columns=[i for j in [gal_colunms, extra_columns, ['class_99']] for i in j])
    
    preds_df_.loc[test_mask, gal_colunms] = preds_gal_
    preds_df_.loc[test_mask, 'class_99'] = 0.018 * preds_99_gal / np.mean(preds_99_gal)
    preds_df_.loc[~test_mask, extra_columns] = preds_extra_
    preds_df_.loc[~test_mask, 'class_99'] = 0.155 * preds_99_extra / np.mean(preds_99_extra)
    
    preds_df_['object_id'] = full_test['object_id']
        
    return preds_df_

def process_test(clfs_gal, clfs_extra, 
                 feat_gal_cols, feat_extra_cols, 
                 featurize_configs, 
                 train_mean_gal,train_mean_extra,
                 filename='predictions.csv',
                 chunks=5000000):

    start = time.time()

    meta_test = process_meta('data/test_set_metadata.csv')
    # meta_test.set_index('object_id',inplace=True)

    remain_df = None
    for i_c, df in enumerate(pd.read_csv('data/test_set.csv', chunksize=chunks, iterator=True)):
        
        new_remain_df = df.loc[df['object_id'] == df['object_id'].iloc[-1]].copy()
        df_len = df.shape[0]
        if remain_df is None:
            df = df[df['object_id']!=df['object_id'].iloc[-1]]
            remain_df = new_remain_df
        elif df_len < chunks:
            df = pd.concat([remain_df, df], axis=0)
        else:
            df = pd.concat([remain_df, df[df['object_id']!=df['object_id'].iloc[-1]]], axis=0)
            remain_df = new_remain_df
        
        preds_df = predict_chunk(df_=df,
                                 clfs_gal_=clfs_gal,
                                 clfs_extra_=clfs_extra,
                                 meta_=meta_test,
                                 feat_gal_cols_=feat_gal_cols,
                                 feat_extra_cols_=feat_extra_cols,
                                 featurize_configs=featurize_configs,
                                 train_mean_gal=train_mean_gal,
                                 train_mean_extra=train_mean_extra)
    
        if i_c == 0:
            preds_df.to_csv(filename, header=True, mode='a', index=False)
        else:
            preds_df.to_csv(filename, header=False, mode='a', index=False)
    
        del preds_df
        gc.collect()
        print('{:15d} done in {:5.1f} minutes' .format(
                chunks * (i_c + 1), (time.time() - start) / 60), flush=True)

    
#     # Compute last object in remain_df
#     preds_df = predict_chunk(df_=remain_df,
#                              clfs_gal_=clfs_gal,
#                              clfs_extra_=clfs_extra,
#                              meta_=meta_test,
#                              feat_gal_cols_=feat_gal_cols,
#                              feat_extra_cols_=feat_extra_cols,
#                              featurize_configs=featurize_configs,
#                              train_mean_gal=train_mean_gal,
#                              train_mean_extra=train_mean_extra)
        
#     preds_df.to_csv(filename, header=False, mode='a', index=False)
    return

# Features to compute with tsfresh library. Fft coefficient is meant to capture periodicity
# agg features
aggs = {
    'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'detected': ['mean'],
    'flux_ratio_sq':['sum', 'skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
}

# tsfresh features
fcp = {
    'flux': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean_change': None,
        'mean_abs_change': None,
        'length': None,
    },

    'flux_by_flux_ratio_sq': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,       
    },

    'flux_passband': {
        'fft_coefficient': [
                {'coeff': 0, 'attr': 'abs'}, 
                {'coeff': 1, 'attr': 'abs'},
                {'coeff': 2, 'attr': 'abs'} ],
        'maximum' : None, 'minimum' : None, 'mean' : None, 'median' : None,
        'variance' : None, 'standard_deviation' : None, 'kurtosis' : None, 'skewness' : None,
    },

    'mjd': {
        'maximum': None, 
        'minimum': None,
        'mean_change': None,
        'mean_abs_change': None,
    },
}

meta_train = process_meta('data/training_set_metadata.csv')

train = pd.read_csv('data/training_set.csv')
full_train = featurize(train, meta_train, aggs, fcp)

# full_train.to_csv('data/full_train.csv', index=False)

if 'target' in full_train:
    y = full_train['target']
    del full_train['target']

classes = sorted(y.unique())    
# Taken from Giba's topic : https://www.kaggle.com/titericz
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# with Kyle Boone's post https://www.kaggle.com/kyleboone
class_weights = {c: 1 for c in classes}
class_weights.update({c:2 for c in [64, 15]})
print('Unique classes : {}, {}'.format(len(classes), classes))
print(class_weights)
#sanity check: classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
#sanity check: class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
#if len(np.unique(y_true)) > 14:
#    classes.append(99)
#    class_weights[99] = 2

if 'object_id' in full_train:
    oof_df = full_train[['object_id']]
    del full_train['object_id'] 
    #del full_train['distmod'] 
#     del full_train['hostgal_specz']
#     del full_train['ra'], full_train['decl'], full_train['gal_l'], full_train['gal_b']
#     del full_train['ddf']

pd.set_option('display.max_rows', 500)
full_train.describe().T

train_mask = full_train['distmod'].isnull().values
feat_extra_li = []
feat_gal_cols = list(full_train.columns.values)
feat_extra_cols = feat_gal_cols + feat_extra_li
train_mean_gal = full_train.loc[train_mask,feat_gal_cols].mean(axis=0)
train_mean_extra = full_train.loc[~train_mask,feat_extra_cols].mean(axis=0)
classes_gal=list(np.unique(y[train_mask]))
classes_extra=list(np.unique(y[~train_mask]))
classes_gal, classes_extra
class_weights_gal = {c: 1 for c in classes_gal}
class_weights_extra = {c: 1 for c in classes_extra}
class_weights_extra.update({c:2 for c in [64, 15]})

best_params = {
        'device': 'cpu',  # or GPU
        'objective': 'multiclass', 
        'num_class': 14, 
        'n_jobs': cpu_num,    #CPU core 4(server) 2(PC)
        'max_depth': 20,    #defoult -1 (no limit)
        'n_estimators': 1000,   #defoult 100
        'subsample_freq': 3,    #defoult 0(disable bagging, k: bagging at every k iteration)
        'subsample_for_bin': 5000,   #default = 200000 number of data that sampled to construct histogram bins
        'metric': 'multi_logloss', 
        'colsample_bytree': 0.5,   #0.0 < feature_fraction <= 1.0  default = 1.0  if you set it to 0.8, LightGBM will select 80% of features before training each tree
        'learning_rate': 0.02,  #default = 0.1  
        'min_child_samples': 9,   #default = 20  minimal number of data in one leaf. Can be used to deal with over-fitting
        'min_child_weight': 130,  # default = 1e-3 minimal sum hessian in one leaf. Like min_data_in_leaf, it can be used to deal with over-fitting
        'min_split_gain': 0.04,  #default = 0.0  the minimal gain to perform split
        'num_leaves': 7,   #default = 31 max number of leaves in one tree
        'reg_alpha': 0.1,   #default = 0.0  L1 regularization
        'reg_lambda': 0.00023,   #default = 0.0  L2 regularization
        'skip_drop': 0.44,    #default = 0.5   0.0 <= skip_drop <= 1.0  used only in dart  probability of skipping the dropout procedure during a boosting iteration
        'subsample': 0.65}   #default = 1.0  like feature_fraction, but this will randomly select part of data without resampling

#import pdb; pdb.set_trace()
full_train.fillna(0, inplace=True)

deleted = []
for i in range(len(feat_gal_cols)-2):
    eval_func_gal = partial(lgbm_modeling_cross_validation_gal, 
                        full_train=full_train.loc[train_mask, feat_gal_cols],
                        y=y.loc[train_mask], 
                        classes=classes_gal, 
                        class_weights=class_weights_gal, 
                        nr_fold=5, 
                        random_state=1234)
    clfs_gal, score_gal, df_importances = eval_func_gal(best_params)
    del_item = df_importances.groupby('feature').mean().sort_values('mean_gain').index[0]
    feat_gal_cols.remove(del_item)
    deleted.append([i, del_item, score_gal])
deleted_items_gal = pd.DataFrame(deleted, columns=['i', 'del_item', 'score_gal'])
deleted_items_gal.to_csv('lgbm_feature_gal.csv', index=False)

deleted = []
for i in range(len(feat_extra_cols)-2):
    eval_func_extra = partial(lgbm_modeling_cross_validation_extra, 
                               full_train=full_train.loc[~train_mask, feat_extra_cols],
                               y=y.loc[~train_mask], 
                               classes=classes_extra, 
                               class_weights=class_weights_extra, 
                               nr_fold=5, 
                               random_state=1234)
    clfs_extra, score_extra, df_importances = eval_func_extra(best_params)
    del_item = df_importances.groupby('feature').mean().sort_values('mean_gain').index[0]
    feat_extra_cols.remove(del_item)
    deleted.append([i, del_item, score_extra])
deleted_items_extra = pd.DataFrame(deleted, columns=['i', 'del_item', 'score_extra'])
deleted_items_extra.to_csv('lgbm_feature_extra.csv', index=False)

filename = 'subm_{:.6f}_{}.csv'.format((score_gal*y.loc[train_mask].shape[0]+score_extra*y.loc[~train_mask].shape[0])/y.shape[0], 
                 dt.now().strftime('%Y-%m-%d-%H-%M'))
print('save to {}'.format(filename))

# TEST
process_test(clfs_gal, clfs_extra,
             feat_gal_cols, feat_extra_cols,
             featurize_configs={'aggs': aggs, 'fcp': fcp}, 
             train_mean_gal=train_mean_gal, train_mean_extra=train_mean_extra,
             filename=filename,
             chunks=5000000)
 z = pd.read_csv(filename)
print("Shape BEFORE grouping: {}".format(z.shape))
z = z.groupby('object_id').mean()
z = z.loc[:,['class_{}'.format(s) for s in [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]]]
print("Shape AFTER grouping: {}".format(z.shape))

z_sum=z.sum[1:](axis=1)
for i in range(z[1:].shape[1]):
    z.iloc[1:,i] /= z_sum
z.to_csv('sub6_{}'.format(filename), index=True)