import pandas
import os
import itertools
import numpy as np
import nibabel as ni
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.formula.api as smf
from glob import glob
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler #, MinMaxScaler
from sklearn import model_selection, linear_model
from sklearn.ensemble import RandomForestRegressor
import lime
import lime.lime_tabular
#from sklearn import mixture
#from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from nilearn import image, plotting
#from sklearn.neighbors import kneighbors_graph
#from sklearn.metrics import calinski_harabaz_score
#from sklearn.metrics import silhouette_score
#from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D



######## UTILITIES ########

def convert_coords(coord, to_what = 'mni', vs = 1):
    origin = [90, 126, 72]
    origin = (np.array(origin) / vs).astype(int).tolist()
    x,y,z = coord[0],coord[1],coord[2]
    
    if to_what == 'mni':
        x = (origin[0]-x)*vs
        y = (y-origin[1])*vs
        z = (z-origin[2])*vs
        
    elif to_what == 'xyz':
        x=origin[0]- x/vs
        y=y/vs + origin[1]
        z=z/vs + origin[2]
        
    else:
        raise IOError('please set to_what to either mni or xyz')
    
    return x,y,z


def make_sphere(coord, radius):
    summers = []
    negrad = (radius*-1)+1
    for x in itertools.product(range(negrad,radius),repeat=3):
        summers.append(np.array(x))
    s_coords = [np.array(coord) + x for x in summers]
    xs = [int(x[0]) for x in s_coords]
    ys = [int(x[1]) for x in s_coords]
    zs = [int(x[2]) for x in s_coords]
    
    return xs, ys, zs

def get_gene_vector(bigdf, gene_vec = [], probe_ids = [], betas = []):

    if len(gene_vec) == 0 and len(probe_ids) == 0:
        raise IOError('please supply either a gene vector or probe IDs')
    if len(gene_vec) > 0 and len(probe_ids) > 0:
        raise IOError('please supply either a gene vector or probe IDs, but NOT BOTH!')
    #if type(clf) != type(None) and len(probe_ids) == 0:
    #   raise IOError('must pass probe_ids with clf')
    if len(betas) > 0 and len(probe_ids)==0:
        raise IOError('please supply probe_ids along with the betas argument')
        
    if len(gene_vec) > 0:
        gene_vec = np.array(gene_vec)
        vals = []
        for i in range(bigdf.shape[-1]):
            try:
                vals.append(stats.pearsonr(gene_vec,
                                        bigdf.loc[:,bigdf.columns[i]])[0])
            except:
                bigdf.columns = ['col_%s'%x for x in range(bigdf.shape[-1])]
                vals.append(stats.pearsonr(gene_vec,
                                        bigdf.loc[:,bigdf.columns[i]])[0])
    elif len(probe_ids) > 0:
        #if type(clf) == type(None):
        if len(betas) == 0:
            vals = bigdf.loc[probe_ids].mean().values
        else:
            X = bigdf.loc[probe_ids].values.T
            vals = np.dot(X,betas)
        #    vals = clf.predict(X)
    return vals

def make_expression_image(vector, coords, projection_space, 
                          wdir = './', nm = 'gene', vrad=5,
                         return_img = False):
    if len(vector) != len(coords):
        print('input and coordinate vectors must have the same length')
    if type(projection_space) == str:
        jnk = ni.load(projection_space)
        aff = jnk.affine
        dat = jnk.get_data()
    else:
        try:
            aff = projection_space.affine
            dat = projection_space.get_data()
        except:
            raise IOError('projection_space must be a NiftiImage object or path to .nii file')
    nimg = np.zeros_like(dat).astype(float)
    for i in range(len(vector)):
        xs,ys,zs = make_sphere(coords[i], vrad)
        nimg[xs,ys,zs] = vector[i]
    nimg = ni.Nifti1Image(nimg,aff)
    flnm = os.path.join(wdir,'%s_xp_image.nii.gz'%nm)
    nimg.to_filename(flnm)
    
    if return_img:
        return nimg

def find_closest_point_along_axis(coords,skel_coords):
    closest_coords = []
    y_coord = []
    for coord in coords:
        dists = []
        for i in range(len(skel_coords[0])):
            dist = sum([abs(skel_coords[0][i] - coord[0]), 
                        abs(skel_coords[1][i] - coord[1]), 
                        abs(skel_coords[2][i] - coord[2])])
            dists.append(dist)
        gind = np.argmin(dists)
        closest_coords.append([skel_coords[0][gind],
                               skel_coords[1][gind],
                                skel_coords[2][gind]])
        y_coord.append(skel_coords[1][gind])

    return y_coord, closest_coords

def plot_3d_render(label_locations, data, outfl = None, r1=0, r2=30, step=1, r_init=30):
    
    plt.close()
    fig = plt.figure()
    ax = Axes3D(fig)
    f = ax.scatter(label_locations[0], label_locations[1], label_locations[2], c = data[(label_locations)],
              cmap='RdBu_r')

    for angle in range(r1, r2, step):
        ax.view_init(r_init, angle)
        plt.draw()
        plt.pause(.001)

    ax.set_xlabel('\nX',fontsize=20,linespacing=4)
    ax.set_xticks(range(120,50,-20))
    plt.setp(ax.get_xticklabels(), fontsize=20)
    ax.set_ylabel('\nY',fontsize=20, linespacing=2)
    ax.set_yticks(range(90,121,10))
    plt.setp(ax.get_yticklabels(), fontsize=20)
    ax.set_zlabel('\n\nZ',fontsize=20, linespacing=6)
    ax.set_zticks(range(45,76,10))
    ax.tick_params(labelsize=20)
    fig.colorbar(f)
    if outfl:
        plt.savefig(outfl, bbox_inches='tight')
    plt.show()
    
   


def label_coordinate_by_atlas(atlas, coordinates, cube_size = 1):
    ''' This function will take a set of coordinates and an atlas and return the
    atlas label for each coordinate. Optionally, a cube can be drawn around the 
    coordinate. In this case, the most frequent non-zero value inside the cube will 
    be selected as the label. The function will output a pandas dataframe with a
    label for each input coordinate.
    
    atlas can be a path to a nifti image, or a nifti object, or a 3D np.ndarray
    
    coordinates can be a pandas Dataframe, a numpy.ndarray or a list of lists, 
    representing the mni coordinates for each input corrdinate. as such, the 
    length of one of the dimensions should be 3
    
    cube_size represents the radius of the cube to be created. leaving it as 1 will
    only retrieve the value at the coordinate specified. Increasing the value will 
    draw a cube of increasing size and collect values from within the cube. The most
    frequent non-zero value within the cube will be selected
    
    '''
    
    # initiate and check inputs
    atl, coords = init_and_chk_inputs(atlas,coordinates)
    
    # convert coordinates to xyz space
    xyz = np.array([convert_coords(coords[x],'xyz') for x in range(coords.shape[0])
                   ]).astype(int)
    
    # create output container
    results = pandas.DataFrame(index=range(coords.shape[0]),columns = ['Label'])
    
    # get labels for each coordinate
    print('extracting labels')
    if cube_size == 1:
        results.loc[:,'Label'] = atl[xyz[:,0].tolist(),xyz[:,1].tolist(),xyz[:,2].tolist()]
    else:
        labs = []
        for i in range(xyz.shape[0]):
            labs.append(extract_value_from_cube(xyz[i,:], cube_size, atl))
        results.loc[:,'Label'] = labs
    
    print('completed')
    print('%s coordinates were outside of the provided atlas'%(
                                            len(results[results.Label==0])))

    return results
    
def init_and_chk_inputs(atlas,coordinates):
    print('checking and initializing inputs')
    if type(atlas) == str:
        atl = ni.load(atlas).get_data()
    elif type(atlas) == ni.nifti1.Nifti1Image: 
        atl = atlas.get_data()
    elif type(atlas) == np.core.memmap.memmap or type(atlas) == np.ndarray:
        atl = atlas
    else:
        raise IOError('atlas must be a nifti object, a numpy array or a path to a nifti file')
    
    if type(coordinates) == pandas.core.frame.DataFrame:
        coords = coordinates.values
       
    elif type(coordinates) == list or type(coordinates) == tuple:
        if len(coordinates)== 3:
            coords = np.zeros((len(coordinates[0]),3))
            for i in range(3):
                coords[:,i] = coords[i]
    elif type(coordinates) == np.ndarray:
        coords = coordinates
    else:
        raise IOError('coordiantes must be a pandas dataframe, numpy array or list of lists')
    
    if not any(x==3 for x in coords.shape):
        raise IOError('Coordinates are 3D and thus 3 columns are expected...')
    elif coords.shape[-1] != 3:
        coords = coords.T
    
    coords = coords.round()
    
    return atl,coords
    
def convert_coords(coord, to_what = 'mni', vs = 1):
    origin = [90, 126, 72]
    origin = (np.array(origin) / vs).astype(int).tolist()
    x,y,z = coord[0],coord[1],coord[2]
    
    if to_what == 'mni':
        x = (origin[0]-x)*vs
        y = (y-origin[1])*vs
        z = (z-origin[2])*vs
        
    elif to_what == 'xyz':
        x=origin[0]- x/vs
        y=y/vs + origin[1]
        z=z/vs + origin[2]
        
    else:
        raise IOError('please set to_what to either mni or xyz')
    
    return x,y,z

def extract_value_from_cube(coord, radius, atl):
    
    xs,ys,zs = make_cube(coord, radius)
    cube_vals = atl[xs,ys,zs]
    if stats.mode(cube_vals)[0][0] == 0:
        if len(cube_vals[cube_vals!=0]) > 0:
            lab = stats.mode(cube_vals[cube_vals!=0])[0][0]
        else:
            lab = 0
    else:
        lab = stats.mode(cube_vals)[0][0]
    
    return lab
        
def make_cube(coord, radius):
    summers = []
    negrad = (radius*-1)+1
    for x in itertools.product(range(negrad,radius),repeat=3):
        summers.append(np.array(x))
    s_coords = [np.array(coord) + x for x in summers]
    xs = [int(x[0]) for x in s_coords]
    ys = [int(x[1]) for x in s_coords]
    zs = [int(x[2]) for x in s_coords]
    
    return xs, ys, zs

def find_bigram(gene_df, target_id, gene_names = [], top_res=10, report = True,
                save_bg = False, check_genes = [], check_type = 'perc', check_val = 0.95):
    
    if type(check_type) != type(None):
        if check_type not in ['perc','r2']:
            raise IOError('check_type must be set to "perc" or "r2" ')
        else:
            if type(check_val) != float:
                raise ValueError('check_val must be a float between 0 and 1')
            else:
                if check_val < 0 or check_val > 1:
                    raise ValueError('check_val must be a float between 0 and 1')
    
    bg = pandas.DataFrame(index=gene_df.index,columns=['r2'])
    res = []
    target = gene_df.loc[target_id].values
    for x in gene_df.index:
        res.append(stats.pearsonr(target,gene_df.loc[x].values)[0])
    bg.loc[:,'r2'] = [round(x**2,4) for x in res]
    bg.loc[:,'r'] = [round(x,4) for x in res]
    if len(gene_names) > 0:
        bg.loc[:,'name'] = gene_names
    if report:
        print(bg.sort_values('r2',ascending=False).head(top_res))
        
    if len(check_genes)>0:
        if len(gene_names)==0:
            print('no gene names provide, cannot look up genes. Sorry.')
        else:
            bg.sort_values('r2',inplace=True)
            bg.loc[:,'rank'] = range(len(bg))
            if type(check_type) == type(None):
                for g in check_genes:
                    print(bg[bg.name==g])
            else:
                for g in check_genes:
                    report_gene = False
                    rank = np.max(bg[bg.name==g]['rank']) 
                    perc = rank / len(bg)
                    if check_type == 'perc':
                        if perc > check_val:
                            report_gene = True
                    else:
                        if max(bg[bg.name==g]['r2'].values) > check_val:
                            report_gene = True
                    if report_gene:
                        print(bg[bg.name==g])
                        print('correlation greater than %s%% of other genes'%(perc))
                        r = bg[bg['rank']==rank]['r'].values[0]
                        plt.close()
                        g = sns.distplot(bg.r)
                        plt.axvline(r, color = 'r', linestyle = 'dashed')
                        plt.show()
                        
    if save_bg:
        return bg
    

def prepare_GO_terms(gene_set, go_sheets, probedf):
    
    ind = probedf.loc[gene_set.index,'gene_symbol'].unique()
    cols = []
    gos = []
    for sht in go_sheets:
        jnk = pandas.ExcelFile(sht)
        go = pandas.ExcelFile(sht).parse(jnk.sheet_names[0])
        gos.append(go)
        cols += go.Description.tolist()
    
    go_gsea = pandas.DataFrame(np.zeros((len(ind),len(cols))), index=ind, columns = cols) 
    
    for go in gos:
        for i,row in go.iterrows():
            jnk = row['Genes']
            jnk = jnk.replace('[','')
            jnk = jnk.replace(']','')
            genes = [x for x in jnk.split(' ') if x.isupper()]
            try:
                go_gsea.loc[genes,row['Description']] = 1
            except:
                misses = [genes.pop(genes.index(x)) for x in genes if x not in go_gsea.index]
                go_gsea.loc[genes,row['Description']] = 1
                for gene in misses:
                    go_gsea.loc[gene,row['Description']] = 1

    go_gsea.fillna(0,inplace=True)
    
    # Drop genes with 0 hits
    go_gsea.drop([x for x in go_gsea.index if all(go_gsea.loc[x].values == 0)],inplace=True)
    
    return go_gsea

######## SCRIPTS #########
def PCA_LR_pipeline(in_mtx, y, pca = PCA(random_state=123), 
                    clf = linear_model.LassoCV(random_state = 123), 
                    cv_strategy = None, cv = 10, test_gene_num = [100], illustrative = False,
                   sanity_check_style = 'separate', cv_labels = [], reverse_axes = True):
    
    final_outputs = {}
    
    if type(in_mtx) == pandas.core.frame.DataFrame:
        in_mtx = in_mtx.values
    
    if type(pca) == type(None):
        pca_tfm = in_mtx
    else:
        print('running PCA')
        dat_pca = pca.fit(in_mtx)
        print('transforming data')
        pca_tfm = dat_pca.transform(in_mtx)
        #if len(y) != pca_tfm.shape[-1]:
        #    raise ValueError('length of y-axis of transformed item must match the length of y')
        final_outputs.update({'pca_object': dat_pca})
    
    
    print('performing model cross-validation')
    if hasattr(cv_strategy, 'get_n_splits'):
        folds = cv_strategy.get_n_splits(pca_tfm)
        #scores = model_selection.cross_val_score(clf, pca_tfm, y=y, groups=folds, cv=cv)
        predicted = model_selection.cross_val_predict(clf, pca_tfm, y=y, groups=folds, cv=cv)
        observed = y
        score = stats.pearsonr(predicted, y)[0]**2
    elif cv_strategy == 'LOLO':
        print('using leave-one-label-out cross-validation')
        observed, predicted = leave_one_x_out(cv_labels, pca_tfm, y, clf)
        score = stats.pearsonr(predicted,observed)[0]**2
    elif cv_strategy == 'balanced':
        print('balancing cross-validation by labels')
        observed, predicted = balanced_cv(cv_labels, pca_tfm, y, clf, cv)
        score = stats.pearsonr(predicted,observed)[0]**2
    elif type(cv_strategy) == int:
        print('using %s iterations of %s-fold cross-validation'%(cv_strategy,cv))
        score = []
        preds = np.zeros((cv_strategy,len(y)))
        for i in range(cv_strategy):
            sel = model_selection.KFold(n_splits=cv, shuffle=True)
            predicted = model_selection.cross_val_predict(clf, pca_tfm, y=y, cv=sel)
            preds[i,:] = predicted
            score.append(stats.pearsonr(predicted, y)[0]**2)
            print('completed iteration',i+1)
    elif type(cv_strategy) != type(None):
        print('using basic %s-fold cross-validation'%cv)
        #scores = model_selection.cross_val_score(clf, pca_tfm, y=y, cv=cv)
        predicted = model_selection.cross_val_predict(clf, pca_tfm, y=y, cv=cv)
        observed = y
        try:
            score = stats.pearsonr(predicted, y)[0]**2
        except:
            score = stats.pearsonr(predicted[:,0], y)[0]**2
    else:
        score = None
    if type(score) != type(None):
        if type(score) == list:
            if illustrative:
                plt.close()
                sns.regplot(preds.mean(0), y, fit_reg=False)
                plt.xlabel('Average CV model predicted position along axis')
                plt.ylabel('Actual position along axis')
                plt.show()

                jnk = pandas.DataFrame(index = range(cv_strategy+1), 
                                       columns = ['score','iteration'])
                jnk.loc[:,'iteration'] = list(range(cv_strategy)) + ['mean']
                jnk.loc[:,'score'] = score + [np.mean(score)]
                plt.close()
                sns.factorplot(x='iteration',y='score',data=jnk)
                plt.show()
                print('model cv scores (r2):')
                print(score)
                print('average r2:',np.mean(score))
                final_outputs.update({'CV_scores': score})
        else:        
            if illustrative:
                plt.close()
                sns.regplot(predicted, observed, fit_reg=False)
                plt.xlabel('CV model predicted position along axis')
                plt.ylabel('Actual position along axis')
                plt.show()
            print('model cv score: r2 = ',score)
            final_outputs.update({'CV_scores': score})
    else:
        print('no valid cross-validation method specified')
    
    print('running final model')
    mod = clf.fit(pca_tfm, y)
    if not hasattr(mod,'coef_'):
        raise IOError('right now, this pipeline can only accept clf objects with a coef_ attribute')
    final_outputs.update({'final_model': mod})
    scr = mod.score(pca_tfm, y)
    print('final model fit r2 = ',scr)
    if illustrative:
        plt.close()
        sns.regplot(x=mod.predict(pca_tfm), y=y)
        plt.xlabel('Model predicted position along A-P axis')
        plt.ylabel('Actual position along A-P axis')
        plt.show()
    
    if type(pca) == type(None):
        f_betas = mod.coef_
    else:
        f_betas = back_transform(dat_pca, mod)
    final_outputs.update({'betas': f_betas})
    
    gene_selections = sanity_check(in_mtx, y, f_betas, test_gene_num, 
                                   illustrative, sanity_check_style, reverse_axes)
    final_outputs.update({'gene_selections': gene_selections})
    
    return final_outputs
    
def back_transform(pca_obj, clf_obj):
    return np.dot(pca_obj.components_.T,clf_obj.coef_)

def sanity_check(in_mtx, y, betas, test_gene_num, illustrative, 
                 sanity_check_style, reverse_axes):
    
    if sanity_check_style == 'separate':
        ascores = []
        pscores = []
    else:
        scores = []
    print('running sanity_check')
    try:
        betas = pandas.Series(betas)
    except:
        betas = pandas.Series(betas[:,0])
    outputs = {}
    for num in test_gene_num:
      
        p_chk = betas.sort_values(ascending=False)[:num].index
        a_chk = betas.sort_values(ascending=False)[-num:].index

        pchk_vals = []
        achk_vals = []
        
        for sample in range(in_mtx.shape[0]):
            to_avg = []
            for gene in p_chk:
                to_avg.append(in_mtx[sample,gene])
            if sanity_check_style == 'model':
                pchk_vals.append(np.mean(np.array(to_avg) * betas.loc[p_chk].values))
            else:
                pchk_vals.append(np.mean(to_avg))
        
        for sample in range(in_mtx.shape[0]):
            to_avg = []
            for gene in a_chk:
                to_avg.append(in_mtx[sample,gene])
            if sanity_check_style == 'separate':
                achk_vals.append(np.mean(to_avg))
            elif sanity_check_style == 'model':
                achk_vals.append(np.mean(np.array(to_avg) * betas.loc[a_chk].values))
            else:
                achk_vals.append(np.mean(to_avg) * -1)

        if sanity_check_style != 'separate':
            chk_vals = np.array(pchk_vals) + np.array(achk_vals)
            
        if sanity_check_style == 'separate':
            pr,pp = stats.pearsonr(pchk_vals, y)
            if illustrative:
                plt.close()
                sns.regplot(x=np.array(pchk_vals), y=y, fit_reg=None)
                if reverse_axes:
                    plt.xlabel('expression of anterior direction genes')
                    plt.ylabel('location along hippocampus (anterior = higher)')
                    plt.show()
                    print('anterior %s genes vs. y:  r2 = %s, p = %s \n\n'%(num, pr**2, pp))
                else:
                    plt.xlabel('expression of posterior direction genes')
                    plt.ylabel('location along hippocampus (posterior = higher)')
                    plt.show()
                    print('posterior %s genes vs. y:  r2 = %s, p = %s \n\n'%(num, pr**2, pp))

            ar,ap = stats.pearsonr(achk_vals, y)
            if illustrative:
                plt.close()
                sns.regplot(x=np.array(achk_vals), y=y, fit_reg=None)
                if reverse_axes:
                    plt.xlabel('expression of posterior direction genes')
                    plt.ylabel('location along hippocampus (anterior = higher)')
                    plt.show()
                    print('posterior %s genes vs. y:  r2 = %s, p = %s \n\n'%(num, ar**2, ap))
                else:
                    plt.xlabel('expression of anterior direction genes')
                    plt.ylabel('location along hippocampus (posterior = higher)')
                    plt.show()
                    print('anterior %s genes vs. y:  r2 = %s, p = %s \n\n'%(num, ar**2, ap))
            
        else:
            r,p = stats.pearsonr(chk_vals, y)
            if illustrative:
                plt.close()
                sns.regplot(x=np.array(chk_vals), y=y, fit_reg=None)
                plt.xlabel('expression of A-P axis genes')
                if reverse_axes:
                    plt.ylabel('location along hippocampus (anterior = higher)')
                else:
                    plt.ylabel('location along hippocampus (posterior = higher)')
                plt.show()
                print('posterior and anterior %s genes vs. y:  r2 = %s, p = %s \n\n'%(
                                                                                num, r**2, p))
            
        if sanity_check_style == 'separate':
            if reverse_axes:
                ascores.append(pr**2)
                pscores.append(ar**2)
            else:
                ascores.append(ar**2)
                pscores.append(pr**2)
        else:
            scores.append(r**2)
        if reverse_axes:
            outputs.update({'posterior_genes_%s'%num: a_chk}) 
            outputs.update({'anterior_genes_%s'%num: p_chk})
        else:
            outputs.update({'posterior_genes_%s'%num: p_chk}) 
            outputs.update({'anterior_genes_%s'%num: a_chk})
            
    if len(test_gene_num) > 1:
        if sanity_check_style == 'separate':
            jnk = pandas.concat([pandas.Series(test_gene_num), 
                                 pandas.Series(ascores),
                                pandas.Series(pscores)],axis=1)
            jnk.columns = ['num','a','p']

            plt.close()
            fig,(ax1,ax2) = plt.subplots(2, figsize=(8,10))
            sns.factorplot(x='num', y='a', data=jnk, ax=ax1)
            sns.factorplot(x='num', y='p', data=jnk, ax=ax2)
            ax1.set(xlabel = 'Number of posterior genes', 
                    ylabel ='Explained variance in \nhippocampus a-p gradient')
            ax2.set(xlabel = 'Number of anterior genes', 
                    ylabel ='Explained variance in \nhippocampus a-p gradient')
            plt.show()
        else:
            jnk = pandas.concat([pandas.Series(test_gene_num), 
                                 pandas.Series(scores)
                                ],axis=1)
            jnk.columns = ['num','score']

            plt.close()
            fig,ax1 = plt.subplots(1, figsize=(8,10))
            sns.factorplot(x='num', y='score', data=jnk, ax=ax1)
            ax1.set(xlabel = 'Number of genes', 
                    ylabel ='Explained variance in \nhippocampus a-p gradient')
            plt.show()

    return outputs

def leave_one_x_out(labels, in_X, in_y, clf):

    obsz = []
    predz = []
    label_map = dict(zip(labels.unique(), range(len(labels.unique()))))
    labels = [label_map[x] for x in labels]
    for label in np.unique(labels):
        Tr_samps = [i for i in labels if labels[i] != label]
        Te_samps = [i for i in labels if labels[i] == label]
        X = in_X[Tr_samps,:]
        in_y = y.iloc[Tr_samps]
        mod = clf.fit(X,y)
        pred = mod.predict(in_X[Te_samps,:])
        for x in range(len(pred)):
            obsz.append(y.values[x])
            predz.append(pred[x])
    
    return obsz, predz

def balanced_cv(labels, in_X, in_y, clf, cv):
    
    obsz = []
    predz = []
    tst = model_selection.StratifiedKFold(n_splits=cv, random_state=123)
    label_map = dict(zip(labels.unique(), range(len(labels.unique()))))
    labels = [label_map[x] for x in labels]
    for tr_idx, te_idx in tst.split(in_X, labels):
        X = in_X[tr_idx,:]
        y = in_y.loc[tr_idx]
        mod = clf.fit(X,y)
        pred = mod.predict(in_X[te_idx,:])
        for i in range(len(predz)):
            obsz.append(y.values[x])
            predz.append(pred[i])
            
    return obsz, predz

def run_hipp_connectivity_analysis(ant_cut, post_cut, df, ycol, 
                                   ccol, wdir, gdf, msk, gcx_col, plabs,
                                   del_img = True, diff_img = True,  vrad = 5, vdim = 1,
                                  in_imgs = [], bootstrap = False, n_iter = 100,
                                   hue_vals=[], return_results=False, return_vectors = False,
                                   illustrative=True, save_dir = None, tspace = None):
    
    if len(in_imgs) == 0:
        a_idx = df.loc[[x for x in df.index if df.loc[x,ycol] < ant_cut]].index
        p_idx = df.loc[[x for x in df.index if df.loc[x,ycol] > post_cut]].index
        print('%s maps used for posterior, %s used for anterior'%(len(p_idx),len(a_idx)))
        print('\n')
        print('processing anterior image')
        #aimg = make_mean_img(df.loc[a_idx,ccol].tolist(),wdir,del_img, 'ant')
        aimg = make_mean_img(df.loc[a_idx,ccol].tolist(),tspace)
        print('processing posterior image')
        pimg = make_mean_img(df.loc[p_idx,ccol].tolist(),tspace)

        imgs = {'post': pimg, 'ant': aimg}
    
        if diff_img:
            print('running analysis')
            diff_img = pimg - aimg
            imgs.update({'diff': diff_img})
            res, vectors = run_gvfcx_analysis(diff_img, gdf, msk, vrad, vdim, gcx_col, plabs, 
                                     bootstrap, n_iter, hue_vals, illustrative)
        else:
            print('running posterior analysis')
            res, vectors = run_gvfcx_analysis(pimg, gdf, msk, vrad, vdim, gcx_col,  plabs, 
                                     bootstrap, n_iter, hue_vals, illustrative)
            print('running anterior analysis')
            res, vectors = run_gvfcx_analysis(aimg, gdf, msk, vrad, vdim, gcx_col, plabs, 
                                     bootstrap, n_iter, hue_vals, illustrative)
        if os.path.isdir(save_dir):
        	print('saving images')
        	for label, img in imgs.items():
        		tosave = ni.Nifti1Image(img,ni.load(tspace).affine)
        		tosave.to_filename(os.path.join(save_dir,'cnx_%s_img'%(label)))

    else:
        if diff_img:
            diff_img = ni.load(in_imgs[1]).get_data() - ni.load(in_imgs[0]).get_data()
            res, vectors = run_gvfcx_analysis(diff_img, gdf, msk, vrad, vdim, gcx_col, plabs, 
                                     bootstrap, n_iter, hue_vals, illustrative)
        else:
            for img in in_imgs:
                print('running analysis for image',img)
                dat = ni.load(img).get_data()
                res, vectors = run_gvfcx_analysis(dat, gdf, msk, vrad, vdim, gcx_col, plabs, 
                                         bootstrap, hue_vals, illustrative)
    
    if return_results and return_vectors:
        return res, vectors
    elif return_results and not return_vectors:
        return res
    elif return_vectors and not return_results:
        return vectors
    
# def make_mean_img(scans, wdir, del_img, lab, tspace)
def make_mean_img(scans, tspace):
    
    print('making mean image')
    img = ni.concat_images(scans)
    x,y,z,q,t = img.shape
    mat = img.get_data().reshape(x,y,z,t)
    mimg = ni.Nifti1Image(mat.mean(axis=3),img.affine)
    print('resampling')
    fimg = image.resample_to_img(mimg, tspace).get_data()
    # fnm = os.path.join(wdir,'del_%s_img.nii'%lab)
    # mimg.to_filename(fnm)
    
    # print('moving to template space')
    # #mni = '/Users/jakevogel/Science/tau/MNI152_T1_1mm_brain.nii'
    # mni = '/home/users/jvogel/Science/templates/templates/MNI152_T1_1mm_brain.nii'
    # #tfm = '/Users/jakevogel/Science/AHBA/cx_maps/2_to_1_mm_tfm'
    # tfm = '/home/users/jvogel/Science/templates/tfms/2_to_1mm_MNI.tfm'
    # nfnm = os.path.join(wdir,'%s_img.nii'%lab)
    # #os.system('flirt -in %s -ref %s -applyxfm -init %s -out %s'%(fnm,mni,tfm,nfnm))
    # os.system('fsl5.0-flirt -in %s -ref %s -applyxfm -init %s -out %s'%(fnm,mni,tfm,nfnm))
    # os.remove(fnm)
    # nfnm = nfnm+'.gz'
    # fimg = ni.load(nfnm).get_data()
    # if del_img:
    #     os.remove(nfnm)
    
    return fimg

def run_gvfcx_analysis(img, gdf, msk, vrad, vdim, gcx_col, plabs,
                       bootstrap, n_iter, hue_vals, illustrative):
    
    if type(vrad) != list:
        vrad = [vrad]
    res = pandas.DataFrame(index=vrad,columns=['r2','p'])
    for vs in vrad:
        g_cx = []
        f_cx = []
        if len(hue_vals) > 0:
            hvs = []

        for i,row in gdf.iterrows():
            #coord = convert_coords([row['mni_x'], row['mni_y'], row['mni_z']], 'xyz')
            coord = convert_coords([row['mni_nlin_x'], row['mni_nlin_y'], row['mni_nlin_z']], 
                                   'xyz', vdim)
            coord = [round(x) for x in coord]
            #if msk[coord[0],coord[1],coord[2]] != 0:
            if msk[coord[0],coord[1],coord[2]] > 0:
                xs,ys,zs = make_sphere(coord, vs)
                val = img[xs,ys,zs]
                f_cx.append(val.mean())
                g_cx.append(row[gcx_col])
                if len(hue_vals) > 0:
                    hvs.append(hue_vals[i])

        if len(hue_vals) == 0:
            if illustrative:
                plt.close()
                sns.regplot(np.array(g_cx), np.array(f_cx))
                plt.title(plabs[0])
                plt.xlabel(plabs[1])
                plt.ylabel(plabs[2])
                plt.show()
        else:
            if illustrative:
                jnk = pandas.DataFrame(index=range(len(g_cx)),columns = plabs[1:])
                jnk.loc[:,plabs[1]] = g_cx
                jnk.loc[:,plabs[2]] = f_cx
                jnk.loc[:,'sample expression cluster'] = hvs
                sns.lmplot(x=plabs[1],y=plabs[2],hue='sample expression cluster',data=jnk)
                plt.show()

        r, p = stats.pearsonr(np.array(g_cx), np.array(f_cx))
        print('standard statistics: r2 = %s, p = %s'%(r**2,p))
        res.loc[vs,['r2','p']] = [r**2,p]
        
        vectors = {'g_vector': g_cx, 'cx_vector': f_cx}
        
        if bootstrap == 'permute':
            distr = []
            for i in range(n_iter):
                distr.append(stats.pearsonr(np.random.permutation(np.array(g_cx)),
                                            np.array(f_cx)
                                           )[0]**2)
            distr = np.array(distr)
            p = (n_iter - len(distr[distr<r**2])) * (1/n_iter)
            ci_l = sorted(distr)[int(n_iter*0.05)]
            ci_u = sorted(distr)[int(n_iter*0.95)]
            mn = np.mean(distr)
            print('Occurence greater than chance: p = %s (chance r2 = %s [%s,%s])'%(
                                                                p, mn, ci_l, ci_u))
            res.loc[vs,'manual_FDR'] = p
            res.loc[vs,'chance_r2'] = mn
            res.loc[vs,'ci_l'] = ci_l
            res.loc[vs,'ci_u'] = ci_u

        elif bootstrap == 'bootstrap':
            # NEED TO MAKE THIS FASTER SO I CAN DO IT AT LEAST 100 TIMES.
            # AS OF NOW, IT TAKES LIKE >15 seconds to run 1
            distr = []
            for n in range(n_iter):
                r_cx = []
                possible_coords = np.where(msk!=0)
                rand_coords = np.random.permutation(range(len(possible_coords[0])))[:len(g_cx)]
                for i in rand_coords:
                    rand_coord = [possible_coords[0][i], 
                                  possible_coords[1][i], 
                                  possible_coords[2][i]]
                    xs,ys,zs = make_sphere(rand_coord, vs)
                    r_cx.append(img[xs,ys,zs].mean())
                distr.append(stats.pearsonr(np.array(g_cx), np.array(r_cx))[0]**2)
            distr = np.array(distr)
            p = (n_iter - len(distr[distr<r**2])) * (1/n_iter)
            ci_l = sorted(distr)[int(n_iter*0.05)]
            ci_u = sorted(distr)[int(n_iter*0.95)]
            mn = np.mean(distr)
            print('Occurence greater than chance: p = %s (chance r2 = %s [%s,%s])'%(
                                                                p, mn, ci_l, ci_u))
            res.loc[vs,'manual_FDR'] = p
            res.loc[vs,'chance_r2'] = mn
            res.loc[vs,'ci_l'] = ci_l
            res.loc[vs,'ci_u'] = ci_u

    return res, vectors

def bootstrap_model(gene_set, all_genes, y, n_iterations=10, 
                   bs_type='bootstrap', inner_set=100, 
                   smallset=False, random_state = 123):
    
    np.random.seed(random_state)

    results = []
    for i in range(n_iterations):
        if bs_type == 'bootstrap':
            rand_samp = np.random.randint(0,len(gene_set.index),len(gene_set))
            X = gene_set.loc[gene_set.index[rand_samp]].values.T
        elif bs_type == 'null':
            rand_samp = np.random.randint(0,len(all_genes),len(gene_set))
            X = all_genes.loc[all_genes.index[rand_samp]].values.T
        elif bs_type == 'inner_set':
            rand_samp = np.random.randint(0,len(gene_set.index),inner_set)
            X = gene_set.loc[gene_set.index[rand_samp]].values.T
        if smallset:
            jnk = PCA_LR_pipeline(X, y, pca=None,
                                clf = linear_model.LassoCV(cv=10, max_iter=5000),
                                cv_strategy='score', illustrative=False,
                                  sanity_check_style='model')
        else:
            jnk = PCA_LR_pipeline(X, y, cv_strategy='score', illustrative=False, 
                                 sanity_check_style ='model')
        results.append(jnk['CV_scores'])
        
        print('>>>>finished round %s<<<<'%i)
    
    return results

def feature_explainer_pipeline(gene_set, y, probedf, tmp_dir = None,
                               clf = RandomForestRegressor(random_state=123,n_estimators=1000),
                               kfold = model_selection.KFold(random_state=123, shuffle=True, n_splits=10),
                               outdir = None, outnm = 'FeatureExplainer',
                               ylim=(0,3.9), nm_thresh = 0.3,
                              verbose = False):
    
    if not tmp_dir:
        tmp_dir = os.path.join(os.getcwd(),'tmp')
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

    trial = 0
    all_results = {}
    names = gene_set.columns
    
    for tri, tei in kfold.split(gene_set):
        print('running trial',trial)
        train = gene_set.loc[gene_set.index[tri]]
        test = gene_set.loc[gene_set.index[tei]]
        labels_train = y[tri]
        labels_test = y[tei]
        clf.fit(train, labels_train)
        explainer = lime.lime_tabular.LimeTabularExplainer(train.values, 
                                                       feature_names=names.values, 
                                                       class_names=['Position'], 
                                                       verbose=True, mode='regression')
        cols = ['score','pred','loc_pred']
        results = pandas.DataFrame(index = range(len(test)), columns = cols)
        for i in range(len(test)):
            exp = explainer.explain_instance(test.values[i], clf.predict, num_features=len(names))
            jnk = exp.as_list()
            results.loc[i,'score'] = exp.score
            results.loc[i,'pred'] = exp.predicted_value
            results.loc[i,'loc_pred'] = exp.local_pred[0]
            for x in range(len(names)):
                if '>' in jnk[x][0]:
                    splt = jnk[x][0].split(' >')
                    if len(splt) == 2:
                        gene = splt[0].strip()
                    else:
                        gene = splt[1].strip()
                else:
                    splt = jnk[x][0].split(' <')
                    if len(splt) == 3:
                        gene = splt[1].strip()
                    else:
                        gene = splt[0].strip()
                scr = abs(jnk[x][1])
                results.loc[i,gene] = scr
            results.loc[:,'actual'] = labels_test.values
            results.loc[:,'error'] = results.pred - results.actual
        all_results.update({trial: results})
        
        outfl = os.path.join(tmp_dir, '%s_Trial%s.csv'%(outnm,trial))
        results.to_csv(outfl)
        trial += 1
        print(results.head())
    
    plotr, cols = assemble_FE_data(tmp_dir, gene_set)
    
    plot_FE_data(plotr, cols, probedf, ylim, nm_thresh, outnm, outdir)
    
    return plotr

def assemble_FE_data(tmp_dir, gene_set):
    all_results = {}
    sheetz = sorted(glob(os.path.join(tmp_dir, '*.csv')))
    for sheet in sheetz:
        trial = sheet.split('_trial')[-1].split('.')[0]
        jnk = pandas.read_csv(sheet, index_col=0)
        all_results.update({trial: jnk[jnk.columns[3:-2]]})
    
    holder = []
    for i,rdf in all_results.items():
        holder.append(rdf)
    combined_results = pandas.concat(holder)
    cols = combined_results.columns[:len(gene_set)]
    plotr = pandas.DataFrame(index = range(170*len(cols)), columns = ['score','gene'])
    flt = combined_results[cols].values.flatten()
    plotr.loc[:,'score'] = flt
    plotr.loc[:,'gene'] = cols.tolist()*170
    
    return plotr, cols

def plot_FE_data(plotr, cols, probedf, ylim, nm_thresh, outnm, outdir = None):
    
    plt.close()
    sns.set_context('notebook')
    sns.set_style('white')
    plt.subplots(figsize = (14,10))
    g = sns.barplot(x='gene',y='score',data=plotr)
    #g.set_xticklabels([g.get_xticklabels(),rotation=90])
    g.set_xticklabels(['' for x in range(len(g.get_xticklabels()))])
    g.set_ylim(ylim)
    g.set_xlabel('Gene', size=25)
    g.set_ylabel('Contribution', size=25)

    for tick in g.get_yticklabels():
        tick.set_fontsize(20)
    for i,gene in enumerate(cols):
        mn = plotr[plotr.gene==gene]['score'].mean()
        sd = plotr[plotr.gene==gene]['score'].std()
        xval = g.patches[i].get_x()
        ht = g.patches[i].get_height()
        wd = g.patches[i].get_width()
        if mn > nm_thresh:
            print(gene,xval,i)
            g.text(xval+wd,ht+(sd*0.25),probedf.loc[int(gene),'gene_symbol'],color='black', ha="right", fontsize=25)
    fig = g.get_figure()
    if outdir:
        outfl = os.path.join(outdir, '%s_FEimg.pdf'%outnm)
        fig.savefig(outfl, bbox_inches='tight')
    plt.show()

def structural_connectivity_analysis(input_img, df, col, ant_cut, post_cut, vdim, 
                                     mask_thr=0.2, outdir = None, outname='strucx'):
    
    print('initializing')
    i4d = input_img.get_data()
    avg_image = i4d.mean(3)
    mask = np.zeros_like(avg_image)
    mask[avg_image>mask_thr] = 1
    mskr = input_data.NiftiMasker(ni.Nifti1Image(mask,input_img.affine))
    i2d = mskr.fit_transform(input_img)
    
    print('creating anterior connectivity map')
    antdf = df[df[col]>=ant_cut][['mni_nlin_x','mni_nlin_y','mni_nlin_z']]
    ant_mtx = get_structural_connectivity(antdf, i4d, i2d, vdim)
    ant_image = mskr.inverse_transform(ant_mtx).get_data().mean(3)
    
    print('creating posterior connectivity map')
    postdf = df[df[col]<=post_cut][['mni_nlin_x','mni_nlin_y','mni_nlin_z']]
    post_mtx = get_structural_connectivity(postdf, i4d, i2d, vdim)
    post_image = mskr.inverse_transform(post_mtx).get_data().mean(3)
    
    diff_image = ni.Nifti1Image((ant_image - post_image), input_img.affine)
    ant_image = ni.Nifti1Image(ant_image,input_img.affine)
    post_image = ni.Nifti1Image(post_image,input_img.affine)
    output = {'anterior': ant_image,
             'posterior': post_image,
             'difference': diff_image}

    if outdir:
        for lab, image in output.items():
            image.to_filename(os.path.join(wdir,'%s_%s'%(outname,lab)))
    
    return output
        
def get_structural_connectivity(cdf, i4d, i2d, vdim, embedded=True):
    
    rmat = np.zeros((len(cdf),i2d.shape[1]))
    for i,c in enumerate(cdf.index):
        coord = [int(round(x)) for x in convert_coords(cdf.loc[c].values, 'xyz', vdim)]
        print('computing %s of %s connectivity maps'%(i+1,len(cdf)))
        cvec = i4d[coord[0],coord[1],coord[2],:]
        rs = [stats.pearsonr(cvec,i2d[:,x])[0] for x in range(i2d.shape[1])]
        rmat[i,:] = rs
    if embedded:
        rvec = rmat.mean(0)
        r_mtx = np.repeat(np.array(rvec)[:, np.newaxis], i2d.shape[0], axis=1).T

        return r_mtx
    else:
        return rmat

def cognitive_metaanalysis_pipeline(scans=None, gdf = None, target_col = None, 
                                    metares = None, labels = None,
                                    min_samples = 500, figtype = 'horizontal', 
                                    return_scans = False, savefig = '',
                                   ATLabs = ['T65','T60','T17','T20','T90'],
                                   PMLabs = ['T56','T24','T40','T14','T75']):

    '''
    Depending on inputs, this script will load neurosynth LDA topic maps, find 
    overlap samples and maps, calculate the average HAGS (or whatever) of 
    samples that overlap with the map, and plot a figure summarizing.

    scans = If passing metares, scans can be left as None. Otherwise, either a
    list of paths pointing to neurosynth LDA images, or a 4D array representing
    all of these images.

    gdf = If passing metares, gdf cane be left as None. Otherwise, a spreadsheet
    having one row for each brain sample and (at least) columns indicating 
    sample coordinates, along with target_col.

    target_col = If passing metares, gdf cane be left as None. Otherwise, a 
    string label corresponding to the column in gdf where HAGS or whatever
    sample-wise value you would like to analyze is.

    metares = If None, you must pass scans, gdf, target_col and labels 
    instead. Otherwise, this should be a samples X maps array, such that
    each cell has a value corresponding to the sample's target value (e.g. HAGS) 
    for map that the sample overlaps with, and an NaN for each map it does not 
    overlap with.

    labels = If passing metares, labels can be left as None. Otherwise, a list 
    of string labels the same length as scans (or the the 4th dimension of 
    scans) indicating the label of each topic map.
    
    min_samples = A threshold indiciating the minimum number of samples
    overlapping with a map that is sufficient to analzye that map. A map
    overlapping with few samples will likely be biased.

    fig_type = What kind of figure to produce:
    * 'horizontal' = Topic x Value
    * 'vertical' = Value x Topic
    * None = Do not generate a figure at all.

    return_scans = if you passed a list of scans, this will return the 4D data
    corresponding to those scans. Note, this can be a very large array.

    savefig = A path where you would like to save the figure created in this
    script. Note that the path should include the desired figure extension
    (e.g. .pdf, .png, etc). If None, no figure will be save.

    ATLabs / PMLabs = A string containing topic labels for topics you wish to
    label as AT or PM in the figure.  
    '''
    
    data_passed = False
    
    if type(scans) == type(None) and type(metares) == type(None):
        raise IOError('a value must be passed for scans or metares')
    
    if type(scans) != type(None) and type(metares) != type(None):
        raise IOError('a value must be passed for scans *or* metares')
    
    if type(metares) == type(None) and any([type(x)==type(None) for x in [gdf, target_col, labels]]):
        raise IOError('if not passing metares, you must pass a value for gdf, target_col and labels')
    
    if type(metares) == type(None):
        data_passed = True
        if type(scans) == list:
            if not os.path.isfile(scans[0]):
                raise IOError('Assuming you passed a list of scan paths, but I couldnt locate the first one...')
            else:
                print('>>>loading data<<<')
                allmetas = image.load_img(scans).get_data()
        else:
            if type(scans) != np.ndarray:
                raise IOError('scans must be 4D array or list of paths to neurosynth maps')
            elif len(scans) < 4:
                raise IOError('scans must be a 4D array, but you passed an array with %s dimensions'%len(scans))
            else:
                allmetas = scans
        
        print('>>>computing overlap of each sample with each map<<<')
        metares_a = compute_sample_overlap(gdf, target_col, allmetas, labels)
        
        map_sizes = []
        for i in range(allmetas.shape[-1]):
            jnk = allmetas[:,:,:,i]
            map_sizes.append(len(jnk[jnk>0]))
        # calculate the number of samples found within each map
            
        if not return_scans:
            del(allmetas)
    else:
        metares_a = metares
        map_sizes = None
    
    map_hits = []
    for col in metares_a.columns:
        jnk = metares_a[col]
        map_hits.append(len([x for x in jnk.values if pandas.notnull(x)]))

    print('>>>Calculating stuff<<<')
    res_sum, metaresb = compute_results_summary(metares_a, map_sizes, map_hits)

    # parse topic number and add system to relevant topics
    for top in res_sum.index:
        tnum = top.split('_')[0]
        if tnum in ATLabs:
            res_sum.loc[top,'system'] = 'AT'
        elif tnum in PMLabs:
            res_sum.loc[top,'system'] = 'PM'
    res_sum2 = res_sum.loc[metaresb.columns]
    
    # removing maps without enough data
    print('>>>removing data with less than %s samples overlapping'%min_samples)
    goodlabs = res_sum[res_sum.map_hits>min_samples].index
    len(goodlabs)
    metares400 = metares_a[goodlabs]
    metares400 = metares400[metares400.mean().sort_values().index]
    res_sum4 = res_sum.loc[metares400.columns]
    print('%s maps remaining'%len(res_sum4))
    
    # plotting
    if figtype != None:
        print('>>>plotting<<<')
        create_metacog_plot(metares400, res_sum4, figtype, savefig)
    
    # preparing results
    all_results = {'res_sum': res_sum4,
              'metares_a': metares_a,
              'metares%s'%min_samples: metares400}
    if data_passed:
        if return_scans:
            all_results.update({'allmetas': allmetas})
    
    return all_results 
    
    
def compute_sample_overlap(gdf, col, allmetas, labels):

    metares_a = pandas.DataFrame(index=gdf.index,columns = labels)
    # for each coordinate
    for i,row in gdf.iterrows():
        # draw a 5mm cube around the coordinate
        xs,ys,zs = make_sphere(convert_coords([row['mni_nlin_x'],row['mni_nlin_y'],row['mni_nlin_z']],'xyz',2), 3)
        # for each image
        for img in range(allmetas.shape[-1]):
            # if there is any data inside the cube (i.e. if the sample falls within the map)
            if allmetas[xs,ys,zs,img].mean() > 0:
                # add data for this sample to spreadsheet (i.e. mark this sample as within the map)
                metares_a.loc[i,labels[img]] = row[col]
        if i%100 == 0:
            print('finished %s of %s'%(i,len(gdf)))

    return metares_a

def compute_results_summary(metares_a, map_sizes, map_hits):

    # create a spreadsheet summarizing all of the relevant information

    # create spreadsheet
    res_sum = pandas.DataFrame(index = metares_a.columns)
    res_sum.loc[:,'mean'] = metares_a.mean().values # average HAGS index of samples within map
    res_sum.loc[:,'sem'] = metares_a.sem().values # SEM of HAGS index of samples within map
    if map_sizes:
        res_sum.loc[:,'map_size'] = map_sizes # size (in voxels) of each map
    res_sum.loc[:,'map_hits'] = map_hits # number of samples falling within each map

    # Just in case someone asks for it, create a new mean weighted by number of samples in each map
    cols = ['mean','sem']
    for col in cols:
        wtd = (res_sum[col].values * res_sum.map_hits.values
              ) / (res_sum[col].values + res_sum.map_hits.values)
        res_sum.loc[:,'wtd_%s'%col] = wtd

    # How many (of the 100 maps) actually overlap with some samples?
    goodlabs = metares_a.mean().dropna().index
    print('%s maps did not overlap with any samples'%len(goodlabs))

    # Get rid of those that don't and sort data
    metaresb = metares_a[goodlabs]
    metaresb = metaresb[metaresb.mean().sort_values().index]
    
    return res_sum, metaresb

def create_metacog_plot(metares400, res_sum4, figtype, savefig):

    sns.set_context('notebook',font_scale=2)

    # get mean and SEM data
    means = (metares400.mean()).tolist()
    stds = metares400.sem().tolist()
    cis = [((means[x]-stds[x]),(means[x]+stds[x])) for x in range(len(means))]

    # get labels
    xlabs = ['%s: %s / %s'%(x.split('_')[0],
                    x.split('_')[1],
                    x.split('_')[2]) for x in metares400.mean().dropna().index.tolist()]

    # get the range of the confidence interval
    y_r = [((cis[i][0] - means[i]) + (means[i] - cis[i][1]))/2 for i in range(len(cis))]

    # color code AT and PM bars
    colors = ['red' if res_sum4.loc[x,'system'] == 'AT' else 'blue' if res_sum4.loc[x,'system'] == 'PM' else 'gray' for x in metares400.columns]

    # Make a barplot
    if figtype == 'horizontal':
        plt.close()
        sns.set_style('white')
        fig, ax = plt.subplots(figsize=(16,6))
        fig = plt.bar(range(len(means)), means, yerr=y_r, alpha=0.5, align='center', color = colors,
                     )
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(xlabs, rotation=90)
        ax.set_ylabel('HAGS Index')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)
    
    elif figtype == 'vertical':
        plt.close()
        sns.set_style('white')
        fig, ax = plt.subplots(figsize=(6,16))
        fig = plt.barh(range(len(means)), means, xerr=y_r, alpha=0.5, align='center', color = colors,
                     )
        ax.set_yticks(range(len(means)))
        ax.set_yticklabels(metares400.mean().dropna().index.tolist())
        ax.set_xlabel('A-P Axis Genomic Similarity\n(Anterior high)')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12) 
    else:
        raise IOError('figtype must be set to "horizontal" or "vertical"')
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    
    plt.show()
    