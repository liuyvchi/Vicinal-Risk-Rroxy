import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

import scipy.stats
import seaborn as sns
from scipy.stats import norm


test_acc = []
test_acc2 = []
test_loss = []
p_overlap_mean = []
p_overlap_mea2 = []
sp_mean = []
invariance_mean = []
aus_invariance_mean = []
integral_mean = []
aus_guided = []
ensemble = []
wj_score = []
colors_options = ['blue', 'red', 'green', 'orange']
colors = []

p_overlap = []
pn_overlap = [] 

aus_overlap = [] 
aus_overlap2 = [] 
aus_PNoverlap = []
aus_PNoverlap2 = []
aus_vrm = []

ensemble_P_halfvrm = []
ensemble_Poverlap = []
ensemble_PNoverlap = [] 
ensemble_PNoverlap2 = [] 
ensemble_overlap = []
ensemble_overlap2 = []
ensemble_vrm = []

mixup_overlap = []
mixup_overlap2 = []
mixup_PNoverlap = []
mixup_PNoverlap2 = []
mixup_vrm = []
mixup_halfvrm = []

aus_base = [] 
wj_score = [] 
wj_score2 = [] 
I_overlap_score = [] 
pnI_overlap_socre = []
ausP_EI = []
ausP_EI2 = []
ausPN_EI = []
wjPlus_score = []
wjPlus_score2 = []
self_vrm_mean = []
self_vrm_mean2 = []

EI_randomPair = []
grey_vrm = []
grey_vrmV2 = []
grey_vrmV3 = []
ensemble_s1s2 = []
ensemble_Ps1s2 = []
ensemble_agreement = []
EI_vrm_1 = []
EI_vrm_all = []
vrm_pairs_1 = []
vrm_pairs_all = []
pair_EI_vrm = []
AC = []
AC_vrp = []


def add(dir, dir2):
    tests = os.listdir(dir)
    tests2 = os.listdir(dir2)

    for test in tests:
        try:
            dic = np.load(os.path.join(dir, test), allow_pickle='TRUE')
            dic2 = np.load(os.path.join(dir2, test), allow_pickle='TRUE')
        except:
            continue
        # if dic.item().get('ensemble_overlap') <= 0.98:
        #     continue


        acc = dic.item().get('test_acc')
        acc2 = dic2.item().get('test_acc')
        # if acc < 0.05 :
        #     print(test)
        #     print(acc)
        #     continue
        # if math.isnan(dic.item().get('aus_PNoverlap')):
        #     print(test)
        #     continue
        
        test_acc.append(norm.cdf(acc))
        test_acc2.append(norm.cdf(acc2))

        # I_overlap_score.append(dic.item()['I_overlap_score']) 
        # aus_overlap2.append(dic2.item()['aus_overlap']) 
        # aus_PNoverlap.append(dic.item().get('aus_PNoverlap'))
        # aus_PNoverlap2.append(dic2.item().get('aus_PNoverlap'))
        # aus_vrm.append(dic.item()['aus_vrm']) 

        # ensemble_P_halfvrm.append(dic.item().get('ensemble_P_halfvrm'))
        # ensemble_Poverlap.append(dic.item().get('ensemble_Poverlap'))
        # ensemble_overlap.append(dic.item().get('ensemble_overlap'))
        # ensemble_overlap2.append(dic2.item().get('ensemble_overlap'))
        # ensemble_vrm.append(dic.item()['ensemble_vrm']) 
        # mixup_halfvrm.append(dic.item()['mixup_halfvrm']) 
        # mixup_vrm.append(dic.item()['mixup_vrm']) 

        wj_score.append(dic.item().get('wj_score')) 
        wj_score2.append(dic2.item().get('wj_score')) 

        EI_randomPair.append(dic.item().get('EI_randomPair')) 
        grey_vrm.append(dic.item().get('grey_vrm')) 
        grey_vrmV2.append(dic.item().get('grey_vrmV2')) 
        grey_vrmV3.append(dic.item().get('grey_vrmV3'))
        # ensemble_s1s2.append(dic.item().get('ensemble_s1s2')) 
        # ensemble_Ps1s2.append(dic.item().get('ensemble_Ps1s2')) 

        EI_vrm_1_node = dic.item().get('EI_vrm_1')
        EI_vrm_1.append(EI_vrm_1_node)
        EI_vrm_all.append(dic.item().get('EI_vrm_all'))
        vrm_pairs_all_node = dic.item().get('vrm_pairs_all')
        vrm_pairs_all.append(vrm_pairs_all_node)
        vrm_pairs_1_node = dic.item().get('vrm_pairs_1')
        vrm_pairs_1.append(vrm_pairs_1_node)

        pair_EI_vrm.append((EI_vrm_1_node*vrm_pairs_1_node))

        ensemble_agreement.append(dic.item().get('ensemble_agreement'))

        AC.append(dic.item().get('AC'))
        AC_vrp.append(dic.item().get('AC_vrp'))

# dir2 = '/home/liuyuchi/afs_get/FER_measure/V7/pretrain_RafRaf_V7'
# dir = '/home/liuyuchi/afs_get/FER_measure/V7/pretrain_RafAffect_V7'
# add(dir, dir2)
# dir2 = '/home/liuyuchi/afs_get/FER_measure/V7/nopre_RafRaf_V7'
# dir = '/home/liuyuchi/afs_get/FER_measure/V7/nopre_RafAffect_V7'
# add(dir, dir2)
dir = '/root/paddlejob/workspace/env_run/output/imagenet_a_out_rotation/'
dir2 = '/root/paddlejob/workspace/env_run/output/imagenet_a_out_rotation/'
add(dir, dir2)
# dir = '/home/liuyuchi/afs_get/FER_measure/vrm_v1/imgenet_a_vrm'
# dir2 = '/home/liuyuchi/afs_get/FER_measure/V7/nopre_AffectAffect_V7'
# add(dir, dir2)



fig = plt.figure(figsize=plt.figaspect(0.8))
plt.rcParams["font.family"] = "Times New Roman"

ax1 = fig.add_subplot(1,1,1)
# ax1.xaxis.tick_top()
ax1.set_facecolor('whitesmoke')
ax1.grid(True, linestyle="--", alpha=0.5, c='white')
x = np.array(wj_score)
y = np.array(test_acc)
sns.regplot(x=x, y=y, ax=ax1, scatter=True, color='grey', scatter_kws={'alpha':0.3, 's':15})
# ax1.set_ylim(y.min(), y.max())
plt.tick_params(width=3, labelsize=15)
# ax1.scatter(x, y, c='blue', s=1)
# ax1.set_ylabel('')
# ax1.set_xlabel('EI with rotation ')

print(scipy.stats.pearsonr(x, y)[0])
print(scipy.stats.spearmanr(x, y)[0])
pearsonr = scipy.stats.pearsonr(x, y)[0]
spearmanr = scipy.stats.spearmanr(x, y)[0]
print('\n')

#the second
ax2 = ax1.twiny()

x2 = np.array(EI_randomPair)
y2 = np.array(test_acc)
sns.regplot(x=x2, y=y2, ax=ax2, scatter=True, color='green', scatter_kws={'alpha':0.5, 's':15})
ax2.
# ax2.xaxis.tick_bottom()
# ax2.set_ylim(y2.min(), y2.max())
plt.tick_params(width=3, labelsize=15)

# ax1.scatter(x, y, c='blue', s=1)
# ax1.set_ylabel('')
# ax1.set_xlabel('EI with rotation ')

print(scipy.stats.pearsonr(x2, y2)[0])
print(scipy.stats.spearmanr(x2, y2)[0])
pearsonr = scipy.stats.pearsonr(x2, y2)[0]
spearmanr = scipy.stats.spearmanr(x2, y2)[0]
print('\n')




# plt.subplots_adjust(wspace=0.3, hspace=0.5)


# plt.savefig('./AffectRaf_measureV6_acc.jpg', format='jpg', bbox_inches='tight')
plt.savefig('./plot/imagenet_a_EI_rotation.jpg', format='jpg', bbox_inches='tight')
plt.savefig('./plot/imagenet_a_EI_rotation.pdf', format='pdf', bbox_inches='tight')



