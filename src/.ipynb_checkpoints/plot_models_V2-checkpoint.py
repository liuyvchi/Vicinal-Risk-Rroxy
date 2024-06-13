import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

import scipy.stats
import seaborn as sns


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
        
        test_acc.append(acc)
        test_acc2.append(acc2)

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
dir = '/root/paddlejob/workspace/env_run/output/imagenet_a_out_rotation'
dir2 = '/root/paddlejob/workspace/env_run/output/imagenet_a_out_rotation'
add(dir, dir2)
# dir = '/home/liuyuchi/afs_get/FER_measure/vrm_v1/imgenet_a_vrm'
# dir2 = '/home/liuyuchi/afs_get/FER_measure/V7/nopre_AffectAffect_V7'
# add(dir, dir2)

fig = plt.figure(figsize=plt.figaspect(0.4))

ax1 = fig.add_subplot(2,4,1)
x = np.array(wj_score)
y = test_acc
sns.regplot(x=x, y=y, ax=ax1, scatter=True, color='green', scatter_kws={'s':1})
# ax1.scatter(x, y, c='blue', s=1)
ax1.set_ylabel('')
ax1.set_xlabel('Effective Inviarance')
print(scipy.stats.pearsonr(x, y)[0])
print(scipy.stats.spearmanr(x, y)[0])
pearsonr = scipy.stats.pearsonr(x, y)[0]
spearmanr = scipy.stats.spearmanr(x, y)[0]
ax1.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax2 = fig.add_subplot(2,4,2)
x = AC
y = test_acc
sns.regplot(x=x, y=y, ax=ax2, scatter=True, color='green', scatter_kws={'s':1})
# ax2.scatter(aus_PNoverlap, test_acc, c='blue', s=1)
# ax4.scatter(aus_invariance_mean, test_loss, c='red', s=1.2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax2.set_ylabel('')
ax2.set_xlabel('AC')
# ax4.legend()
index_without_nan = [i for i in range(len(x)) if np.isnan(x[i]) == False]
print(scipy.stats.pearsonr([x[i] for i in index_without_nan], [y[i] for i in index_without_nan])[0])
print(scipy.stats.spearmanr([x[i] for i in index_without_nan], [y[i] for i in index_without_nan])[0])
pearsonr = scipy.stats.pearsonr([x[i] for i in index_without_nan], [y[i] for i in index_without_nan])[0]
spearmanr = scipy.stats.spearmanr([x[i] for i in index_without_nan], [y[i] for i in index_without_nan])[0]
ax2.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax3 = fig.add_subplot(2,4,3)
x=np.array(EI_randomPair)
y=test_acc
sns.regplot(x=x, y=y, ax=ax3, scatter=True, color='green', scatter_kws={'s':1})
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax3.set_ylabel('')
ax3.set_xlabel('EI_randomPair')
print(scipy.stats.pearsonr(x, y)[0])
print(scipy.stats.spearmanr(x, y)[0])
pearsonr = scipy.stats.pearsonr(x, y)[0]
spearmanr = scipy.stats.spearmanr(x, y)[0]
ax3.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax4 = fig.add_subplot(2,4,4)
x= np.array(AC_vrp)
y= test_acc
ax4.set_ylabel('')
ax4.set_xlabel('AC_vrp')
index_without_nan = [i for i in range(len(x)) if np.isnan(x[i]) == False]
print(scipy.stats.pearsonr([x[i] for i in index_without_nan], [y[i] for i in index_without_nan])[0])
print(scipy.stats.spearmanr([x[i] for i in index_without_nan], [y[i] for i in index_without_nan])[0])
x= [x[i] for i in index_without_nan]
y= [y[i] for i in index_without_nan]
sns.regplot(x=x, y=y, ax=ax4, scatter=True, color='green', scatter_kws={'s':1})

pearsonr = scipy.stats.pearsonr([x[i] for i in index_without_nan], [y[i] for i in index_without_nan])[0]
spearmanr = scipy.stats.spearmanr([x[i] for i in index_without_nan], [y[i] for i in index_without_nan])[0]
ax4.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax5 = fig.add_subplot(2,4,5)
x=np.array(vrm_pairs_1)
y=test_acc
sns.regplot(x=x, y=y, ax=ax5, scatter=True, color='green', scatter_kws={'s':1})
# ax6.scatter(pnI_overlap_socre, test_acc, c='blue', s=1)
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax5.set_ylabel('')
ax5.set_xlabel('vrm_pairs_1')
print(scipy.stats.pearsonr(x, y)[0])
print(scipy.stats.spearmanr(x, y)[0])
pearsonr = scipy.stats.pearsonr(x, y)[0]
spearmanr = scipy.stats.spearmanr(x, y)[0]
ax5.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax6 = fig.add_subplot(2,4,6)
x = np.array(vrm_pairs_all)
y = np.array(test_acc)
sns.regplot(x=x, y=y, ax=ax6, scatter=True, color='green', scatter_kws={'s':1})
# ax6.scatter(pnI_overlap_socre, test_acc, c='blue', s=1) 
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax6.set_xlabel('vrm_pairs_all')
ax6.set_ylabel('')
print(scipy.stats.pearsonr(x, y)[0])
print(scipy.stats.spearmanr(x, y)[0])
pearsonr = scipy.stats.pearsonr(x, y)[0]
spearmanr = scipy.stats.spearmanr(x, y)[0]
ax6.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax7 = fig.add_subplot(2,4,7)
x =  np.array(EI_vrm_1)
y = np.array(test_acc)
sns.regplot(x=x, y=y, ax=ax7, scatter=True, color='green', scatter_kws={'s':1})
ax7.set_ylabel('')
ax7.set_xlabel('EI_vrm_1')
# ax5.legend()
print(scipy.stats.pearsonr(x, y)[0])
print(scipy.stats.spearmanr(x, y)[0])
pearsonr = scipy.stats.pearsonr(x, y)[0]
spearmanr = scipy.stats.spearmanr(x, y)[0]
ax7.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax8 = fig.add_subplot(2,4,8)
x =  np.array(EI_vrm_all)
y = np.array(test_acc)
sns.regplot(x=x, y=y, ax=ax8, scatter=True, color='green', scatter_kws={'s':1})
ax8.set_ylabel('')
ax8.set_xlabel('EI_vrm_all')
# ax5.legend()
print(scipy.stats.pearsonr(x, y)[0])
print(scipy.stats.spearmanr(x, y)[0])
pearsonr = scipy.stats.pearsonr(x, y)[0]
spearmanr = scipy.stats.spearmanr(x, y)[0]
ax8.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')


plt.subplots_adjust(wspace=0.3, hspace=0.5)


# plt.savefig('./AffectRaf_measureV6_acc.jpg', format='jpg', bbox_inches='tight')
plt.savefig('./imagenet_a_out_rotation.jpg', format='jpg', bbox_inches='tight')
# plt.savefig('./aus_metrics_loss.svg', format='svg', bbox_inches='tight')

# 绘制曲面
from matplotlib import cm
fig2 = plt.figure(figsize=plt.figaspect(0.4))

ax1 = fig2.add_subplot(241, projection='3d')

X = np.linspace(0, 1, 50)
Y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(X, Y)
Z = 1-X*Y
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_zlabel('')
ax1.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
ax1.view_init(15,15)

ax2 = fig2.add_subplot(242, projection='3d')
Z = X*(1-Y)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_zlabel('')
ax2.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
ax2.view_init(15,15)

ax3 = fig2.add_subplot(243, projection='3d')
Z = np.divide((X*(1-Y)+Y*(1-X)), X+Y+1e-8)
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_zlabel('')
ax3.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
ax3.view_init(15,15)

ax4 = fig2.add_subplot(244, projection='3d')
Z = ((1-Y)+(1-X))
ax4.set_xlabel('S1')
ax4.set_ylabel('S2')
ax4.set_zlabel('score')
ax4.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
ax4.view_init(15,15)


ax5 = fig2.add_subplot(245, projection='3d')
distance1 = 1 - X
distance2 = 1 - Y
sigma = 0.7
Z = np.divide((np.exp( -np.abs(distance1) ** 2 / (2 * sigma ** 2)))*(1-Y) + (np.exp(-np.abs(distance2) ** 2 / (2 * sigma ** 2)))*(1-X), (np.exp(-np.abs(distance1) ** 2 / (2 * sigma ** 2)))+(np.exp(-np.abs(distance2) ** 2 / (2 * sigma ** 2))))
ax5.set_xlabel('S1')
ax5.set_ylabel('S2')
ax5.set_zlabel('score')
ax5.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
ax5.view_init(15,15)

ax6 = fig2.add_subplot(246, projection='3d')
distance1 = 1 - X
distance2 = 1 - Y
sigma = 0.5
Z = np.divide((np.exp( -np.abs(distance1) ** 2 / (2 * sigma ** 2)))*(1-Y) + (np.exp(-np.abs(distance2) ** 2 / (2 * sigma ** 2)))*(1-X), (np.exp(-np.abs(distance1) ** 2 / (2 * sigma ** 2)))+(np.exp(-np.abs(distance2) ** 2 / (2 * sigma ** 2))))
ax6.set_xlabel('S1')
ax6.set_ylabel('S2')
ax6.set_zlabel('score')
ax6.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
ax6.view_init(15,15)


##高斯核函数 np.exp(-dists ** 2 / (2 * sigma ** 2))
ax7 = fig2.add_subplot(247, projection='3d')
distance1 = 1 - X
distance2 = 1 - Y
sigma = 0.3
Z = np.divide((np.exp( -np.abs(distance1) ** 2 / (2 * sigma ** 2)))*(1-Y) + (np.exp(-np.abs(distance2) ** 2 / (2 * sigma ** 2)))*(1-X), (np.exp(-np.abs(distance1) ** 2 / (2 * sigma ** 2)))+(np.exp(-np.abs(distance2) ** 2 / (2 * sigma ** 2))))
# Z = np.exp(-np.abs(X-Y) ** 2 / (2 * 1 ** 2))
ax7.set_xlabel('X')
ax7.set_ylabel('Y')
ax7.set_zlabel('score')
ax7.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
ax7.view_init(15,0)

# ##高斯核函数 np.exp(-dists ** 2 / (2 * sigma ** 2))
# ax8 = fig2.add_subplot(247, projection='3d')
# distance1 = 1 - X
# distance2 = 1 - Y
# sigma = 0.01
# Z = np.divide((np.exp( -np.abs(distance1) ** 2 / (2 * sigma ** 2)))*(1-Y) + (np.exp(-np.abs(distance2) ** 2 / (2 * sigma ** 2)))*(1-X), (np.exp(-np.abs(distance1) ** 2 / (2 * sigma ** 2)))+(np.exp(-np.abs(distance2) ** 2 / (2 * sigma ** 2))))
# # Z = np.exp(-np.abs(X-Y) ** 2 / (2 * 1 ** 2))
# ax8.set_xlabel('X')
# ax8.set_ylabel('Y')
# ax8.set_zlabel('score')
# ax8.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
# # ax7.view_init(15,45)


plt.subplots_adjust( wspace=0.5, hspace=0.2)

plt.savefig('./integral_surface.jpg', format='jpg', bbox_inches='tight')

