import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

import scipy.stats
import seaborn as sns

test_acc0=[]
test_acc = []
test_acc2 = []
test_loss = []
p_overlap_mean = []
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
aus_PNoverlap = []
ensemble_PNoverlap = [] 
aus_base = [] 
wj_score = [] 
I_overlap_score = [] 
pnI_overlap_socre = []
ausP_EI = []
ausPN_EI = []
wjPlus_score = []


dir = '/home/liuyuchi/afs_get/FER_measure/RafRaf_measuresV5'
dir2= '/home/liuyuchi/afs_get/FER_measure/RafAffect_measuresV5'

tests = os.listdir(dir)
tests2 = os.listdir(dir2)
for test in tests:
    try:
        dic = np.load(os.path.join(dir, test), allow_pickle='TRUE')
        dic2 = np.load(os.path.join(dir2, test), allow_pickle='TRUE')
    except:
        continue
    # if dic.item().get('test_acc') <= 0.72:
    #     continue
    # if dic.item().get('test_acc') < 0.15 :
    #     print(test)
    #     continue
    if math.isnan(dic.item().get('aus_PNoverlap')):
        print(test)
        continue
    
    test_acc0.append(dic.item().get('test_acc'))
    test_acc.append(dic2.item().get('test_acc'))
    test_acc2.append(dic2.item().get('test_acc'))

    p_overlap.append(dic.item().get('p_overlap'))
    pn_overlap.append(dic.item().get('pn_overlap')) 
    aus_overlap.append(dic.item()['aus_overlap']) 
    aus_PNoverlap.append(dic.item().get('aus_PNoverlap'))
    ensemble_PNoverlap.append(dic.item().get('ensemble_PNoverlap'))
    aus_base.append(dic.item().get('aus_base')) 
    wj_score.append(dic.item().get('wj_score')) 
    I_overlap_score.append(dic.item().get('I_overlap_score') )
    pnI_overlap_socre.append(dic.item().get('pnI_overlap_score'))
    ausP_EI.append(dic.item().get('ausP_EI'))
    ausPN_EI.append(dic.item().get('ausPN_EI'))
    wjPlus_score.append(dic.item().get('wjPlus_score'))

fig = plt.figure(figsize=plt.figaspect(0.5))

ax1 = fig.add_subplot(2,4,1)
x = np.array(wj_score)
y = test_acc
sns.regplot(x=x, y=y, ax=ax1, scatter=True, color='green', scatter_kws={'s':1})
# ax1.scatter(x, y, c='blue', s=1)
ax1.set_ylabel('acc')
ax1.set_xlabel('Effective Inviarance')
print(scipy.stats.pearsonr(wj_score, test_acc)[0])
print(scipy.stats.spearmanr(wj_score, test_acc)[0])
pearsonr = scipy.stats.pearsonr(wj_score, test_acc)[0]
spearmanr = scipy.stats.spearmanr(wj_score, test_acc)[0]
ax1.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax2 = fig.add_subplot(2,4,2)
sns.regplot(x=aus_PNoverlap, y=test_acc, ax=ax2, scatter=True, color='green', scatter_kws={'s':1})
# ax2.scatter(aus_PNoverlap, test_acc, c='blue', s=1)
# ax4.scatter(aus_invariance_mean, test_loss, c='red', s=1.2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax2.set_ylabel('acc')
ax2.set_xlabel('aus_PNoverlap')
# ax4.legend()
index_without_nan = [i for i in range(len(aus_PNoverlap)) if np.isnan(aus_PNoverlap[i]) == False]
print(scipy.stats.pearsonr([aus_PNoverlap[i] for i in index_without_nan], [test_acc[i] for i in index_without_nan])[0])
print(scipy.stats.spearmanr([aus_PNoverlap[i] for i in index_without_nan], [test_acc[i] for i in index_without_nan])[0])
pearsonr = scipy.stats.pearsonr([aus_PNoverlap[i] for i in index_without_nan], [test_acc[i] for i in index_without_nan])[0]
spearmanr = scipy.stats.spearmanr([aus_PNoverlap[i] for i in index_without_nan], [test_acc[i] for i in index_without_nan])[0]
ax2.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax3 = fig.add_subplot(2,4,3)
sns.regplot(x=I_overlap_score, y=test_acc, ax=ax3, scatter=True, color='green', scatter_kws={'s':1})
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax3.set_ylabel('acc')
ax3.set_xlabel('I_overlap_score')
print(scipy.stats.pearsonr(I_overlap_score, test_acc)[0])
print(scipy.stats.spearmanr(I_overlap_score, test_acc)[0])
pearsonr = scipy.stats.pearsonr(I_overlap_score, test_acc)[0]
spearmanr = scipy.stats.spearmanr(I_overlap_score, test_acc)[0]
ax3.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax4 = fig.add_subplot(2,4,4)
sns.regplot(x=ausP_EI, y=test_acc, ax=ax4, scatter=True, color='green', scatter_kws={'s':1})
# ax4.scatter(ausPN_EI, test_acc, c='blue', s=1)
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax4.set_ylabel('acc')
ax4.set_xlabel('ausP_EI')
# ax5.legend()
print(scipy.stats.pearsonr(ausP_EI, test_acc2)[0])
print(scipy.stats.spearmanr(ausP_EI, test_acc2)[0])
pearsonr = scipy.stats.pearsonr(ausP_EI, test_acc2)[0]
spearmanr = scipy.stats.spearmanr(ausP_EI, test_acc2)[0]
ax4.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax5 = fig.add_subplot(2,4,5)
# ax5.scatter(ensemble_PNoverlap, test_acc, c='blue', s=1)
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax5.set_ylabel('acc')
ax5.set_xlabel('ensemble_PNoverlap')
index_without_nan = [i for i in range(len(ensemble_PNoverlap)) if np.isnan(ensemble_PNoverlap[i]) == False]
print(scipy.stats.pearsonr([ensemble_PNoverlap[i] for i in index_without_nan], [test_acc[i] for i in index_without_nan])[0])
print(scipy.stats.spearmanr([ensemble_PNoverlap[i] for i in index_without_nan], [test_acc[i] for i in index_without_nan])[0])
x= [ensemble_PNoverlap[i] for i in index_without_nan]
y= [test_acc[i] for i in index_without_nan]
sns.regplot(x=x, y=y, ax=ax5, scatter=True, color='green', scatter_kws={'s':1})

pearsonr = scipy.stats.pearsonr([ensemble_PNoverlap[i] for i in index_without_nan], [test_acc[i] for i in index_without_nan])[0]
spearmanr = scipy.stats.spearmanr([ensemble_PNoverlap[i] for i in index_without_nan], [test_acc[i] for i in index_without_nan])[0]
ax5.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax6 = fig.add_subplot(2,4,6)
sns.regplot(x=pnI_overlap_socre, y=test_acc, ax=ax6, scatter=True, color='green', scatter_kws={'s':1})
# ax6.scatter(pnI_overlap_socre, test_acc, c='blue', s=1)
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax6.set_ylabel('acc')
ax6.set_xlabel('pnI_overlap_socre')
print(scipy.stats.pearsonr(pnI_overlap_socre, test_acc2)[0])
print(scipy.stats.spearmanr(pnI_overlap_socre, test_acc2)[0])
pearsonr = scipy.stats.pearsonr(pnI_overlap_socre, test_acc2)[0]
spearmanr = scipy.stats.spearmanr(pnI_overlap_socre, test_acc2)[0]
ax6.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

ax7 = fig.add_subplot(2,4,7)
# sns.regplot(x=wjPlus_score, y=test_acc, ax=ax7, scatter=True, color='green', scatter_kws={'s':1})
# ax7.scatter(wjPlus_score, test_acc, c='blue', s=1)
x = np.array(wjPlus_score)
y = test_acc
sns.regplot(x=x, y=y, ax=ax7, scatter=True, color='green', scatter_kws={'s':1})
ax7.scatter(x, y, c='brown', s=1)
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax7.set_ylabel('acc')
ax7.set_xlabel('wjPlus_score')
# ax5.legend()
print(scipy.stats.pearsonr(wjPlus_score, test_acc2)[0])
print(scipy.stats.spearmanr(wjPlus_score, test_acc2)[0])
pearsonr = scipy.stats.pearsonr(wjPlus_score, test_acc2)[0]
spearmanr = scipy.stats.spearmanr(wjPlus_score, test_acc2)[0]
ax7.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')


ax8 = fig.add_subplot(2,4,8)
# sns.regplot(x=wjPlus_score, y=test_acc, ax=ax7, scatter=True, color='green', scatter_kws={'s':1})
# ax7.scatter(wjPlus_score, test_acc, c='blue', s=1)
x = np.array(test_acc0)
y = test_acc2
sns.regplot(x=x, y=y, ax=ax8, scatter=True, color='green', scatter_kws={'s':1})
ax8.scatter(x, y, c='brown', s=1)
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax8.set_ylabel('acc')
ax8.set_xlabel('IID_OOD')
# ax5.legend()
print(scipy.stats.pearsonr(test_acc0, test_acc2)[0])
print(scipy.stats.spearmanr(test_acc0, test_acc2)[0])
pearsonr = scipy.stats.pearsonr(test_acc0, test_acc2)[0]
spearmanr = scipy.stats.spearmanr(test_acc0, test_acc2)[0]
ax8.set_title(r'$\rho=%.2f,\gamma$=%.2f'%(pearsonr,spearmanr))
print('\n')

# ax8 = fig.add_subplot(1,8,8)
# ax8.scatter(ausPN_EI, test_acc, c='blue', s=1)
# # ax5.scatter(integral_mean, test_loss, c='red', s=2)
# # ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
# ax8.set_ylabel('acc')
# ax8.set_xlabel('ausPN_EI')
# # ax5.legend()
# print(scipy.stats.pearsonr(ausPN_EI, test_acc)[0])
# print(scipy.stats.spearmanr(ausPN_EI, test_acc)[0])
# print('\n')


plt.subplots_adjust(wspace=0.3, hspace=0.5)


plt.savefig('./RafAffect_measureV7_acc.jpg', format='jpg', bbox_inches='tight')
# plt.savefig('./aus_metrics_loss.svg', format='svg', bbox_inches='tight')

# # 绘制曲面
# from matplotlib import cm
# fig2 = plt.figure(figsize=plt.figaspect(0.25))

# ax1 = fig2.add_subplot(141, projection='3d')

# X = np.linspace(0, 1, 50)
# Y = np.linspace(0, 1, 50)
# X, Y = np.meshgrid(X, Y)
# Z = 2*(2*X*Y - X - Y + 0.5)
# ax1.set_xlabel('s_au')
# ax1.set_ylabel('O')
# ax1.set_zlabel('score')
# ax1.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)


# ax2 = fig2.add_subplot(142, projection='3d')
# Z = 2*X*Y - X - Y + 1
# ax2.set_xlabel('s_au')
# ax2.set_ylabel('O')
# ax2.set_zlabel('score')
# ax2.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

# ax3 = fig2.add_subplot(143, projection='3d')
# Z = -2*X*Y + X + Y
# ax3.set_xlabel('s_au')
# ax3.set_ylabel('O')
# ax3.set_zlabel('score')
# ax3.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

# ax4 = fig2.add_subplot(144, projection='3d')
# Z = (2*X*Y - X - Y + 1) + (-2*X*Y + X + Y)
# ax4.set_xlabel('s_au')
# ax4.set_ylabel('O')
# ax4.set_zlabel('score')
# ax4.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

# # plt.subplots_adjust(wspace=0.2, hspace=0.2)
# # ax.set_zlim(0, 1)

# plt.savefig('./integral_surface.jpg', format='jpg', bbox_inches='tight')

