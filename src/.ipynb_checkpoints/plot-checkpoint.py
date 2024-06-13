import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


test_acc = []
test_loss = []
confidence_mean = []
sp_mean = []
invariance_mean = []
aus_invariance_mean = []
integral_mean = []
aus_guided = []
ensemble = []
wj_score = []
colors_options = ['blue', 'red', 'green', 'orange']
colors = []

log_file0 = './log/R18MS1M_raf_invariance_hard/log.txt'
log_file1 = './log/R18_Affect_Affect_integralV2/log.txt'
log_file2 = './log/R18_Affect_Affect_integralV2_noAug/log.txt'
log_file3 = './log/R18_Affect_Affect_integralV2_noAug/log.txt'
log_file4 = './log/R18_Raf_Raf_integralV3/log.txt'

log_file5 = './log/R18_Affect_Affect_Grayscale_test/log.txt'
log_file6 = './log/R18_Affect_Raf_RandomErase_test/log.txt'
log_file7 = './log/R18_Affect_Raf_Gaussian_test/log.txt'
log_file8 = './log/R18_Affect_Raf_noAug_test/log.txt'


files = [log_file6, log_file7, log_file8]
# files = [log_file5]

file_ctn = 0
for file in files:
    print(file)
    file_ctn+=1
    color = colors_options[file_ctn-1]
    with open (file, 'r') as rf:
        lines = rf.readlines()
        lines = lines[:-2]
        print(len(lines))
        for l in lines:
            l = l.strip()
            l = l.split(' ')
            print(l)
            if len(l)<4:
                continue
            else:
                test_acc.append(float(l[1]))
                test_loss.append(float(l[2]))

                # confidence_mean.append(float(l[5]))
                # sp_mean.append(l[6])
                invariance_mean.append(float(l[7]))
                aus_invariance_mean.append(float(l[8]))
                aus_guided.append(float(l[9])*2)
                ensemble.append(float(l[10])*2)
                wj_score.append(float(l[12]))
                colors.append(color)
                print(colors)


fig = plt.figure(figsize=plt.figaspect(1.5))

# ax1 = fig.add_subplot(4,1,1)
# ax1.scatter(confidence_mean, test_acc, c='red', label='confidence_mean')
# ax1.set_ylim(125,170)
# ax1.legend()
#
# ax2 = fig.add_subplot(4,1,2)
# ax2.scatter(sp_mean, test_acc, c='red', label='sp_mean')
# ax2.legend()

ax3 = fig.add_subplot(3,1,1)
ax3.scatter(wj_score, test_acc, c=colors, s=2)
# ax3.scatter(wj_score, test_loss, c='red', s=2)
# ax3.scatter(invariance_mean, test_loss, c='blue', s=0.3, label='invariance_mean')
ax3.set_ylabel('acc')
ax3.set_xlabel('Effective Inviarance')
# ax3.legend()

# ax4 = fig.add_subplot(3,1,2)
# ax4.scatter(aus_invariance_mean, test_acc, c='red', s=2)
# # ax4.scatter(aus_invariance_mean, test_loss, c='red', s=1.2)
# # ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
# ax4.set_ylabel('acc')
# ax4.set_xlabel('Aus_Invariance')
# # ax4.legend()

ax5 = fig.add_subplot(3,1,3)
ax5.scatter(aus_guided, test_acc, c=colors, s=2)
# ax5.scatter(integral_mean, test_loss, c='red', s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax5.set_ylabel('acc')
ax5.set_xlabel('AUs Guided')
# ax5.legend()

ax6 = fig.add_subplot(3,1,2)
ax6.scatter(aus_invariance_mean, test_acc, c=colors, s=2)
# ax4.scatter(aus_invariance_mean, test_loss, c='red', s=1.2)
# ax4.scatter(aus_invariance_mean, test_loss, c='blue', s=0.3, label='aus_invariance_mean')
ax6.set_ylabel('acc')
ax6.set_xlabel('Ensemble')
# ax4.legend()

print(invariance_mean)
print(aus_invariance_mean)
print(test_acc)

plt.subplots_adjust(hspace=0.5)


plt.savefig('./aus_metrics_acc.svg', format='svg', bbox_inches='tight')
# plt.savefig('./aus_metrics_loss.svg', format='svg', bbox_inches='tight')

# 绘制曲面
from matplotlib import cm
fig2 = plt.figure(figsize=plt.figaspect(0.25))

ax1 = fig2.add_subplot(141, projection='3d')

X = np.linspace(0, 1, 50)
Y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(X, Y)
Z = 2*(2*X*Y - X - Y + 0.5)
ax1.set_xlabel('s_au')
ax1.set_ylabel('O')
ax1.set_zlabel('score')
ax1.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)



# X2 = np.linspace(0.2, 0.4, 20)
# Y2 = np.linspace(0, 0.6, 20)
# X2, Y2 = np.meshgrid(X2, Y2)
# Z2 = 2*(2*X2*Y2 - X2 - Y2 + 0.5)
# ax1.set_xlabel('s_au')
# ax1.set_ylabel('O')
# ax1.set_zlabel('score')
# ax1.plot_surface(X2, Y2, Z2, cmap=plt.cm.PuOr)

ax2 = fig2.add_subplot(142, projection='3d')
Z = 2*X*Y - X - Y + 1
ax2.set_xlabel('s_au')
ax2.set_ylabel('O')
ax2.set_zlabel('score')
ax2.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

ax3 = fig2.add_subplot(143, projection='3d')
Z = -2*X*Y + X + Y
ax3.set_xlabel('s_au')
ax3.set_ylabel('O')
ax3.set_zlabel('score')
ax3.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

ax4 = fig2.add_subplot(144, projection='3d')
Z = (2*X*Y - X - Y + 1) + (-2*X*Y + X + Y)
ax4.set_xlabel('s_au')
ax4.set_ylabel('O')
ax4.set_zlabel('score')
ax4.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

# plt.subplots_adjust(wspace=0.2, hspace=0.2)
# ax.set_zlim(0, 1)

plt.savefig('./integral_surface.svg', format='svg', bbox_inches='tight')

